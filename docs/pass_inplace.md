# Pass 2: In-Place Op Rewrite

## Problem

RoPE 当前实现需要额外的 `device_copy`：先把 Q/K 从 Linear 输出 buffer 拷贝到 RoPE 输出 buffer，
再 in-place 修改。这是因为 SSA 语义要求每个 Value 有独立的 buffer。

但如果 Linear 的输出 **只被 RoPE 消费**（last_use == RoPE node），那 RoPE 可以直接写入 Linear 的 buffer，省掉拷贝。

## Design

### In-Place 条件

一个 op 的输入 Value 可以被原地覆盖（in-place rewrite），当且仅当：

1. 该 Value 的 **last use** 就是当前 op（之后没人需要它了）
2. 该 Value 的 buffer 大小 **>= op 输出的 buffer 大小**（不会越界）

满足这两个条件时，op 的输出可以直接复用输入的 buffer slot（在 memory plan 中指向同一个 slot）。

### 受益的 Op

| Op | In-Place 输入 | 效果 |
|----|-------------|------|
| **RoPE** | Q (from q_proj Linear), K (from k_proj Linear) | 消除 2 次 device_copy per layer = **48 次/forward** |

RoPE 的 Q 输入只被 RoPE 消费（之后用的是 RoPE 的输出 Q'），K 同理。
Buffer 大小完全相同（seq_len * num_heads * head_dim）。完美满足 in-place 条件。

### Implementation

修改 `plan_memory()`: 检测 in-place 条件，当满足时让输出 Value 的 slot_index 直接指向输入 Value 的 slot。

修改 `Executable::run()` 的 RoPE case: 当检测到输入和输出共享 slot 时，跳过 `device_copy`，直接在输入 buffer 上 in-place 执行 `backend_.rope()`。

```
// Before (memory plan treats Q_in and Q_out as different slots):
slot[3] = Q (from Linear)
slot[7] = Q' (RoPE output)    ← device_copy(slot[7], slot[3])
                                 rope(slot[7], ...)

// After (in-place: Q_out reuses Q_in's slot):
slot[3] = Q (from Linear)     ← rope(slot[3], ...)
                                 Q' is now also slot[3]
```

### Expected Impact

- 消除 48 次 device_copy per forward (24 layers × 2 copies)
- 减少 2 个 buffer slot（Q_out 和 K_out 不再需要独立 slot）
- Decode 阶段每次 copy 约 896*2=1792 bytes (FP16)，开销主要在 kernel launch latency
