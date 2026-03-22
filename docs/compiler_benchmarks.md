# Compiler Execution Benchmarks

Hardware: NVIDIA GeForce RTX 4060 (SM 8.9, 8188 MB), x86_64
Model: Qwen2.5-0.5B (24 layers, hidden=896, int8 quantized, FP16 compute)

## Baseline: Interpreted vs Compiled (no optimizations)

Compiled path 每个 op 独立分配 Tensor（~270 次 cudaMalloc per forward）。

| Prompt | Mode | Prefill tok/s | Decode tok/s | Decode avg latency |
|--------|------|--------------|-------------|-------------------|
| 1 tok  | Interpreted | 2.46 | **597.5** | 1.7 ms |
| 1 tok  | Compiled | 2.49 | 236.7 | 4.2 ms |
| 4 tok  | Interpreted | 10.06 | **481.5** | 2.1 ms |
| 4 tok  | Compiled | 8.93 | 309.2 | 3.2 ms |
| 144 tok | Interpreted | 318.0 | **521.2** | 1.9 ms |
| 144 tok | Compiled | 296.4 | 335.5 | 3.0 ms |

**Compiled decode 比 interpreted 慢 ~36%**，瓶颈在 per-op buffer 分配。

## Pass 1: Memory Planning

Liveness analysis + greedy best-fit buffer reuse。预分配固定 pool，forward 内 0 次分配。

| Prompt | Mode | Prefill tok/s | Decode tok/s | Decode avg latency |
|--------|------|--------------|-------------|-------------------|
| 1 tok  | Interpreted | 2.46 | **597.5** | 1.7 ms |
| 1 tok  | Compiled + MemPlan | 2.63 | 362.8 | 2.8 ms |
| 4 tok  | Interpreted | 9.49 | **479.3** | 2.1 ms |
| 4 tok  | Compiled + MemPlan | 11.08 | 344.1 | 2.9 ms |
| 144 tok | Interpreted | 318.0 | **521.2** | 1.9 ms |
| 144 tok | Compiled + MemPlan | 289.2 | 327.5 | 3.1 ms |

**Memory Planning 效果**: decode +53% over no-plan (237→363)，但仍比 interpreted 慢 ~40%。

### 剩余 gap 分析

1. **Hash map 查找**: 每个 op 查 `slot_index.at(v->id)` + `weights_.at(name)` — O(1) 但常数大
2. **RoPE device_copy**: 编译路径把 Q/K 复制到输出 buffer 再 in-place — interpreted 直接原地改
3. **Switch dispatch**: 533 次 switch-case — interpreted 是紧凑模板循环
4. **Weight 名字查找**: 每次 run 都 string hash lookup — interpreted 是直接指针

## Pass 2: In-Place RoPE + Pass 3: Execution Plan Pre-compilation

两个 pass 一起实现：

- **In-Place RoPE**: RoPE 输出复用输入 buffer slot，消除 48 次 device_copy/forward (24 layers × 2)
- **Pre-compilation**: construction 时遍历 graph 一次，构建扁平 `vector<ResolvedOp>`，
  预解析所有 slot index、weight 指针、op 参数。run 时 zero hash lookup。

| Prompt | Mode | Prefill tok/s | Decode tok/s | Decode avg latency |
|--------|------|--------------|-------------|-------------------|
| 1 tok  | Interpreted | 2.46 | **597.5** | 1.7 ms |
| 1 tok  | Compiled (all passes) | 2.67 | 418.4 | 2.4 ms |
| 4 tok  | Interpreted | 10.17 | **509.5** | 2.0 ms |
| 4 tok  | Compiled (all passes) | 10.74 | 381.3 | 2.6 ms |
| 144 tok | Interpreted | 318.0 | **521.2** | 1.9 ms |
| 144 tok | Compiled (all passes) | 244.2 | 381.4 | 2.6 ms |

### Cumulative improvement (decode, 1 tok prompt)

| Stage | tok/s | vs Baseline | vs Interpreted |
|-------|-------|-------------|----------------|
| Compiled (no opt) | 236.7 | — | -60% |
| + Memory Planning | 362.8 | +53% | -39% |
| + InPlace + Precompile | **418.4** | **+77%** | **-30%** |
| Interpreted | 597.5 | — | — |

### 剩余 gap 分析

Compiled 仍比 interpreted 慢 ~30%。剩余开销：

1. **Logits copy**: 每次 forward 从 pool 拷贝 vocab_size (151936) 个 FP16 值到新 Tensor — interpreted 直接返回
2. **Re-plan check**: 每次 run 检查 seq_len 是否变化（比较 slot sizes） — 可缓存 decode plan
3. **Switch dispatch**: 仍有 ~270 次 switch-case（去掉 Weight/Input 后） — interpreted 是无分支模板循环
4. **Per-token loops**: SiLUMul/RMSNorm 的 per-token 循环有 reinterpret_cast 开销 — interpreted 完全相同

后续优化方向：Op Fusion（减少 kernel launch 数量）才是真正超越 interpreted 的手段。
