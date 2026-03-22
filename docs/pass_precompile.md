# Pass 3: Execution Plan Pre-compilation

## Problem

当前 `Executable::run()` 每次 forward 都要：

1. **遍历 graph nodes**: `for (const auto& node : graph_->nodes())` — 533 个 unique_ptr 的 indirection
2. **Switch dispatch**: 每个 node 做 switch(op) — 分支预测不友好
3. **Hash map 查找**: `plan_.slot_index.at(v->id)` 查 Value→slot 映射 — 每个 buf()/cbuf() 调用一次
4. **Weight string lookup**: `weights_.at(name)` 用 string hash — 每个 Linear/RMSNorm 一次

这些 overhead 在 decode 阶段（seq_len=1，kernel 本身很快）占比大。

## Design

### 核心思想

把 graph walk + switch + hash lookup 从 run-time 移到 construction-time。
预编译出一个扁平的 **instruction list**，每条 instruction 包含执行一个 op 所需的全部信息（函数指针 + buffer 指针 + 参数），不需要任何间接查找。

### Data Structures

```cpp
/// Pre-resolved buffer pointer: 直接指向 pool_[slot_id] 的 data 指针
/// 在 pool 分配后计算一次，每次 run 直接用
struct ResolvedOp {
    OpType op;

    // Pre-resolved buffer pointers (indices into pool_)
    uint32_t out_slot;          // 输出 buffer 的 slot index
    uint32_t in_slots[4];      // 输入 buffer 的 slot indices（最多 4 个）
    uint8_t num_inputs;

    // Pre-resolved weight pointers (not going through hash map)
    const float* weight_fp32;
    const int8_t* weight_int8;
    const float* weight_scales;
    const float* bias;
    bool weight_quantized;

    // Op-specific parameters (pre-extracted from Attributes, no variant lookup)
    union {
        struct { std::size_t N, K; bool has_bias; } linear;
        struct { float eps; std::size_t hidden; } rms_norm;
        struct { std::size_t nh, nkv, hd; float theta; bool inplace_q, inplace_k; } rope;
        struct { std::size_t nh, nkv, hd, layer; } attention;
        struct { std::size_t N, K; } linear_add;
        struct { std::size_t intermediate; } silu_mul;
    } params;
};
```

### Construction-time Compilation

```cpp
// 在 Executable 构造函数中，graph walk 只做一次：
for (const auto& node : graph_->nodes()) {
    ResolvedOp rop;
    rop.op = node->op;
    rop.out_slot = plan_.slot_index.at(out->id);
    // Pre-resolve all inputs to slot indices
    // Pre-resolve weight name → WeightRef pointer
    // Pre-extract all attributes into rop.params
    ops_.push_back(rop);
}
```

### Run-time Execution

```cpp
// run() 变成一个紧凑的循环，无 hash lookup，无 switch：
for (const auto& op : ops_) {
    float* out = pool_[op.out_slot].data<float>();
    // dispatch table: function pointer array indexed by OpType
    dispatch_[static_cast<int>(op.op)](this, op, out, seq_len, pos);
}
```

或者保留 switch 但操作的是扁平的 `ResolvedOp`（无 hash 查找）。

### Expected Impact

| Overhead | Before | After |
|----------|--------|-------|
| slot_index hash lookup | ~1600/forward (533 ops × ~3 buf calls) | **0** (pre-resolved to slot index) |
| weight string hash | ~240/forward | **0** (pre-resolved to pointer) |
| Attribute variant get | ~1000/forward | **0** (pre-extracted to plain fields) |
| unique_ptr indirection | 533/forward | **0** (flat vector) |

总共消除约 3000+ 次 hash map lookup per forward。
对 decode (kernel ~1.5ms total)，这些 lookup 的 overhead 可能有 0.5-1ms。
