# vvllm Compiler IR Design

## Motivation

当前 vvllm 采用解释执行：`transformer_forward()` 逐个调用 `Backend` 虚函数，每个算子立即执行。
这导致：
- 每个算子独立 launch CUDA kernel，overhead 累积（24 层 × ~20 ops = ~480 次 kernel launch）
- 中间结果反复写回显存再读出，浪费内存带宽
- 无法跨算子优化（如 RMSNorm + Linear 融合）

编译执行模式：先构建完整计算图（IR），再做优化 pass，最后生成优化后的执行计划。

## Design Principles

1. **SSA（Static Single Assignment）**：每个 Value 只有一个 producer，不可变
2. **从模型配置直接建图**：不走 trace，直接从 `ModelConfig` + weight names 构建图结构
3. **LLM 推理专用**：不支持训练、不支持动态 control flow，图是静态 DAG
4. **Prefill / Decode 分离**：两种 workload 各自构建独立的图

## Core Data Structures

### DType & Shape

```cpp
// 复用 vvllm::DType (Float32, Float16, Int8)
using Shape = std::vector<std::size_t>;
```

### Value（SSA 值）

每个中间结果是一个 `Value`，定义一次不可变。

```cpp
struct Value {
    uint32_t id;            // %0, %1, %2... 全图唯一
    DType dtype;            // 数据类型
    Shape shape;            // 张量形状，如 {seq_len, hidden_size}
    OpNode* producer;       // 产生此值的算子（输入节点为 nullptr）
    vector<OpNode*> users;  // 消费此值的算子列表
};
```

### OpType（算子类型）

直接对应 `Backend` 接口上的算子，加上若干复合算子：

```cpp
enum class OpType {
    // 输入/常量
    Input,          // 模型输入（token embeddings）
    Weight,         // 模型权重引用

    // 基础算子（对应 Backend 方法）
    Embedding,      // embedding lookup
    RMSNorm,        // RMS normalization
    Linear,         // out = inp @ W^T + bias（含 FP32 和 INT8 两种）
    RoPE,           // rotary position embedding（in-place on Q, K）
    Attention,      // scaled dot-product attention with causal mask + KV cache
    SiLUMul,        // silu(gate) * up，fused activation
    Add,            // element-wise residual add
    Softmax,        // softmax（Attention 内部使用，也可独立出现）

    // 融合算子（优化 pass 产生）
    LinearAdd,      // linear + residual add（已有 backend 支持）
    // 未来可扩展：FusedRMSNormLinear, FusedAttention 等
};
```

### OpNode（算子节点）

```cpp
struct OpNode {
    uint32_t id;                // 节点编号
    OpType op;                  // 算子类型
    std::vector<Value*> inputs; // 输入值列表
    std::vector<Value*> outputs;// 输出值列表（大多数算子只有一个输出）
    Attributes attrs;           // 算子参数（如 eps, num_heads, weight_name 等）
};
```

### Attributes（算子属性）

```cpp
// 用 variant 存储不同类型的属性值
using AttrValue = std::variant<int64_t, double, std::string, bool>;
using Attributes = std::unordered_map<std::string, AttrValue>;
```

常用属性：
| 算子 | 属性 |
|------|------|
| Embedding | `hidden_size`, `vocab_size` |
| RMSNorm | `eps` |
| Linear | `M`, `N`, `K`, `has_bias`, `quantized` |
| RoPE | `num_heads`, `num_kv_heads`, `head_dim`, `theta` |
| Attention | `num_heads`, `num_kv_heads`, `head_dim`, `scale` |
| SiLUMul | `intermediate_size` |
| Weight | `name`（权重名，如 `"layers.0.q_proj.weight"`） |

### Graph（计算图）

```cpp
struct Graph {
    std::vector<std::unique_ptr<OpNode>> nodes;   // 所有节点（拓扑序）
    std::vector<std::unique_ptr<Value>> values;   // 所有 SSA 值
    std::vector<Value*> inputs;                   // 图输入
    std::vector<Value*> outputs;                  // 图输出（logits）

    // 建图辅助方法
    Value* create_value(Shape shape, DType dtype);
    Value* add_op(OpType op, std::vector<Value*> inputs, Shape out_shape,
                  DType out_dtype, Attributes attrs = {});
    Value* add_weight(std::string name, Shape shape, DType dtype);
};
```

## Graph Construction: `build_transformer_graph()`

从 `ModelConfig` 直接构建图，不走 trace。每种模型架构一个 builder。

### 输入

- `ModelConfig`：包含 `num_hidden_layers`, `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads` 等
- `Mode`：`Prefill`（seq_len = 动态）或 `Decode`（seq_len = 1）
- `seq_len`：具体序列长度

### 构建流程

```
Input(token_ids)
  │
  ▼
Embedding ──→ x: [seq_len, hidden]
  │
  ▼
╔══════════════════════════════════════════╗
║  for layer_idx in 0..num_hidden_layers:  ║
║                                          ║
║  ┌─ RMSNorm(x, attn_norm_weight)        ║
║  │                                       ║
║  ├─ Linear(norm, q_proj) → Q            ║
║  ├─ Linear(norm, k_proj) → K            ║
║  ├─ Linear(norm, v_proj) → V            ║
║  │                                       ║
║  ├─ RoPE(Q, K) → Q', K'                ║
║  │                                       ║
║  ├─ Attention(Q', K', V) → attn_out    ║
║  │                                       ║
║  ├─ LinearAdd(attn_out, o_proj, x) → x ║
║  │                                       ║
║  ├─ RMSNorm(x, ffn_norm_weight)        ║
║  │                                       ║
║  ├─ Linear(norm, gate_up_proj)          ║
║  ├─ SiLUMul(gate_up) → activated       ║
║  │                                       ║
║  └─ LinearAdd(activated, down_proj, x)  ║
║       → x                               ║
╚══════════════════════════════════════════╝
  │
  ▼
RMSNorm(x, final_norm_weight)
  │
  ▼
Linear(norm, embed_tokens) → logits
```

### Weight Naming Convention

权重节点通过 name 属性引用实际权重数据，命名遵循 safetensors 约定：

```
model.embed_tokens.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.self_attn.q_proj.weight
model.layers.{i}.self_attn.q_proj.bias        (Qwen2 only)
model.layers.{i}.self_attn.k_proj.weight
model.layers.{i}.self_attn.k_proj.bias        (Qwen2 only)
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.v_proj.bias        (Qwen2 only)
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.mlp.gate_up_proj.weight      (fused gate + up)
model.layers.{i}.mlp.down_proj.weight
model.layers.{i}.post_attention_layernorm.weight
model.norm.weight
```

### IR Dump Example

对 Qwen2-0.5B (24 layers, hidden=896) 的第一个 layer，IR 大概如下：

```
%0 = Input() -> [seq, 896]
%1 = Weight("model.embed_tokens.weight") -> [151936, 896]
%2 = Embedding(%0, %1) {hidden_size=896, vocab_size=151936} -> [seq, 896]
// --- Layer 0: Attention ---
%3 = Weight("model.layers.0.input_layernorm.weight") -> [896]
%4 = RMSNorm(%2, %3) {eps=1e-6} -> [seq, 896]
%5 = Weight("model.layers.0.self_attn.q_proj.weight") -> [896, 896]
%6 = Weight("model.layers.0.self_attn.q_proj.bias") -> [896]
%7 = Linear(%4, %5, %6) {M=seq, N=896, K=896} -> [seq, 896]
%8 = Weight("model.layers.0.self_attn.k_proj.weight") -> [128, 896]
%9 = Weight("model.layers.0.self_attn.k_proj.bias") -> [128]
%10 = Linear(%4, %8, %9) {M=seq, N=128, K=896} -> [seq, 128]
%11 = Weight("model.layers.0.self_attn.v_proj.weight") -> [128, 896]
%12 = Weight("model.layers.0.self_attn.v_proj.bias") -> [128]
%13 = Linear(%4, %11, %12) {M=seq, N=128, K=896} -> [seq, 128]
%14 = RoPE(%7, %10) {num_heads=14, num_kv_heads=2, head_dim=64, theta=1e6} -> [seq, 896], [seq, 128]
%15 = Attention(%14.q, %14.k, %13) {num_heads=14, num_kv_heads=2, head_dim=64} -> [seq, 896]
%16 = Weight("model.layers.0.self_attn.o_proj.weight") -> [896, 896]
%17 = LinearAdd(%15, %16, %2) {M=seq, N=896, K=896} -> [seq, 896]
// --- Layer 0: MLP ---
%18 = Weight("model.layers.0.post_attention_layernorm.weight") -> [896]
%19 = RMSNorm(%17, %18) {eps=1e-6} -> [seq, 896]
%20 = Weight("model.layers.0.mlp.gate_up_proj.weight") -> [9728, 896]
%21 = Linear(%19, %20) {M=seq, N=9728, K=896} -> [seq, 9728]
%22 = SiLUMul(%21) {intermediate_size=4864} -> [seq, 4864]
%23 = Weight("model.layers.0.mlp.down_proj.weight") -> [896, 4864]
%24 = LinearAdd(%22, %23, %17) {M=seq, N=896, K=4864} -> [seq, 896]
// --- Layer 1..23: same pattern ---
// ...
// --- Output ---
%N   = Weight("model.norm.weight") -> [896]
%N+1 = RMSNorm(%last_x, %N) {eps=1e-6} -> [seq, 896]
%N+2 = Linear(%N+1, %1) {M=1, N=151936, K=896} -> [1, 151936]
// Output: %N+2 (logits)
```

## Memory Planning Pass

### Problem

无 memory planning 时，每个 OpNode 的输出分配独立的 GPU buffer。
Qwen2-0.5B 有 533 个 op，每次 forward 触发数百次 cudaMalloc/cudaFree。
对比 interpreted path 只用 7 个 scratch buffer 复用 24 层，overhead 巨大。

### Algorithm: Liveness-Based Greedy Buffer Reuse

核心思想：分析每个 Value 的生命周期（从产生到最后一次被消费），生命周期不重叠的 Value 可以共享同一块 buffer。

**Step 1 — Liveness Analysis**

遍历所有 node（拓扑序），记录每个 Value 的 last use（最后一个消费它的 node 的序号）：

```
last_use[value_id] = max node_index where value appears as input
```

Graph output 标记为 "永不释放"（last_use = ∞）。

**Step 2 — Greedy Best-Fit Allocation**

按拓扑序遍历每个 node：

```
for each node_i in topological order:
    // 1. Free: 释放 last_use == i 的 input value 的 buffer slot
    for each input of node_i:
        if last_use[input] == i:
            free_pool.add(slot_of[input])

    // 2. Allocate: 为 output value 分配 buffer slot
    for each output of node_i:
        needed = compute_buffer_size(output, seq_len)
        slot = free_pool.best_fit(needed)  // 找 >= needed 的最小 slot
        if slot found:
            reuse slot
        else:
            allocate new slot (with size = needed)
        slot_of[output] = slot
```

**Step 3 — Pool Pre-allocation**

`Executable` 构造时根据 plan 预分配固定数量的 Tensor：

```cpp
pool_.resize(plan.num_slots());
for (i = 0; i < num_slots; i++)
    pool_[i] = Tensor({slot_sizes[i]}, dtype, device);
```

每次 `run()` 直接复用 pool，zero allocation。

### Result

对 Qwen2-0.5B (24 layers, 533 ops)：

| | 无 Memory Planning | 有 Memory Planning |
|---|---|---|
| **Buffer 分配数** | ~270 per forward | **预分配固定 pool，forward 内 0 次** |
| **Decode tok/s** | 237 | **363 (+53%)** |

Buffer 复用模式和 interpreted path 的 scratch buffer 等效：同类 op（如每层的 RMSNorm 输出）共享同一个 slot。

### Data Structure

```cpp
struct MemoryPlan {
    unordered_map<uint32_t, uint32_t> slot_index;  // value_id → slot_id
    vector<size_t> slot_sizes;                      // slot_id → max elements
};
```

### Limitations & Future Work

- 当前 plan 是 per-seq_len 的，seq_len 变化时需重新 plan（prefill vs decode 各一次）
- Best-fit 策略简单但够用；更优的做法是 linear scan + interval graph coloring
- 未来可以 plan 一次、覆盖 max_seq_len，避免 re-plan

## Future: Optimization Passes

图构建完成后，后续可以添加的优化 pass：

1. **Op Fusion**: RMSNorm + Linear → FusedRMSNormLinear（省一次显存读写）
2. ~~**Memory Planning**: 静态分配 scratch buffer，复用不再需要的中间值内存~~ ✅ 已实现
3. **Dead Code Elimination**: 删除 users 为空的节点
4. **Constant Folding**: 编译期计算 RoPE frequency table 等常量

## Codegen & Execution

编译路径的完整流水线：

```
ModelConfig + Weights
       │
       ▼
  build_graph()          ✅ 已实现
       │
       ▼
  optimize(graph)        ← no-op placeholder (future: fusion passes)
       │
       ▼
  plan_memory(graph)     ✅ 已实现 — liveness analysis + greedy reuse
       │
       ▼
  Executable(graph)      ✅ 已实现 — pre-allocate pool, walk graph, dispatch to Backend
       │
       ▼
  Executable::run()      ✅ 已实现 — zero-alloc forward pass
```
