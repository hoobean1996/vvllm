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

## Pass 2: In-Place RoPE + Pass 3: Execution Plan Pre-compilation

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

## Pass 4: Plan Caching + Logits Buffer Reuse

- **Plan caching**: 缓存 `compiled_seq_len_`，decode 阶段（seq_len 恒为 1）跳过 re-plan
- **Logits reuse**: 预分配 logits Tensor，每次 forward 复用（消除 per-run Tensor 分配）
- **Non-owning view**: 返回 pool buffer 的 view 而非 copy ownership

| Prompt | Mode | Decode tok/s | Decode avg latency | P50 |
|--------|------|-------------|-------------------|-----|
| 4 tok, 100 gen | Interpreted | **535.0** | 1.9 ms | 1.7 ms |
| 4 tok, 100 gen | Compiled (final) | 450.6 | 2.2 ms | 2.0 ms |

## Cumulative Improvement (decode)

| Stage | tok/s | vs Baseline | vs Interpreted |
|-------|-------|-------------|----------------|
| Compiled (no opt) | 236.7 | — | -60% |
| + Memory Planning | 362.8 | +53% | -39% |
| + InPlace + Precompile | 418.4 | +77% | -30% |
| + Plan Cache + Logits Reuse | **450.6** | **+90%** | **-16%** |
| Interpreted | 535.0 | — | — |

## Conclusions

### 为什么 compiled 没有超过 interpreted？

1. **Interpreted 本身就是最优的 "compiled" 代码**：`transformer_forward()` 是 C++ 模板，
   编译器内联所有调用，7 个 scratch buffer 手动复用，直接指针访问，零运行时开销。
   手写的 C++ 模板代码就是最好的 codegen。

2. **计算热点在 cuBLAS 里**：Linear（matmul）占 95%+ 的计算时间，两条路径调用的是
   完全相同的 cuBLAS Tensor Core kernel。编译路径无法改变 cuBLAS 的执行效率。

3. **Op Fusion 受限于 cuBLAS**：RMSNorm + Linear 融合需要自己写 WMMA kernel 替代 cuBLAS，
   工程量大且性能不一定更好。SiLUMul + Linear 同理。

4. **剩余 overhead 是结构性的**：switch dispatch (~270 次/forward)、logits copy
   (vocab_size=151936 FP16 values)、per-run 的 seq_len check — 这些是编译路径
   graph-walking 架构的固有开销，interpreted 的紧凑循环没有。

### 编译模式的价值

虽然在当前场景（0.5B 模型、单 request、cuBLAS backend）下 compiled 没有超过 interpreted，
但编译模式的架构为以下场景打下了基础：

- **自定义 fused kernel**：如果写了 fused RMSNorm+GEMM WMMA kernel，compiled 是唯一能插入它的地方
- **更大模型**：graph-level memory planning 比手写 scratch buffer 更灵活、更节省显存
- **Batching**：多 request 共享计算图时 memory plan 收益更大
- **跨设备优化**：同一份 IR 可以 codegen 到不同 backend（CPU/CUDA/未来的其他加速器）
- **离线编译**：graph + memory plan 可序列化到磁盘，下次启动秒加载

### 学到的东西

| 组件 | 实现内容 |
|------|---------|
| **IR 设计** | SSA Value/OpNode/Graph，OpType 枚举对应 Backend 接口 |
| **Graph Builder** | 从 ModelConfig 直接建图（不走 trace），支持 Prefill/Decode 两种模式 |
| **Memory Planning** | Liveness analysis + greedy best-fit buffer reuse，+53% decode |
| **In-Place Pass** | 检测 last-use 条件复用 buffer，消除 RoPE 的 48 次 device_copy |
| **Pre-compilation** | `vector<ResolvedOp>` 扁平指令列表，消除 run-time hash lookup |
| **Plan Caching** | 缓存 seq_len→plan 映射，跳过重复 plan_memory() |
| **Codegen** | 遍历 ResolvedOp 列表 dispatch 到 Backend 方法，zero-alloc forward |
