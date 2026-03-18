# vvllm

A minimal LLM inference engine built from scratch in C++20. No frameworks, no libraries for the core — just raw C++ to understand how LLM inference actually works.

Currently runs Qwen2.5-0.5B and SmolLM-135M. Supports OpenBLAS for accelerated linear projections, batched GQA attention, INT8 per-channel weight quantization, and a CUDA backend with GPU mirror system, GPU-resident KV cache, fused operators, and FP16 inference for minimal PCIe transfers and maximum throughput.

## Quick Start

```bash
# Build
bazel build //bin/qwen:qwen

# Run (naive CPU backend)
bazel run //bin/qwen:qwen -- --model /path/to/Qwen2.5-0.5B --prompt "The capital of France is" --max_tokens 50

# Run with OpenBLAS (requires libopenblas-dev)
bazel run //bin/qwen:qwen -- --model /path/to/Qwen2.5-0.5B --prompt "The capital of France is" --max_tokens 50 --backend blas

# Run with INT8 quantization (2.7x faster decode)
bazel run //bin/qwen:qwen -- --model /path/to/Qwen2.5-0.5B --prompt "The capital of France is" --max_tokens 50 --backend blas --quantize int8

# Run with CUDA backend (requires NVIDIA GPU + CUDA toolkit)
bazel run -c opt //bin/qwen:qwen -- --model /path/to/Qwen2.5-0.5B --prompt "The capital of France is" --max_tokens 50 --backend cuda --quantize int8

# Run with FP16 inference (CUDA only, ~29% faster decode)
bazel run -c opt //bin/qwen:qwen -- --model /path/to/Qwen2.5-0.5B --prompt "The capital of France is" --max_tokens 50 --backend cuda --quantize int8 --fp16
```
### Example Output

```
Model: qwen2 (896d, 24 layers)
Loading weights...
Loaded 290 tensors
Model initialized
Prompt: "The capital of France is" (5 tokens)

The capital of France is Paris, where the most important cultural institutions are located. The city is famous for its museums, including the Lou
```

## Project Structure

```
include/vvllm/          # Public headers
  backend/backend.h     # Abstract backend interface
  tensor/tensor.h       # Tensor template class
  safetensors/          # SafeTensors file loader
  config/config.h       # Model config (config.json)
  tokenizer/tokenizer.h # BPE tokenizer
  model/model.h         # Qwen2 model architecture
  sampler/sampler.h     # Temperature + top-p sampler

src/                    # Implementations + tests
  backend/              # CPU backend (naive + NEON) + BLAS backend (OpenBLAS) + CUDA backend
  tensor/               # Tensor implementation
  safetensors/          # SafeTensors loader (mmap + BF16->F32)
  config/               # Config parser
  tokenizer/            # BPE tokenizer
  model/                # Transformer forward pass
  sampler/              # Sampler implementation
  quantize/             # INT8 per-channel weight quantization

bin/qwen/               # Qwen inference binary
models/                 # Model weights (not checked in)
```

## What Works

- [x] Backend abstraction + CPU implementation
- [x] Tensor class (RAII, move semantics)
- [x] SafeTensors loader (mmap, BF16/F32 conversion)
- [x] Config loader (config.json parsing)
- [x] BPE tokenizer (encode/decode)
- [x] Backend ops (matmul, add, mul, silu, rms_norm, softmax, rope, embedding)
- [x] Full Qwen2 forward pass (24 layers, GQA attention, SwiGLU MLP)
- [x] Autoregressive text generation
- [x] Sampler with temperature and top-p (nucleus) sampling
- [x] KV cache (~56x decode speedup)
- [x] OpenBLAS backend (~9x linear projection speedup)
- [x] Batched GQA attention via BLAS (~6x attention speedup, +14% end-to-end decode)
- [x] INT8 per-channel weight quantization with NEON-optimized `linear_q8` kernel (~2.7x decode speedup)
- [x] sgemv dispatch for M=1 decode (eliminates sgemm packing overhead, +42% decode throughput)
- [x] CUDA backend with GPU mirror system, GPU-resident KV cache, and fused operators (4.5x decode speedup over BLAS on Qwen2.5-0.5B)
- [x] FP16 inference — all GPU ops use half precision with FP32 accumulation for stability (+29% decode throughput)

## Benchmarks

Component-level microbenchmarks using Google Benchmark with a synthetic tiny model (64-dim, 2 layers, no model files needed):

```bash
bazel run //src/model:model_bench
```

### Linear projection: CPU vs OpenBLAS (`-c opt`, aarch64)

| Dim | CPU (naive) | BLAS (OpenBLAS) | Speedup |
|-----|-------------|-----------------|---------|
| 64  | 2,227 ns    | 884 ns          | **2.5x** |
| 128 | 12,148 ns   | 3,804 ns        | **3.2x** |
| 256 | 58,317 ns   | 16,672 ns       | **3.5x** |
| 576 | 366,660 ns  | 42,761 ns       | **8.6x** |

### Attention: CPU vs batched BLAS (`-c opt`, aarch64)

GQA config matching Qwen2.5-0.5B (14 query heads, 2 KV heads, head_dim=64).
Batches 7 query heads per KV group into a single `cblas_sgemm` call.

| attend_len | CPU (naive) | BLAS (batched) | Speedup |
|------------|-------------|----------------|---------|
| 16         | 20,377 ns   | 3,390 ns       | **6.0x** |
| 64         | 79,466 ns   | 11,539 ns      | **6.9x** |
| 256        | 363,499 ns  | 70,969 ns      | **5.1x** |
| 1024       | 1,407,478 ns | 308,249 ns    | **4.6x** |

### INT8 quantized linear: NEON vs fp32 BLAS (`-c opt`, aarch64)

Per-channel absmax W8A32 quantization. NEON kernel loads int8 weights (4x less memory traffic), widens to float in-register via `vmovl` + `vcvtq`, accumulates with 4-way `vfmaq_f32`.

| Dim | BLAS fp32 | NEON int8 | Speedup |
|-----|-----------|-----------|---------|
| 64  | 1,136 ns  | 423 ns    | **2.7x** |
| 128 | 4,383 ns  | 1,662 ns  | **2.6x** |
| 256 | 20,399 ns | 6,187 ns  | **3.3x** |
| 576 | 82,233 ns | 30,379 ns | **2.7x** |

### End-to-end decode (`-c opt`, `--backend blas`, `--quantize int8`, aarch64)

| Model | Decode (fp32) | Decode (int8) | Speedup | Prefill (int8) |
|-------|---------------|---------------|---------|----------------|
| SmolLM-135M (576d, 30L) | 12.3 tok/s | **88.4 tok/s** | **7.2x** | 80.0 tok/s |
| Qwen2.5-0.5B (896d, 24L) | 12.3 tok/s | **33.4 tok/s** | **2.7x** | 77.7 tok/s |

### sgemv optimization for single-token decode

`perf record` profiling revealed that OpenBLAS `cblas_sgemm` was spending 14% of total inference time in matrix packing (`sgemm_incopy`) and only 2% in actual computation (`sgemm_kernel`). During decode, every `linear()` call is M=1 (matrix-vector), so the O(N*K) packing cost equals the computation cost but is used only once and discarded.

Fix: dispatch M=1 to `cblas_sgemv` instead of `cblas_sgemm`, eliminating packing and thread synchronization overhead entirely.

| Metric | Before (sgemm) | After (sgemv) | Improvement |
|--------|-----------------|---------------|-------------|
| Prefill throughput | 26.95 tok/s | 109.46 tok/s | **4.1x** |
| Decode throughput | 35.43 tok/s | 50.34 tok/s | **42%** |
| Decode avg latency | 28.2 ms | 19.9 ms | **-29%** |
| Decode P50 latency | 21.4 ms | 17.8 ms | **-17%** |

*Measured on SmolLM-135M (576d, 30L), `--backend blas --quantize int8`, aarch64.*

### CUDA backend (`-c opt`, `--backend cuda`, `--quantize int8`, x86_64 + RTX 4060)

Weights are cached on GPU (uploaded once). A mirror system keeps intermediate results on GPU between operations — `gpu_input` uses a 3-path lookup (exact mirror → parent mirror for sub-buffers → CPU upload with D2D sub-mirror overlay) so data only crosses PCIe for KV cache append and final logits.

| Model | BLAS (int8) | CUDA (int8) | Speedup |
|-------|-------------|-------------|---------|
| Qwen2.5-0.5B (896d, 24L) | 24.4 tok/s | **109.7 tok/s** | **4.5x** |

Optimization progression on Qwen2.5-0.5B decode:

| Stage | Decode | Improvement |
|-------|--------|-------------|
| GPU mirror system | 86.3 tok/s | baseline |
| + GPU-resident KV cache | 107.8 tok/s | **+24.9%** |
| + Fused operators | 109.7 tok/s | **+1.7%** |

**GPU-resident KV cache**: KV cache stays on GPU, eliminating 96 PCIe memcpy per decode token (2 tensors × 2 KV × 24 layers).

**FP16 inference** (`--fp16`): All GPU operations (linear, attention, RMSNorm, RoPE, SiLU) use half precision with FP32 accumulation for numerical stability. Halves memory bandwidth and enables Tensor Core utilization for all GEMMs. KV cache stored in FP16, halving GPU memory usage.

| Mode | Decode | Improvement |
|------|--------|-------------|
| CUDA INT8 (FP32) | 113.7 tok/s | baseline |
| CUDA INT8 (FP16) | **146.9 tok/s** | **+29.2%** |

*Measured on Qwen2.5-0.5B, RTX 4060, `--backend cuda --quantize int8`.*

**Fused operators**: Three operator fusions to reduce kernel launches and temporary buffers:
- Gate/Up projection fusion — concatenate `gate_proj` + `up_proj` into a single `[2*intermediate, hidden]` weight, one matmul instead of two
- Fused `silu_mul` — `silu(gate) * up` in a single kernel instead of separate `silu` + `mul`
- Fused `linear_add` — matmul + residual add in one kernel, eliminating temporary buffers for `proj_out` and `mlp_out`

### KV cache impact

`DecodeWithCache/64` at ~362us vs `DecodeNoCache/64` at ~20ms — **56x speedup**.

### BPE byte token decoding

**Problem**: During streaming decode, newlines displayed as `Ċ` and other control characters showed as garbled Unicode. For example:

```
...with a math problem.ĊĊOllie said    ← broken
...with a math problem.                 ← fixed (actual newlines)

Ollie said
```

**Root cause**: BPE tokenizers encode raw byte N as `chr(256 + N)`. For example, `Ċ` = chr(266) = 256 + 10 = newline byte `0x0A`. The `decode()` function only hardcoded `Ġ` (chr(288) = space) but left all other byte tokens as raw Unicode surrogates.

**Solution**: Generalized the byte token conversion in `decode()` — detect all codepoints in the chr(256..511) range and convert back to raw byte `cp - 256`:

```cpp
unsigned int cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
if (cp >= 256 && cp < 512)
    output += static_cast<char>(cp - 256);  // chr(256+N) → byte N
```

One universal rule replaces the previous hardcoded special case, fixing newlines, tabs, and all other byte-encoded characters.

## Known Issues

### Performance
- [x] **~~Naive matmul~~** — OpenBLAS backend available (`--backend blas`), ~9x faster linear projections + ~6x faster GQA attention
- [x] **~~No SIMD~~** — NEON-optimized INT8 linear kernel, 2.7x faster than BLAS fp32
- [ ] **No parallelism** — single-threaded
- [x] **~~High memory usage~~** — INT8 quantization reduces weight memory from ~1.9GB to ~0.5GB (`--quantize int8`)

### Correctness / Features
- [ ] **No stop tokens** — only stops on single EOS token
- [x] **~~Tokenizer edge cases~~** — fixed, see [BPE byte token decoding](#bpe-byte-token-decoding) below

### Architecture
- [ ] **No streaming** — generates all tokens then prints (currently prints per-token, but could be cleaner)
- [ ] **No batching** — single sequence only
- [x] **~~CPU only~~** — CUDA backend available (`--backend cuda`), 4.5x faster decode on Qwen2.5-0.5B
- [ ] **Limited model support** — only Qwen2 and Llama architectures

## Roadmap

1. ~~**KV cache**~~ — done
2. ~~**BLAS integration**~~ — done, OpenBLAS backend with ~9x linear speedup + ~6x batched GQA attention
3. ~~**INT8 weight quantization**~~ — done, per-channel absmax with NEON kernel, 2.7x decode speedup
4. **INT4 quantization** — further bandwidth reduction
5. **Multi-threading** — parallelize matmul and independent ops
6. ~~**GPU backend**~~ — done, CUDA backend with GPU mirror system + GPU-resident KV cache + fused ops, 4.5x decode speedup
7. ~~**FP16 inference**~~ — done, +29% decode throughput with half-precision GPU ops

## Build Requirements

- Bazel 7.4.1
- C++20 compiler
- `libopenblas-dev` (optional, for `--backend blas`)
- CUDA toolkit 12.x (optional, for `--backend cuda`)

```bash
# Build everything
bazel build //...

# Run tests
bazel test //...

# Refresh compile commands (after adding new files)
bazel run //:refresh_compile_commands
```
