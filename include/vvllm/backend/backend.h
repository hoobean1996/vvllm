#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace vvllm
{

enum class BackendKind
{
    CPU,
    BLAS,
};

class Backend
{
public:
    virtual ~Backend() = default;

    /// Memory
    virtual void* allocate(std::size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;

    /// Compute

    /// out[M,N] = A[M,K] * B[K,N]
    virtual void matmul(float* out, const float* A, const float* B, std::size_t M, std::size_t N,
                        std::size_t K) = 0;

    /// out[i] = x[i] + y[i], element-wise addition
    virtual void add(float* out, const float* x, const float* y, std::size_t n) = 0;

    /// out[i] = x[i] * y[i], element-wise multiplication
    virtual void mul(float* out, const float* x, const float* y, std::size_t n) = 0;

    /// out[i] = x[i] / (1 + exp(-x[i])), SiLU activation
    virtual void silu(float* out, const float* x, std::size_t n) = 0;

    /// RMSNorm: out[i] = (x[i] / rms) * weight[i]
    /// where rms = sqrt(mean(x^2) + eps)
    virtual void rms_norm(float* out, const float* x, const float* weight, std::size_t n,
                          float eps) = 0;

    /// Softmax: out[i] = exp(x[i]) / sum(exp(x[j]))
    virtual void softmax(float* out, const float* x, std::size_t n) = 0;

    /// RoPE: apply rotary position embeddings in-place
    /// q[seq_len, num_heads, head_dim], k[seq_len, num_kv_heads, head_dim]
    virtual void rope(float* q, float* k, std::size_t seq_len, std::size_t num_heads,
                      std::size_t num_kv_heads, std::size_t head_dim, std::size_t pos,
                      float theta) = 0;

    /// out[M, N] = inp[M, K] @ weight^T[K, N] + bias[N]
    /// Weight is stored row-major as [N, K] (standard PyTorch convention).
    /// bias can be nullptr.
    virtual void linear(float* out, const float* inp, const float* weight, const float* bias,
                        std::size_t M, std::size_t N, std::size_t K) = 0;

    /// out[M, N] = inp[M, K] @ int8_weight^T[K, N] * scales[N] + bias[N]
    /// int8_weight is [N, K] row-major (int8). scales is [N]. bias can be nullptr.
    virtual void linear_q8(float* out, const float* inp, const int8_t* weight, const float* scales,
                           const float* bias, std::size_t M, std::size_t N, std::size_t K) = 0;

    /// out[i] = silu(gate[i]) * up[i], fused SiLU + element-wise multiply
    virtual void silu_mul(float* out, const float* gate, const float* up, std::size_t n) = 0;

    /// out[M,N] = inp[M,K] @ weight^T[K,N] + bias[N] + residual[M,N]
    virtual void linear_add(float* out, const float* inp, const float* weight, const float* bias,
                            const float* residual, std::size_t M, std::size_t N,
                            std::size_t K) = 0;

    /// out[M,N] = inp[M,K] @ int8_weight^T * scales + bias + residual
    virtual void linear_q8_add(float* out, const float* inp, const int8_t* weight,
                               const float* scales, const float* bias, const float* residual,
                               std::size_t M, std::size_t N, std::size_t K) = 0;

    /// Copy embedding row: out[hidden_size] = table[token_id * hidden_size ..]
    virtual void embedding(float* out, const float* table, std::size_t token_id,
                           std::size_t hidden_size) = 0;

    /// Multi-head causal attention for a single query position.
    /// q: [seq_len, num_heads, head_dim], k/v: [attend_len, num_kv_heads, head_dim]
    /// out: [num_heads * head_dim]
    virtual void causal_attention(float* out, const float* q, std::size_t q_idx, const float* k,
                                  const float* v, std::size_t attend_len, std::size_t num_heads,
                                  std::size_t num_kv_heads, std::size_t head_dim, float scale) = 0;

    /// Called at the start of each forward pass.
    /// GPU backends use this to invalidate stale mirrors from previous calls.
    virtual void begin_forward() {}

    /// Ensure the CPU buffer has the latest data from the device.
    /// No-op for CPU backends. GPU backends download if the device copy is newer.
    virtual void flush(const void* /*ptr*/, std::size_t /*bytes*/) {}

    /// GPU-resident KV cache interface.
    /// CPU backends use the no-op defaults; BackendCUDA overrides these.

    /// Allocate GPU KV buffers for the given number of layers (idempotent).
    virtual void kv_cache_init(std::size_t /*num_layers*/, std::size_t /*kv_dim*/) {}

    /// Append new K/V tokens to the GPU KV buffer for a layer.
    /// k and v point to GPU-mirrored data (seq_len * kv_dim floats each).
    virtual void kv_cache_append(std::size_t /*layer*/, const float* /*k*/, const float* /*v*/,
                                 std::size_t /*num_tokens*/) {}

    /// Return GPU pointer to cached K data for a layer, or nullptr if no GPU KV cache.
    virtual const float* kv_cache_k(std::size_t /*layer*/) const { return nullptr; }

    /// Return GPU pointer to cached V data for a layer, or nullptr if no GPU KV cache.
    virtual const float* kv_cache_v(std::size_t /*layer*/) const { return nullptr; }

    /// Bump the GPU KV cache sequence length after all layers are processed.
    virtual void kv_cache_advance(std::size_t /*num_tokens*/) {}

    /// Reset the GPU KV cache (e.g. for --kv_cache=false recompute mode).
    virtual void kv_cache_reset() {}

    /// Truncate the GPU KV cache to the given sequence length.
    /// Used by best-of-N to rewind to the prefill state between candidates.
    virtual void kv_cache_truncate(std::size_t /*new_seq_len*/) {}
};

std::unique_ptr<Backend> create_backend(BackendKind kind);
}  // namespace vvllm
