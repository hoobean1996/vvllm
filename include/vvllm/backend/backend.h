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
                           std::size_t hidden_size, std::size_t vocab_size) = 0;

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

    /// Return the device pointer for a host buffer if it has a valid GPU mirror.
    /// Returns nullptr for CPU backends or when no valid mirror exists.
    virtual const void* device_ptr(const void* /*host_ptr*/) const { return nullptr; }

    /// Sample a token from logits on the device. Returns -1 if not supported (CPU).
    virtual int sample_gpu(const float* /*logits*/, std::size_t /*n*/, float /*temperature*/,
                           float /*top_p*/, std::uint64_t /*seed*/, int /*step*/) { return -1; }
};

std::unique_ptr<Backend> create_backend(BackendKind kind);
}  // namespace vvllm
