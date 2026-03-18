#pragma once

#include "vvllm/backend/backend.h"

namespace vvllm
{

/// element-wise: add, mul, silu
/// reduction: softmax, rms_norm
/// structured: matmul, rope, embedding
class BackendCPU : public Backend
{
public:
    void* allocate(std::size_t bytes) override;
    void deallocate(void* ptr) override;

    void matmul(float* out, const float* A, const float* B, std::size_t M, std::size_t N,
                std::size_t K) override;
    void add(float* out, const float* x, const float* y, std::size_t n) override;
    void mul(float* out, const float* x, const float* y, std::size_t n) override;
    void silu(float* out, const float* x, std::size_t n) override;
    void rms_norm(float* out, const float* x, const float* weight, std::size_t n,
                  float eps) override;
    void softmax(float* out, const float* x, std::size_t n) override;
    void rope(float* q, float* k, std::size_t seq_len, std::size_t num_heads,
              std::size_t num_kv_heads, std::size_t head_dim, std::size_t pos,
              float theta) override;
    void linear(float* out, const float* inp, const float* weight, const float* bias, std::size_t M,
                std::size_t N, std::size_t K) override;
    void linear_q8(float* out, const float* inp, const int8_t* weight, const float* scales,
                   const float* bias, std::size_t M, std::size_t N, std::size_t K) override;
    void silu_mul(float* out, const float* gate, const float* up, std::size_t n) override;
    void linear_add(float* out, const float* inp, const float* weight, const float* bias,
                    const float* residual, std::size_t M, std::size_t N, std::size_t K) override;
    void linear_q8_add(float* out, const float* inp, const int8_t* weight, const float* scales,
                       const float* bias, const float* residual, std::size_t M, std::size_t N,
                       std::size_t K) override;
    void embedding(float* out, const float* table, std::size_t token_id,
                   std::size_t hidden_size) override;
    void causal_attention(float* out, const float* q, std::size_t q_idx, const float* k,
                          const float* v, std::size_t attend_len, std::size_t num_heads,
                          std::size_t num_kv_heads, std::size_t head_dim, float scale) override;
};

}  // namespace vvllm
