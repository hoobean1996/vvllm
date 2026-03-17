#pragma once

#include "src/backend/backend_naive.h"

namespace vvllm
{

/// OpenBLAS-accelerated backend. Overrides linear() with cblas_sgemm;
/// all other ops fall through to BackendCPU.
class BackendBLAS : public BackendCPU
{
public:
    void linear(float* out, const float* inp, const float* weight, const float* bias, std::size_t M,
                std::size_t N, std::size_t K) override;
    void causal_attention(float* out, const float* q, std::size_t q_idx, const float* k,
                          const float* v, std::size_t attend_len, std::size_t num_heads,
                          std::size_t num_kv_heads, std::size_t head_dim, float scale) override;
};

}  // namespace vvllm
