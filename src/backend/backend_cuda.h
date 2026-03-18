#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "src/backend/backend_naive.h"

namespace vvllm
{

/// CUDA-accelerated backend. Weights are cached on GPU. Intermediate buffers
/// are GPU-resident Tensors (Device::CUDA), so all float* are device pointers.
class BackendCUDA : public BackendCPU
{
public:
    explicit BackendCUDA(bool fp16 = false);
    ~BackendCUDA() override;

    bool fp16() const { return fp16_; }

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
                   std::size_t hidden_size, std::size_t vocab_size) override;
    void causal_attention(float* out, const float* q, std::size_t q_idx, const float* k,
                          const float* v, std::size_t attend_len, std::size_t num_heads,
                          std::size_t num_kv_heads, std::size_t head_dim, float scale) override;
    bool is_fp16() const override;
    Device device() const override;

private:
    struct GpuBuf
    {
        void* ptr;
        std::size_t bytes;
    };

    // Weight cache: host address → device copy (uploaded once)
    std::unordered_map<const void*, GpuBuf> weight_cache_;

    // FP16 weight cache: host ptr → dequantized/converted FP16 device buffer
    std::unordered_map<const void*, GpuBuf> weight_fp16_cache_;

    // Temp buffers for causal_attention kv uploads (CPU fallback)
    void* d_tmp_aux_[2];
    std::size_t d_tmp_aux_bytes_[2];

    // FP16 temp buffer for input conversion (FP32 → FP16 before GEMM)
    void* d_tmp_fp16_in_ = nullptr;
    std::size_t d_tmp_fp16_in_bytes_ = 0;

    void ensure_buf(void*& buf, std::size_t& current, std::size_t needed);
    void* cache_weight(const void* host_ptr, std::size_t bytes);
    void* cache_weight_fp16(const int8_t* weight, const float* scales,
                            std::size_t N, std::size_t K);

    // Cache FP32 weight as FP16 on GPU
    void* cache_weight_as_fp16(const void* host_ptr, std::size_t fp32_bytes);

    bool fp16_;
};

}  // namespace vvllm
