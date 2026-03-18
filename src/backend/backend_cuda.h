#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "src/backend/backend_naive.h"

namespace vvllm
{

/// CUDA-accelerated backend. Weights are cached on GPU. Intermediate results
/// stay on GPU between operations via a mirror system, minimizing PCIe transfers.
class BackendCUDA : public BackendCPU
{
public:
    BackendCUDA();
    ~BackendCUDA() override;

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
    void begin_forward() override;
    void flush(const void* ptr, std::size_t bytes) override;

private:
    struct GpuBuf
    {
        void* ptr;
        std::size_t bytes;
    };

    struct GpuMirror
    {
        void* d_ptr;
        std::size_t capacity;
        bool valid;  // true = GPU has the latest data
    };

    // Weight cache: host address → device copy (uploaded once)
    std::unordered_map<const void*, GpuBuf> weight_cache_;

    // FP16 weight cache: INT8 host ptr → dequantized FP16 device buffer
    std::unordered_map<const void*, GpuBuf> weight_fp16_cache_;

    // GPU mirrors: host address → device mirror (for intermediate results)
    std::unordered_map<const void*, GpuMirror> mirrors_;

    // Temp buffers for non-mirrored inputs
    void* d_tmp_in_;
    std::size_t d_tmp_in_bytes_;
    void* d_tmp_in2_;
    std::size_t d_tmp_in2_bytes_;
    void* d_tmp_aux_[2];  // for causal_attention kv uploads
    std::size_t d_tmp_aux_bytes_[2];

    // FP16 temp buffer for input conversion
    void* d_tmp_fp16_in_ = nullptr;
    std::size_t d_tmp_fp16_in_bytes_ = 0;

    void ensure_buf(void*& buf, std::size_t& current, std::size_t needed);
    void* cache_weight(const void* host_ptr, std::size_t bytes);
    void* cache_weight_fp16(const int8_t* weight, const float* scales,
                            std::size_t N, std::size_t K);

    // Mirror helpers
    void* gpu_input(const void* cpu_ptr, std::size_t bytes, void*& fallback,
                    std::size_t& fallback_cap);

    // Returns GPU output pointer. Sets reused=true if mirror had matching capacity.
    void* gpu_output(const void* cpu_ptr, std::size_t bytes, bool& reused);

    void mirror_set_valid(const void* cpu_ptr);

    // After kernel: mark valid. If mirror was newly created (!reused), download to
    // keep CPU in sync (needed for prefill sub-buffer patterns).
    void finish_output(void* cpu_ptr, void* d_ptr, std::size_t bytes, bool reused);

};

}  // namespace vvllm
