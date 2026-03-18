#include "src/backend/backend_cuda.h"

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "src/backend/cuda_kernels.h"

namespace vvllm
{

BackendCUDA::BackendCUDA(bool fp16)
    : d_tmp_aux_{nullptr, nullptr},
      d_tmp_aux_bytes_{0, 0},
      fp16_(fp16)
{
    char name[256];
    if (cuda_init_device(name, sizeof(name)) != 0)
    {
        fprintf(stderr, "No CUDA devices found\n");
        exit(1);
    }
    fprintf(stderr, "CUDA device: %s\n", name);
    cuda_cublas_init();
}

BackendCUDA::~BackendCUDA()
{
    cuda_cublas_destroy();
    for (auto& [_, buf] : weight_cache_)
        cuda_free(buf.ptr);
    for (auto& [_, buf] : weight_fp16_cache_)
        cuda_free(buf.ptr);
    if (d_tmp_fp16_in_) cuda_free(d_tmp_fp16_in_);
    for (int i = 0; i < 2; i++)
        if (d_tmp_aux_[i]) cuda_free(d_tmp_aux_[i]);
}

void* BackendCUDA::allocate(std::size_t bytes) { return cuda_malloc_host(bytes); }
void BackendCUDA::deallocate(void* ptr) { cuda_free_host(ptr); }

void BackendCUDA::ensure_buf(void*& buf, std::size_t& current, std::size_t needed)
{
    if (needed <= current) return;
    if (buf) cuda_free(buf);
    buf = cuda_malloc(needed);
    current = needed;
}

void* BackendCUDA::cache_weight(const void* host_ptr, std::size_t bytes)
{
    auto it = weight_cache_.find(host_ptr);
    if (it != weight_cache_.end()) return it->second.ptr;
    void* d = cuda_malloc(bytes);
    cuda_memcpy_h2d(d, host_ptr, bytes);
    weight_cache_.emplace(host_ptr, GpuBuf{d, bytes});
    return d;
}

void* BackendCUDA::cache_weight_fp16(const int8_t* weight, const float* scales,
                                      std::size_t N, std::size_t K)
{
    auto it = weight_fp16_cache_.find(weight);
    if (it != weight_fp16_cache_.end()) return it->second.ptr;

    // Upload INT8 weight and scales to GPU, then dequantize to FP16
    int8_t* d_w = static_cast<int8_t*>(cache_weight(weight, N * K * sizeof(int8_t)));
    float* d_scales = static_cast<float*>(cache_weight(scales, N * sizeof(float)));

    std::size_t fp16_bytes = N * K * sizeof(uint16_t);  // sizeof(half) == 2
    void* d_fp16 = cuda_malloc(fp16_bytes);
    cuda_int8_dequant_to_fp16(d_fp16, d_w, d_scales, N, K);

    weight_fp16_cache_.emplace(weight, GpuBuf{d_fp16, fp16_bytes});
    return d_fp16;
}

void* BackendCUDA::cache_weight_as_fp16(const void* host_ptr, std::size_t fp32_bytes)
{
    auto it = weight_fp16_cache_.find(host_ptr);
    if (it != weight_fp16_cache_.end()) return it->second.ptr;

    std::size_t n = fp32_bytes / sizeof(float);
    std::size_t fp16_bytes = n * sizeof(uint16_t);

    void* d_fp32 = cuda_malloc(fp32_bytes);
    cuda_memcpy_h2d(d_fp32, host_ptr, fp32_bytes);
    void* d_fp16 = cuda_malloc(fp16_bytes);
    cuda_fp32_to_fp16(d_fp16, static_cast<float*>(d_fp32), n);
    cuda_free(d_fp32);

    weight_fp16_cache_.emplace(host_ptr, GpuBuf{d_fp16, fp16_bytes});
    return d_fp16;
}

// ============================================================
// Element-wise ops
// ============================================================

void BackendCUDA::add(float* out, const float* x, const float* y, std::size_t n)
{
    if (fp16_) { cuda_add_fp16(out, x, y, n); return; }
    cuda_add(out, x, y, n);
}

void BackendCUDA::mul(float* out, const float* x, const float* y, std::size_t n)
{
    if (fp16_) { cuda_mul_fp16(out, x, y, n); return; }
    cuda_mul(out, x, y, n);
}

void BackendCUDA::silu(float* out, const float* x, std::size_t n)
{
    if (fp16_) { cuda_silu_fp16(out, x, n); return; }
    cuda_silu(out, x, n);
}

void BackendCUDA::silu_mul(float* out, const float* gate, const float* up, std::size_t n)
{
    if (fp16_) { cuda_silu_mul_fp16(out, gate, up, n); return; }
    cuda_silu_mul(out, gate, up, n);
}

void BackendCUDA::softmax(float* out, const float* x, std::size_t n)
{
    if (fp16_) { cuda_softmax_fp16(out, x, n); return; }
    cuda_softmax(out, x, n);
}

// ============================================================
// Reductions
// ============================================================

void BackendCUDA::rms_norm(float* out, const float* x, const float* weight, std::size_t n,
                           float eps)
{
    if (fp16_)
    {
        void* d_w = cache_weight_as_fp16(weight, n * sizeof(float));
        cuda_rms_norm_fp16(out, x, d_w, n, eps);
        return;
    }
    float* d_w = static_cast<float*>(cache_weight(weight, n * sizeof(float)));
    cuda_rms_norm(out, x, d_w, n, eps);
}

// ============================================================
// RoPE (in-place: q and k are both input and output)
// ============================================================

void BackendCUDA::rope(float* q, float* k, std::size_t seq_len, std::size_t num_heads,
                       std::size_t num_kv_heads, std::size_t head_dim, std::size_t pos,
                       float theta)
{
    if (fp16_)
    {
        cuda_rope_fp16(q, k, seq_len, num_heads, num_kv_heads, head_dim, pos, theta);
        return;
    }
    cuda_rope(q, k, seq_len, num_heads, num_kv_heads, head_dim, pos, theta);
}

// ============================================================
// Linear
// ============================================================

void BackendCUDA::linear(float* out, const float* inp, const float* weight, const float* bias,
                         std::size_t M, std::size_t N, std::size_t K)
{
    if (fp16_)
    {
        // In FP16 mode, inp and out are actually half* (from FP16 Tensors)
        void* d_w = cache_weight_as_fp16(weight, N * K * sizeof(float));
        void* d_bias = bias ? cache_weight_as_fp16(bias, N * sizeof(float)) : nullptr;
        cuda_cublas_hgemm_fp16(out, inp, d_w, M, N, K, 0.0f);
        if (d_bias) cuda_add_bias_fp16(out, d_bias, M, N);
        return;
    }

    float* d_w = static_cast<float*>(cache_weight(weight, N * K * sizeof(float)));
    float* d_bias = bias ? static_cast<float*>(cache_weight(bias, N * sizeof(float))) : nullptr;

    cuda_cublas_sgemm(out, inp, d_w, M, N, K, 0.0f);
    if (d_bias) cuda_add_bias(out, d_bias, M, N);
}

void BackendCUDA::linear_q8(float* out, const float* inp, const int8_t* weight,
                             const float* scales, const float* bias, std::size_t M, std::size_t N,
                             std::size_t K)
{
    void* d_w_fp16 = cache_weight_fp16(weight, scales, N, K);

    if (fp16_)
    {
        // inp and out are FP16 (from FP16 Tensors)
        void* d_bias = bias ? cache_weight_as_fp16(bias, N * sizeof(float)) : nullptr;
        cuda_cublas_hgemm_fp16(out, inp, d_w_fp16, M, N, K, 0.0f);
        if (d_bias) cuda_add_bias_fp16(out, d_bias, M, N);
        return;
    }

    float* d_bias = bias ? static_cast<float*>(cache_weight(bias, N * sizeof(float))) : nullptr;

    // Convert FP32 input to FP16 for hgemm
    std::size_t fp16_inp_bytes = M * K * sizeof(uint16_t);
    ensure_buf(d_tmp_fp16_in_, d_tmp_fp16_in_bytes_, fp16_inp_bytes);
    cuda_fp32_to_fp16(d_tmp_fp16_in_, inp, M * K);

    // FP16 input, FP16 weight → FP32 output
    cuda_cublas_hgemm(out, d_tmp_fp16_in_, d_w_fp16, M, N, K, 0.0f);
    if (d_bias) cuda_add_bias(out, d_bias, M, N);
}

// ============================================================
// Linear + residual add
// ============================================================

void BackendCUDA::linear_add(float* out, const float* inp, const float* weight, const float* bias,
                              const float* residual, std::size_t M, std::size_t N, std::size_t K)
{
    if (fp16_)
    {
        std::size_t fp16_out_bytes = M * N * sizeof(uint16_t);
        void* d_w = cache_weight_as_fp16(weight, N * K * sizeof(float));
        if (residual != out) cuda_memcpy_d2d(out, residual, fp16_out_bytes);
        cuda_cublas_hgemm_fp16(out, inp, d_w, M, N, K, 1.0f);
        if (bias)
        {
            void* d_bias = cache_weight_as_fp16(bias, N * sizeof(float));
            cuda_add_bias_fp16(out, d_bias, M, N);
        }
        return;
    }

    std::size_t out_bytes = M * N * sizeof(float);
    float* d_w = static_cast<float*>(cache_weight(weight, N * K * sizeof(float)));
    float* d_bias = bias ? static_cast<float*>(cache_weight(bias, N * sizeof(float))) : nullptr;

    if (residual != out) cuda_memcpy_d2d(out, residual, out_bytes);

    cuda_cublas_sgemm(out, inp, d_w, M, N, K, 1.0f);
    if (d_bias) cuda_add_bias(out, d_bias, M, N);
}

void BackendCUDA::linear_q8_add(float* out, const float* inp, const int8_t* weight,
                                  const float* scales, const float* bias, const float* residual,
                                  std::size_t M, std::size_t N, std::size_t K)
{
    void* d_w_fp16 = cache_weight_fp16(weight, scales, N, K);

    if (fp16_)
    {
        std::size_t fp16_out_bytes = M * N * sizeof(uint16_t);
        void* d_bias = bias ? cache_weight_as_fp16(bias, N * sizeof(float)) : nullptr;
        if (residual != out) cuda_memcpy_d2d(out, residual, fp16_out_bytes);
        cuda_cublas_hgemm_fp16(out, inp, d_w_fp16, M, N, K, 1.0f);
        if (d_bias) cuda_add_bias_fp16(out, d_bias, M, N);
        return;
    }

    std::size_t out_bytes = M * N * sizeof(float);
    float* d_bias = bias ? static_cast<float*>(cache_weight(bias, N * sizeof(float))) : nullptr;

    std::size_t fp16_inp_bytes = M * K * sizeof(uint16_t);
    ensure_buf(d_tmp_fp16_in_, d_tmp_fp16_in_bytes_, fp16_inp_bytes);
    cuda_fp32_to_fp16(d_tmp_fp16_in_, inp, M * K);

    // Copy residual into output so cuBLAS beta=1.0 accumulates into it
    if (residual != out) cuda_memcpy_d2d(out, residual, out_bytes);

    // FP16 input, FP16 weight → FP32 output (accumulated with residual)
    cuda_cublas_hgemm(out, d_tmp_fp16_in_, d_w_fp16, M, N, K, 1.0f);
    if (d_bias) cuda_add_bias(out, d_bias, M, N);
}

// ============================================================
// Embedding
// ============================================================

void BackendCUDA::embedding(float* out, const float* table, std::size_t token_id,
                            std::size_t hidden_size, std::size_t vocab_size)
{
    std::size_t table_bytes = vocab_size * hidden_size * sizeof(float);
    float* d_table = static_cast<float*>(cache_weight(table, table_bytes));
    cuda_embedding(out, d_table, token_id, hidden_size);
}

// ============================================================
// Matmul
// ============================================================

void BackendCUDA::matmul(float* out, const float* A, const float* B, std::size_t M, std::size_t N,
                         std::size_t K)
{
    float* d_b = static_cast<float*>(cache_weight(B, K * N * sizeof(float)));
    cuda_matmul(out, A, d_b, M, N, K);
}

// ============================================================
// Causal Attention
// ============================================================

void BackendCUDA::causal_attention(float* out, const float* q, std::size_t q_idx, const float* k,
                                   const float* v, std::size_t attend_len, std::size_t num_heads,
                                   std::size_t num_kv_heads, std::size_t head_dim, float scale)
{
    std::size_t elem = fp16_ ? sizeof(uint16_t) : sizeof(float);
    std::size_t kv_bytes = attend_len * num_kv_heads * head_dim * elem;

    // Detect if k/v are already GPU-resident (e.g. from KVCacheCUDA)
    float* d_k = nullptr;
    float* d_v = nullptr;
    cudaPointerAttributes attrs;
    if (cudaPointerGetAttributes(&attrs, k) == cudaSuccess &&
        attrs.type == cudaMemoryTypeDevice)
    {
        d_k = const_cast<float*>(k);
        d_v = const_cast<float*>(v);
    }
    else
    {
        // Fallback: upload from CPU
        ensure_buf(d_tmp_aux_[0], d_tmp_aux_bytes_[0], kv_bytes);
        cuda_memcpy_h2d(d_tmp_aux_[0], k, kv_bytes);
        ensure_buf(d_tmp_aux_[1], d_tmp_aux_bytes_[1], kv_bytes);
        cuda_memcpy_h2d(d_tmp_aux_[1], v, kv_bytes);
        d_k = static_cast<float*>(d_tmp_aux_[0]);
        d_v = static_cast<float*>(d_tmp_aux_[1]);
    }

    if (fp16_)
    {
        cuda_causal_attention_fp16(out, q, q_idx, d_k, d_v, attend_len, num_heads, num_kv_heads,
                                    head_dim, scale);
    }
    else
    {
        cuda_causal_attention(out, q, q_idx, d_k, d_v, attend_len, num_heads, num_kv_heads,
                              head_dim, scale);
    }
}

bool BackendCUDA::is_fp16() const { return fp16_; }

Device BackendCUDA::device() const { return Device::CUDA; }

void BackendCUDA::fp32_to_fp16(void* dst, const float* src, std::size_t n)
{
    cuda_fp32_to_fp16(dst, src, n);
}

void BackendCUDA::fp16_to_fp32(float* dst, const void* src, std::size_t n)
{
    cuda_fp16_to_fp32(dst, src, n);
}

}  // namespace vvllm
