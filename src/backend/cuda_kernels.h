#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// GPU memory management wrappers
void* cuda_malloc(size_t bytes);
void cuda_free(void* ptr);
void* cuda_malloc_host(size_t bytes);  // pinned (page-locked) CPU memory
void cuda_free_host(void* ptr);
void cuda_memcpy_h2d(void* dst, const void* src, size_t bytes);
void cuda_memcpy_d2h(void* dst, const void* src, size_t bytes);
void cuda_memcpy_d2d(void* dst, const void* src, size_t bytes);
void cuda_memcpy_auto(void* dst, const void* src, size_t bytes);  // auto-detect direction
void cuda_sync();

// Returns 0 on success, fills name buffer with device name
int cuda_init_device(char* name_buf, size_t name_buf_size);

// Kernel wrappers
void cuda_linear_q8(float* out, const float* inp, const int8_t* weight, const float* scales,
                    const float* bias, size_t M, size_t N, size_t K);

void cuda_linear(float* out, const float* inp, const float* weight, const float* bias, size_t M,
                 size_t N, size_t K);

void cuda_add(float* out, const float* x, const float* y, size_t n);

void cuda_mul(float* out, const float* x, const float* y, size_t n);

void cuda_silu(float* out, const float* x, size_t n);

void cuda_silu_mul(float* out, const float* gate, const float* up, size_t n);

void cuda_rms_norm(float* out, const float* x, const float* weight, size_t n, float eps);

void cuda_softmax(float* out, const float* x, size_t n);

void cuda_rope(float* q, float* k, size_t seq_len, size_t num_heads, size_t num_kv_heads,
               size_t head_dim, size_t pos, float theta);

void cuda_linear_add(float* out, const float* inp, const float* weight, const float* bias,
                     const float* residual, size_t M, size_t N, size_t K);

void cuda_linear_q8_add(float* out, const float* inp, const int8_t* weight, const float* scales,
                         const float* bias, const float* residual, size_t M, size_t N, size_t K);

void cuda_embedding(float* out, const float* table, size_t token_id, size_t hidden_size);

void cuda_causal_attention(float* out, const float* q, size_t q_idx, const float* k, const float* v,
                           size_t attend_len, size_t num_heads, size_t num_kv_heads, size_t head_dim,
                           float scale);

void cuda_matmul(float* out, const float* A, const float* B, size_t M, size_t N, size_t K);

// cuBLAS lifecycle
void cuda_cublas_init();
void cuda_cublas_destroy();

// FP32 GEMM: out = alpha * inp @ weight^T + beta * out
void cuda_cublas_sgemm(float* out, const float* inp, const float* weight,
                       size_t M, size_t N, size_t K, float beta);

// FP16 GEMM with FP32 output: out = alpha * inp_fp16 @ weight_fp16^T + beta * out
void cuda_cublas_hgemm(float* out, const void* inp_fp16, const void* weight_fp16,
                       size_t M, size_t N, size_t K, float beta);

// FP32 → FP16 conversion
void cuda_fp32_to_fp16(void* out, const float* in, size_t n);

// INT8 dequantize to FP16: out[j*K+k] = (float)weight[j*K+k] * scales[j]
void cuda_int8_dequant_to_fp16(void* out, const int8_t* weight, const float* scales,
                               size_t N, size_t K);

// Bias broadcast: out[i*N+j] += bias[j] for i in [0,M)
void cuda_add_bias(float* out, const float* bias, size_t M, size_t N);

// ============================================================
// FP16 kernel variants (void* = half* on device)
// ============================================================

void cuda_add_fp16(void* out, const void* x, const void* y, size_t n);
void cuda_mul_fp16(void* out, const void* x, const void* y, size_t n);
void cuda_silu_fp16(void* out, const void* x, size_t n);
void cuda_silu_mul_fp16(void* out, const void* gate, const void* up, size_t n);
void cuda_rms_norm_fp16(void* out, const void* x, const void* weight, size_t n, float eps);
void cuda_softmax_fp16(void* out, const void* x, size_t n);
void cuda_rope_fp16(void* q, void* k, size_t seq_len, size_t num_heads,
                    size_t num_kv_heads, size_t head_dim, size_t pos, float theta);
void cuda_embedding_fp16(void* out, const void* table, size_t token_id, size_t hidden_size);
void cuda_causal_attention_fp16(void* out, const void* q, size_t q_idx, const void* k,
                                const void* v, size_t attend_len, size_t num_heads,
                                size_t num_kv_heads, size_t head_dim, float scale);
void cuda_add_bias_fp16(void* out, const void* bias, size_t M, size_t N);

// FP16 GEMM with FP16 output
void cuda_cublas_hgemm_fp16(void* out_fp16, const void* inp_fp16, const void* weight_fp16,
                            size_t M, size_t N, size_t K, float beta);

// FP16 → FP32 conversion
void cuda_fp16_to_fp32(float* out, const void* in, size_t n);

// GPU sampling: returns sampled token ID
// Greedy (temperature=0): argmax
// Stochastic (temperature>0): temperature scale + Gumbel-max trick
int cuda_sample(const void* logits, size_t n, float temperature, float top_p,
                unsigned long long seed, int step, int fp16);

#ifdef __cplusplus
}
#endif
