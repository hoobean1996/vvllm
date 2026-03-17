#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "src/backend/backend_naive.h"

namespace vvllm
{

void* BackendCPU::allocate(std::size_t bytes) { return std::malloc(bytes); }

void BackendCPU::deallocate(void* ptr) { std::free(ptr); }

void BackendCPU::matmul(float* out, const float* A, const float* B, std::size_t M, std::size_t N,
                        std::size_t K)
{
    // out(M×N) = A(M×K) × B(K×N)
    for (std::size_t i = 0; i < M; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (std::size_t k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
}

void BackendCPU::add(float* out, const float* x, const float* y, std::size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        out[i] = x[i] + y[i];
    }
}

void BackendCPU::mul(float* out, const float* x, const float* y, std::size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        out[i] = x[i] * y[i];
    }
}

void BackendCPU::silu(float* out, const float* x, std::size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        out[i] = x[i] * (1 / (1.0f + std::exp(-x[i])));
    }
}

void BackendCPU::rms_norm(float* out, const float* x, const float* weight, std::size_t n, float eps)
{
    float sum = std::accumulate(x, x + n, 0.0f, [](float acc, float x) { return acc + x * x; });
    float rms = sqrt(sum / n + eps);  // avoid 0
    for (size_t i = 0; i < n; i++)
    {
        out[i] = x[i] / rms * weight[i];
    }
}

void BackendCPU::softmax(float* out, const float* x, std::size_t n)
{
    float max_val = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; i++)
    {
        out[i] = std::exp(x[i] - max_val);
        sum += out[i];
    }
    for (std::size_t i = 0; i < n; i++)
    {
        out[i] /= sum;
    }
}

void BackendCPU::rope(float* q, float* k, std::size_t seq_len, std::size_t num_heads,
                      std::size_t num_kv_heads, std::size_t head_dim, std::size_t pos, float theta)
{
    for (std::size_t s = 0; s < seq_len; s++)
    {
        std::size_t position = pos + s;

        // Rotate Q heads (split-half convention: pairs (i, i+half))
        std::size_t half = head_dim / 2;
        for (std::size_t h = 0; h < num_heads; h++)
        {
            float* head = q + s * num_heads * head_dim + h * head_dim;
            for (std::size_t i = 0; i < half; i++)
            {
                float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim);
                float angle = position * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);
                float x0 = head[i];
                float x1 = head[i + half];
                head[i] = x0 * cos_a - x1 * sin_a;
                head[i + half] = x0 * sin_a + x1 * cos_a;
            }
        }

        // Rotate K heads (split-half convention: pairs (i, i+half))
        for (std::size_t h = 0; h < num_kv_heads; h++)
        {
            float* head = k + s * num_kv_heads * head_dim + h * head_dim;
            for (std::size_t i = 0; i < half; i++)
            {
                float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim);
                float angle = position * freq;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);
                float x0 = head[i];
                float x1 = head[i + half];
                head[i] = x0 * cos_a - x1 * sin_a;
                head[i + half] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

void BackendCPU::linear(float* out, const float* inp, const float* weight, const float* bias,
                        std::size_t M, std::size_t N, std::size_t K)
{
    // out[M, N] = inp[M, K] @ weight^T[K, N] + bias[N]
    // weight is [N, K] row-major
    for (std::size_t i = 0; i < M; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            float sum = bias ? bias[j] : 0.0f;
            for (std::size_t k = 0; k < K; k++)
            {
                sum += inp[i * K + k] * weight[j * K + k];
            }
            out[i * N + j] = sum;
        }
    }
}

void BackendCPU::linear_q8(float* out, const float* inp, const int8_t* weight, const float* scales,
                           const float* bias, std::size_t M, std::size_t N, std::size_t K)
{
    // out[M, N] = inp[M, K] @ int8_weight^T[K, N] * scales[N] + bias[N]
    // weight is [N, K] row-major (int8)
    for (std::size_t i = 0; i < M; i++)
    {
        const float* in_row = inp + i * K;
        for (std::size_t j = 0; j < N; j++)
        {
            const int8_t* w_row = weight + j * K;
            float sum;

#ifdef __ARM_NEON
            // NEON: process 16 int8 weights per iteration.
            // 4 independent accumulators hide FMA latency.
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            std::size_t k = 0;
            for (; k + 16 <= K; k += 16)
            {
                // Load 16 int8 weights
                int8x16_t w8 = vld1q_s8(w_row + k);

                // Widen: int8 → int16
                int16x8_t w16_lo = vmovl_s8(vget_low_s8(w8));
                int16x8_t w16_hi = vmovl_s8(vget_high_s8(w8));

                // Widen: int16 → int32 → float32
                float32x4_t wf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_lo)));
                float32x4_t wf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_lo)));
                float32x4_t wf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16_hi)));
                float32x4_t wf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16_hi)));

                // Load 16 float inputs
                float32x4_t in0 = vld1q_f32(in_row + k);
                float32x4_t in1 = vld1q_f32(in_row + k + 4);
                float32x4_t in2 = vld1q_f32(in_row + k + 8);
                float32x4_t in3 = vld1q_f32(in_row + k + 12);

                // Fused multiply-accumulate
                acc0 = vfmaq_f32(acc0, wf0, in0);
                acc1 = vfmaq_f32(acc1, wf1, in1);
                acc2 = vfmaq_f32(acc2, wf2, in2);
                acc3 = vfmaq_f32(acc3, wf3, in3);
            }

            // Reduce 4 accumulators → scalar
            acc0 = vaddq_f32(acc0, acc1);
            acc2 = vaddq_f32(acc2, acc3);
            acc0 = vaddq_f32(acc0, acc2);
            sum = vaddvq_f32(acc0);

            // Scalar tail
            for (; k < K; k++)
            {
                sum += in_row[k] * static_cast<float>(w_row[k]);
            }
#elif defined(__AVX2__) && defined(__FMA__)
            // AVX2+FMA: process 32 int8 weights per iteration.
            // 4 independent accumulators hide FMA latency.
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            std::size_t k = 0;
            for (; k + 32 <= K; k += 32)
            {
                // Load 32 int8 weights
                __m256i w8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w_row + k));

                // Extract 4 groups of 8 int8 → int32 → float32
                __m128i w8_0 = _mm256_castsi256_si128(w8);
                __m128i w8_2 = _mm256_extracti128_si256(w8, 1);

                // First 8 (bytes 0-7)
                __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8_0));
                // Second 8 (bytes 8-15)
                __m256 wf1 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_srli_si128(w8_0, 8)));
                // Third 8 (bytes 16-23)
                __m256 wf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8_2));
                // Fourth 8 (bytes 24-31)
                __m256 wf3 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_srli_si128(w8_2, 8)));

                // Load 32 float inputs (8 at a time)
                __m256 in0 = _mm256_loadu_ps(in_row + k);
                __m256 in1 = _mm256_loadu_ps(in_row + k + 8);
                __m256 in2 = _mm256_loadu_ps(in_row + k + 16);
                __m256 in3 = _mm256_loadu_ps(in_row + k + 24);

                // Fused multiply-accumulate
                acc0 = _mm256_fmadd_ps(wf0, in0, acc0);
                acc1 = _mm256_fmadd_ps(wf1, in1, acc1);
                acc2 = _mm256_fmadd_ps(wf2, in2, acc2);
                acc3 = _mm256_fmadd_ps(wf3, in3, acc3);
            }

            // Reduce 4 accumulators → scalar
            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc0 = _mm256_add_ps(acc0, acc2);
            // 256-bit → two 128-bit halves, add them
            __m128 lo = _mm256_castps256_ps128(acc0);
            __m128 hi = _mm256_extractf128_ps(acc0, 1);
            __m128 r = _mm_add_ps(lo, hi);
            r = _mm_hadd_ps(r, r);
            r = _mm_hadd_ps(r, r);
            sum = _mm_cvtss_f32(r);

            // Scalar tail
            for (; k < K; k++)
            {
                sum += in_row[k] * static_cast<float>(w_row[k]);
            }
#else
            sum = 0.0f;
            for (std::size_t k = 0; k < K; k++)
            {
                sum += in_row[k] * static_cast<float>(w_row[k]);
            }
#endif

            sum *= scales[j];
            if (bias)
            {
                sum += bias[j];
            }
            out[i * N + j] = sum;
        }
    }
}

void BackendCPU::silu_mul(float* out, const float* gate, const float* up, std::size_t n)
{
    for (std::size_t i = 0; i < n; i++)
    {
        float v = gate[i];
        out[i] = (v / (1.0f + std::exp(-v))) * up[i];
    }
}

void BackendCPU::linear_add(float* out, const float* inp, const float* weight, const float* bias,
                             const float* residual, std::size_t M, std::size_t N, std::size_t K)
{
    // Fused: out[i,j] = dot(inp[i], weight[j]) + bias[j] + residual[i,j]
    // Must be single-pass to handle out == residual (in-place residual connection).
    for (std::size_t i = 0; i < M; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            float sum = residual[i * N + j];
            if (bias) sum += bias[j];
            for (std::size_t k = 0; k < K; k++)
            {
                sum += inp[i * K + k] * weight[j * K + k];
            }
            out[i * N + j] = sum;
        }
    }
}

void BackendCPU::linear_q8_add(float* out, const float* inp, const int8_t* weight,
                                const float* scales, const float* bias, const float* residual,
                                std::size_t M, std::size_t N, std::size_t K)
{
    // Handle out == residual aliasing: save residual before linear_q8 clobbers it.
    std::vector<float> res_buf;
    if (out == residual)
    {
        res_buf.assign(residual, residual + M * N);
        residual = res_buf.data();
    }
    linear_q8(out, inp, weight, scales, bias, M, N, K);
    add(out, out, residual, M * N);
}

void BackendCPU::embedding(float* out, const float* table, std::size_t token_id,
                           std::size_t hidden_size)
{
    std::memcpy(out, table + token_id * hidden_size, hidden_size * sizeof(float));
}

void BackendCPU::causal_attention(float* out, const float* q, std::size_t q_idx, const float* k,
                                  const float* v, std::size_t attend_len, std::size_t num_heads,
                                  std::size_t num_kv_heads, std::size_t head_dim, float scale)
{
    std::size_t groups = num_heads / num_kv_heads;
    std::vector<float> scores(attend_len);

    for (std::size_t h = 0; h < num_heads; h++)
    {
        std::size_t kv_h = h / groups;
        const float* q_head = q + q_idx * num_heads * head_dim + h * head_dim;

        for (std::size_t t = 0; t < attend_len; t++)
        {
            const float* k_head = k + t * num_kv_heads * head_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (std::size_t d = 0; d < head_dim; d++)
            {
                dot += q_head[d] * k_head[d];
            }
            scores[t] = dot * scale;
        }

        softmax(scores.data(), scores.data(), attend_len);

        float* head_out = out + h * head_dim;
        std::memset(head_out, 0, head_dim * sizeof(float));
        for (std::size_t t = 0; t < attend_len; t++)
        {
            const float* v_head = v + t * num_kv_heads * head_dim + kv_h * head_dim;
            for (std::size_t d = 0; d < head_dim; d++)
            {
                head_out[d] += scores[t] * v_head[d];
            }
        }
    }
}

// --- KV cache (CPU-resident) ---

void BackendCPU::kv_cache_init(std::size_t num_layers, std::size_t kv_dim)
{
    if (!kv_k_cache_.empty()) return;
    kv_k_cache_.resize(num_layers);
    kv_v_cache_.resize(num_layers);
    kv_dim_ = kv_dim;
}

void BackendCPU::kv_cache_append(std::size_t layer, const float* k, const float* v,
                                 std::size_t num_tokens)
{
    std::size_t count = num_tokens * kv_dim_;
    auto& kc = kv_k_cache_[layer];
    auto& vc = kv_v_cache_[layer];
    kc.insert(kc.end(), k, k + count);
    vc.insert(vc.end(), v, v + count);
}

const float* BackendCPU::kv_cache_k(std::size_t layer) const
{
    return kv_k_cache_[layer].data();
}

const float* BackendCPU::kv_cache_v(std::size_t layer) const
{
    return kv_v_cache_[layer].data();
}

void BackendCPU::kv_cache_advance(std::size_t num_tokens) { kv_seq_len_ += num_tokens; }

void BackendCPU::kv_cache_reset()
{
    for (auto& kc : kv_k_cache_) kc.clear();
    for (auto& vc : kv_v_cache_) vc.clear();
    kv_seq_len_ = 0;
}

void BackendCPU::kv_cache_truncate(std::size_t new_seq_len)
{
    if (new_seq_len >= kv_seq_len_) return;
    std::size_t new_size = new_seq_len * kv_dim_;
    for (auto& kc : kv_k_cache_) kc.resize(new_size);
    for (auto& vc : kv_v_cache_) vc.resize(new_size);
    kv_seq_len_ = new_seq_len;
}

}  // namespace vvllm
