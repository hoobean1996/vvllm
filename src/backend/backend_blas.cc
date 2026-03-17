#include "src/backend/backend_blas.h"

#include <vector>

#include <cblas.h>

namespace vvllm
{

void BackendBLAS::linear(float* out, const float* inp, const float* weight, const float* bias,
                         std::size_t M, std::size_t N, std::size_t K)
{
    // out[M, N] = inp[M, K] @ weight^T[K, N]
    // weight is [N, K] row-major.
    if (M == 1)
    {
        // M=1: matrix-vector multiply. sgemv avoids sgemm's packing and thread sync overhead.
        // y = alpha * A^T * x + beta * y, where A = weight[N,K] and we want weight^T * x.
        cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(N), static_cast<int>(K), 1.0f,
                    weight, static_cast<int>(K), inp, 1, 0.0f, out, 1);
    }
    else
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(M),
                    static_cast<int>(N), static_cast<int>(K), 1.0f, inp, static_cast<int>(K),
                    weight, static_cast<int>(K), 0.0f, out, static_cast<int>(N));
    }

    if (bias)
    {
        for (std::size_t i = 0; i < M; i++)
        {
            for (std::size_t j = 0; j < N; j++)
            {
                out[i * N + j] += bias[j];
            }
        }
    }
}

void BackendBLAS::causal_attention(float* out, const float* q, std::size_t q_idx, const float* k,
                                   const float* v, std::size_t attend_len, std::size_t num_heads,
                                   std::size_t num_kv_heads, std::size_t head_dim, float scale)
{
    // Batch across GQA groups: instead of per-head sgemv, use one sgemm per KV head.
    // For Qwen2 (14 query heads, 2 KV heads): 2 sgemm calls instead of 14 sgemv calls.
    std::size_t groups = num_heads / num_kv_heads;

    int G = static_cast<int>(groups);       // query heads per KV head
    int T = static_cast<int>(attend_len);
    int D = static_cast<int>(head_dim);
    int kv_stride = static_cast<int>(num_kv_heads * head_dim);

    std::vector<float> scores(groups * attend_len);

    for (std::size_t kv_h = 0; kv_h < num_kv_heads; kv_h++)
    {
        // Q_group is [groups, head_dim], contiguous because heads are stored sequentially.
        const float* q_group = q + q_idx * num_heads * head_dim + kv_h * groups * head_dim;

        // scores[G, T] = Q_group[G, D] @ K_head[T, D]^T * scale
        // K_head rows are strided by kv_stride in the KV cache.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, G, T, D, scale, q_group, D,
                    k + kv_h * head_dim, kv_stride, 0.0f, scores.data(), T);

        // Softmax per query head (per row of scores)
        for (std::size_t g = 0; g < groups; g++)
        {
            softmax(scores.data() + g * attend_len, scores.data() + g * attend_len, attend_len);
        }

        // out_group[G, D] = scores[G, T] @ V_head[T, D]
        float* out_group = out + kv_h * groups * head_dim;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, G, D, T, 1.0f, scores.data(), T,
                    v + kv_h * head_dim, kv_stride, 0.0f, out_group, D);
    }
}

}  // namespace vvllm
