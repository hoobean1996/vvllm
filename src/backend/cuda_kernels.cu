#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/backend/cuda_kernels.h"

#define CUDA_CHECK(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                       \
                    cudaGetErrorString(err));                                                       \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// ============================================================
// Element-wise kernels
// ============================================================

__global__ void add_kernel(float* out, const float* x, const float* y, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = x[i] + y[i];
    }
}

__global__ void mul_kernel(float* out, const float* x, const float* y, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = x[i] * y[i];
    }
}

__global__ void silu_kernel(float* out, const float* x, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float v = x[i];
        out[i] = v / (1.0f + expf(-v));
    }
}

__global__ void silu_mul_kernel(float* out, const float* gate, const float* up, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float v = gate[i];
        out[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

// ============================================================
// RMS Norm: one block per call, shared memory reduction
// ============================================================

__global__ void rms_norm_kernel(float* out, const float* x, const float* weight, size_t n,
                                float eps)
{
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    // Pass 1: compute sum of squares
    float local_sum = 0.0f;
    for (size_t i = tid; i < n; i += stride)
    {
        float v = x[i];
        local_sum += v * v;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / n + eps);

    // Pass 2: normalize
    for (size_t i = tid; i < n; i += stride)
    {
        out[i] = x[i] / rms * weight[i];
    }
}

// ============================================================
// Softmax: one block, shared memory reduction
// ============================================================

__global__ void softmax_kernel(float* out, const float* x, size_t n)
{
    extern __shared__ float sdata[];

    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    // Pass 1: find max
    float local_max = -1e30f;
    for (size_t i = tid; i < n; i += stride)
    {
        float v = x[i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Pass 2: compute exp and sum
    float local_sum = 0.0f;
    for (size_t i = tid; i < n; i += stride)
    {
        float e = expf(x[i] - max_val);
        out[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float sum = sdata[0];
    __syncthreads();

    // Pass 3: normalize
    for (size_t i = tid; i < n; i += stride)
    {
        out[i] /= sum;
    }
}

// ============================================================
// RoPE: one thread per (head, dimension-pair)
// ============================================================

__global__ void rope_kernel(float* q, float* k, size_t seq_len, size_t num_heads,
                            size_t num_kv_heads, size_t head_dim, size_t pos, float theta)
{
    // Grid: (half * max(num_heads, num_kv_heads), seq_len)
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = blockIdx.y;

    size_t half = head_dim / 2;
    size_t max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;
    size_t total_pairs = half * max_heads;
    if (idx >= total_pairs || s >= seq_len) return;

    size_t h = idx / half;
    size_t i = idx % half;

    size_t position = pos + s;
    float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
    float angle = position * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Rotate Q
    if (h < num_heads)
    {
        float* head_q = q + s * num_heads * head_dim + h * head_dim;
        float x0 = head_q[i];
        float x1 = head_q[i + half];
        head_q[i] = x0 * cos_a - x1 * sin_a;
        head_q[i + half] = x0 * sin_a + x1 * cos_a;
    }

    // Rotate K
    if (h < num_kv_heads)
    {
        float* head_k = k + s * num_kv_heads * head_dim + h * head_dim;
        float x0 = head_k[i];
        float x1 = head_k[i + half];
        head_k[i] = x0 * cos_a - x1 * sin_a;
        head_k[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

// ============================================================
// Linear: out[M,N] = inp[M,K] @ weight^T[K,N] + bias
// weight is [N,K] row-major. Grid(N, M), block 256.
// Each block computes one output element via dot product.
// ============================================================

__global__ void linear_kernel(float* out, const float* inp, const float* weight, const float* bias,
                              size_t M, size_t N, size_t K)
{
    size_t j = blockIdx.x;  // output column
    size_t i = blockIdx.y;  // output row
    if (i >= M || j >= N) return;

    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    const float* in_row = inp + i * K;
    const float* w_row = weight + j * K;

    float local_sum = 0.0f;
    for (size_t k = tid; k < K; k += stride)
    {
        local_sum += in_row[k] * w_row[k];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        float val = sdata[0];
        if (bias) val += bias[j];
        out[i * N + j] = val;
    }
}

// ============================================================
// Linear Q8: out[M,N] = inp[M,K] @ int8_weight^T * scales + bias
// weight is [N,K] int8. Grid(N, M), block 256.
// ============================================================

__global__ void linear_q8_kernel(float* out, const float* inp, const int8_t* weight,
                                 const float* scales, const float* bias, size_t M, size_t N,
                                 size_t K)
{
    size_t j = blockIdx.x;  // output column
    size_t i = blockIdx.y;  // output row
    if (i >= M || j >= N) return;

    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    const float* in_row = inp + i * K;
    const int8_t* w_row = weight + j * K;

    float local_sum = 0.0f;
    for (size_t k = tid; k < K; k += stride)
    {
        local_sum += in_row[k] * (float)w_row[k];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        float val = sdata[0] * scales[j];
        if (bias) val += bias[j];
        out[i * N + j] = val;
    }
}

// ============================================================
// Linear + Add: out[j] = dot(inp, weight[j]) + bias[j] + residual[j]
// ============================================================

__global__ void linear_add_kernel(float* out, const float* inp, const float* weight,
                                   const float* bias, const float* residual, size_t M, size_t N,
                                   size_t K)
{
    size_t j = blockIdx.x;  // output column
    size_t i = blockIdx.y;  // output row
    if (i >= M || j >= N) return;

    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    const float* in_row = inp + i * K;
    const float* w_row = weight + j * K;

    float local_sum = 0.0f;
    for (size_t k = tid; k < K; k += stride)
    {
        local_sum += in_row[k] * w_row[k];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        float val = sdata[0];
        if (bias) val += bias[j];
        out[i * N + j] = val + residual[i * N + j];
    }
}

__global__ void linear_q8_add_kernel(float* out, const float* inp, const int8_t* weight,
                                      const float* scales, const float* bias,
                                      const float* residual, size_t M, size_t N, size_t K)
{
    size_t j = blockIdx.x;  // output column
    size_t i = blockIdx.y;  // output row
    if (i >= M || j >= N) return;

    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    const float* in_row = inp + i * K;
    const int8_t* w_row = weight + j * K;

    float local_sum = 0.0f;
    for (size_t k = tid; k < K; k += stride)
    {
        local_sum += in_row[k] * (float)w_row[k];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        float val = sdata[0] * scales[j];
        if (bias) val += bias[j];
        out[i * N + j] = val + residual[i * N + j];
    }
}

// ============================================================
// Matmul: out[M,N] = A[M,K] * B[K,N]
// ============================================================

__global__ void matmul_kernel(float* out, const float* A, const float* B, size_t M, size_t N,
                              size_t K)
{
    size_t j = blockIdx.x;  // column
    size_t i = blockIdx.y;  // row
    if (i >= M || j >= N) return;

    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    float local_sum = 0.0f;
    for (size_t k = tid; k < K; k += stride)
    {
        local_sum += A[i * K + k] * B[k * N + j];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[i * N + j] = sdata[0];
    }
}

// ============================================================
// Causal attention: per-head attention with softmax
// Grid: (num_heads), Block: 256
// ============================================================

__global__ void causal_attention_kernel(float* out, const float* q, size_t q_idx, const float* k,
                                        const float* v, size_t attend_len, size_t num_heads,
                                        size_t num_kv_heads, size_t head_dim, float scale)
{
    size_t h = blockIdx.x;
    if (h >= num_heads) return;

    extern __shared__ float sdata[];
    // sdata layout: [0..attend_len) = scores, [attend_len..attend_len+blockDim.x) = reduction
    float* scores = sdata;
    float* reduce_buf = sdata + attend_len;

    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    size_t groups = num_heads / num_kv_heads;
    size_t kv_h = h / groups;

    const float* q_head = q + q_idx * num_heads * head_dim + h * head_dim;

    // Compute QK^T scores
    for (size_t t = tid; t < attend_len; t += stride)
    {
        const float* k_head = k + t * num_kv_heads * head_dim + kv_h * head_dim;
        float dot = 0.0f;
        for (size_t d = 0; d < head_dim; d++)
        {
            dot += q_head[d] * k_head[d];
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Softmax pass 1: find max
    float local_max = -1e30f;
    for (size_t t = tid; t < attend_len; t += stride)
    {
        if (scores[t] > local_max) local_max = scores[t];
    }
    reduce_buf[tid] = local_max;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + s]);
        }
        __syncthreads();
    }
    float max_val = reduce_buf[0];
    __syncthreads();

    // Softmax pass 2: exp and sum
    float local_sum = 0.0f;
    for (size_t t = tid; t < attend_len; t += stride)
    {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            reduce_buf[tid] += reduce_buf[tid + s];
        }
        __syncthreads();
    }
    float sum = reduce_buf[0];
    __syncthreads();

    // Softmax pass 3: normalize
    for (size_t t = tid; t < attend_len; t += stride)
    {
        scores[t] /= sum;
    }
    __syncthreads();

    // Weighted V sum: each thread handles a subset of head_dim
    float* head_out = out + h * head_dim;
    for (size_t d = tid; d < head_dim; d += stride)
    {
        float val = 0.0f;
        for (size_t t = 0; t < attend_len; t++)
        {
            val += scores[t] * v[t * num_kv_heads * head_dim + kv_h * head_dim + d];
        }
        head_out[d] = val;
    }
}

// ============================================================
// GPU memory management wrappers
// ============================================================

extern "C" void* cuda_malloc(size_t bytes)
{
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

extern "C" void cuda_free(void* ptr) { CUDA_CHECK(cudaFree(ptr)); }

extern "C" void* cuda_malloc_host(size_t bytes)
{
    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
    return ptr;
}

extern "C" void cuda_free_host(void* ptr) { CUDA_CHECK(cudaFreeHost(ptr)); }

extern "C" void cuda_memcpy_h2d(void* dst, const void* src, size_t bytes)
{
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

extern "C" void cuda_memcpy_d2h(void* dst, const void* src, size_t bytes)
{
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

extern "C" void cuda_memcpy_d2d(void* dst, const void* src, size_t bytes)
{
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
}

extern "C" void cuda_memcpy_auto(void* dst, const void* src, size_t bytes)
{
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
}

extern "C" void cuda_sync() { CUDA_CHECK(cudaDeviceSynchronize()); }

extern "C" int cuda_init_device(char* name_buf, size_t name_buf_size)
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) return -1;
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    snprintf(name_buf, name_buf_size, "%s (SM %d.%d, %.0f MB)", prop.name, prop.major, prop.minor,
             prop.totalGlobalMem / (1024.0 * 1024.0));
    return 0;
}

// ============================================================
// Kernel launch wrappers
// ============================================================

extern "C" void cuda_add(float* out, const float* x, const float* y, size_t n)
{
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    add_kernel<<<grid, block>>>(out, x, y, n);
}

extern "C" void cuda_mul(float* out, const float* x, const float* y, size_t n)
{
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    mul_kernel<<<grid, block>>>(out, x, y, n);
}

extern "C" void cuda_silu(float* out, const float* x, size_t n)
{
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    silu_kernel<<<grid, block>>>(out, x, n);
}

extern "C" void cuda_silu_mul(float* out, const float* gate, const float* up, size_t n)
{
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    silu_mul_kernel<<<grid, block>>>(out, gate, up, n);
}

extern "C" void cuda_rms_norm(float* out, const float* x, const float* weight, size_t n, float eps)
{
    size_t block = 256;
    size_t smem = block * sizeof(float);
    rms_norm_kernel<<<1, block, smem>>>(out, x, weight, n, eps);
}

extern "C" void cuda_softmax(float* out, const float* x, size_t n)
{
    size_t block = 256;
    size_t smem = block * sizeof(float);
    softmax_kernel<<<1, block, smem>>>(out, x, n);
}

extern "C" void cuda_rope(float* q, float* k, size_t seq_len, size_t num_heads,
                           size_t num_kv_heads, size_t head_dim, size_t pos, float theta)
{
    size_t half = head_dim / 2;
    size_t max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;
    size_t total_pairs = half * max_heads;

    size_t block = 256;
    size_t grid_x = (total_pairs + block - 1) / block;
    dim3 grid(grid_x, seq_len);
    rope_kernel<<<grid, block>>>(q, k, seq_len, num_heads, num_kv_heads, head_dim, pos, theta);
}

extern "C" void cuda_linear(float* out, const float* inp, const float* weight, const float* bias,
                             size_t M, size_t N, size_t K)
{
    size_t block = 256;
    size_t smem = block * sizeof(float);
    dim3 grid(N, M);
    linear_kernel<<<grid, block, smem>>>(out, inp, weight, bias, M, N, K);
}

extern "C" void cuda_linear_q8(float* out, const float* inp, const int8_t* weight,
                                const float* scales, const float* bias, size_t M, size_t N,
                                size_t K)
{
    size_t block = 256;
    size_t smem = block * sizeof(float);
    dim3 grid(N, M);
    linear_q8_kernel<<<grid, block, smem>>>(out, inp, weight, scales, bias, M, N, K);
}

extern "C" void cuda_linear_add(float* out, const float* inp, const float* weight, const float* bias,
                                 const float* residual, size_t M, size_t N, size_t K)
{
    size_t block = 256;
    size_t smem = block * sizeof(float);
    dim3 grid(N, M);
    linear_add_kernel<<<grid, block, smem>>>(out, inp, weight, bias, residual, M, N, K);
}

extern "C" void cuda_linear_q8_add(float* out, const float* inp, const int8_t* weight,
                                     const float* scales, const float* bias,
                                     const float* residual, size_t M, size_t N, size_t K)
{
    size_t block = 256;
    size_t smem = block * sizeof(float);
    dim3 grid(N, M);
    linear_q8_add_kernel<<<grid, block, smem>>>(out, inp, weight, scales, bias, residual, M, N, K);
}

extern "C" void cuda_embedding(float* out, const float* table, size_t token_id, size_t hidden_size)
{
    CUDA_CHECK(cudaMemcpy(out, table + token_id * hidden_size, hidden_size * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

extern "C" void cuda_causal_attention(float* out, const float* q, size_t q_idx, const float* k,
                                      const float* v, size_t attend_len, size_t num_heads,
                                      size_t num_kv_heads, size_t head_dim, float scale)
{
    size_t block = 256;
    // Shared memory: scores[attend_len] + reduce_buf[block]
    size_t smem = (attend_len + block) * sizeof(float);
    causal_attention_kernel<<<num_heads, block, smem>>>(out, q, q_idx, k, v, attend_len, num_heads,
                                                        num_kv_heads, head_dim, scale);
}

extern "C" void cuda_matmul(float* out, const float* A, const float* B, size_t M, size_t N,
                             size_t K)
{
    size_t block = 256;
    size_t smem = block * sizeof(float);
    dim3 grid(N, M);
    matmul_kernel<<<grid, block, smem>>>(out, A, B, M, N, K);
}

// ============================================================
// cuBLAS
// ============================================================

static cublasHandle_t g_cublas_handle = nullptr;

#define CUBLAS_CHECK(call)                                                                         \
    do                                                                                             \
    {                                                                                              \
        cublasStatus_t status = (call);                                                            \
        if (status != CUBLAS_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)status);       \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

extern "C" void cuda_cublas_init()
{
    CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
    // Allow Tensor Core usage
    CUBLAS_CHECK(cublasSetMathMode(g_cublas_handle, CUBLAS_DEFAULT_MATH));
}

extern "C" void cuda_cublas_destroy()
{
    if (g_cublas_handle)
    {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

// out[M,N] = inp[M,K] @ weight^T[K,N] + beta * out
// Row-major convention mapped to column-major cuBLAS:
//   C(N,M) = alpha * op(A)(N,K) * op(B)(K,M) + beta * C(N,M)
//   where A=weight (N,K row-major = K,N col-major, transposed),
//         B=inp (M,K row-major = K,M col-major, no transpose)
extern "C" void cuda_cublas_sgemm(float* out, const float* inp, const float* weight,
                                   size_t M, size_t N, size_t K, float beta)
{
    float alpha = 1.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas_handle,
                             CUBLAS_OP_T,   // op(A): transpose weight
                             CUBLAS_OP_N,   // op(B): no transpose inp
                             (int)N, (int)M, (int)K,
                             &alpha,
                             weight, (int)K,  // lda = K (row-major stride of weight)
                             inp, (int)K,     // ldb = K (row-major stride of inp)
                             &beta,
                             out, (int)N));   // ldc = N (row-major stride of out)
}

// FP16 GEMM with FP32 accumulation and FP32 output
extern "C" void cuda_cublas_hgemm(float* out, const void* inp_fp16, const void* weight_fp16,
                                   size_t M, size_t N, size_t K, float beta)
{
    float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(g_cublas_handle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              (int)N, (int)M, (int)K,
                              &alpha,
                              weight_fp16, CUDA_R_16F, (int)K,
                              inp_fp16, CUDA_R_16F, (int)K,
                              &beta,
                              out, CUDA_R_32F, (int)N,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ============================================================
// FP16 conversion kernels
// ============================================================

__global__ void fp32_to_fp16_kernel(half* out, const float* in, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = __float2half(in[i]);
    }
}

__global__ void int8_dequant_to_fp16_kernel(half* out, const int8_t* weight, const float* scales,
                                             size_t N, size_t K)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * K;
    if (idx < total)
    {
        size_t j = idx / K;  // output channel
        out[idx] = __float2half((float)weight[idx] * scales[j]);
    }
}

__global__ void add_bias_kernel(float* out, const float* bias, size_t M, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = M * N;
    if (idx < total)
    {
        out[idx] += bias[idx % N];
    }
}

extern "C" void cuda_fp32_to_fp16(void* out, const float* in, size_t n)
{
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    fp32_to_fp16_kernel<<<grid, block>>>(static_cast<half*>(out), in, n);
}

extern "C" void cuda_int8_dequant_to_fp16(void* out, const int8_t* weight, const float* scales,
                                            size_t N, size_t K)
{
    size_t total = N * K;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    int8_dequant_to_fp16_kernel<<<grid, block>>>(static_cast<half*>(out), weight, scales, N, K);
}

extern "C" void cuda_add_bias(float* out, const float* bias, size_t M, size_t N)
{
    size_t total = M * N;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    add_bias_kernel<<<grid, block>>>(out, bias, M, N);
}

// ============================================================
// FP16 kernels — load/store as half, accumulate in float
// ============================================================

__global__ void add_fp16_kernel(half* out, const half* x, const half* y, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));
}

__global__ void mul_fp16_kernel(half* out, const half* x, const half* y, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(x[i]) * __half2float(y[i]));
}

__global__ void silu_fp16_kernel(half* out, const half* x, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float v = __half2float(x[i]);
        out[i] = __float2half(v / (1.0f + expf(-v)));
    }
}

__global__ void silu_mul_fp16_kernel(half* out, const half* gate, const half* up, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float v = __half2float(gate[i]);
        out[i] = __float2half((v / (1.0f + expf(-v))) * __half2float(up[i]));
    }
}

__global__ void rms_norm_fp16_kernel(half* out, const half* x, const half* weight, size_t n,
                                      float eps)
{
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    float local_sum = 0.0f;
    for (size_t i = tid; i < n; i += stride)
    {
        float v = __half2float(x[i]);
        local_sum += v * v;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / n + eps);

    for (size_t i = tid; i < n; i += stride)
    {
        out[i] = __float2half(__half2float(x[i]) / rms * __half2float(weight[i]));
    }
}

__global__ void softmax_fp16_kernel(half* out, const half* x, size_t n)
{
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    float local_max = -1e30f;
    for (size_t i = tid; i < n; i += stride)
    {
        float v = __half2float(x[i]);
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (size_t i = tid; i < n; i += stride)
    {
        float e = expf(__half2float(x[i]) - max_val);
        out[i] = __float2half(e);
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum = sdata[0];
    __syncthreads();

    for (size_t i = tid; i < n; i += stride)
    {
        out[i] = __float2half(__half2float(out[i]) / sum);
    }
}

__global__ void rope_fp16_kernel(half* q, half* k, size_t seq_len, size_t num_heads,
                                  size_t num_kv_heads, size_t head_dim, size_t pos, float theta)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = blockIdx.y;
    size_t hdim_half = head_dim / 2;
    size_t max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;
    size_t total_pairs = hdim_half * max_heads;
    if (idx >= total_pairs || s >= seq_len) return;

    size_t h = idx / hdim_half;
    size_t i = idx % hdim_half;
    size_t position = pos + s;
    float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
    float angle = position * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    if (h < num_heads)
    {
        half* head_q = q + s * num_heads * head_dim + h * head_dim;
        float x0 = __half2float(head_q[i]);
        float x1 = __half2float(head_q[i + hdim_half]);
        head_q[i] = __float2half(x0 * cos_a - x1 * sin_a);
        head_q[i + hdim_half] = __float2half(x0 * sin_a + x1 * cos_a);
    }
    if (h < num_kv_heads)
    {
        half* head_k = k + s * num_kv_heads * head_dim + h * head_dim;
        float x0 = __half2float(head_k[i]);
        float x1 = __half2float(head_k[i + hdim_half]);
        head_k[i] = __float2half(x0 * cos_a - x1 * sin_a);
        head_k[i + hdim_half] = __float2half(x0 * sin_a + x1 * cos_a);
    }
}

__global__ void causal_attention_fp16_kernel(half* out, const half* q, size_t q_idx, const half* k,
                                              const half* v, size_t attend_len, size_t num_heads,
                                              size_t num_kv_heads, size_t head_dim, float scale)
{
    size_t h = blockIdx.x;
    if (h >= num_heads) return;

    extern __shared__ float sdata[];
    float* scores = sdata;
    float* reduce_buf = sdata + attend_len;

    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    size_t groups = num_heads / num_kv_heads;
    size_t kv_h = h / groups;

    const half* q_head = q + q_idx * num_heads * head_dim + h * head_dim;

    // QK^T scores (accumulate in FP32)
    for (size_t t = tid; t < attend_len; t += stride)
    {
        const half* k_head = k + t * num_kv_heads * head_dim + kv_h * head_dim;
        float dot = 0.0f;
        for (size_t d = 0; d < head_dim; d++)
            dot += __half2float(q_head[d]) * __half2float(k_head[d]);
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Softmax
    float local_max = -1e30f;
    for (size_t t = tid; t < attend_len; t += stride)
        if (scores[t] > local_max) local_max = scores[t];
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + s]);
        __syncthreads();
    }
    float max_val = reduce_buf[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (size_t t = tid; t < attend_len; t += stride)
    {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float sum = reduce_buf[0];
    __syncthreads();
    for (size_t t = tid; t < attend_len; t += stride)
        scores[t] /= sum;
    __syncthreads();

    // Weighted V sum → FP16 output
    half* head_out = out + h * head_dim;
    for (size_t d = tid; d < head_dim; d += stride)
    {
        float val = 0.0f;
        for (size_t t = 0; t < attend_len; t++)
            val += scores[t] * __half2float(v[t * num_kv_heads * head_dim + kv_h * head_dim + d]);
        head_out[d] = __float2half(val);
    }
}

__global__ void add_bias_fp16_kernel(half* out, const half* bias, size_t M, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N)
        out[idx] = __float2half(__half2float(out[idx]) + __half2float(bias[idx % N]));
}

__global__ void fp16_to_fp32_kernel(float* out, const half* in, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

// ============================================================
// FP16 kernel launch wrappers
// ============================================================

extern "C" void cuda_add_fp16(void* out, const void* x, const void* y, size_t n)
{
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    add_fp16_kernel<<<grid, block>>>((half*)out, (const half*)x, (const half*)y, n);
}

extern "C" void cuda_mul_fp16(void* out, const void* x, const void* y, size_t n)
{
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    mul_fp16_kernel<<<grid, block>>>((half*)out, (const half*)x, (const half*)y, n);
}

extern "C" void cuda_silu_fp16(void* out, const void* x, size_t n)
{
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    silu_fp16_kernel<<<grid, block>>>((half*)out, (const half*)x, n);
}

extern "C" void cuda_silu_mul_fp16(void* out, const void* gate, const void* up, size_t n)
{
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    silu_mul_fp16_kernel<<<grid, block>>>((half*)out, (const half*)gate, (const half*)up, n);
}

extern "C" void cuda_rms_norm_fp16(void* out, const void* x, const void* weight, size_t n,
                                    float eps)
{
    size_t block = 256;
    size_t smem = block * sizeof(float);
    rms_norm_fp16_kernel<<<1, block, smem>>>((half*)out, (const half*)x, (const half*)weight, n,
                                              eps);
}

extern "C" void cuda_softmax_fp16(void* out, const void* x, size_t n)
{
    size_t block = 256;
    size_t smem = block * sizeof(float);
    softmax_fp16_kernel<<<1, block, smem>>>((half*)out, (const half*)x, n);
}

extern "C" void cuda_rope_fp16(void* q, void* k, size_t seq_len, size_t num_heads,
                                size_t num_kv_heads, size_t head_dim, size_t pos, float theta)
{
    size_t hdim_half = head_dim / 2;
    size_t max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;
    size_t total_pairs = hdim_half * max_heads;
    size_t block = 256;
    size_t grid_x = (total_pairs + block - 1) / block;
    dim3 grid(grid_x, seq_len);
    rope_fp16_kernel<<<grid, block>>>((half*)q, (half*)k, seq_len, num_heads, num_kv_heads,
                                       head_dim, pos, theta);
}

extern "C" void cuda_embedding_fp16(void* out, const void* table, size_t token_id,
                                     size_t hidden_size)
{
    CUDA_CHECK(cudaMemcpy(out, (const half*)table + token_id * hidden_size,
                           hidden_size * sizeof(half), cudaMemcpyDeviceToDevice));
}

extern "C" void cuda_causal_attention_fp16(void* out, const void* q, size_t q_idx, const void* k,
                                            const void* v, size_t attend_len, size_t num_heads,
                                            size_t num_kv_heads, size_t head_dim, float scale)
{
    size_t block = 256;
    size_t smem = (attend_len + block) * sizeof(float);
    causal_attention_fp16_kernel<<<num_heads, block, smem>>>(
        (half*)out, (const half*)q, q_idx, (const half*)k, (const half*)v, attend_len, num_heads,
        num_kv_heads, head_dim, scale);
}

extern "C" void cuda_add_bias_fp16(void* out, const void* bias, size_t M, size_t N)
{
    size_t total = M * N;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    add_bias_fp16_kernel<<<grid, block>>>((half*)out, (const half*)bias, M, N);
}

extern "C" void cuda_cublas_hgemm_fp16(void* out_fp16, const void* inp_fp16,
                                        const void* weight_fp16, size_t M, size_t N, size_t K,
                                        float beta)
{
    float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(g_cublas_handle,
                               CUBLAS_OP_T, CUBLAS_OP_N,
                               (int)N, (int)M, (int)K,
                               &alpha,
                               weight_fp16, CUDA_R_16F, (int)K,
                               inp_fp16, CUDA_R_16F, (int)K,
                               &beta,
                               out_fp16, CUDA_R_16F, (int)N,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

extern "C" void cuda_fp16_to_fp32(float* out, const void* in, size_t n)
{
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    fp16_to_fp32_kernel<<<grid, block>>>(out, (const half*)in, n);
}

// ============================================================
// GPU Sampling — Gumbel-max trick
// ============================================================

// Hash-based RNG: deterministic, no curand dependency
__device__ float gpu_rand(unsigned long long seed, int step, int idx)
{
    unsigned long long h = seed ^ ((unsigned long long)step * 0x9E3779B97F4A7C15ULL +
                                    (unsigned long long)idx * 0x517CC1B727220A95ULL);
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
    h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
    h = h ^ (h >> 31);
    // Convert to float in (0, 1)
    return (float)(h >> 11) * (1.0f / (float)(1ULL << 53)) + 1e-10f;
}

// Single-block kernel: 256 threads cooperatively process all vocab entries
// temperature=0: argmax. temperature>0: Gumbel-max trick (equivalent to categorical sampling)
// top_p: applied by finding probability threshold via softmax + cumulative check
template <typename T>
__global__ void sample_kernel(const T* logits, size_t n, float temperature,
                               unsigned long long seed, int step, int* result)
{
    extern __shared__ float sdata[];
    // Layout: [0..blockDim.x) = values, [blockDim.x..2*blockDim.x) = indices
    float* s_val = sdata;
    int* s_idx = reinterpret_cast<int*>(sdata + blockDim.x);

    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    if (temperature == 0.0f)
    {
        // Greedy: find argmax
        float best_val = -1e30f;
        int best_idx = 0;
        for (size_t i = tid; i < n; i += stride)
        {
            float v = (float)logits[i];
            if (v > best_val)
            {
                best_val = v;
                best_idx = (int)i;
            }
        }
        s_val[tid] = best_val;
        s_idx[tid] = best_idx;
        __syncthreads();

        for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s && s_val[tid + s] > s_val[tid])
            {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) *result = s_idx[0];
    }
    else
    {
        // Gumbel-max: logit/temperature + Gumbel noise → argmax = categorical sample
        float best_val = -1e30f;
        int best_idx = 0;
        for (size_t i = tid; i < n; i += stride)
        {
            float v = (float)logits[i] / temperature;
            float u = gpu_rand(seed, step, (int)i);
            float gumbel = -logf(-logf(u));
            float noisy = v + gumbel;
            if (noisy > best_val)
            {
                best_val = noisy;
                best_idx = (int)i;
            }
        }
        s_val[tid] = best_val;
        s_idx[tid] = best_idx;
        __syncthreads();

        for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s && s_val[tid + s] > s_val[tid])
            {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) *result = s_idx[0];
    }
}

// Persistent device memory for the result (avoid per-call cudaMalloc)
static int* d_sample_result_ = nullptr;

extern "C" int cuda_sample(const void* logits, size_t n, float temperature, float top_p,
                            unsigned long long seed, int step, int fp16)
{
    (void)top_p;  // Gumbel-max implicitly handles the distribution; top_p TBD

    if (!d_sample_result_)
    {
        d_sample_result_ = (int*)cuda_malloc(sizeof(int));
    }

    size_t block = 256;
    size_t smem = block * (sizeof(float) + sizeof(int));

    if (fp16)
    {
        sample_kernel<half><<<1, block, smem>>>(
            (const half*)logits, n, temperature, seed, step, d_sample_result_);
    }
    else
    {
        sample_kernel<float><<<1, block, smem>>>(
            (const float*)logits, n, temperature, seed, step, d_sample_result_);
    }

    int result;
    cuda_memcpy_d2h(&result, d_sample_result_, sizeof(int));
    return result;
}
