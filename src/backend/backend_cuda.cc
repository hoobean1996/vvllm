#include "src/backend/backend_cuda.h"

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "src/backend/cuda_kernels.h"

namespace vvllm
{

BackendCUDA::BackendCUDA(bool fp16)
    : d_tmp_in_(nullptr),
      d_tmp_in_bytes_(0),
      d_tmp_in2_(nullptr),
      d_tmp_in2_bytes_(0),
      d_tmp_aux_{nullptr, nullptr},
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
    for (auto& [_, m] : mirrors_)
        cuda_free(m.d_ptr);
    if (d_tmp_in_) cuda_free(d_tmp_in_);
    if (d_tmp_in2_) cuda_free(d_tmp_in2_);
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

// ============================================================
// GPU Mirror System
// ============================================================

void* BackendCUDA::gpu_input(const void* cpu_ptr, std::size_t bytes, void*& fallback,
                             std::size_t& fallback_cap)
{
    // Path 1: exact mirror match with sufficient capacity
    auto it = mirrors_.find(cpu_ptr);
    if (it != mirrors_.end() && it->second.valid && it->second.capacity >= bytes)
    {
        return it->second.d_ptr;
    }

    const char* range_start = static_cast<const char*>(cpu_ptr);
    const char* range_end = range_start + bytes;

    // Path 2: check if a larger mirror fully covers this range
    // (handles sub-buffer reads, e.g. rms_norm reading x+offset when x has a full mirror)
    for (const auto& [addr, mirror] : mirrors_)
    {
        if (!mirror.valid) continue;
        const char* m_start = static_cast<const char*>(addr);
        const char* m_end = m_start + mirror.capacity;
        if (m_start <= range_start && m_end >= range_end)
        {
            std::size_t offset = static_cast<std::size_t>(range_start - m_start);
            return static_cast<char*>(mirror.d_ptr) + offset;
        }
    }

    // Path 3: upload from CPU, then overlay any valid sub-mirrors
    // (handles full-buffer reads when only per-token sub-mirrors exist)
    ensure_buf(fallback, fallback_cap, bytes);
    cuda_memcpy_h2d(fallback, cpu_ptr, bytes);

    for (const auto& [addr, mirror] : mirrors_)
    {
        if (!mirror.valid) continue;
        const char* m_start = static_cast<const char*>(addr);
        if (m_start >= range_start && m_start < range_end)
        {
            std::size_t offset = static_cast<std::size_t>(m_start - range_start);
            std::size_t copy_bytes = std::min(mirror.capacity, bytes - offset);
            cuda_memcpy_d2d(static_cast<char*>(fallback) + offset, mirror.d_ptr, copy_bytes);
        }
    }

    return fallback;
}

void* BackendCUDA::gpu_output(const void* cpu_ptr, std::size_t bytes, bool& reused)
{
    auto it = mirrors_.find(cpu_ptr);
    if (it != mirrors_.end())
    {
        if (it->second.capacity >= bytes)
        {
            reused = true;
            return it->second.d_ptr;
        }
        // Resize
        cuda_free(it->second.d_ptr);
        void* d = cuda_malloc(bytes);
        it->second.d_ptr = d;
        it->second.capacity = bytes;
        it->second.valid = false;
        reused = false;
        return d;
    }
    void* d = cuda_malloc(bytes);
    mirrors_.emplace(cpu_ptr, GpuMirror{d, bytes, false});
    reused = false;
    return d;
}

const void* BackendCUDA::device_ptr(const void* host_ptr) const
{
    auto it = mirrors_.find(host_ptr);
    if (it != mirrors_.end() && it->second.valid)
    {
        return it->second.d_ptr;
    }
    return nullptr;
}

void BackendCUDA::begin_forward()
{
    for (auto& [_, mirror] : mirrors_)
    {
        mirror.valid = false;
    }
}

void BackendCUDA::mirror_set_valid(const void* cpu_ptr)
{
    auto it = mirrors_.find(cpu_ptr);
    if (it != mirrors_.end())
    {
        it->second.valid = true;
    }
}

void BackendCUDA::finish_output(void* cpu_ptr, void* d_ptr, std::size_t bytes, bool reused)
{
    mirror_set_valid(cpu_ptr);
    if (!reused)
    {
        // New mirror: download to keep CPU in sync for sub-buffer patterns
        cuda_memcpy_d2h(cpu_ptr, d_ptr, bytes);
    }
}

void BackendCUDA::flush(const void* ptr, std::size_t bytes)
{
    auto it = mirrors_.find(ptr);
    if (it != mirrors_.end() && it->second.valid)
    {
        if (fp16_)
        {
            std::size_t n = bytes / sizeof(float);
            ensure_buf(d_tmp_fp16_in_, d_tmp_fp16_in_bytes_, bytes);
            cuda_fp16_to_fp32(static_cast<float*>(d_tmp_fp16_in_), it->second.d_ptr, n);
            cuda_memcpy_d2h(const_cast<void*>(ptr), d_tmp_fp16_in_, bytes);
        }
        else
        {
            cuda_memcpy_d2h(const_cast<void*>(ptr), it->second.d_ptr, bytes);
        }
    }
}

// ============================================================
// FP16 mirror helpers
// ============================================================

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

void* BackendCUDA::gpu_input_fp16(const void* cpu_ptr, std::size_t fp32_bytes, void*& fallback,
                                   std::size_t& fallback_cap)
{
    std::size_t fp16_bytes = fp32_bytes / 2;

    // Path 1: exact mirror match
    auto it = mirrors_.find(cpu_ptr);
    if (it != mirrors_.end() && it->second.valid && it->second.capacity >= fp16_bytes)
    {
        return it->second.d_ptr;
    }

    // Path 2: parent mirror covers this range
    const char* range_start = static_cast<const char*>(cpu_ptr);
    const char* range_end = range_start + fp32_bytes;
    for (const auto& [addr, mirror] : mirrors_)
    {
        if (!mirror.valid) continue;
        const char* m_start = static_cast<const char*>(addr);
        // Mirror stores FP16 → covers capacity*2 FP32 bytes of CPU range
        const char* m_end = m_start + mirror.capacity * 2;
        if (m_start <= range_start && m_end >= range_end)
        {
            std::size_t cpu_offset = static_cast<std::size_t>(range_start - m_start);
            return static_cast<char*>(mirror.d_ptr) + cpu_offset / 2;
        }
    }

    // Path 3: upload FP32 → convert to FP16
    ensure_buf(d_tmp_fp16_in_, d_tmp_fp16_in_bytes_, fp32_bytes);
    cuda_memcpy_h2d(d_tmp_fp16_in_, cpu_ptr, fp32_bytes);
    ensure_buf(fallback, fallback_cap, fp16_bytes);
    cuda_fp32_to_fp16(fallback, static_cast<float*>(d_tmp_fp16_in_), fp32_bytes / sizeof(float));

    // Overlay valid sub-mirrors
    for (const auto& [addr, mirror] : mirrors_)
    {
        if (!mirror.valid) continue;
        const char* m_start = static_cast<const char*>(addr);
        if (m_start >= range_start && m_start < range_end)
        {
            std::size_t cpu_offset = static_cast<std::size_t>(m_start - range_start);
            std::size_t dev_offset = cpu_offset / 2;
            std::size_t copy_bytes = std::min(mirror.capacity, fp16_bytes - dev_offset);
            cuda_memcpy_d2d(static_cast<char*>(fallback) + dev_offset, mirror.d_ptr, copy_bytes);
        }
    }

    return fallback;
}

void* BackendCUDA::gpu_output_fp16(const void* cpu_ptr, std::size_t fp32_bytes, bool& reused)
{
    std::size_t fp16_bytes = fp32_bytes / 2;
    auto it = mirrors_.find(cpu_ptr);
    if (it != mirrors_.end())
    {
        if (it->second.capacity >= fp16_bytes)
        {
            reused = true;
            return it->second.d_ptr;
        }
        cuda_free(it->second.d_ptr);
        void* d = cuda_malloc(fp16_bytes);
        it->second.d_ptr = d;
        it->second.capacity = fp16_bytes;
        it->second.valid = false;
        reused = false;
        return d;
    }
    void* d = cuda_malloc(fp16_bytes);
    mirrors_.emplace(cpu_ptr, GpuMirror{d, fp16_bytes, false});
    reused = false;
    return d;
}

void BackendCUDA::finish_output_fp16(void* cpu_ptr, void* d_ptr, std::size_t fp32_bytes,
                                      bool reused)
{
    mirror_set_valid(cpu_ptr);
    if (!reused)
    {
        std::size_t n = fp32_bytes / sizeof(float);
        ensure_buf(d_tmp_fp16_in_, d_tmp_fp16_in_bytes_, fp32_bytes);
        cuda_fp16_to_fp32(static_cast<float*>(d_tmp_fp16_in_), d_ptr, n);
        cuda_memcpy_d2h(cpu_ptr, d_tmp_fp16_in_, fp32_bytes);
    }
}

// ============================================================
// Element-wise ops
// ============================================================

void BackendCUDA::add(float* out, const float* x, const float* y, std::size_t n)
{
    std::size_t bytes = n * sizeof(float);
    if (fp16_)
    {
        void* d_x = gpu_input_fp16(x, bytes, d_tmp_in_, d_tmp_in_bytes_);
        void* d_y = gpu_input_fp16(y, bytes, d_tmp_in2_, d_tmp_in2_bytes_);
        bool reused;
        void* d_out = gpu_output_fp16(out, bytes, reused);
        cuda_add_fp16(d_out, d_x, d_y, n);
        finish_output_fp16(out, d_out, bytes, reused);
        return;
    }
    float* d_x = static_cast<float*>(gpu_input(x, bytes, d_tmp_in_, d_tmp_in_bytes_));
    float* d_y = static_cast<float*>(gpu_input(y, bytes, d_tmp_in2_, d_tmp_in2_bytes_));

    bool reused;
    float* d_out = static_cast<float*>(gpu_output(out, bytes, reused));

    cuda_add(d_out, d_x, d_y, n);
    finish_output(out, d_out, bytes, reused);
}

void BackendCUDA::mul(float* out, const float* x, const float* y, std::size_t n)
{
    std::size_t bytes = n * sizeof(float);
    if (fp16_)
    {
        void* d_x = gpu_input_fp16(x, bytes, d_tmp_in_, d_tmp_in_bytes_);
        void* d_y = gpu_input_fp16(y, bytes, d_tmp_in2_, d_tmp_in2_bytes_);
        bool reused;
        void* d_out = gpu_output_fp16(out, bytes, reused);
        cuda_mul_fp16(d_out, d_x, d_y, n);
        finish_output_fp16(out, d_out, bytes, reused);
        return;
    }
    float* d_x = static_cast<float*>(gpu_input(x, bytes, d_tmp_in_, d_tmp_in_bytes_));
    float* d_y = static_cast<float*>(gpu_input(y, bytes, d_tmp_in2_, d_tmp_in2_bytes_));

    bool reused;
    float* d_out = static_cast<float*>(gpu_output(out, bytes, reused));

    cuda_mul(d_out, d_x, d_y, n);
    finish_output(out, d_out, bytes, reused);
}

void BackendCUDA::silu(float* out, const float* x, std::size_t n)
{
    std::size_t bytes = n * sizeof(float);
    if (fp16_)
    {
        void* d_x = gpu_input_fp16(x, bytes, d_tmp_in_, d_tmp_in_bytes_);
        bool reused;
        void* d_out = gpu_output_fp16(out, bytes, reused);
        cuda_silu_fp16(d_out, d_x, n);
        finish_output_fp16(out, d_out, bytes, reused);
        return;
    }
    float* d_x = static_cast<float*>(gpu_input(x, bytes, d_tmp_in_, d_tmp_in_bytes_));

    bool reused;
    float* d_out = static_cast<float*>(gpu_output(out, bytes, reused));

    cuda_silu(d_out, d_x, n);
    finish_output(out, d_out, bytes, reused);
}

void BackendCUDA::silu_mul(float* out, const float* gate, const float* up, std::size_t n)
{
    std::size_t bytes = n * sizeof(float);
    if (fp16_)
    {
        void* d_gate = gpu_input_fp16(gate, bytes, d_tmp_in_, d_tmp_in_bytes_);
        void* d_up = gpu_input_fp16(up, bytes, d_tmp_in2_, d_tmp_in2_bytes_);
        bool reused;
        void* d_out = gpu_output_fp16(out, bytes, reused);
        cuda_silu_mul_fp16(d_out, d_gate, d_up, n);
        finish_output_fp16(out, d_out, bytes, reused);
        return;
    }
    float* d_gate = static_cast<float*>(gpu_input(gate, bytes, d_tmp_in_, d_tmp_in_bytes_));
    float* d_up = static_cast<float*>(gpu_input(up, bytes, d_tmp_in2_, d_tmp_in2_bytes_));

    bool reused;
    float* d_out = static_cast<float*>(gpu_output(out, bytes, reused));

    cuda_silu_mul(d_out, d_gate, d_up, n);
    finish_output(out, d_out, bytes, reused);
}

void BackendCUDA::linear_add(float* out, const float* inp, const float* weight, const float* bias,
                              const float* residual, std::size_t M, std::size_t N, std::size_t K)
{
    std::size_t inp_bytes = M * K * sizeof(float);
    std::size_t out_bytes = M * N * sizeof(float);

    if (fp16_)
    {
        void* d_inp = gpu_input_fp16(inp, inp_bytes, d_tmp_in_, d_tmp_in_bytes_);
        void* d_w = cache_weight_as_fp16(weight, N * K * sizeof(float));
        void* d_bias = bias ? cache_weight_as_fp16(bias, N * sizeof(float)) : nullptr;
        void* d_res = gpu_input_fp16(residual, out_bytes, d_tmp_in2_, d_tmp_in2_bytes_);
        bool reused;
        void* d_out = gpu_output_fp16(out, out_bytes, reused);
        std::size_t fp16_out_bytes = M * N * sizeof(uint16_t);
        if (d_res != d_out) cuda_memcpy_d2d(d_out, d_res, fp16_out_bytes);
        cuda_cublas_hgemm_fp16(d_out, d_inp, d_w, M, N, K, 1.0f);
        if (d_bias) cuda_add_bias_fp16(d_out, d_bias, M, N);
        finish_output_fp16(out, d_out, out_bytes, reused);
        return;
    }

    float* d_inp = static_cast<float*>(gpu_input(inp, inp_bytes, d_tmp_in_, d_tmp_in_bytes_));
    float* d_w = static_cast<float*>(cache_weight(weight, N * K * sizeof(float)));
    float* d_bias = bias ? static_cast<float*>(cache_weight(bias, N * sizeof(float))) : nullptr;
    float* d_res = static_cast<float*>(gpu_input(residual, out_bytes, d_tmp_in2_, d_tmp_in2_bytes_));

    bool reused;
    float* d_out = static_cast<float*>(gpu_output(out, out_bytes, reused));

    // Copy residual into d_out so cuBLAS beta=1.0 accumulates into it
    if (d_res != d_out) cuda_memcpy_d2d(d_out, d_res, out_bytes);

    cuda_cublas_sgemm(d_out, d_inp, d_w, M, N, K, 1.0f);
    if (d_bias) cuda_add_bias(d_out, d_bias, M, N);
    finish_output(out, d_out, out_bytes, reused);
}

void BackendCUDA::linear_q8_add(float* out, const float* inp, const int8_t* weight,
                                  const float* scales, const float* bias, const float* residual,
                                  std::size_t M, std::size_t N, std::size_t K)
{
    std::size_t inp_bytes = M * K * sizeof(float);
    std::size_t out_bytes = M * N * sizeof(float);

    if (fp16_)
    {
        void* d_inp = gpu_input_fp16(inp, inp_bytes, d_tmp_in_, d_tmp_in_bytes_);
        void* d_w_fp16 = cache_weight_fp16(weight, scales, N, K);
        void* d_bias = bias ? cache_weight_as_fp16(bias, N * sizeof(float)) : nullptr;
        void* d_res = gpu_input_fp16(residual, out_bytes, d_tmp_in2_, d_tmp_in2_bytes_);
        bool reused;
        void* d_out = gpu_output_fp16(out, out_bytes, reused);
        std::size_t fp16_out_bytes = M * N * sizeof(uint16_t);
        if (d_res != d_out) cuda_memcpy_d2d(d_out, d_res, fp16_out_bytes);
        cuda_cublas_hgemm_fp16(d_out, d_inp, d_w_fp16, M, N, K, 1.0f);
        if (d_bias) cuda_add_bias_fp16(d_out, d_bias, M, N);
        finish_output_fp16(out, d_out, out_bytes, reused);
        return;
    }

    float* d_inp = static_cast<float*>(gpu_input(inp, inp_bytes, d_tmp_in_, d_tmp_in_bytes_));
    void* d_w_fp16 = cache_weight_fp16(weight, scales, N, K);
    float* d_bias = bias ? static_cast<float*>(cache_weight(bias, N * sizeof(float))) : nullptr;
    float* d_res = static_cast<float*>(gpu_input(residual, out_bytes, d_tmp_in2_, d_tmp_in2_bytes_));

    // Convert FP32 input to FP16
    std::size_t fp16_inp_bytes = M * K * sizeof(uint16_t);
    ensure_buf(d_tmp_fp16_in_, d_tmp_fp16_in_bytes_, fp16_inp_bytes);
    cuda_fp32_to_fp16(d_tmp_fp16_in_, d_inp, M * K);

    bool reused;
    float* d_out = static_cast<float*>(gpu_output(out, out_bytes, reused));

    // Copy residual into d_out so cuBLAS beta=1.0 accumulates into it
    if (d_res != d_out) cuda_memcpy_d2d(d_out, d_res, out_bytes);

    cuda_cublas_hgemm(d_out, d_tmp_fp16_in_, d_w_fp16, M, N, K, 1.0f);
    if (d_bias) cuda_add_bias(d_out, d_bias, M, N);
    finish_output(out, d_out, out_bytes, reused);
}

// ============================================================
// Reductions
// ============================================================

void BackendCUDA::rms_norm(float* out, const float* x, const float* weight, std::size_t n,
                           float eps)
{
    std::size_t bytes = n * sizeof(float);
    if (fp16_)
    {
        void* d_x = gpu_input_fp16(x, bytes, d_tmp_in_, d_tmp_in_bytes_);
        void* d_w = cache_weight_as_fp16(weight, bytes);
        bool reused;
        void* d_out = gpu_output_fp16(out, bytes, reused);
        cuda_rms_norm_fp16(d_out, d_x, d_w, n, eps);
        finish_output_fp16(out, d_out, bytes, reused);
        return;
    }
    float* d_x = static_cast<float*>(gpu_input(x, bytes, d_tmp_in_, d_tmp_in_bytes_));
    float* d_w = static_cast<float*>(cache_weight(weight, bytes));

    bool reused;
    float* d_out = static_cast<float*>(gpu_output(out, bytes, reused));

    cuda_rms_norm(d_out, d_x, d_w, n, eps);
    finish_output(out, d_out, bytes, reused);
}

void BackendCUDA::softmax(float* out, const float* x, std::size_t n)
{
    std::size_t bytes = n * sizeof(float);
    if (fp16_)
    {
        void* d_x = gpu_input_fp16(x, bytes, d_tmp_in_, d_tmp_in_bytes_);
        bool reused;
        void* d_out = gpu_output_fp16(out, bytes, reused);
        cuda_softmax_fp16(d_out, d_x, n);
        finish_output_fp16(out, d_out, bytes, reused);
        return;
    }
    float* d_x = static_cast<float*>(gpu_input(x, bytes, d_tmp_in_, d_tmp_in_bytes_));

    bool reused;
    float* d_out = static_cast<float*>(gpu_output(out, bytes, reused));

    cuda_softmax(d_out, d_x, n);
    finish_output(out, d_out, bytes, reused);
}

// ============================================================
// RoPE (in-place: q and k are both input and output)
// ============================================================

void BackendCUDA::rope(float* q, float* k, std::size_t seq_len, std::size_t num_heads,
                       std::size_t num_kv_heads, std::size_t head_dim, std::size_t pos,
                       float theta)
{
    std::size_t q_bytes = seq_len * num_heads * head_dim * sizeof(float);
    std::size_t k_bytes = seq_len * num_kv_heads * head_dim * sizeof(float);

    if (fp16_)
    {
        bool q_reused, k_reused;
        void* d_q = gpu_output_fp16(q, q_bytes, q_reused);
        void* d_k = gpu_output_fp16(k, k_bytes, k_reused);
        auto q_it = mirrors_.find(q);
        if (q_it == mirrors_.end() || !q_it->second.valid)
        {
            std::size_t n_q = seq_len * num_heads * head_dim;
            ensure_buf(d_tmp_fp16_in_, d_tmp_fp16_in_bytes_, q_bytes);
            cuda_memcpy_h2d(d_tmp_fp16_in_, q, q_bytes);
            cuda_fp32_to_fp16(d_q, static_cast<float*>(d_tmp_fp16_in_), n_q);
        }
        auto k_it = mirrors_.find(k);
        if (k_it == mirrors_.end() || !k_it->second.valid)
        {
            std::size_t n_k = seq_len * num_kv_heads * head_dim;
            ensure_buf(d_tmp_fp16_in_, d_tmp_fp16_in_bytes_, k_bytes);
            cuda_memcpy_h2d(d_tmp_fp16_in_, k, k_bytes);
            cuda_fp32_to_fp16(d_k, static_cast<float*>(d_tmp_fp16_in_), n_k);
        }
        cuda_rope_fp16(d_q, d_k, seq_len, num_heads, num_kv_heads, head_dim, pos, theta);
        finish_output_fp16(q, d_q, q_bytes, q_reused);
        finish_output_fp16(k, d_k, k_bytes, k_reused);
        return;
    }

    // Get or create mirrors
    bool q_reused, k_reused;
    float* d_q = static_cast<float*>(gpu_output(q, q_bytes, q_reused));
    float* d_k = static_cast<float*>(gpu_output(k, k_bytes, k_reused));

    // Upload if mirrors don't have valid data
    auto q_it = mirrors_.find(q);
    if (q_it == mirrors_.end() || !q_it->second.valid)
    {
        cuda_memcpy_h2d(d_q, q, q_bytes);
    }
    auto k_it = mirrors_.find(k);
    if (k_it == mirrors_.end() || !k_it->second.valid)
    {
        cuda_memcpy_h2d(d_k, k, k_bytes);
    }

    cuda_rope(d_q, d_k, seq_len, num_heads, num_kv_heads, head_dim, pos, theta);
    finish_output(q, d_q, q_bytes, q_reused);
    finish_output(k, d_k, k_bytes, k_reused);
}

// ============================================================
// Linear
// ============================================================

void BackendCUDA::linear(float* out, const float* inp, const float* weight, const float* bias,
                         std::size_t M, std::size_t N, std::size_t K)
{
    std::size_t inp_bytes = M * K * sizeof(float);
    std::size_t out_bytes = M * N * sizeof(float);

    if (fp16_)
    {
        void* d_inp = gpu_input_fp16(inp, inp_bytes, d_tmp_in_, d_tmp_in_bytes_);
        void* d_w = cache_weight_as_fp16(weight, N * K * sizeof(float));
        void* d_bias = bias ? cache_weight_as_fp16(bias, N * sizeof(float)) : nullptr;
        bool reused;
        void* d_out = gpu_output_fp16(out, out_bytes, reused);
        cuda_cublas_hgemm_fp16(d_out, d_inp, d_w, M, N, K, 0.0f);
        if (d_bias) cuda_add_bias_fp16(d_out, d_bias, M, N);
        finish_output_fp16(out, d_out, out_bytes, reused);
        return;
    }

    float* d_inp = static_cast<float*>(gpu_input(inp, inp_bytes, d_tmp_in_, d_tmp_in_bytes_));
    float* d_w = static_cast<float*>(cache_weight(weight, N * K * sizeof(float)));
    float* d_bias = bias ? static_cast<float*>(cache_weight(bias, N * sizeof(float))) : nullptr;

    bool reused;
    float* d_out = static_cast<float*>(gpu_output(out, out_bytes, reused));

    cuda_cublas_sgemm(d_out, d_inp, d_w, M, N, K, 0.0f);
    if (d_bias) cuda_add_bias(d_out, d_bias, M, N);
    finish_output(out, d_out, out_bytes, reused);
}

void BackendCUDA::linear_q8(float* out, const float* inp, const int8_t* weight,
                             const float* scales, const float* bias, std::size_t M, std::size_t N,
                             std::size_t K)
{
    std::size_t inp_bytes = M * K * sizeof(float);
    std::size_t out_bytes = M * N * sizeof(float);

    if (fp16_)
    {
        void* d_inp = gpu_input_fp16(inp, inp_bytes, d_tmp_in_, d_tmp_in_bytes_);
        void* d_w_fp16 = cache_weight_fp16(weight, scales, N, K);
        void* d_bias = bias ? cache_weight_as_fp16(bias, N * sizeof(float)) : nullptr;
        bool reused;
        void* d_out = gpu_output_fp16(out, out_bytes, reused);
        cuda_cublas_hgemm_fp16(d_out, d_inp, d_w_fp16, M, N, K, 0.0f);
        if (d_bias) cuda_add_bias_fp16(d_out, d_bias, M, N);
        finish_output_fp16(out, d_out, out_bytes, reused);
        return;
    }

    float* d_inp = static_cast<float*>(gpu_input(inp, inp_bytes, d_tmp_in_, d_tmp_in_bytes_));
    void* d_w_fp16 = cache_weight_fp16(weight, scales, N, K);
    float* d_bias = bias ? static_cast<float*>(cache_weight(bias, N * sizeof(float))) : nullptr;

    // Convert FP32 input to FP16
    std::size_t fp16_inp_bytes = M * K * sizeof(uint16_t);
    ensure_buf(d_tmp_fp16_in_, d_tmp_fp16_in_bytes_, fp16_inp_bytes);
    cuda_fp32_to_fp16(d_tmp_fp16_in_, d_inp, M * K);

    bool reused;
    float* d_out = static_cast<float*>(gpu_output(out, out_bytes, reused));

    cuda_cublas_hgemm(d_out, d_tmp_fp16_in_, d_w_fp16, M, N, K, 0.0f);
    if (d_bias) cuda_add_bias(d_out, d_bias, M, N);
    finish_output(out, d_out, out_bytes, reused);
}

// ============================================================
// Embedding — CPU fallback, invalidate mirror
// ============================================================

void BackendCUDA::embedding(float* out, const float* table, std::size_t token_id,
                            std::size_t hidden_size, std::size_t vocab_size)
{
    std::size_t table_bytes = vocab_size * hidden_size * sizeof(float);
    std::size_t out_bytes = hidden_size * sizeof(float);

    if (fp16_)
    {
        void* d_table = cache_weight_as_fp16(table, table_bytes);
        bool reused;
        void* d_out = gpu_output_fp16(out, out_bytes, reused);
        cuda_embedding_fp16(d_out, d_table, token_id, hidden_size);
        finish_output_fp16(out, d_out, out_bytes, reused);
        return;
    }

    // GPU lookup: cache table, D2D copy one row
    float* d_table = static_cast<float*>(cache_weight(table, table_bytes));
    bool reused;
    float* d_out = static_cast<float*>(gpu_output(out, out_bytes, reused));
    cuda_embedding(d_out, d_table, token_id, hidden_size);
    finish_output(out, d_out, out_bytes, reused);
}

// ============================================================
// Matmul
// ============================================================

void BackendCUDA::matmul(float* out, const float* A, const float* B, std::size_t M, std::size_t N,
                         std::size_t K)
{
    std::size_t a_bytes = M * K * sizeof(float);
    std::size_t out_bytes = M * N * sizeof(float);

    float* d_a = static_cast<float*>(gpu_input(A, a_bytes, d_tmp_in_, d_tmp_in_bytes_));
    float* d_b = static_cast<float*>(cache_weight(B, K * N * sizeof(float)));

    bool reused;
    float* d_out = static_cast<float*>(gpu_output(out, out_bytes, reused));

    cuda_matmul(d_out, d_a, d_b, M, N, K);
    finish_output(out, d_out, out_bytes, reused);
}

// ============================================================
// Causal Attention
// ============================================================

void BackendCUDA::causal_attention(float* out, const float* q, std::size_t q_idx, const float* k,
                                   const float* v, std::size_t attend_len, std::size_t num_heads,
                                   std::size_t num_kv_heads, std::size_t head_dim, float scale)
{
    std::size_t q_bytes = (q_idx + 1) * num_heads * head_dim * sizeof(float);
    std::size_t kv_bytes = attend_len * num_kv_heads * head_dim * sizeof(float);
    std::size_t out_bytes = num_heads * head_dim * sizeof(float);

    if (fp16_)
    {
        void* d_q = gpu_input_fp16(q, q_bytes, d_tmp_in_, d_tmp_in_bytes_);

        // k/v from KVCacheCUDA are already FP16 device pointers
        void* d_k = nullptr;
        void* d_v = nullptr;
        cudaPointerAttributes attrs;
        if (cudaPointerGetAttributes(&attrs, k) == cudaSuccess &&
            attrs.type == cudaMemoryTypeDevice)
        {
            d_k = const_cast<float*>(k);
            d_v = const_cast<float*>(v);
        }
        else
        {
            std::size_t fp16_kv_bytes = kv_bytes / 2;
            ensure_buf(d_tmp_aux_[0], d_tmp_aux_bytes_[0], fp16_kv_bytes);
            ensure_buf(d_tmp_fp16_in_, d_tmp_fp16_in_bytes_, kv_bytes);
            cuda_memcpy_h2d(d_tmp_fp16_in_, k, kv_bytes);
            std::size_t n_kv = attend_len * num_kv_heads * head_dim;
            cuda_fp32_to_fp16(d_tmp_aux_[0], static_cast<float*>(d_tmp_fp16_in_), n_kv);
            ensure_buf(d_tmp_aux_[1], d_tmp_aux_bytes_[1], fp16_kv_bytes);
            cuda_memcpy_h2d(d_tmp_fp16_in_, v, kv_bytes);
            cuda_fp32_to_fp16(d_tmp_aux_[1], static_cast<float*>(d_tmp_fp16_in_), n_kv);
            d_k = d_tmp_aux_[0];
            d_v = d_tmp_aux_[1];
        }

        bool reused;
        void* d_out = gpu_output_fp16(out, out_bytes, reused);
        cuda_causal_attention_fp16(d_out, d_q, q_idx, d_k, d_v, attend_len, num_heads,
                                    num_kv_heads, head_dim, scale);
        finish_output_fp16(out, d_out, out_bytes, reused);
        return;
    }

    // q likely has a mirror
    float* d_q = static_cast<float*>(gpu_input(q, q_bytes, d_tmp_in_, d_tmp_in_bytes_));

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

    bool reused;
    float* d_out = static_cast<float*>(gpu_output(out, out_bytes, reused));

    cuda_causal_attention(d_out, d_q, q_idx, d_k, d_v, attend_len, num_heads, num_kv_heads,
                          head_dim, scale);
    finish_output(out, d_out, out_bytes, reused);
}

int BackendCUDA::sample_gpu(const float* logits, std::size_t n, float temperature, float top_p,
                             std::uint64_t seed, int step)
{
    // Get logits from GPU mirror (they're already on GPU after forward pass)
    auto it = mirrors_.find(logits);
    if (it == mirrors_.end() || !it->second.valid) return -1;

    return cuda_sample(it->second.d_ptr, n, temperature, top_p, seed, step, fp16_ ? 1 : 0);
}

}  // namespace vvllm
