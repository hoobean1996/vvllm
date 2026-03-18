#include "src/kv_cache/kv_cache_cuda.h"

#include <algorithm>
#include <cstdint>

#include "src/backend/cuda_kernels.h"

namespace vvllm
{

KVCacheCUDA::KVCacheCUDA(bool fp16) : fp16_(fp16) {}

KVCacheCUDA::~KVCacheCUDA()
{
    for (auto& kv : layers_)
    {
        if (kv.d_k) cuda_free(kv.d_k);
        if (kv.d_v) cuda_free(kv.d_v);
    }
}

void KVCacheCUDA::init(std::size_t num_layers, std::size_t kv_dim)
{
    if (!layers_.empty()) return;
    layers_.resize(num_layers);
    kv_dim_ = kv_dim;
    seq_len_ = 0;

    std::size_t elem_size = fp16_ ? sizeof(uint16_t) : sizeof(float);
    std::size_t init_cap = 128 * kv_dim;
    std::size_t init_bytes = init_cap * elem_size;
    for (auto& kv : layers_)
    {
        kv.d_k = cuda_malloc(init_bytes);
        kv.d_v = cuda_malloc(init_bytes);
        kv.capacity = init_cap;
    }
}

void KVCacheCUDA::append(std::size_t layer, const float* k, const float* v,
                         std::size_t num_tokens)
{
    auto& kv = layers_[layer];
    std::size_t new_seq_len = seq_len_ + num_tokens;
    std::size_t needed = new_seq_len * kv_dim_;
    std::size_t elem_size = fp16_ ? sizeof(uint16_t) : sizeof(float);

    // Grow buffer if needed (2x growth factor)
    if (needed > kv.capacity)
    {
        std::size_t new_cap = std::max(needed, kv.capacity * 2);
        std::size_t new_bytes = new_cap * elem_size;
        void* new_dk = cuda_malloc(new_bytes);
        void* new_dv = cuda_malloc(new_bytes);
        if (kv.d_k)
        {
            std::size_t old_bytes = seq_len_ * kv_dim_ * elem_size;
            cuda_memcpy_d2d(new_dk, kv.d_k, old_bytes);
            cuda_memcpy_d2d(new_dv, kv.d_v, old_bytes);
            cuda_free(kv.d_k);
            cuda_free(kv.d_v);
        }
        kv.d_k = new_dk;
        kv.d_v = new_dv;
        kv.capacity = new_cap;
    }

    // Copy new tokens to GPU KV buffer
    // In FP16 mode, source pointers are FP16 device pointers (from BackendCUDA mirrors)
    std::size_t token_bytes = num_tokens * kv_dim_ * elem_size;
    std::size_t offset_bytes = seq_len_ * kv_dim_ * elem_size;
    cuda_memcpy_auto(static_cast<char*>(kv.d_k) + offset_bytes, k, token_bytes);
    cuda_memcpy_auto(static_cast<char*>(kv.d_v) + offset_bytes, v, token_bytes);
}

const float* KVCacheCUDA::k(std::size_t layer) const
{
    if (layer >= layers_.size()) return nullptr;
    return static_cast<const float*>(layers_[layer].d_k);
}

const float* KVCacheCUDA::v(std::size_t layer) const
{
    if (layer >= layers_.size()) return nullptr;
    return static_cast<const float*>(layers_[layer].d_v);
}

void KVCacheCUDA::reset() { seq_len_ = 0; }

void KVCacheCUDA::truncate(std::size_t new_seq_len)
{
    if (new_seq_len < seq_len_) seq_len_ = new_seq_len;
}

}  // namespace vvllm
