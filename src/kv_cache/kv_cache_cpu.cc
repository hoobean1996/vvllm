#include "src/kv_cache/kv_cache_cpu.h"

namespace vvllm
{

void KVCacheCPU::init(std::size_t num_layers, std::size_t kv_dim)
{
    if (!k_cache_.empty()) return;
    k_cache_.resize(num_layers);
    v_cache_.resize(num_layers);
    kv_dim_ = kv_dim;
}

void KVCacheCPU::append(std::size_t layer, const float* k, const float* v,
                        std::size_t num_tokens)
{
    std::size_t count = num_tokens * kv_dim_;
    auto& kc = k_cache_[layer];
    auto& vc = v_cache_[layer];
    kc.insert(kc.end(), k, k + count);
    vc.insert(vc.end(), v, v + count);
}

const float* KVCacheCPU::k(std::size_t layer) const { return k_cache_[layer].data(); }

const float* KVCacheCPU::v(std::size_t layer) const { return v_cache_[layer].data(); }

void KVCacheCPU::reset()
{
    for (auto& kc : k_cache_) kc.clear();
    for (auto& vc : v_cache_) vc.clear();
    seq_len_ = 0;
}

void KVCacheCPU::truncate(std::size_t new_seq_len)
{
    if (new_seq_len >= seq_len_) return;
    std::size_t new_size = new_seq_len * kv_dim_;
    for (auto& kc : k_cache_) kc.resize(new_size);
    for (auto& vc : v_cache_) vc.resize(new_size);
    seq_len_ = new_seq_len;
}

}  // namespace vvllm
