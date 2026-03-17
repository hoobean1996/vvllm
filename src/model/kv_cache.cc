#include "vvllm/model/kv_cache.h"

namespace vvllm
{

KVCache::KVCache(std::size_t num_layers) : k_cache(num_layers), v_cache(num_layers) {}

void KVCache::append(std::size_t layer_idx, const float* new_k, const float* new_v,
                     std::size_t num_new_tokens, std::size_t kv_dim)
{
    std::size_t count = num_new_tokens * kv_dim;
    auto& kc = k_cache[layer_idx];
    auto& vc = v_cache[layer_idx];
    kc.insert(kc.end(), new_k, new_k + count);
    vc.insert(vc.end(), new_v, new_v + count);
}

const float* KVCache::k_data(std::size_t layer_idx) const { return k_cache[layer_idx].data(); }

const float* KVCache::v_data(std::size_t layer_idx) const { return v_cache[layer_idx].data(); }

void KVCache::advance(std::size_t num_new_tokens) { seq_len += num_new_tokens; }

void KVCache::reset()
{
    for (auto& kc : k_cache)
    {
        kc.clear();
    }
    for (auto& vc : v_cache)
    {
        vc.clear();
    }
    seq_len = 0;
}

void KVCache::truncate(std::size_t new_seq_len)
{
    if (new_seq_len >= seq_len) return;
    if (seq_len == 0) return;

    std::size_t kv_dim = k_cache[0].size() / seq_len;
    std::size_t new_size = new_seq_len * kv_dim;
    for (auto& kc : k_cache)
    {
        kc.resize(new_size);
    }
    for (auto& vc : v_cache)
    {
        vc.resize(new_size);
    }
    seq_len = new_seq_len;
}

}  // namespace vvllm
