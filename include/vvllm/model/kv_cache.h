#pragma once

#include <cstddef>
#include <vector>

namespace vvllm
{

struct KVCache
{
    std::vector<std::vector<float>> k_cache;  // [num_layers][cached_positions * kv_dim]
    std::vector<std::vector<float>> v_cache;  // [num_layers][cached_positions * kv_dim]
    std::size_t seq_len = 0;                  // total cached positions

    explicit KVCache(std::size_t num_layers);

    /// Append new K/V data for a single layer into the cache.
    void append(std::size_t layer_idx, const float* new_k, const float* new_v,
                std::size_t num_new_tokens, std::size_t kv_dim);

    const float* k_data(std::size_t layer_idx) const;
    const float* v_data(std::size_t layer_idx) const;

    /// Bump seq_len after all layers have been processed.
    void advance(std::size_t num_new_tokens);

    void reset();

    /// Truncate the cache to the given sequence length.
    /// Used by best-of-N to rewind to the prefill state between candidates.
    void truncate(std::size_t new_seq_len);
};

}  // namespace vvllm
