#pragma once

#include <vector>

#include "vvllm/kv_cache/kv_cache.h"

namespace vvllm
{

class KVCacheCPU : public KVCache
{
public:
    void init(std::size_t num_layers, std::size_t kv_dim) override;
    void append(std::size_t layer, const float* k, const float* v,
                std::size_t num_tokens) override;
    const float* k(std::size_t layer) const override;
    const float* v(std::size_t layer) const override;
    void reset() override;
    void truncate(std::size_t new_seq_len) override;

private:
    std::vector<std::vector<float>> k_cache_;
    std::vector<std::vector<float>> v_cache_;
};

}  // namespace vvllm
