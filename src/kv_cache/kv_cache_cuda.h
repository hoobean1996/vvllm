#pragma once

#include <cstddef>
#include <vector>

#include "vvllm/kv_cache/kv_cache.h"

namespace vvllm
{

class KVCacheCUDA : public KVCache
{
public:
    explicit KVCacheCUDA(bool fp16 = false);
    ~KVCacheCUDA() override;

    void init(std::size_t num_layers, std::size_t kv_dim) override;
    void append(std::size_t layer, const float* k, const float* v,
                std::size_t num_tokens) override;
    const float* k(std::size_t layer) const override;
    const float* v(std::size_t layer) const override;
    void reset() override;
    void truncate(std::size_t new_seq_len) override;

private:
    struct GpuKVLayer
    {
        void* d_k = nullptr;
        void* d_v = nullptr;
        std::size_t capacity = 0;
    };
    std::vector<GpuKVLayer> layers_;
    bool fp16_;
};

}  // namespace vvllm
