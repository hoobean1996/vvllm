#pragma once

#include <cstddef>

namespace vvllm
{

class KVCache
{
public:
    virtual ~KVCache() = default;

    virtual void init(std::size_t num_layers, std::size_t kv_dim) = 0;
    virtual void append(std::size_t layer, const float* k, const float* v,
                        std::size_t num_tokens) = 0;
    virtual const float* k(std::size_t layer) const = 0;
    virtual const float* v(std::size_t layer) const = 0;
    virtual void advance(std::size_t num_tokens) = 0;
    virtual void reset() = 0;
    virtual void truncate(std::size_t new_seq_len) = 0;
};

}  // namespace vvllm
