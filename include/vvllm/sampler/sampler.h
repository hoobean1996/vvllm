#pragma once

#include <cstdint>
#include <vector>

namespace vvllm
{

class Sampler
{
public:
    virtual ~Sampler() = default;

    /// Sample a token ID from the logits vector.
    virtual int sample(const std::vector<float>& logits, int step) = 0;
};

}  // namespace vvllm
