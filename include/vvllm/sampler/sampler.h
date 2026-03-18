#pragma once

#include <cstdint>

#include "vvllm/tensor/tensor.h"

namespace vvllm
{

class Sampler
{
public:
    virtual ~Sampler() = default;

    /// Sample a token ID from the logits tensor (CPU or GPU).
    virtual int sample(const Tensor& logits, int step) = 0;
};

}  // namespace vvllm
