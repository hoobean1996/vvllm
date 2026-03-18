#pragma once

#include <cstdint>

#include "vvllm/sampler/sampler.h"

namespace vvllm
{

class SamplerCUDA : public Sampler
{
public:
    SamplerCUDA(float temperature, float top_p, std::uint64_t seed);

    int sample(const Tensor& logits, int step) override;

private:
    float temperature_;
    float top_p_;
    std::uint64_t seed_;
};

}  // namespace vvllm
