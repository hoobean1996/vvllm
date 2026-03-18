#pragma once

#include <cstdint>
#include <random>

#include "vvllm/sampler/sampler.h"

namespace vvllm
{

class SamplerCPU : public Sampler
{
public:
    SamplerCPU(float temperature, float top_p, std::uint64_t seed);

    int sample(const std::vector<float>& logits, int step) override;

private:
    float temperature_;
    float top_p_;
    std::mt19937 rng_;
};

}  // namespace vvllm
