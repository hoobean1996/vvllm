#pragma once

#include <cstdint>

#include "vvllm/backend/backend.h"
#include "vvllm/sampler/sampler.h"

namespace vvllm
{

class SamplerCUDA : public Sampler
{
public:
    SamplerCUDA(float temperature, float top_p, std::uint64_t seed, Backend& backend);

    int sample(const std::vector<float>& logits, int step) override;

private:
    float temperature_;
    float top_p_;
    std::uint64_t seed_;
    Backend& backend_;
};

}  // namespace vvllm
