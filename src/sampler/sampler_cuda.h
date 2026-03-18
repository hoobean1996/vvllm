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
    ~SamplerCUDA() override;

    int sample(const std::vector<float>& logits, int step) override;

private:
    float temperature_;
    float top_p_;
    std::uint64_t seed_;
    Backend& backend_;

    // GPU temp buffer for uploading logits
    void* d_logits_ = nullptr;
    std::size_t d_logits_bytes_ = 0;
};

}  // namespace vvllm
