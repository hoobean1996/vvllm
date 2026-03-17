#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace vvllm
{

class Sampler
{
public:
    /// temperature: scaling factor for logits (0 = greedy argmax).
    /// top_p: nucleus sampling threshold in (0, 1].
    /// seed: random seed for reproducibility.
    Sampler(float temperature, float top_p, std::uint64_t seed);

    /// Sample a token ID from the logits vector.
    int sample(const std::vector<float>& logits);

private:
    float temperature_;
    float top_p_;
    std::mt19937 rng_;
};

}  // namespace vvllm
