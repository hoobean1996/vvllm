#include <algorithm>
#include <cmath>
#include <numeric>

#include "vvllm/sampler/sampler.h"

namespace vvllm
{

Sampler::Sampler(float temperature, float top_p, std::uint64_t seed)
    : temperature_(temperature), top_p_(top_p), rng_(seed)
{
}

int Sampler::sample(const std::vector<float>& logits)
{
    const std::size_t n = logits.size();

    // Greedy argmax when temperature is 0
    if (temperature_ == 0.0f)
    {
        return static_cast<int>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    // Temperature scaling
    std::vector<float> scaled(n);
    for (std::size_t i = 0; i < n; i++)
    {
        scaled[i] = logits[i] / temperature_;
    }

    // Softmax (subtract max for numerical stability)
    float max_val = *std::max_element(scaled.begin(), scaled.end());
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; i++)
    {
        scaled[i] = std::exp(scaled[i] - max_val);
        sum += scaled[i];
    }
    for (std::size_t i = 0; i < n; i++)
    {
        scaled[i] /= sum;
    }

    // Top-p: partial sort — only sort enough tokens to cover top_p.
    // For typical distributions, top-p is satisfied within a few hundred tokens.
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    auto cmp = [&scaled](int a, int b) { return scaled[a] > scaled[b]; };

    constexpr std::size_t kInitialK = 256;
    std::size_t k = std::min(kInitialK, n);
    auto k_it = indices.begin() + static_cast<std::ptrdiff_t>(k);
    std::nth_element(indices.begin(), k_it, indices.end(), cmp);
    std::sort(indices.begin(), k_it, cmp);

    float cumsum = 0.0f;
    std::size_t cutoff = n;
    for (std::size_t i = 0; i < k; i++)
    {
        cumsum += scaled[indices[i]];
        if (cumsum >= top_p_)
        {
            cutoff = i + 1;
            break;
        }
    }

    // Fallback: if top-256 didn't cover top_p, sort the remainder
    if (cumsum < top_p_)
    {
        std::sort(indices.begin() + static_cast<std::ptrdiff_t>(k), indices.end(), cmp);
        for (std::size_t i = k; i < n; i++)
        {
            cumsum += scaled[indices[i]];
            if (cumsum >= top_p_)
            {
                cutoff = i + 1;
                break;
            }
        }
    }

    // Renormalize the kept tokens
    float kept_sum = 0.0f;
    for (std::size_t i = 0; i < cutoff; i++)
    {
        kept_sum += scaled[indices[i]];
    }

    // Sample from the kept distribution
    std::uniform_real_distribution<float> dist(0.0f, kept_sum);
    float r = dist(rng_);

    float acc = 0.0f;
    for (std::size_t i = 0; i < cutoff; i++)
    {
        acc += scaled[indices[i]];
        if (r < acc)
        {
            return indices[i];
        }
    }

    return indices[cutoff - 1];
}

}  // namespace vvllm
