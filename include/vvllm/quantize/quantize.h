#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vvllm
{

struct QuantizedWeight
{
    std::vector<int8_t> data;   // [N * K] row-major
    std::vector<float> scales;  // [N] per-row scale factors
};

/// Quantize an fp32 weight matrix to int8 per-channel (per-row) absmax.
/// weight is [N, K] row-major. Returns owned int8 data + scales.
QuantizedWeight quantize_per_channel(const float* weight, std::size_t N, std::size_t K);

}  // namespace vvllm
