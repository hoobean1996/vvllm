#include "vvllm/quantize/quantize.h"

#include <algorithm>
#include <cmath>

namespace vvllm
{

QuantizedWeight quantize_per_channel(const float* weight, std::size_t N, std::size_t K)
{
    QuantizedWeight qw;
    qw.data.resize(N * K);
    qw.scales.resize(N);

    for (std::size_t j = 0; j < N; j++)
    {
        // Find absmax for this row
        float absmax = 0.0f;
        for (std::size_t k = 0; k < K; k++)
        {
            absmax = std::max(absmax, std::abs(weight[j * K + k]));
        }

        float scale = absmax / 127.0f;
        qw.scales[j] = scale;

        if (scale == 0.0f)
        {
            // Zero row: all int8 values are 0
            for (std::size_t k = 0; k < K; k++)
            {
                qw.data[j * K + k] = 0;
            }
        }
        else
        {
            float inv_scale = 1.0f / scale;
            for (std::size_t k = 0; k < K; k++)
            {
                float val = weight[j * K + k] * inv_scale;
                val = std::max(-128.0f, std::min(127.0f, std::round(val)));
                qw.data[j * K + k] = static_cast<int8_t>(val);
            }
        }
    }

    return qw;
}

}  // namespace vvllm
