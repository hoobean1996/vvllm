#include <cmath>
#include <cstddef>
#include <vector>

#include "gtest/gtest.h"
#include "vvllm/quantize/quantize.h"

namespace vvllm
{
namespace
{

TEST(QuantizeTest, RoundtripAccuracy)
{
    // Small matrix with known values
    // clang-format off
    std::vector<float> weight = {
        0.5f, -0.3f,  0.1f,  0.9f,
        1.0f,  0.0f, -1.0f,  0.5f,
        0.0f,  0.2f,  0.4f, -0.8f,
    };
    // clang-format on
    const std::size_t N = 3, K = 4;

    auto qw = quantize_per_channel(weight.data(), N, K);

    ASSERT_EQ(qw.data.size(), N * K);
    ASSERT_EQ(qw.scales.size(), N);

    // Verify roundtrip: dequantized value should be close to original
    for (std::size_t j = 0; j < N; j++)
    {
        for (std::size_t k = 0; k < K; k++)
        {
            float dequant = static_cast<float>(qw.data[j * K + k]) * qw.scales[j];
            float orig = weight[j * K + k];
            // Max quantization error per element is scale/2 (rounding)
            EXPECT_NEAR(dequant, orig, qw.scales[j] + 1e-6f)
                << "at [" << j << "," << k << "]";
        }
    }
}

TEST(QuantizeTest, ZeroRow)
{
    // A row of all zeros should produce zero scale and zero int8 values
    std::vector<float> weight = {0.0f, 0.0f, 0.0f, 0.0f};
    auto qw = quantize_per_channel(weight.data(), 1, 4);

    EXPECT_FLOAT_EQ(qw.scales[0], 0.0f);
    for (std::size_t k = 0; k < 4; k++)
    {
        EXPECT_EQ(qw.data[k], 0);
    }
}

TEST(QuantizeTest, Saturation)
{
    // Values at the extreme should clamp to -128..127
    std::vector<float> weight = {127.0f, -128.0f, 0.0f, 50.0f};
    auto qw = quantize_per_channel(weight.data(), 1, 4);

    // scale = 128/127 ≈ 1.0079
    // int8(-128) should be exactly -127 (because 128/scale = 127, clamped to 127,
    // but -128/scale = -127, which rounds to -127)
    // The absolute value of the largest element is 128, so scale = 128/127
    EXPECT_FLOAT_EQ(qw.scales[0], 128.0f / 127.0f);

    // 127 / scale = 127 * 127/128 ≈ 126.0 -> rounds to 126
    // -128 / scale = -128 * 127/128 = -127.0 -> -127
    EXPECT_EQ(qw.data[1], -127);  // -128/scale = -127

    // Verify all values are in [-128, 127]
    for (std::size_t k = 0; k < 4; k++)
    {
        EXPECT_GE(qw.data[k], -128);
        EXPECT_LE(qw.data[k], 127);
    }
}

TEST(QuantizeTest, LargeMatrix)
{
    // Larger matrix to test performance doesn't break anything
    const std::size_t N = 64, K = 128;
    std::vector<float> weight(N * K);
    for (std::size_t i = 0; i < N * K; i++)
    {
        weight[i] = static_cast<float>(i % 255) / 127.0f - 1.0f;
    }

    auto qw = quantize_per_channel(weight.data(), N, K);
    ASSERT_EQ(qw.data.size(), N * K);
    ASSERT_EQ(qw.scales.size(), N);

    // All scales should be positive (non-zero rows)
    for (std::size_t j = 0; j < N; j++)
    {
        EXPECT_GT(qw.scales[j], 0.0f);
    }
}

}  // namespace
}  // namespace vvllm
