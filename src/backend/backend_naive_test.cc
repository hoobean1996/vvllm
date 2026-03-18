#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "src/backend/backend_naive.h"

namespace vvllm
{
namespace
{

TEST(BackendCPUTest, AllocateDeallocate)
{
    BackendCPU backend;
    void* ptr = backend.allocate(1024);
    ASSERT_NE(ptr, nullptr);
    // Write to the allocation to verify it's usable.
    std::memset(ptr, 0xAB, 1024);
    backend.deallocate(ptr);
}

TEST(BackendCPUTest, MatmulIdentity)
{
    BackendCPU backend;
    // A(2×3) × I(3×3) = A(2×3)
    const std::size_t M = 2, K = 3, N = 3;
    // clang-format off
    const std::vector<float> A = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    const std::vector<float> I = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    // clang-format on
    std::vector<float> out(M * N, 0.0f);

    backend.matmul(out.data(), A.data(), I.data(), M, N, K);

    for (std::size_t i = 0; i < M * N; i++)
    {
        EXPECT_FLOAT_EQ(out[i], A[i]) << "mismatch at index " << i;
    }
}

TEST(BackendCPUTest, MatmulKnownResult)
{
    BackendCPU backend;
    // A(2×3) × B(3×2) = C(2×2)
    const std::size_t M = 2, K = 3, N = 2;
    // clang-format off
    const std::vector<float> A = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    const std::vector<float> B = {
        7.0f,  8.0f,
        9.0f,  10.0f,
        11.0f, 12.0f,
    };
    // Expected: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    //           [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    const std::vector<float> expected = {
        58.0f,  64.0f,
        139.0f, 154.0f,
    };
    // clang-format on
    std::vector<float> out(M * N, 0.0f);

    backend.matmul(out.data(), A.data(), B.data(), M, N, K);

    for (std::size_t i = 0; i < M * N; i++)
    {
        EXPECT_FLOAT_EQ(out[i], expected[i]) << "mismatch at index " << i;
    }
}

TEST(BackendCPUTest, MatmulSingleElement)
{
    BackendCPU backend;
    // A(1×1) × B(1×1) = C(1×1)
    const float A = 3.0f;
    const float B = 5.0f;
    float out = 0.0f;

    backend.matmul(&out, &A, &B, 1, 1, 1);

    EXPECT_FLOAT_EQ(out, 15.0f);
}

TEST(BackendCPUTest, MatmulVectorDotProduct)
{
    BackendCPU backend;
    // row(1×4) × col(4×1) = scalar(1×1)
    const std::size_t K = 4;
    const std::vector<float> row = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> col = {5.0f, 6.0f, 7.0f, 8.0f};
    // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    float out = 0.0f;

    backend.matmul(&out, row.data(), col.data(), 1, 1, K);

    EXPECT_FLOAT_EQ(out, 70.0f);
}

// ── add ──

TEST(BackendCPUTest, Add)
{
    BackendCPU backend;
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> y = {4.0f, 5.0f, 6.0f};
    std::vector<float> out(3);

    backend.add(out.data(), x.data(), y.data(), 3);

    EXPECT_FLOAT_EQ(out[0], 5.0f);
    EXPECT_FLOAT_EQ(out[1], 7.0f);
    EXPECT_FLOAT_EQ(out[2], 9.0f);
}

TEST(BackendCPUTest, AddInPlace)
{
    BackendCPU backend;
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> y = {4.0f, 5.0f, 6.0f};

    backend.add(x.data(), x.data(), y.data(), 3);

    EXPECT_FLOAT_EQ(x[0], 5.0f);
    EXPECT_FLOAT_EQ(x[1], 7.0f);
    EXPECT_FLOAT_EQ(x[2], 9.0f);
}

// ── mul ──

TEST(BackendCPUTest, Mul)
{
    BackendCPU backend;
    std::vector<float> x = {2.0f, 3.0f, 4.0f};
    std::vector<float> y = {5.0f, 6.0f, 7.0f};
    std::vector<float> out(3);

    backend.mul(out.data(), x.data(), y.data(), 3);

    EXPECT_FLOAT_EQ(out[0], 10.0f);
    EXPECT_FLOAT_EQ(out[1], 18.0f);
    EXPECT_FLOAT_EQ(out[2], 28.0f);
}

// ── silu ──

TEST(BackendCPUTest, SiluZero)
{
    BackendCPU backend;
    float x = 0.0f;
    float out = 0.0f;

    backend.silu(&out, &x, 1);

    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    EXPECT_FLOAT_EQ(out, 0.0f);
}

TEST(BackendCPUTest, SiluPositive)
{
    BackendCPU backend;
    std::vector<float> x = {1.0f, 5.0f};
    std::vector<float> out(2);

    backend.silu(out.data(), x.data(), 2);

    for (std::size_t i = 0; i < 2; i++)
    {
        float expected = x[i] / (1.0f + std::exp(-x[i]));
        EXPECT_NEAR(out[i], expected, 1e-6f) << "mismatch at index " << i;
    }
}

TEST(BackendCPUTest, SiluNegative)
{
    BackendCPU backend;
    float x = -10.0f;
    float out = 0.0f;

    backend.silu(&out, &x, 1);

    // silu(-10) ≈ -10 * sigmoid(-10) ≈ -10 * 0.0000454 ≈ near zero
    EXPECT_NEAR(out, 0.0f, 1e-3f);
}

// ── rms_norm ──

TEST(BackendCPUTest, RmsNormUniform)
{
    BackendCPU backend;
    // All same value: rms = sqrt(v^2 + eps) ≈ v, out[i] ≈ weight[i]
    std::vector<float> x = {2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> weight = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(4);

    backend.rms_norm(out.data(), x.data(), weight.data(), 4, 1e-6f);

    // rms = sqrt(4/4 + eps) = sqrt(4 + eps) ≈ 2.0
    // out[i] = 2.0 / 2.0 * 1.0 = 1.0
    for (std::size_t i = 0; i < 4; i++)
    {
        EXPECT_NEAR(out[i], 1.0f, 1e-5f);
    }
}

TEST(BackendCPUTest, RmsNormWithWeight)
{
    BackendCPU backend;
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> weight = {0.5f, 1.0f, 2.0f};
    std::vector<float> out(3);
    float eps = 1e-6f;

    backend.rms_norm(out.data(), x.data(), weight.data(), 3, eps);

    // rms = sqrt((1+4+9)/3 + eps) = sqrt(4.6667 + eps)
    float sum_sq = 1.0f + 4.0f + 9.0f;
    float rms = std::sqrt(sum_sq / 3.0f + eps);
    for (std::size_t i = 0; i < 3; i++)
    {
        float expected = x[i] / rms * weight[i];
        EXPECT_NEAR(out[i], expected, 1e-5f) << "mismatch at index " << i;
    }
}

// ── softmax ──

TEST(BackendCPUTest, SoftmaxSumsToOne)
{
    BackendCPU backend;
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4);

    backend.softmax(out.data(), x.data(), 4);

    float sum = std::accumulate(out.begin(), out.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST(BackendCPUTest, SoftmaxOrdering)
{
    BackendCPU backend;
    std::vector<float> x = {1.0f, 3.0f, 2.0f};
    std::vector<float> out(3);

    backend.softmax(out.data(), x.data(), 3);

    // Larger input → larger probability
    EXPECT_GT(out[1], out[2]);
    EXPECT_GT(out[2], out[0]);
}

TEST(BackendCPUTest, SoftmaxLargeValues)
{
    BackendCPU backend;
    // Without max subtraction, exp(1000) would overflow
    std::vector<float> x = {1000.0f, 1001.0f, 1002.0f};
    std::vector<float> out(3);

    backend.softmax(out.data(), x.data(), 3);

    float sum = std::accumulate(out.begin(), out.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
    EXPECT_GT(out[2], out[1]);
    EXPECT_GT(out[1], out[0]);
}

// ── rope ──

TEST(BackendCPUTest, RopePositionZeroIsIdentity)
{
    BackendCPU backend;
    // At position 0, angle = 0 for all freqs → cos=1, sin=0 → no change
    std::vector<float> q = {1.0f, 2.0f, 3.0f, 4.0f};  // 1 head, head_dim=4
    std::vector<float> k = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> q_orig = q;
    std::vector<float> k_orig = k;

    backend.rope(q.data(), k.data(), 1, 1, 1, 4, 0, 10000.0f);

    for (std::size_t i = 0; i < 4; i++)
    {
        EXPECT_NEAR(q[i], q_orig[i], 1e-6f);
        EXPECT_NEAR(k[i], k_orig[i], 1e-6f);
    }
}

TEST(BackendCPUTest, RopePreservesNorm)
{
    BackendCPU backend;
    // Rotation preserves vector length
    std::vector<float> q = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> k = {5.0f, 6.0f, 7.0f, 8.0f};

    float q_norm_before = 0.0f;
    for (auto v : q) q_norm_before += v * v;

    backend.rope(q.data(), k.data(), 1, 1, 1, 4, 7, 10000.0f);

    float q_norm_after = 0.0f;
    for (auto v : q) q_norm_after += v * v;

    EXPECT_NEAR(q_norm_before, q_norm_after, 1e-4f);
}

// ── embedding ──

TEST(BackendCPUTest, EmbeddingLookup)
{
    BackendCPU backend;
    // Table: 3 tokens, hidden_size=4
    // clang-format off
    std::vector<float> table = {
        0.1f, 0.2f, 0.3f, 0.4f,   // token 0
        0.5f, 0.6f, 0.7f, 0.8f,   // token 1
        0.9f, 1.0f, 1.1f, 1.2f,   // token 2
    };
    // clang-format on
    std::vector<float> out(4);

    backend.embedding(out.data(), table.data(), 1, 4, 3);

    EXPECT_FLOAT_EQ(out[0], 0.5f);
    EXPECT_FLOAT_EQ(out[1], 0.6f);
    EXPECT_FLOAT_EQ(out[2], 0.7f);
    EXPECT_FLOAT_EQ(out[3], 0.8f);
}

TEST(BackendCPUTest, EmbeddingFirstToken)
{
    BackendCPU backend;
    std::vector<float> table = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(3);

    backend.embedding(out.data(), table.data(), 0, 3, 2);

    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
}

// ── linear_q8 ──

TEST(BackendCPUTest, LinearQ8MatchesFp32)
{
    BackendCPU backend;

    // weight [3, 4] row-major
    // clang-format off
    std::vector<float> weight_fp32 = {
        0.5f, -0.3f,  0.1f,  0.9f,
        1.0f,  0.0f, -1.0f,  0.5f,
        0.0f,  0.2f,  0.4f, -0.8f,
    };
    // clang-format on
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::size_t N = 3, K = 4;

    // Compute fp32 reference
    std::vector<float> out_fp32(N);
    backend.linear(out_fp32.data(), input.data(), weight_fp32.data(), nullptr, 1, N, K);

    // Manually quantize
    std::vector<int8_t> weight_q8(N * K);
    std::vector<float> scales(N);
    for (std::size_t j = 0; j < N; j++)
    {
        float absmax = 0.0f;
        for (std::size_t k = 0; k < K; k++)
        {
            absmax = std::max(absmax, std::abs(weight_fp32[j * K + k]));
        }
        scales[j] = absmax / 127.0f;
        for (std::size_t k = 0; k < K; k++)
        {
            float val = weight_fp32[j * K + k] / scales[j];
            val = std::max(-128.0f, std::min(127.0f, std::round(val)));
            weight_q8[j * K + k] = static_cast<int8_t>(val);
        }
    }

    // Compute int8 result
    std::vector<float> out_q8(N);
    backend.linear_q8(out_q8.data(), input.data(), weight_q8.data(), scales.data(), nullptr, 1, N,
                      K);

    // Should be close to fp32 result (within quantization error)
    for (std::size_t j = 0; j < N; j++)
    {
        // Tolerance: scales[j] * K (worst case: each element off by scale/2)
        float tol = scales[j] * static_cast<float>(K) * 0.5f + 1e-5f;
        EXPECT_NEAR(out_q8[j], out_fp32[j], tol) << "mismatch at output " << j;
    }
}

TEST(BackendCPUTest, LinearQ8WithBias)
{
    BackendCPU backend;
    // Simple case: identity-like int8 weight with bias
    const std::size_t N = 2, K = 2;
    std::vector<int8_t> weight = {127, 0, 0, 127};
    std::vector<float> scales = {1.0f / 127.0f, 1.0f / 127.0f};  // scale so 127 -> 1.0
    std::vector<float> bias = {10.0f, 20.0f};
    std::vector<float> input = {3.0f, 5.0f};
    std::vector<float> output(N);

    backend.linear_q8(output.data(), input.data(), weight.data(), scales.data(), bias.data(), 1, N,
                      K);

    // out[0] = scale[0] * (127*3 + 0*5) + 10 = (1/127)*381 + 10 = 3.0 + 10 = 13.0
    // out[1] = scale[1] * (0*3 + 127*5) + 20 = (1/127)*635 + 20 = 5.0 + 20 = 25.0
    EXPECT_NEAR(output[0], 13.0f, 1e-4f);
    EXPECT_NEAR(output[1], 25.0f, 1e-4f);
}

}  // namespace
}  // namespace vvllm
