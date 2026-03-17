#include "gtest/gtest.h"
#include "src/backend/backend_naive.h"
#include "vvllm/tensor/tensor.h"

namespace vvllm
{
namespace
{

TEST(TensorTest, ConstructAndDestruct)
{
    BackendCPU backend;
    Tensor<float> t(backend, {2, 3});

    ASSERT_NE(t.data(), nullptr);
    EXPECT_EQ(t.size(), 6);
    EXPECT_EQ(t.bytes(), 6 * sizeof(float));
    ASSERT_EQ(t.shape().size(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST(TensorTest, ConstructDifferentTypes)
{
    BackendCPU backend;
    Tensor<float> f(backend, {4, 4});
    Tensor<double> d(backend, {4, 4});

    EXPECT_EQ(f.size(), 16);
    EXPECT_EQ(f.bytes(), 16 * sizeof(float));
    EXPECT_EQ(d.size(), 16);
    EXPECT_EQ(d.bytes(), 16 * sizeof(double));
}

TEST(TensorTest, MoveConstruct)
{
    BackendCPU backend;
    Tensor<float> a(backend, {2, 3});
    float* original_data = a.data();

    Tensor<float> b(std::move(a));

    EXPECT_EQ(b.data(), original_data);
    EXPECT_EQ(b.size(), 6);
    ASSERT_EQ(b.shape().size(), 2);
    EXPECT_EQ(b.shape()[0], 2);
    EXPECT_EQ(b.shape()[1], 3);

    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0);
}

TEST(TensorTest, MoveAssign)
{
    BackendCPU backend;
    Tensor<float> a(backend, {2, 3});
    Tensor<float> b(backend, {4, 4});
    float* original_data = a.data();

    b = std::move(a);

    EXPECT_EQ(b.data(), original_data);
    EXPECT_EQ(b.size(), 6);
    ASSERT_EQ(b.shape().size(), 2);
    EXPECT_EQ(b.shape()[0], 2);
    EXPECT_EQ(b.shape()[1], 3);

    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0);
}

TEST(TensorTest, DataIsWritable)
{
    BackendCPU backend;
    Tensor<float> t(backend, {2, 3});

    for (std::size_t i = 0; i < t.size(); i++)
    {
        t.data()[i] = static_cast<float>(i);
    }

    for (std::size_t i = 0; i < t.size(); i++)
    {
        EXPECT_FLOAT_EQ(t.data()[i], static_cast<float>(i));
    }
}

}  // namespace
}  // namespace vvllm
