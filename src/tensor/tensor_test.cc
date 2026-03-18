#include "gtest/gtest.h"
#include "vvllm/tensor/tensor.h"

namespace vvllm
{
namespace
{

TEST(TensorTest, ConstructAndDestruct)
{
    Tensor t({2, 3}, DType::Float32, Device::CPU);

    ASSERT_NE(t.data<float>(), nullptr);
    EXPECT_EQ(t.size(), 6);
    EXPECT_EQ(t.bytes(), 6 * sizeof(float));
    ASSERT_EQ(t.shape().size(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST(TensorTest, ConstructDifferentTypes)
{
    Tensor f({4, 4}, DType::Float32, Device::CPU);
    Tensor h({4, 4}, DType::Float16, Device::CPU);

    EXPECT_EQ(f.size(), 16);
    EXPECT_EQ(f.bytes(), 16 * sizeof(float));
    EXPECT_EQ(h.size(), 16);
    EXPECT_EQ(h.bytes(), 16 * 2);
}

TEST(TensorTest, MoveConstruct)
{
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    float* original_data = a.data<float>();

    Tensor b(std::move(a));

    EXPECT_EQ(b.data<float>(), original_data);
    EXPECT_EQ(b.size(), 6);
    ASSERT_EQ(b.shape().size(), 2);
    EXPECT_EQ(b.shape()[0], 2);
    EXPECT_EQ(b.shape()[1], 3);

    EXPECT_EQ(a.data<float>(), nullptr);
    EXPECT_EQ(a.size(), 0);
}

TEST(TensorTest, MoveAssign)
{
    Tensor a({2, 3}, DType::Float32, Device::CPU);
    Tensor b({4, 4}, DType::Float32, Device::CPU);
    float* original_data = a.data<float>();

    b = std::move(a);

    EXPECT_EQ(b.data<float>(), original_data);
    EXPECT_EQ(b.size(), 6);
    ASSERT_EQ(b.shape().size(), 2);
    EXPECT_EQ(b.shape()[0], 2);
    EXPECT_EQ(b.shape()[1], 3);

    EXPECT_EQ(a.data<float>(), nullptr);
    EXPECT_EQ(a.size(), 0);
}

TEST(TensorTest, DataIsWritable)
{
    Tensor t({2, 3}, DType::Float32, Device::CPU);

    for (std::size_t i = 0; i < t.size(); i++)
    {
        t.data<float>()[i] = static_cast<float>(i);
    }

    for (std::size_t i = 0; i < t.size(); i++)
    {
        EXPECT_FLOAT_EQ(t.data<float>()[i], static_cast<float>(i));
    }
}

}  // namespace
}  // namespace vvllm
