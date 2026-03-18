#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "vvllm/safetensors/safetensors.h"

namespace vvllm
{
namespace
{

/// Helper: write a minimal safetensors file to disk.
/// Format: 8-byte header_size (LE) | JSON header | raw tensor data
std::string write_test_safetensors(const nlohmann::json& header, const std::vector<uint8_t>& data)
{
    std::string path = std::tmpnam(nullptr);
    std::string header_str = header.dump();
    uint64_t header_size = header_str.size();

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header_size), 8);
    out.write(header_str.data(), header_size);
    out.write(reinterpret_cast<const char*>(data.data()), data.size());
    out.close();

    return path;
}

TEST(SafeTensorsLoaderTest, ParseSingleTensor)
{
    // Create a tensor with 3 float32 values: [1.0, 2.0, 3.0]
    std::vector<uint8_t> data(3 * sizeof(float));
    float values[] = {1.0f, 2.0f, 3.0f};
    std::memcpy(data.data(), values, data.size());

    nlohmann::json header = {
        {"test_tensor",
         {{"dtype", "F32"}, {"shape", {3}}, {"data_offsets", {0, 3 * sizeof(float)}}}}};

    std::string path = write_test_safetensors(header, data);
    SafeTensorsLoader loader(path);
    loader.parse();

    auto& infos = loader.tensor_infos();
    ASSERT_EQ(infos.size(), 1);
    ASSERT_TRUE(infos.count("test_tensor"));
    auto& info = infos.at("test_tensor");
    EXPECT_EQ(info.name, "test_tensor");
    EXPECT_EQ(info.dtype, "F32");
    ASSERT_EQ(info.shape.size(), 1);
    EXPECT_EQ(info.shape[0], 3);
    EXPECT_EQ(info.offset_begin, 0);
    EXPECT_EQ(info.offset_end, 3 * sizeof(float));

    std::remove(path.c_str());
}

TEST(SafeTensorsLoaderTest, ParseMultipleTensors)
{
    // Two tensors back to back
    // tensor_a: 2x2 BF16 = 4 elements * 2 bytes = 8 bytes at [0, 8)
    // tensor_b: 3 F32 = 3 elements * 4 bytes = 12 bytes at [8, 20)
    std::vector<uint8_t> data(20, 0);

    nlohmann::json header = {
        {"tensor_a", {{"dtype", "BF16"}, {"shape", {2, 2}}, {"data_offsets", {0, 8}}}},
        {"tensor_b", {{"dtype", "F32"}, {"shape", {3}}, {"data_offsets", {8, 20}}}},
    };

    std::string path = write_test_safetensors(header, data);
    SafeTensorsLoader loader(path);
    loader.parse();

    auto& infos = loader.tensor_infos();
    ASSERT_EQ(infos.size(), 2);

    ASSERT_TRUE(infos.count("tensor_a"));
    auto& a = infos.at("tensor_a");
    EXPECT_EQ(a.dtype, "BF16");
    ASSERT_EQ(a.shape.size(), 2);
    EXPECT_EQ(a.shape[0], 2);
    EXPECT_EQ(a.shape[1], 2);
    EXPECT_EQ(a.offset_begin, 0);
    EXPECT_EQ(a.offset_end, 8);

    ASSERT_TRUE(infos.count("tensor_b"));
    auto& b = infos.at("tensor_b");
    EXPECT_EQ(b.dtype, "F32");
    ASSERT_EQ(b.shape.size(), 1);
    EXPECT_EQ(b.shape[0], 3);
    EXPECT_EQ(b.offset_begin, 8);
    EXPECT_EQ(b.offset_end, 20);

    std::remove(path.c_str());
}

TEST(SafeTensorsLoaderTest, ParseSkipsMetadata)
{
    std::vector<uint8_t> data(4, 0);

    nlohmann::json header = {
        {"__metadata__", {{"format", "pt"}}},
        {"weight", {{"dtype", "F32"}, {"shape", {1}}, {"data_offsets", {0, 4}}}},
    };

    std::string path = write_test_safetensors(header, data);
    SafeTensorsLoader loader(path);
    loader.parse();

    auto& infos = loader.tensor_infos();
    ASSERT_EQ(infos.size(), 1);
    ASSERT_TRUE(infos.count("weight"));

    std::remove(path.c_str());
}

TEST(SafeTensorsLoaderTest, ParseInvalidFileThrows)
{
    EXPECT_THROW(
        {
            SafeTensorsLoader loader("/nonexistent/path/model.safetensors");
            loader.parse();
        },
        std::runtime_error);
}

TEST(SafeTensorsLoaderTest, LoadTensorF32)
{
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<uint8_t> data(sizeof(values));
    std::memcpy(data.data(), values, sizeof(values));

    nlohmann::json header = {
        {"weight", {{"dtype", "F32"}, {"shape", {2, 3}}, {"data_offsets", {0, sizeof(values)}}}},
    };

    std::string path = write_test_safetensors(header, data);
    SafeTensorsLoader loader(path);
    loader.parse();

    auto tensor = loader.load_tensor("weight");

    ASSERT_EQ(tensor.shape().size(), 2);
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 3);
    EXPECT_EQ(tensor.size(), 6);

    for (size_t i = 0; i < 6; i++)
    {
        EXPECT_FLOAT_EQ(tensor.data<float>()[i], values[i]) << "mismatch at index " << i;
    }

    std::remove(path.c_str());
}

TEST(SafeTensorsLoaderTest, LoadTensorBF16)
{
    // BF16 for 1.0f = 0x3F80, 2.0f = 0x4000, -3.0f = 0xC040
    uint16_t bf16_values[] = {0x3F80, 0x4000, 0xC040};
    float expected[] = {1.0f, 2.0f, -3.0f};

    std::vector<uint8_t> data(sizeof(bf16_values));
    std::memcpy(data.data(), bf16_values, sizeof(bf16_values));

    nlohmann::json header = {
        {"weight",
         {{"dtype", "BF16"}, {"shape", {3}}, {"data_offsets", {0, sizeof(bf16_values)}}}},
    };

    std::string path = write_test_safetensors(header, data);
    SafeTensorsLoader loader(path);
    loader.parse();

    auto tensor = loader.load_tensor("weight");

    ASSERT_EQ(tensor.size(), 3);
    for (size_t i = 0; i < 3; i++)
    {
        EXPECT_FLOAT_EQ(tensor.data<float>()[i], expected[i]) << "mismatch at index " << i;
    }

    std::remove(path.c_str());
}

TEST(SafeTensorsLoaderTest, LoadTensorUnsupportedDtypeThrows)
{
    std::vector<uint8_t> data(8, 0);

    nlohmann::json header = {
        {"weight", {{"dtype", "F16"}, {"shape", {4}}, {"data_offsets", {0, 8}}}},
    };

    std::string path = write_test_safetensors(header, data);
    SafeTensorsLoader loader(path);
    loader.parse();

    EXPECT_THROW(loader.load_tensor("weight"), std::runtime_error);

    std::remove(path.c_str());
}

TEST(SafeTensorsLoaderTest, LoadTensorNotFoundThrows)
{
    std::vector<uint8_t> data(4, 0);

    nlohmann::json header = {
        {"weight", {{"dtype", "F32"}, {"shape", {1}}, {"data_offsets", {0, 4}}}},
    };

    std::string path = write_test_safetensors(header, data);
    SafeTensorsLoader loader(path);
    loader.parse();

    EXPECT_THROW(loader.load_tensor("nonexistent"), std::out_of_range);

    std::remove(path.c_str());
}

TEST(SafeTensorsLoaderTest, LoadAll)
{
    // Two F32 tensors: a=[1.0, 2.0], b=[3.0]
    float a_vals[] = {1.0f, 2.0f};
    float b_vals[] = {3.0f};
    std::vector<uint8_t> data(sizeof(a_vals) + sizeof(b_vals));
    std::memcpy(data.data(), a_vals, sizeof(a_vals));
    std::memcpy(data.data() + sizeof(a_vals), b_vals, sizeof(b_vals));

    nlohmann::json header = {
        {"tensor_a", {{"dtype", "F32"}, {"shape", {2}}, {"data_offsets", {0, sizeof(a_vals)}}}},
        {"tensor_b",
         {{"dtype", "F32"},
          {"shape", {1}},
          {"data_offsets", {sizeof(a_vals), sizeof(a_vals) + sizeof(b_vals)}}}},
    };

    std::string path = write_test_safetensors(header, data);
    SafeTensorsLoader loader(path);
    loader.parse();

    auto tensors = loader.load_all();

    ASSERT_EQ(tensors.size(), 2);

    ASSERT_TRUE(tensors.count("tensor_a"));
    auto& a = tensors.at("tensor_a");
    EXPECT_EQ(a.size(), 2);
    EXPECT_FLOAT_EQ(a.data<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(a.data<float>()[1], 2.0f);

    ASSERT_TRUE(tensors.count("tensor_b"));
    auto& b = tensors.at("tensor_b");
    EXPECT_EQ(b.size(), 1);
    EXPECT_FLOAT_EQ(b.data<float>()[0], 3.0f);

    std::remove(path.c_str());
}

}  // namespace
}  // namespace vvllm
