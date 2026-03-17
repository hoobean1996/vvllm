#include <cstdio>
#include <fstream>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "vvllm/config/config.h"

namespace vvllm
{
namespace
{

std::string write_test_config(const nlohmann::json& json)
{
    std::string path = std::tmpnam(nullptr);
    std::ofstream out(path);
    out << json.dump();
    out.close();
    return path;
}

nlohmann::json make_qwen_config()
{
    return {
        {"model_type", "qwen2"},       {"vocab_size", 151936},
        {"hidden_size", 896},          {"intermediate_size", 4864},
        {"num_hidden_layers", 24},     {"num_attention_heads", 14},
        {"num_key_value_heads", 2},    {"max_position_embeddings", 32768},
        {"rms_norm_eps", 1e-06},       {"rope_theta", 1000000.0},
        {"bos_token_id", 151643},      {"eos_token_id", 151643},
        {"tie_word_embeddings", true},
    };
}

TEST(ConfigTest, LoadQwenConfig)
{
    auto json = make_qwen_config();
    std::string path = write_test_config(json);

    auto config = load_config(path);

    EXPECT_EQ(config.model_type, "qwen2");
    EXPECT_EQ(config.vocab_size, 151936);
    EXPECT_EQ(config.hidden_size, 896);
    EXPECT_EQ(config.intermediate_size, 4864);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 14);
    EXPECT_EQ(config.num_key_value_heads, 2);
    EXPECT_EQ(config.max_position_embeddings, 32768);
    EXPECT_DOUBLE_EQ(config.rms_norm_eps, 1e-06);
    EXPECT_DOUBLE_EQ(config.rope_theta, 1000000.0);
    EXPECT_EQ(config.bos_token_id, 151643);
    ASSERT_EQ(config.eos_token_ids.size(), 1);
    EXPECT_EQ(config.eos_token_ids[0], 151643);
    EXPECT_TRUE(config.tie_word_embeddings);

    std::remove(path.c_str());
}

TEST(ConfigTest, LoadDifferentModel)
{
    auto json = make_qwen_config();
    json["model_type"] = "llama";
    json["hidden_size"] = 4096;
    json["num_hidden_layers"] = 32;
    json["num_attention_heads"] = 32;
    json["num_key_value_heads"] = 8;
    json["tie_word_embeddings"] = false;

    std::string path = write_test_config(json);
    auto config = load_config(path);

    EXPECT_EQ(config.model_type, "llama");
    EXPECT_EQ(config.hidden_size, 4096);
    EXPECT_EQ(config.num_hidden_layers, 32);
    EXPECT_EQ(config.num_attention_heads, 32);
    EXPECT_EQ(config.num_key_value_heads, 8);
    EXPECT_FALSE(config.tie_word_embeddings);

    std::remove(path.c_str());
}

TEST(ConfigTest, LoadArrayEosTokenIds)
{
    auto json = make_qwen_config();
    json["eos_token_id"] = {151645, 151643};

    std::string path = write_test_config(json);
    auto config = load_config(path);

    ASSERT_EQ(config.eos_token_ids.size(), 2);
    EXPECT_EQ(config.eos_token_ids[0], 151645);
    EXPECT_EQ(config.eos_token_ids[1], 151643);

    std::remove(path.c_str());
}

TEST(ConfigTest, LoadInvalidFileThrows)
{
    EXPECT_THROW(load_config("/nonexistent/config.json"), std::runtime_error);
}

}  // namespace
}  // namespace vvllm
