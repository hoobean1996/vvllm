#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "vvllm/config/config.h"

namespace vvllm
{

ModelConfig load_config(const std::string& filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("failed to open config: " + filepath);
    }
    auto json = nlohmann::json::parse(file);
    ModelConfig config;
    // model_type
    // qwen, llama, mistral, gemma
    config.model_type = json["model_type"].get<std::string>();
    // vocab_size
    // number of tokens in total
    // qwen: 151936
    config.vocab_size = json["vocab_size"].get<std::size_t>();
    // hidden_size
    // the width of the vectors flowing through the model.
    // qwen: 896
    // 896 / 14 (heads) = 64 per head
    config.hidden_size = json["hidden_size"].get<std::size_t>();
    // intermediate_size
    // the width of the MLP's hidden layer.
    // Think of it as MLP think wider temporarily
    config.intermediate_size = json["intermediate_size"].get<std::size_t>();
    // num_hideen_layers
    // awen: 24
    config.num_hidden_layers = json["num_hidden_layers"].get<std::size_t>();
    // num_attention_heads
    // qwen: 14
    config.num_attention_heads = json["num_attention_heads"].get<std::size_t>();
    // num_key_value_heads
    // qwen: 2
    config.num_key_value_heads = json["num_key_value_heads"].get<std::size_t>();
    config.max_position_embeddings = json["max_position_embeddings"].get<std::size_t>();
    config.rms_norm_eps = json["rms_norm_eps"].get<double>();
    config.rope_theta = json["rope_theta"].get<double>();
    config.bos_token_id = json["bos_token_id"].get<int>();
    auto& eos = json["eos_token_id"];
    if (eos.is_array())
    {
        config.eos_token_ids = eos.get<std::vector<int>>();
    }
    else
    {
        config.eos_token_ids = {eos.get<int>()};
    }
    config.tie_word_embeddings = json["tie_word_embeddings"].get<bool>();
    return config;
}

}  // namespace vvllm
