#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace vvllm
{

struct ModelConfig
{
    std::string model_type;
    std::size_t vocab_size;
    std::size_t hidden_size;
    std::size_t intermediate_size;
    std::size_t num_hidden_layers;
    std::size_t num_attention_heads;
    std::size_t num_key_value_heads;
    std::size_t max_position_embeddings;
    double rms_norm_eps;
    double rope_theta;
    int bos_token_id;
    std::vector<int> eos_token_ids;
    bool tie_word_embeddings;
};

/// Load model config from a config.json file.
ModelConfig load_config(const std::string& filepath);

}  // namespace vvllm
