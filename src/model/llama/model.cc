#include "vvllm/model/llama/model.h"

#include <cstring>

namespace vvllm
{

LlamaModel::LlamaModel(const ModelConfig& config, Backend& backend)
    : config_(config),
      backend_(backend),
      embed_tokens_(nullptr),
      final_norm_weight_(nullptr),
      layers_(config.num_hidden_layers)
{
}

void LlamaModel::load_weights(const std::unordered_map<std::string, Tensor<float>>& weights,
                              bool quantize)
{
    embed_tokens_ = weights.at("model.embed_tokens.weight").data();
    final_norm_weight_ = weights.at("model.norm.weight").data();

    const std::size_t head_dim = config_.hidden_size / config_.num_attention_heads;
    const std::size_t kv_dim = config_.num_key_value_heads * head_dim;

    auto set_weight = [&](LinearWeight& lw, const float* fp32, std::size_t N, std::size_t K) {
        if (quantize)
        {
            quantized_weights_.push_back(quantize_per_channel(fp32, N, K));
            auto& qw = quantized_weights_.back();
            lw = LinearWeight{nullptr, qw.data.data(), qw.scales.data()};
        }
        else
        {
            lw = LinearWeight{fp32, nullptr, nullptr};
        }
    };

    for (std::size_t i = 0; i < config_.num_hidden_layers; i++)
    {
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        auto& layer = layers_[i];

        layer.input_layernorm_weight = weights.at(prefix + "input_layernorm.weight").data();
        layer.post_attention_layernorm_weight =
            weights.at(prefix + "post_attention_layernorm.weight").data();

        auto& attn = layer.attention;
        set_weight(attn.q_proj_weight,
                   weights.at(prefix + "self_attn.q_proj.weight").data(),
                   config_.num_attention_heads * head_dim, config_.hidden_size);
        set_weight(attn.k_proj_weight,
                   weights.at(prefix + "self_attn.k_proj.weight").data(),
                   kv_dim, config_.hidden_size);
        set_weight(attn.v_proj_weight,
                   weights.at(prefix + "self_attn.v_proj.weight").data(),
                   kv_dim, config_.hidden_size);
        set_weight(attn.o_proj_weight,
                   weights.at(prefix + "self_attn.o_proj.weight").data(),
                   config_.hidden_size, config_.hidden_size);

        auto& mlp = layer.mlp;

        // Concatenate gate_proj and up_proj into a single [2*intermediate, hidden] weight
        const float* gate_fp32 = weights.at(prefix + "mlp.gate_proj.weight").data();
        const float* up_fp32 = weights.at(prefix + "mlp.up_proj.weight").data();
        std::size_t gate_up_size = 2 * config_.intermediate_size * config_.hidden_size;
        fused_weights_.emplace_back(gate_up_size);
        auto& gate_up = fused_weights_.back();
        std::size_t half_size = config_.intermediate_size * config_.hidden_size;
        std::memcpy(gate_up.data(), gate_fp32, half_size * sizeof(float));
        std::memcpy(gate_up.data() + half_size, up_fp32, half_size * sizeof(float));
        set_weight(mlp.gate_up_proj_weight, gate_up.data(),
                   2 * config_.intermediate_size, config_.hidden_size);

        set_weight(mlp.down_proj_weight,
                   weights.at(prefix + "mlp.down_proj.weight").data(),
                   config_.hidden_size, config_.intermediate_size);
    }
}

std::vector<float> LlamaModel::forward(const std::vector<int>& token_ids, std::size_t pos,
                                       KVCache& kv_cache)
{
    return transformer_forward(layers_, embed_tokens_, final_norm_weight_, config_, backend_,
                               kv_cache, token_ids, pos);
}

}  // namespace vvllm
