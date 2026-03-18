#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "vvllm/backend/backend.h"
#include "vvllm/config/config.h"
#include "vvllm/model/transformer.h"
#include "vvllm/quantize/quantize.h"
#include "vvllm/tensor/tensor.h"

namespace vvllm
{

struct LlamaAttention
{
    LinearWeight q_proj_weight;  // [num_heads * head_dim, hidden_size]
    LinearWeight k_proj_weight;  // [num_kv_heads * head_dim, hidden_size]
    LinearWeight v_proj_weight;  // [num_kv_heads * head_dim, hidden_size]
    LinearWeight o_proj_weight;  // [hidden_size, hidden_size]
};

inline const float* q_bias(const LlamaAttention&) { return nullptr; }
inline const float* k_bias(const LlamaAttention&) { return nullptr; }
inline const float* v_bias(const LlamaAttention&) { return nullptr; }

class LlamaModel
{
public:
    LlamaModel(const ModelConfig& config, Backend& backend);

    void load_weights(const std::unordered_map<std::string, Tensor<float>>& weights,
                      bool quantize = false);
    std::vector<float> forward(const std::vector<int>& token_ids, std::size_t pos,
                               KVCache& kv_cache);

private:
    const ModelConfig& config_;
    Backend& backend_;

    const float* embed_tokens_;       // [vocab_size, hidden_size]
    const float* final_norm_weight_;  // [hidden_size]
    std::vector<TransformerBlock<LlamaAttention>> layers_;
    std::vector<QuantizedWeight> quantized_weights_;
    std::vector<std::vector<float>> fused_weights_;
};

}  // namespace vvllm
