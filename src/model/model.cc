#include <stdexcept>

#include "vvllm/model/model.h"

namespace vvllm
{

AnyModel create_model(const ModelConfig& config, Backend& backend)
{
    if (config.model_type == "qwen2")
    {
        return Qwen2Model(config, backend);
    }
    if (config.model_type == "llama")
    {
        return LlamaModel(config, backend);
    }
    throw std::runtime_error("Unsupported model type: " + config.model_type);
}

void load_weights(AnyModel& model, const std::unordered_map<std::string, Tensor<float>>& weights,
                  bool quantize)
{
    std::visit([&](auto& m) { m.load_weights(weights, quantize); }, model);
}

std::vector<float> forward(AnyModel& model, const std::vector<int>& token_ids, std::size_t pos,
                           KVCache& kv_cache)
{
    return std::visit([&](auto& m) { return m.forward(token_ids, pos, kv_cache); }, model);
}

}  // namespace vvllm
