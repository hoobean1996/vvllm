#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "vvllm/backend/backend.h"
#include "vvllm/config/config.h"
#include "vvllm/model/llama/model.h"
#include "vvllm/model/qwen2/model.h"
#include "vvllm/tensor/tensor.h"

namespace vvllm
{

using AnyModel = std::variant<Qwen2Model, LlamaModel>;

/// Create a model based on config.model_type ("qwen2" or "llama").
AnyModel create_model(const ModelConfig& config, Backend& backend);

/// Load weights into a model.
void load_weights(AnyModel& model, const std::unordered_map<std::string, Tensor<float>>& weights,
                  bool quantize = false);

/// Run forward pass: token_ids -> logits for the last token.
std::vector<float> forward(AnyModel& model, const std::vector<int>& token_ids, std::size_t pos);

}  // namespace vvllm
