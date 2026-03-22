#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "vvllm/backend/backend.h"
#include "vvllm/config/config.h"
#include "vvllm/ir/graph.h"
#include "vvllm/kv_cache/kv_cache.h"
#include "vvllm/tensor/tensor.h"

namespace vvllm
{
namespace ir
{

/// A weight map resolves IR weight names to raw pointers + metadata.
/// Built from safetensors weights after the same fusing/quantization as the
/// interpreted path.
struct WeightRef
{
    const float* fp32 = nullptr;
    const int8_t* int8 = nullptr;
    const float* scales = nullptr;
    bool quantized = false;
};

using WeightMap = std::unordered_map<std::string, WeightRef>;

/// Compiled execution plan: walks the IR graph and dispatches to Backend.
/// Created once at startup, called for every forward pass.
class Executable
{
public:
    Executable(Graph* graph, const WeightMap& weights, Backend& backend,
               const ModelConfig& config);

    /// Run forward pass. Returns logits tensor (FP32).
    Tensor run(const std::vector<int>& token_ids, std::size_t pos, KVCache& kv_cache);

private:
    Graph* graph_;
    const WeightMap& weights_;
    Backend& backend_;
    const ModelConfig& config_;

    /// Resolve a Weight node's name attribute to a WeightRef.
    const WeightRef& resolve_weight(const OpNode* node) const;

    /// Get the raw float pointer for a Value from the buffer map.
    float* buf(Value* v) const;
    const float* cbuf(Value* v) const;

    /// Per-run buffer allocation: maps Value id -> Tensor.
    mutable std::unordered_map<uint32_t, Tensor> buffers_;
    /// Per-run weight pointer cache: maps Value id -> raw pointer.
    mutable std::unordered_map<uint32_t, const void*> weight_ptrs_;
};

}  // namespace ir
}  // namespace vvllm
