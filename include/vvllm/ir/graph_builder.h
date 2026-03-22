#pragma once

#include <memory>

#include "vvllm/config/config.h"
#include "vvllm/ir/graph.h"

namespace vvllm
{
namespace ir
{

enum class GraphMode
{
    Prefill,  // seq_len = prompt length (compute-bound, large matmul)
    Decode,   // seq_len = 1 (memory-bound, matrix-vector)
};

struct GraphBuildOptions
{
    GraphMode mode = GraphMode::Decode;
    std::size_t seq_len = 1;     // concrete seq_len for shape inference
    bool has_qkv_bias = false;   // Qwen2 has bias, Llama does not
};

/// Build a Transformer compute graph directly from ModelConfig.
/// No tracing needed — the graph structure is determined entirely by config.
std::unique_ptr<Graph> build_transformer_graph(const ModelConfig& config,
                                               const GraphBuildOptions& opts);

}  // namespace ir
}  // namespace vvllm
