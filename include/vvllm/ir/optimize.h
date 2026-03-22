#pragma once

#include "vvllm/ir/graph.h"

namespace vvllm
{
namespace ir
{

/// Optimize a compute graph (currently a no-op pass-through).
/// Future passes: op fusion, memory planning, dead code elimination.
/// Takes ownership of the input graph and returns the (possibly modified) graph.
Graph* optimize(Graph* graph);

}  // namespace ir
}  // namespace vvllm
