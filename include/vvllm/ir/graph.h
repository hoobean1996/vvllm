#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "vvllm/tensor/tensor.h"

namespace vvllm
{
namespace ir
{

// ---------------------------------------------------------------------------
// OpType — one-to-one with Backend methods, plus composite ops for fusion
// ---------------------------------------------------------------------------
enum class OpType
{
    // Structural
    Input,      // graph input (token ids)
    Weight,     // reference to a named weight tensor

    // Core ops (map to Backend interface)
    Embedding,  // embedding lookup
    RMSNorm,    // RMS normalization
    Linear,     // out = inp @ W^T + bias (fp32 or int8)
    RoPE,       // rotary position embedding
    Attention,  // scaled dot-product attention + causal mask + KV cache
    SiLUMul,    // silu(gate) * up
    Add,        // element-wise add (residual)
    Softmax,    // softmax

    // Fused ops (produced by optimization passes, already supported by Backend)
    LinearAdd,  // linear + residual add
};

const char* op_type_name(OpType op);

// ---------------------------------------------------------------------------
// Attributes — key-value bag for op-specific parameters
// ---------------------------------------------------------------------------
using AttrValue = std::variant<int64_t, double, std::string, bool>;
using Attributes = std::unordered_map<std::string, AttrValue>;

// ---------------------------------------------------------------------------
// Value — SSA value (defined once, used many times)
// ---------------------------------------------------------------------------
using Shape = std::vector<std::size_t>;

struct OpNode;

struct Value
{
    uint32_t id;
    DType dtype;
    Shape shape;
    OpNode* producer;             // nullptr for graph inputs
    std::vector<OpNode*> users;   // consumers of this value
};

// ---------------------------------------------------------------------------
// OpNode — a single operation in the graph
// ---------------------------------------------------------------------------
struct OpNode
{
    uint32_t id;
    OpType op;
    std::vector<Value*> inputs;
    std::vector<Value*> outputs;
    Attributes attrs;
};

// ---------------------------------------------------------------------------
// Graph — the complete computation graph (DAG)
// ---------------------------------------------------------------------------
class Graph
{
public:
    /// Create a new SSA value (not produced by any op).
    Value* create_value(Shape shape, DType dtype = DType::Float32);

    /// Add an operation node. Returns the primary output value.
    /// Creates one output value with the given shape/dtype.
    Value* add_op(OpType op, std::vector<Value*> inputs, Shape out_shape,
                  DType out_dtype = DType::Float32, Attributes attrs = {});

    /// Add an operation node with multiple outputs.
    std::vector<Value*> add_op_multi(OpType op, std::vector<Value*> inputs,
                                     std::vector<Shape> out_shapes,
                                     DType out_dtype = DType::Float32,
                                     Attributes attrs = {});

    /// Add a weight reference node.
    Value* add_weight(std::string name, Shape shape, DType dtype = DType::Float32);

    /// Mark a value as a graph input.
    void mark_input(Value* v);

    /// Mark a value as a graph output.
    void mark_output(Value* v);

    /// Dump the graph in human-readable SSA format.
    std::string dump() const;

    // Accessors
    const std::vector<Value*>& inputs() const { return inputs_; }
    const std::vector<Value*>& outputs() const { return outputs_; }
    const std::vector<std::unique_ptr<OpNode>>& nodes() const { return nodes_; }
    const std::vector<std::unique_ptr<Value>>& values() const { return values_; }
    std::size_t num_nodes() const { return nodes_.size(); }
    std::size_t num_values() const { return values_.size(); }

private:
    std::vector<std::unique_ptr<OpNode>> nodes_;
    std::vector<std::unique_ptr<Value>> values_;
    std::vector<Value*> inputs_;
    std::vector<Value*> outputs_;
    uint32_t next_value_id_ = 0;
    uint32_t next_node_id_ = 0;
};

}  // namespace ir
}  // namespace vvllm
