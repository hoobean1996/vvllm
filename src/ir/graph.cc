#include "vvllm/ir/graph.h"

#include <sstream>

namespace vvllm
{
namespace ir
{

const char* op_type_name(OpType op)
{
    switch (op)
    {
        case OpType::Input: return "Input";
        case OpType::Weight: return "Weight";
        case OpType::Embedding: return "Embedding";
        case OpType::RMSNorm: return "RMSNorm";
        case OpType::Linear: return "Linear";
        case OpType::RoPE: return "RoPE";
        case OpType::Attention: return "Attention";
        case OpType::SiLUMul: return "SiLUMul";
        case OpType::Add: return "Add";
        case OpType::Softmax: return "Softmax";
        case OpType::LinearAdd: return "LinearAdd";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

Value* Graph::create_value(Shape shape, DType dtype)
{
    auto v = std::make_unique<Value>();
    v->id = next_value_id_++;
    v->dtype = dtype;
    v->shape = std::move(shape);
    v->producer = nullptr;
    Value* ptr = v.get();
    values_.push_back(std::move(v));
    return ptr;
}

Value* Graph::add_op(OpType op, std::vector<Value*> inputs, Shape out_shape,
                     DType out_dtype, Attributes attrs)
{
    auto node = std::make_unique<OpNode>();
    node->id = next_node_id_++;
    node->op = op;
    node->inputs = inputs;
    node->attrs = std::move(attrs);

    // Create output value
    Value* out = create_value(std::move(out_shape), out_dtype);
    out->producer = node.get();
    node->outputs.push_back(out);

    // Register this node as a user of each input
    for (Value* in : node->inputs)
    {
        in->users.push_back(node.get());
    }

    nodes_.push_back(std::move(node));
    return out;
}

std::vector<Value*> Graph::add_op_multi(OpType op, std::vector<Value*> inputs,
                                         std::vector<Shape> out_shapes,
                                         DType out_dtype, Attributes attrs)
{
    auto node = std::make_unique<OpNode>();
    node->id = next_node_id_++;
    node->op = op;
    node->inputs = inputs;
    node->attrs = std::move(attrs);

    std::vector<Value*> outs;
    for (auto& shape : out_shapes)
    {
        Value* out = create_value(std::move(shape), out_dtype);
        out->producer = node.get();
        node->outputs.push_back(out);
        outs.push_back(out);
    }

    for (Value* in : node->inputs)
    {
        in->users.push_back(node.get());
    }

    nodes_.push_back(std::move(node));
    return outs;
}

Value* Graph::add_weight(std::string name, Shape shape, DType dtype)
{
    Attributes attrs;
    attrs.emplace("name", std::move(name));
    return add_op(OpType::Weight, {}, std::move(shape), dtype, std::move(attrs));
}

void Graph::mark_input(Value* v) { inputs_.push_back(v); }

void Graph::mark_output(Value* v) { outputs_.push_back(v); }

// ---------------------------------------------------------------------------
// dump() — human-readable SSA format
// ---------------------------------------------------------------------------

static std::string shape_str(const Shape& s)
{
    std::string out = "[";
    for (std::size_t i = 0; i < s.size(); i++)
    {
        if (i > 0) out += ", ";
        out += std::to_string(s[i]);
    }
    out += "]";
    return out;
}

static std::string dtype_str(DType dt)
{
    switch (dt)
    {
        case DType::Float32: return "f32";
        case DType::Float16: return "f16";
        case DType::Int8: return "i8";
    }
    return "?";
}

static std::string attr_str(const Attributes& attrs)
{
    if (attrs.empty()) return "";
    std::string out = " {";
    bool first = true;
    for (const auto& [k, v] : attrs)
    {
        if (!first) out += ", ";
        first = false;
        out += k + "=";
        std::visit(
            [&out](auto&& val)
            {
                using T = std::decay_t<decltype(val)>;
                if constexpr (std::is_same_v<T, std::string>)
                    out += "\"" + val + "\"";
                else if constexpr (std::is_same_v<T, bool>)
                    out += val ? "true" : "false";
                else if constexpr (std::is_same_v<T, int64_t>)
                    out += std::to_string(val);
                else if constexpr (std::is_same_v<T, double>)
                    out += std::to_string(val);
            },
            v);
    }
    out += "}";
    return out;
}

std::string Graph::dump() const
{
    std::ostringstream ss;
    ss << "// Graph: " << nodes_.size() << " ops, " << values_.size() << " values\n";

    for (const auto& node : nodes_)
    {
        // Output values
        if (node->outputs.size() == 1)
        {
            auto* out = node->outputs[0];
            ss << "%" << out->id << " = ";
        }
        else if (node->outputs.size() > 1)
        {
            ss << "(";
            for (std::size_t i = 0; i < node->outputs.size(); i++)
            {
                if (i > 0) ss << ", ";
                ss << "%" << node->outputs[i]->id;
            }
            ss << ") = ";
        }

        // Op name
        ss << op_type_name(node->op);

        // Inputs
        ss << "(";
        for (std::size_t i = 0; i < node->inputs.size(); i++)
        {
            if (i > 0) ss << ", ";
            ss << "%" << node->inputs[i]->id;
        }
        ss << ")";

        // Attributes
        ss << attr_str(node->attrs);

        // Output shapes
        ss << " -> ";
        if (node->outputs.size() == 1)
        {
            auto* out = node->outputs[0];
            ss << shape_str(out->shape) << " " << dtype_str(out->dtype);
        }
        else
        {
            for (std::size_t i = 0; i < node->outputs.size(); i++)
            {
                if (i > 0) ss << ", ";
                ss << shape_str(node->outputs[i]->shape) << " " << dtype_str(node->outputs[i]->dtype);
            }
        }
        ss << "\n";
    }

    // Mark outputs
    ss << "// Outputs:";
    for (auto* v : outputs_)
    {
        ss << " %" << v->id;
    }
    ss << "\n";

    return ss.str();
}

}  // namespace ir
}  // namespace vvllm
