#include "vvllm/ir/graph_builder.h"

#include <string>

namespace vvllm
{
namespace ir
{

static std::string layer_weight(std::size_t layer, const std::string& suffix)
{
    return "model.layers." + std::to_string(layer) + "." + suffix;
}

std::unique_ptr<Graph> build_transformer_graph(const ModelConfig& config,
                                               const GraphBuildOptions& opts)
{
    auto g = std::make_unique<Graph>();

    const std::size_t seq = opts.seq_len;
    const std::size_t H = config.hidden_size;
    const std::size_t I = config.intermediate_size;
    const std::size_t head_dim = H / config.num_attention_heads;
    const std::size_t num_heads = config.num_attention_heads;
    const std::size_t num_kv_heads = config.num_key_value_heads;
    const std::size_t kv_dim = num_kv_heads * head_dim;
    const std::size_t vocab = config.vocab_size;

    // --- Graph input: token ids ---
    Value* token_ids = g->create_value({seq}, DType::Int8);  // conceptually int32, but shape is what matters
    g->mark_input(token_ids);

    // --- Embedding ---
    Value* embed_w = g->add_weight("model.embed_tokens.weight", {vocab, H});
    Value* x = g->add_op(OpType::Embedding, {token_ids, embed_w}, {seq, H},
                         DType::Float32,
                         {{"hidden_size", static_cast<int64_t>(H)},
                          {"vocab_size", static_cast<int64_t>(vocab)}});

    // --- Transformer layers ---
    for (std::size_t i = 0; i < config.num_hidden_layers; i++)
    {
        // ===== Self-Attention =====

        // RMSNorm
        Value* attn_norm_w = g->add_weight(
            layer_weight(i, "input_layernorm.weight"), {H});
        Value* norm1 = g->add_op(OpType::RMSNorm, {x, attn_norm_w}, {seq, H},
                                 DType::Float32,
                                 {{"eps", config.rms_norm_eps}});

        // Q projection
        Value* q_w = g->add_weight(
            layer_weight(i, "self_attn.q_proj.weight"), {num_heads * head_dim, H});
        std::vector<Value*> q_inputs = {norm1, q_w};
        if (opts.has_qkv_bias)
        {
            Value* q_b = g->add_weight(
                layer_weight(i, "self_attn.q_proj.bias"), {num_heads * head_dim});
            q_inputs.push_back(q_b);
        }
        Value* q = g->add_op(OpType::Linear, q_inputs, {seq, num_heads * head_dim},
                             DType::Float32,
                             {{"M", static_cast<int64_t>(seq)},
                              {"N", static_cast<int64_t>(num_heads * head_dim)},
                              {"K", static_cast<int64_t>(H)},
                              {"has_bias", opts.has_qkv_bias}});

        // K projection
        Value* k_w = g->add_weight(
            layer_weight(i, "self_attn.k_proj.weight"), {kv_dim, H});
        std::vector<Value*> k_inputs = {norm1, k_w};
        if (opts.has_qkv_bias)
        {
            Value* k_b = g->add_weight(
                layer_weight(i, "self_attn.k_proj.bias"), {kv_dim});
            k_inputs.push_back(k_b);
        }
        Value* k = g->add_op(OpType::Linear, k_inputs, {seq, kv_dim},
                             DType::Float32,
                             {{"M", static_cast<int64_t>(seq)},
                              {"N", static_cast<int64_t>(kv_dim)},
                              {"K", static_cast<int64_t>(H)},
                              {"has_bias", opts.has_qkv_bias}});

        // V projection
        Value* v_w = g->add_weight(
            layer_weight(i, "self_attn.v_proj.weight"), {kv_dim, H});
        std::vector<Value*> v_inputs = {norm1, v_w};
        if (opts.has_qkv_bias)
        {
            Value* v_b = g->add_weight(
                layer_weight(i, "self_attn.v_proj.bias"), {kv_dim});
            v_inputs.push_back(v_b);
        }
        Value* v = g->add_op(OpType::Linear, v_inputs, {seq, kv_dim},
                             DType::Float32,
                             {{"M", static_cast<int64_t>(seq)},
                              {"N", static_cast<int64_t>(kv_dim)},
                              {"K", static_cast<int64_t>(H)},
                              {"has_bias", opts.has_qkv_bias}});

        // RoPE (two outputs: rotated Q and K)
        auto rope_outs = g->add_op_multi(
            OpType::RoPE, {q, k},
            {{seq, num_heads * head_dim}, {seq, kv_dim}},
            DType::Float32,
            {{"num_heads", static_cast<int64_t>(num_heads)},
             {"num_kv_heads", static_cast<int64_t>(num_kv_heads)},
             {"head_dim", static_cast<int64_t>(head_dim)},
             {"theta", config.rope_theta}});
        Value* q_rot = rope_outs[0];
        Value* k_rot = rope_outs[1];

        // Attention
        Value* attn_out = g->add_op(
            OpType::Attention, {q_rot, k_rot, v}, {seq, H},
            DType::Float32,
            {{"num_heads", static_cast<int64_t>(num_heads)},
             {"num_kv_heads", static_cast<int64_t>(num_kv_heads)},
             {"head_dim", static_cast<int64_t>(head_dim)},
             {"layer", static_cast<int64_t>(i)}});

        // Output projection + residual: x = attn_out @ o_proj + x
        Value* o_w = g->add_weight(
            layer_weight(i, "self_attn.o_proj.weight"), {H, H});
        x = g->add_op(OpType::LinearAdd, {attn_out, o_w, x}, {seq, H},
                       DType::Float32,
                       {{"M", static_cast<int64_t>(seq)},
                        {"N", static_cast<int64_t>(H)},
                        {"K", static_cast<int64_t>(H)}});

        // ===== MLP =====

        // RMSNorm
        Value* ffn_norm_w = g->add_weight(
            layer_weight(i, "post_attention_layernorm.weight"), {H});
        Value* norm2 = g->add_op(OpType::RMSNorm, {x, ffn_norm_w}, {seq, H},
                                 DType::Float32,
                                 {{"eps", config.rms_norm_eps}});

        // Fused gate+up projection
        Value* gate_up_w = g->add_weight(
            layer_weight(i, "mlp.gate_up_proj.weight"), {2 * I, H});
        Value* gate_up = g->add_op(OpType::Linear, {norm2, gate_up_w}, {seq, 2 * I},
                                   DType::Float32,
                                   {{"M", static_cast<int64_t>(seq)},
                                    {"N", static_cast<int64_t>(2 * I)},
                                    {"K", static_cast<int64_t>(H)},
                                    {"has_bias", false}});

        // SiLU(gate) * up
        Value* activated = g->add_op(OpType::SiLUMul, {gate_up}, {seq, I},
                                     DType::Float32,
                                     {{"intermediate_size", static_cast<int64_t>(I)}});

        // Down projection + residual: x = activated @ down_proj + x
        Value* down_w = g->add_weight(
            layer_weight(i, "mlp.down_proj.weight"), {H, I});
        x = g->add_op(OpType::LinearAdd, {activated, down_w, x}, {seq, H},
                       DType::Float32,
                       {{"M", static_cast<int64_t>(seq)},
                        {"N", static_cast<int64_t>(H)},
                        {"K", static_cast<int64_t>(I)}});
    }

    // --- Final RMSNorm ---
    Value* final_norm_w = g->add_weight("model.norm.weight", {H});
    Value* final_norm = g->add_op(OpType::RMSNorm, {x, final_norm_w}, {seq, H},
                                  DType::Float32,
                                  {{"eps", config.rms_norm_eps}});

    // --- Logits projection (tied with embedding weights) ---
    Value* logits = g->add_op(OpType::Linear, {final_norm, embed_w}, {1, vocab},
                              DType::Float32,
                              {{"M", static_cast<int64_t>(1)},
                               {"N", static_cast<int64_t>(vocab)},
                               {"K", static_cast<int64_t>(H)},
                               {"has_bias", false}});
    g->mark_output(logits);

    return g;
}

}  // namespace ir
}  // namespace vvllm
