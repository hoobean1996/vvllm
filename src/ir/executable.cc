#include "vvllm/ir/executable.h"

#include <cmath>
#include <cstring>
#include <string>
#include <variant>

namespace vvllm
{
namespace ir
{

// ---------------------------------------------------------------------------
// Helper: extract typed attribute from an OpNode
// ---------------------------------------------------------------------------

static int64_t attr_i64(const OpNode* node, const std::string& key)
{
    return std::get<int64_t>(node->attrs.at(key));
}

static double attr_f64(const OpNode* node, const std::string& key)
{
    return std::get<double>(node->attrs.at(key));
}

static bool attr_bool(const OpNode* node, const std::string& key)
{
    return std::get<bool>(node->attrs.at(key));
}

static const std::string& attr_str(const OpNode* node, const std::string& key)
{
    return std::get<std::string>(node->attrs.at(key));
}

// ---------------------------------------------------------------------------
// Executable
// ---------------------------------------------------------------------------

Executable::Executable(Graph* graph, const WeightMap& weights, Backend& backend,
                       const ModelConfig& config)
    : graph_(graph), weights_(weights), backend_(backend), config_(config)
{
}

const WeightRef& Executable::resolve_weight(const OpNode* node) const
{
    const std::string& name = attr_str(node, "name");
    return weights_.at(name);
}

float* Executable::buf(Value* v) const
{
    return buffers_.at(v->id).data<float>();
}

const float* Executable::cbuf(Value* v) const
{
    return buffers_.at(v->id).data<float>();
}

Tensor Executable::run(const std::vector<int>& token_ids, std::size_t pos,
                       KVCache& kv_cache)
{
    buffers_.clear();
    weight_ptrs_.clear();

    // Runtime dimensions — these override graph-baked shapes
    const std::size_t seq_len = token_ids.size();
    const std::size_t H = config_.hidden_size;
    const std::size_t I = config_.intermediate_size;
    const std::size_t head_dim = H / config_.num_attention_heads;
    const std::size_t num_heads = config_.num_attention_heads;
    const std::size_t num_kv_heads = config_.num_key_value_heads;
    const std::size_t kv_dim = num_kv_heads * head_dim;

    Device dev = backend_.device();
    DType compute_dtype = backend_.is_fp16() ? DType::Float16 : DType::Float32;
    std::size_t elem = dtype_size(compute_dtype);

    backend_.begin_forward();
    kv_cache.init(config_.num_hidden_layers, kv_dim);
    if (pos == 0) kv_cache.reset();

    // Walk graph nodes in topological order (they are already sorted)
    for (const auto& node : graph_->nodes())
    {
        switch (node->op)
        {
            case OpType::Weight:
            case OpType::Input:
            {
                break;
            }

            case OpType::Embedding:
            {
                Value* out = node->outputs[0];
                Value* weight_val = node->inputs[1];
                const WeightRef& wref = resolve_weight(weight_val->producer);
                const float* embed_table = wref.fp32;

                buffers_.emplace(out->id, Tensor({seq_len * H}, compute_dtype, dev));

                if (compute_dtype == DType::Float16)
                {
                    Tensor tmp({seq_len * H}, DType::Float32, dev);
                    for (std::size_t s = 0; s < seq_len; s++)
                    {
                        backend_.embedding(tmp.data<float>() + s * H, embed_table,
                                           token_ids[s], H, config_.vocab_size);
                    }
                    backend_.fp32_to_fp16(buffers_.at(out->id).raw_data(),
                                          tmp.data<float>(), seq_len * H);
                }
                else
                {
                    for (std::size_t s = 0; s < seq_len; s++)
                    {
                        backend_.embedding(buf(out) + s * H, embed_table,
                                           token_ids[s], H, config_.vocab_size);
                    }
                }
                break;
            }

            case OpType::RMSNorm:
            {
                Value* x_val = node->inputs[0];
                Value* w_val = node->inputs[1];
                Value* out = node->outputs[0];
                float eps = static_cast<float>(attr_f64(node.get(), "eps"));

                const WeightRef& wref = resolve_weight(w_val->producer);
                const float* weight = wref.fp32;

                // Use graph shape for hidden dim, runtime for seq_len
                std::size_t norm_hidden = out->shape[1];

                buffers_.emplace(out->id, Tensor({seq_len * norm_hidden}, compute_dtype, dev));

                for (std::size_t s = 0; s < seq_len; s++)
                {
                    std::size_t byte_off = s * norm_hidden * elem;
                    auto* out_s = reinterpret_cast<float*>(
                        reinterpret_cast<char*>(buf(out)) + byte_off);
                    auto* x_s = reinterpret_cast<const float*>(
                        reinterpret_cast<const char*>(cbuf(x_val)) + byte_off);
                    backend_.rms_norm(out_s, x_s, weight, norm_hidden, eps);
                }
                break;
            }

            case OpType::Linear:
            {
                Value* inp_val = node->inputs[0];
                Value* w_val = node->inputs[1];
                Value* out = node->outputs[0];

                // N and K are fixed by weight dimensions; M is runtime seq_len
                std::size_t N = static_cast<std::size_t>(attr_i64(node.get(), "N"));
                std::size_t K = static_cast<std::size_t>(attr_i64(node.get(), "K"));
                bool has_bias = attr_bool(node.get(), "has_bias");

                // For the final logits projection, M=1 (last token only)
                // Detect by checking if this is the output node
                bool is_logits = (!graph_->outputs().empty() &&
                                  out == graph_->outputs()[0]);
                std::size_t M = is_logits ? 1 : seq_len;

                const WeightRef& wref = resolve_weight(w_val->producer);

                const float* bias = nullptr;
                if (has_bias && node->inputs.size() >= 3)
                {
                    Value* b_val = node->inputs[2];
                    const WeightRef& bref = resolve_weight(b_val->producer);
                    bias = bref.fp32;
                }

                buffers_.emplace(out->id, Tensor({M * N}, compute_dtype, dev));

                // For logits: input is the final norm output, only last token
                const float* inp_ptr = cbuf(inp_val);
                if (is_logits && seq_len > 1)
                {
                    // Point to the last token's hidden state
                    inp_ptr = reinterpret_cast<const float*>(
                        reinterpret_cast<const char*>(inp_ptr) +
                        (seq_len - 1) * K * elem);
                }

                if (wref.quantized)
                {
                    backend_.linear_q8(buf(out), inp_ptr, wref.int8, wref.scales,
                                       bias, M, N, K);
                }
                else
                {
                    backend_.linear(buf(out), inp_ptr, wref.fp32, bias, M, N, K);
                }
                break;
            }

            case OpType::RoPE:
            {
                Value* q_val = node->inputs[0];
                Value* k_val = node->inputs[1];
                Value* q_out = node->outputs[0];
                Value* k_out = node->outputs[1];

                std::size_t nh = static_cast<std::size_t>(attr_i64(node.get(), "num_heads"));
                std::size_t nkv = static_cast<std::size_t>(attr_i64(node.get(), "num_kv_heads"));
                std::size_t hd = static_cast<std::size_t>(attr_i64(node.get(), "head_dim"));
                float theta = static_cast<float>(attr_f64(node.get(), "theta"));

                std::size_t q_bytes = seq_len * nh * hd * elem;
                std::size_t k_bytes = seq_len * nkv * hd * elem;

                buffers_.emplace(q_out->id, Tensor({seq_len * nh * hd}, compute_dtype, dev));
                buffers_.emplace(k_out->id, Tensor({seq_len * nkv * hd}, compute_dtype, dev));

                device_copy(buf(q_out), dev, cbuf(q_val), dev, q_bytes);
                device_copy(buf(k_out), dev, cbuf(k_val), dev, k_bytes);

                backend_.rope(buf(q_out), buf(k_out), seq_len, nh, nkv, hd, pos, theta);
                break;
            }

            case OpType::Attention:
            {
                Value* q_val = node->inputs[0];
                Value* k_val = node->inputs[1];
                Value* v_val = node->inputs[2];
                Value* out = node->outputs[0];

                std::size_t nh = static_cast<std::size_t>(attr_i64(node.get(), "num_heads"));
                std::size_t nkv = static_cast<std::size_t>(attr_i64(node.get(), "num_kv_heads"));
                std::size_t hd = static_cast<std::size_t>(attr_i64(node.get(), "head_dim"));
                std::size_t layer = static_cast<std::size_t>(attr_i64(node.get(), "layer"));

                kv_cache.append(layer, cbuf(k_val), cbuf(v_val), seq_len);
                const float* cache_k = kv_cache.k(layer);
                const float* cache_v = kv_cache.v(layer);

                float scale = 1.0f / std::sqrt(static_cast<float>(hd));

                buffers_.emplace(out->id, Tensor({seq_len * H}, compute_dtype, dev));

                for (std::size_t s = 0; s < seq_len; s++)
                {
                    std::size_t byte_off = s * H * elem;
                    auto* out_s = reinterpret_cast<float*>(
                        reinterpret_cast<char*>(buf(out)) + byte_off);
                    backend_.causal_attention(
                        out_s, cbuf(q_val), s,
                        cache_k, cache_v,
                        pos + s + 1, nh, nkv, hd, scale);
                }
                break;
            }

            case OpType::LinearAdd:
            {
                Value* inp_val = node->inputs[0];
                Value* w_val = node->inputs[1];
                Value* residual_val = node->inputs[2];
                Value* out = node->outputs[0];

                std::size_t N = static_cast<std::size_t>(attr_i64(node.get(), "N"));
                std::size_t K = static_cast<std::size_t>(attr_i64(node.get(), "K"));

                const WeightRef& wref = resolve_weight(w_val->producer);

                buffers_.emplace(out->id, Tensor({seq_len * N}, compute_dtype, dev));

                if (wref.quantized)
                {
                    backend_.linear_q8_add(buf(out), cbuf(inp_val), wref.int8, wref.scales,
                                           nullptr, cbuf(residual_val), seq_len, N, K);
                }
                else
                {
                    backend_.linear_add(buf(out), cbuf(inp_val), wref.fp32,
                                        nullptr, cbuf(residual_val), seq_len, N, K);
                }
                break;
            }

            case OpType::SiLUMul:
            {
                Value* gate_up_val = node->inputs[0];
                Value* out = node->outputs[0];

                std::size_t inter = static_cast<std::size_t>(
                    attr_i64(node.get(), "intermediate_size"));

                buffers_.emplace(out->id, Tensor({seq_len * inter}, compute_dtype, dev));

                for (std::size_t s = 0; s < seq_len; s++)
                {
                    std::size_t byte_off_gate = s * 2 * inter * elem;
                    std::size_t byte_off_up = byte_off_gate + inter * elem;
                    auto* gate_s = reinterpret_cast<const float*>(
                        reinterpret_cast<const char*>(cbuf(gate_up_val)) + byte_off_gate);
                    auto* up_s = reinterpret_cast<const float*>(
                        reinterpret_cast<const char*>(cbuf(gate_up_val)) + byte_off_up);
                    std::size_t out_byte_off = s * inter * elem;
                    auto* out_s = reinterpret_cast<float*>(
                        reinterpret_cast<char*>(buf(out)) + out_byte_off);
                    backend_.silu_mul(out_s, gate_s, up_s, inter);
                }
                break;
            }

            case OpType::Add:
            {
                Value* x_val = node->inputs[0];
                Value* y_val = node->inputs[1];
                Value* out = node->outputs[0];

                buffers_.emplace(out->id, Tensor({seq_len * H}, compute_dtype, dev));
                backend_.add(buf(out), cbuf(x_val), cbuf(y_val), seq_len * H);
                break;
            }

            case OpType::Softmax:
            {
                Value* x_val = node->inputs[0];
                Value* out = node->outputs[0];

                std::size_t n = 1;
                for (auto d : out->shape) n *= d;

                buffers_.emplace(out->id, Tensor({n}, compute_dtype, dev));
                backend_.softmax(buf(out), cbuf(x_val), n);
                break;
            }
        }
    }

    // Extract logits from the graph output
    Value* logits_val = graph_->outputs()[0];
    Tensor logits = std::move(buffers_.at(logits_val->id));

    // Advance KV cache after all layers
    kv_cache.advance(seq_len);

    // If FP16, convert to FP32 for the sampler
    if (compute_dtype == DType::Float16)
    {
        std::size_t vocab = config_.vocab_size;
        Tensor logits_fp32({vocab}, DType::Float32, dev);
        backend_.fp16_to_fp32(logits_fp32.data<float>(), logits.raw_data(), vocab);
        return logits_fp32;
    }

    return logits;
}

}  // namespace ir
}  // namespace vvllm
