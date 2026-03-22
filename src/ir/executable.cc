#include "vvllm/ir/executable.h"

#include <algorithm>
#include <cmath>
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
// Memory Planning (unchanged)
// ---------------------------------------------------------------------------

static std::size_t value_buffer_size(const Value* v, const ModelConfig& config,
                                     std::size_t seq_len)
{
    if (!v->producer) return 0;
    OpType op = v->producer->op;
    if (op == OpType::Weight || op == OpType::Input) return 0;

    const std::size_t H = config.hidden_size;
    const std::size_t I = config.intermediate_size;
    const std::size_t head_dim = H / config.num_attention_heads;
    const std::size_t num_heads = config.num_attention_heads;
    const std::size_t num_kv_heads = config.num_key_value_heads;

    switch (op)
    {
        case OpType::Embedding: return seq_len * H;
        case OpType::RMSNorm: return seq_len * v->shape[1];
        case OpType::Linear:
        {
            std::size_t N = static_cast<std::size_t>(attr_i64(v->producer, "N"));
            bool is_logits = (N == config.vocab_size && v->users.empty());
            return (is_logits ? 1 : seq_len) * N;
        }
        case OpType::RoPE:
            return (v == v->producer->outputs[0])
                ? seq_len * num_heads * head_dim
                : seq_len * num_kv_heads * head_dim;
        case OpType::Attention: return seq_len * H;
        case OpType::LinearAdd:
            return seq_len * static_cast<std::size_t>(attr_i64(v->producer, "N"));
        case OpType::SiLUMul: return seq_len * I;
        case OpType::Add: return seq_len * H;
        case OpType::Softmax:
        {
            std::size_t n = 1;
            for (auto d : v->shape) n *= d;
            return n;
        }
        default: return 0;
    }
}

MemoryPlan plan_memory(const Graph& graph, const ModelConfig& config,
                       std::size_t seq_len)
{
    MemoryPlan plan;
    std::unordered_map<uint32_t, uint32_t> last_use;
    const auto& nodes = graph.nodes();

    for (uint32_t i = 0; i < nodes.size(); i++)
        for (Value* in : nodes[i]->inputs)
            last_use[in->id] = i;
    for (Value* out : graph.outputs())
        last_use[out->id] = static_cast<uint32_t>(nodes.size());

    struct FreeSlot { uint32_t slot_id; std::size_t size; };
    std::vector<FreeSlot> free_pool;
    std::vector<std::size_t> slot_sizes;
    std::unordered_map<uint32_t, uint32_t> active;

    for (uint32_t i = 0; i < nodes.size(); i++)
    {
        auto& node = nodes[i];

        for (Value* in : node->inputs)
        {
            auto lu = last_use.find(in->id);
            if (lu != last_use.end() && lu->second == i)
            {
                auto act = active.find(in->id);
                if (act != active.end())
                {
                    free_pool.push_back({act->second, slot_sizes[act->second]});
                    active.erase(act);
                }
            }
        }

        // In-place RoPE: outputs reuse input buffer slots
        if (node->op == OpType::RoPE && node->inputs.size() == 2 &&
            node->outputs.size() == 2)
        {
            for (std::size_t out_idx = 0; out_idx < 2; out_idx++)
            {
                Value* in_val = node->inputs[out_idx];
                Value* out_val = node->outputs[out_idx];
                auto act = active.find(in_val->id);
                if (act != active.end() && last_use.at(in_val->id) == i)
                {
                    uint32_t slot_id = act->second;
                    plan.slot_index[out_val->id] = slot_id;
                    plan.inplace_values.insert(out_val->id);
                    active.erase(act);
                    active[out_val->id] = slot_id;
                }
            }
            if (plan.slot_index.count(node->outputs[0]->id) &&
                plan.slot_index.count(node->outputs[1]->id))
                continue;
        }

        for (Value* out : node->outputs)
        {
            if (plan.slot_index.count(out->id)) continue;
            std::size_t needed = value_buffer_size(out, config, seq_len);
            if (needed == 0) continue;

            int best_idx = -1;
            std::size_t best_size = SIZE_MAX;
            for (int j = 0; j < static_cast<int>(free_pool.size()); j++)
            {
                if (free_pool[j].size >= needed && free_pool[j].size < best_size)
                {
                    best_idx = j;
                    best_size = free_pool[j].size;
                }
            }

            uint32_t slot_id;
            if (best_idx >= 0)
            {
                slot_id = free_pool[best_idx].slot_id;
                if (needed > slot_sizes[slot_id]) slot_sizes[slot_id] = needed;
                free_pool.erase(free_pool.begin() + best_idx);
            }
            else
            {
                slot_id = static_cast<uint32_t>(slot_sizes.size());
                slot_sizes.push_back(needed);
            }

            plan.slot_index[out->id] = slot_id;
            active[out->id] = slot_id;
        }
    }

    plan.slot_sizes = std::move(slot_sizes);
    return plan;
}

// ---------------------------------------------------------------------------
// Executable — pre-compiled instruction list
// ---------------------------------------------------------------------------

Executable::Executable(Graph* graph, const WeightMap& weights, Backend& backend,
                       const ModelConfig& config)
    : graph_(graph), weights_(weights), backend_(backend), config_(config),
      logits_slot_(UINT32_MAX)
{
}

void Executable::compile(std::size_t seq_len)
{
    ops_.clear();
    const auto& graph_outputs = graph_->outputs();

    for (const auto& node : graph_->nodes())
    {
        if (node->op == OpType::Weight || node->op == OpType::Input)
            continue;

        ResolvedOp rop{};
        rop.op = node->op;
        rop.out_slot = UINT32_MAX;
        rop.out_slot2 = UINT32_MAX;
        rop.residual_slot = UINT32_MAX;
        rop.num_inputs = 0;
        rop.weight_fp32 = nullptr;
        rop.weight_int8 = nullptr;
        rop.weight_scales = nullptr;
        rop.bias = nullptr;
        rop.weight_quantized = false;
        rop.is_logits = false;
        rop.rope_q_inplace = false;
        rop.rope_k_inplace = false;
        rop.N = 0;
        rop.K = 0;
        rop.eps = 0;
        rop.hidden = 0;
        rop.nh = 0;
        rop.nkv = 0;
        rop.hd = 0;
        rop.theta = 0;
        rop.layer = 0;
        rop.intermediate = 0;

        // Resolve output slot(s)
        if (!node->outputs.empty())
        {
            auto it = plan_.slot_index.find(node->outputs[0]->id);
            if (it != plan_.slot_index.end()) rop.out_slot = it->second;
        }
        if (node->outputs.size() > 1)
        {
            auto it = plan_.slot_index.find(node->outputs[1]->id);
            if (it != plan_.slot_index.end()) rop.out_slot2 = it->second;
        }

        // Resolve input slots
        for (std::size_t j = 0; j < node->inputs.size() && j < 4; j++)
        {
            Value* in = node->inputs[j];
            auto it = plan_.slot_index.find(in->id);
            if (it != plan_.slot_index.end())
                rop.in_slots[j] = it->second;
            else
                rop.in_slots[j] = UINT32_MAX;
            rop.num_inputs++;
        }

        // Resolve weight pointers (from Weight producer nodes)
        auto resolve_weight = [&](Value* v) -> const WeightRef* {
            if (!v->producer || v->producer->op != OpType::Weight) return nullptr;
            const std::string& name = attr_str(v->producer, "name");
            auto it = weights_.find(name);
            return (it != weights_.end()) ? &it->second : nullptr;
        };

        switch (node->op)
        {
            case OpType::Embedding:
            {
                const WeightRef* wref = resolve_weight(node->inputs[1]);
                if (wref) rop.weight_fp32 = wref->fp32;
                break;
            }
            case OpType::RMSNorm:
            {
                const WeightRef* wref = resolve_weight(node->inputs[1]);
                if (wref) rop.weight_fp32 = wref->fp32;
                rop.eps = static_cast<float>(attr_f64(node.get(), "eps"));
                rop.hidden = node->outputs[0]->shape[1];
                break;
            }
            case OpType::Linear:
            {
                const WeightRef* wref = resolve_weight(node->inputs[1]);
                if (wref)
                {
                    rop.weight_fp32 = wref->fp32;
                    rop.weight_int8 = wref->int8;
                    rop.weight_scales = wref->scales;
                    rop.weight_quantized = wref->quantized;
                }
                rop.N = static_cast<std::size_t>(attr_i64(node.get(), "N"));
                rop.K = static_cast<std::size_t>(attr_i64(node.get(), "K"));
                bool has_bias = attr_bool(node.get(), "has_bias");
                if (has_bias && node->inputs.size() >= 3)
                {
                    const WeightRef* bref = resolve_weight(node->inputs[2]);
                    if (bref) rop.bias = bref->fp32;
                }
                rop.is_logits = (!graph_outputs.empty() &&
                                 node->outputs[0] == graph_outputs[0]);
                break;
            }
            case OpType::RoPE:
            {
                rop.nh = static_cast<std::size_t>(attr_i64(node.get(), "num_heads"));
                rop.nkv = static_cast<std::size_t>(attr_i64(node.get(), "num_kv_heads"));
                rop.hd = static_cast<std::size_t>(attr_i64(node.get(), "head_dim"));
                rop.theta = static_cast<float>(attr_f64(node.get(), "theta"));
                rop.rope_q_inplace = plan_.inplace_values.count(node->outputs[0]->id) > 0;
                rop.rope_k_inplace = plan_.inplace_values.count(node->outputs[1]->id) > 0;
                break;
            }
            case OpType::Attention:
            {
                rop.nh = static_cast<std::size_t>(attr_i64(node.get(), "num_heads"));
                rop.nkv = static_cast<std::size_t>(attr_i64(node.get(), "num_kv_heads"));
                rop.hd = static_cast<std::size_t>(attr_i64(node.get(), "head_dim"));
                rop.layer = static_cast<std::size_t>(attr_i64(node.get(), "layer"));
                break;
            }
            case OpType::LinearAdd:
            {
                const WeightRef* wref = resolve_weight(node->inputs[1]);
                if (wref)
                {
                    rop.weight_fp32 = wref->fp32;
                    rop.weight_int8 = wref->int8;
                    rop.weight_scales = wref->scales;
                    rop.weight_quantized = wref->quantized;
                }
                rop.N = static_cast<std::size_t>(attr_i64(node.get(), "N"));
                rop.K = static_cast<std::size_t>(attr_i64(node.get(), "K"));
                // Residual is input[2]
                if (node->inputs.size() >= 3)
                {
                    auto it = plan_.slot_index.find(node->inputs[2]->id);
                    if (it != plan_.slot_index.end()) rop.residual_slot = it->second;
                }
                break;
            }
            case OpType::SiLUMul:
            {
                rop.intermediate = static_cast<std::size_t>(
                    attr_i64(node.get(), "intermediate_size"));
                break;
            }
            default:
                break;
        }

        ops_.push_back(rop);
    }

    // Record logits slot
    if (!graph_outputs.empty())
    {
        auto it = plan_.slot_index.find(graph_outputs[0]->id);
        if (it != plan_.slot_index.end()) logits_slot_ = it->second;
    }
}

void Executable::allocate_pool(std::size_t seq_len)
{
    Device dev = backend_.device();
    DType compute_dtype = backend_.is_fp16() ? DType::Float16 : DType::Float32;

    plan_ = plan_memory(*graph_, config_, seq_len);

    pool_.clear();
    pool_.reserve(plan_.num_slots());
    for (std::size_t i = 0; i < plan_.num_slots(); i++)
        pool_.emplace_back(Tensor({plan_.slot_sizes[i]}, compute_dtype, dev));

    compile(seq_len);
}

Tensor Executable::run(const std::vector<int>& token_ids, std::size_t pos,
                       KVCache& kv_cache)
{
    const std::size_t seq_len = token_ids.size();
    const std::size_t H = config_.hidden_size;
    const std::size_t head_dim = H / config_.num_attention_heads;
    const std::size_t num_kv_heads = config_.num_key_value_heads;
    const std::size_t kv_dim = num_kv_heads * head_dim;

    Device dev = backend_.device();
    DType compute_dtype = backend_.is_fp16() ? DType::Float16 : DType::Float32;
    std::size_t elem = dtype_size(compute_dtype);

    // Allocate pool + compile on first run or if seq_len changed
    if (ops_.empty())
    {
        allocate_pool(seq_len);
    }
    else
    {
        // Check if we need to re-plan for different seq_len
        MemoryPlan new_plan = plan_memory(*graph_, config_, seq_len);
        bool needs_realloc = (new_plan.num_slots() != plan_.num_slots());
        if (!needs_realloc)
        {
            for (std::size_t i = 0; i < new_plan.num_slots(); i++)
            {
                if (new_plan.slot_sizes[i] > plan_.slot_sizes[i])
                {
                    needs_realloc = true;
                    break;
                }
            }
        }
        if (needs_realloc)
        {
            allocate_pool(seq_len);
        }
        else if (new_plan.slot_index != plan_.slot_index ||
                 new_plan.inplace_values != plan_.inplace_values)
        {
            plan_ = std::move(new_plan);
            compile(seq_len);
        }
    }

    backend_.begin_forward();
    kv_cache.init(config_.num_hidden_layers, kv_dim);
    if (pos == 0) kv_cache.reset();

    // Execute pre-compiled instruction list — zero hash lookups
    for (const auto& op : ops_)
    {
        switch (op.op)
        {
            case OpType::Embedding:
            {
                float* out = slot_buf(op.out_slot);
                const float* embed_table = op.weight_fp32;

                if (compute_dtype == DType::Float16)
                {
                    Tensor tmp({seq_len * H}, DType::Float32, dev);
                    for (std::size_t s = 0; s < seq_len; s++)
                        backend_.embedding(tmp.data<float>() + s * H, embed_table,
                                           token_ids[s], H, config_.vocab_size);
                    backend_.fp32_to_fp16(out, tmp.data<float>(), seq_len * H);
                }
                else
                {
                    for (std::size_t s = 0; s < seq_len; s++)
                        backend_.embedding(out + s * H, embed_table,
                                           token_ids[s], H, config_.vocab_size);
                }
                break;
            }

            case OpType::RMSNorm:
            {
                float* out = slot_buf(op.out_slot);
                const float* x = slot_buf(op.in_slots[0]);
                std::size_t nh = op.hidden;

                for (std::size_t s = 0; s < seq_len; s++)
                {
                    std::size_t off = s * nh * elem;
                    backend_.rms_norm(
                        reinterpret_cast<float*>(reinterpret_cast<char*>(out) + off),
                        reinterpret_cast<const float*>(reinterpret_cast<const char*>(x) + off),
                        op.weight_fp32, nh, op.eps);
                }
                break;
            }

            case OpType::Linear:
            {
                float* out = slot_buf(op.out_slot);
                const float* inp = slot_buf(op.in_slots[0]);
                std::size_t M = op.is_logits ? 1 : seq_len;

                if (op.is_logits && seq_len > 1)
                {
                    inp = reinterpret_cast<const float*>(
                        reinterpret_cast<const char*>(inp) +
                        (seq_len - 1) * op.K * elem);
                }

                if (op.weight_quantized)
                    backend_.linear_q8(out, inp, op.weight_int8, op.weight_scales,
                                       op.bias, M, op.N, op.K);
                else
                    backend_.linear(out, inp, op.weight_fp32, op.bias, M, op.N, op.K);
                break;
            }

            case OpType::RoPE:
            {
                float* q_out = slot_buf(op.out_slot);
                float* k_out = slot_buf(op.out_slot2);

                if (!op.rope_q_inplace)
                {
                    const float* q_in = slot_buf(op.in_slots[0]);
                    device_copy(q_out, dev, q_in, dev, seq_len * op.nh * op.hd * elem);
                }
                if (!op.rope_k_inplace)
                {
                    const float* k_in = slot_buf(op.in_slots[1]);
                    device_copy(k_out, dev, k_in, dev, seq_len * op.nkv * op.hd * elem);
                }

                backend_.rope(q_out, k_out, seq_len, op.nh, op.nkv, op.hd, pos, op.theta);
                break;
            }

            case OpType::Attention:
            {
                float* out = slot_buf(op.out_slot);
                const float* q = slot_buf(op.in_slots[0]);
                const float* k = slot_buf(op.in_slots[1]);
                const float* v = slot_buf(op.in_slots[2]);

                kv_cache.append(op.layer, k, v, seq_len);
                const float* cache_k = kv_cache.k(op.layer);
                const float* cache_v = kv_cache.v(op.layer);
                float scale = 1.0f / std::sqrt(static_cast<float>(op.hd));

                for (std::size_t s = 0; s < seq_len; s++)
                {
                    auto* out_s = reinterpret_cast<float*>(
                        reinterpret_cast<char*>(out) + s * H * elem);
                    backend_.causal_attention(out_s, q, s, cache_k, cache_v,
                                              pos + s + 1, op.nh, op.nkv, op.hd, scale);
                }
                break;
            }

            case OpType::LinearAdd:
            {
                float* out = slot_buf(op.out_slot);
                const float* inp = slot_buf(op.in_slots[0]);
                const float* residual = slot_buf(op.residual_slot);

                if (op.weight_quantized)
                    backend_.linear_q8_add(out, inp, op.weight_int8, op.weight_scales,
                                           nullptr, residual, seq_len, op.N, op.K);
                else
                    backend_.linear_add(out, inp, op.weight_fp32,
                                        nullptr, residual, seq_len, op.N, op.K);
                break;
            }

            case OpType::SiLUMul:
            {
                float* out = slot_buf(op.out_slot);
                const float* gate_up = slot_buf(op.in_slots[0]);
                std::size_t inter = op.intermediate;

                for (std::size_t s = 0; s < seq_len; s++)
                {
                    std::size_t gate_off = s * 2 * inter * elem;
                    std::size_t up_off = gate_off + inter * elem;
                    std::size_t out_off = s * inter * elem;
                    backend_.silu_mul(
                        reinterpret_cast<float*>(reinterpret_cast<char*>(out) + out_off),
                        reinterpret_cast<const float*>(reinterpret_cast<const char*>(gate_up) + gate_off),
                        reinterpret_cast<const float*>(reinterpret_cast<const char*>(gate_up) + up_off),
                        inter);
                }
                break;
            }

            case OpType::Add:
            {
                float* out = slot_buf(op.out_slot);
                backend_.add(out, slot_buf(op.in_slots[0]), slot_buf(op.in_slots[1]),
                             seq_len * H);
                break;
            }

            case OpType::Softmax:
            {
                float* out = slot_buf(op.out_slot);
                // Use N field to store total elements
                backend_.softmax(out, slot_buf(op.in_slots[0]), op.N);
                break;
            }

            default:
                break;
        }
    }

    // Copy logits out of pool
    std::size_t logits_elems = config_.vocab_size;
    Tensor logits({logits_elems}, compute_dtype, dev);
    device_copy(logits.data<float>(), dev, pool_[logits_slot_].data<float>(), dev,
                logits_elems * elem);

    kv_cache.advance(seq_len);

    if (compute_dtype == DType::Float16)
    {
        Tensor logits_fp32({logits_elems}, DType::Float32, dev);
        backend_.fp16_to_fp32(logits_fp32.data<float>(), logits.raw_data(), logits_elems);
        return logits_fp32;
    }

    return logits;
}

}  // namespace ir
}  // namespace vvllm
