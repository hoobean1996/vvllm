#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "vvllm/backend/backend.h"
#include "vvllm/config/config.h"
#include "vvllm/kv_cache/kv_cache.h"
#include "vvllm/tensor/tensor.h"

namespace vvllm
{

struct LinearWeight
{
    const float* fp32 = nullptr;
    const int8_t* int8 = nullptr;
    const float* scales = nullptr;
    bool is_quantized() const { return int8 != nullptr; }
};

struct MLP
{
    LinearWeight gate_up_proj_weight;  // [2 * intermediate_size, hidden_size]
    LinearWeight down_proj_weight;     // [hidden_size, intermediate_size]
};

template <typename Attn>
struct TransformerBlock
{
    const float* input_layernorm_weight;           // [hidden_size]
    const float* post_attention_layernorm_weight;  // [hidden_size]
    Attn attention;
    MLP mlp;
};

// out[M, out_dim] = inp[M, in_dim] @ weight^T + bias
// bias can be nullptr for no-bias projections.
void linear(float* out, const float* inp, const float* weight, const float* bias,
            std::size_t M, std::size_t out_dim, std::size_t in_dim, Backend& backend);

// LinearWeight overload: dispatches to int8 or fp32 path.
void linear(float* out, const float* inp, const LinearWeight& weight, const float* bias,
            std::size_t M, std::size_t out_dim, std::size_t in_dim, Backend& backend);

// out = inp @ weight^T + bias + residual  (fused linear + residual add)
void linear_add(float* out, const float* inp, const LinearWeight& weight, const float* bias,
                const float* residual, std::size_t M, std::size_t out_dim, std::size_t in_dim,
                Backend& backend);

// Apply RMSNorm per token across a sequence.
// elem_size: sizeof(float) for FP32, sizeof(uint16_t) for FP16
void rms_norm_seq(float* out, const float* x, const float* weight, std::size_t seq_len,
                  std::size_t hidden, float eps, Backend& backend, std::size_t elem_size = 4);

/// Shared forward pass for transformer models.
/// Attn must provide: q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight.
/// ADL free functions q_bias(Attn&), k_bias(Attn&), v_bias(Attn&) resolve bias.
template <typename Attn>
std::vector<float> transformer_forward(const std::vector<TransformerBlock<Attn>>& layers,
                                       const float* embed_tokens, const float* final_norm_weight,
                                       const ModelConfig& config, Backend& backend,
                                       KVCache& kv_cache, const std::vector<int>& token_ids,
                                       std::size_t pos)
{
    const std::size_t head_dim = config.hidden_size / config.num_attention_heads;
    const std::size_t seq_len = token_ids.size();
    const std::size_t hidden = config.hidden_size;
    const std::size_t num_heads = config.num_attention_heads;
    const std::size_t num_kv_heads = config.num_key_value_heads;
    const std::size_t kv_dim = num_kv_heads * head_dim;
    const std::size_t intermediate = config.intermediate_size;
    const float eps = static_cast<float>(config.rms_norm_eps);
    const float rope_theta = static_cast<float>(config.rope_theta);

    Device dev = backend.device();
    DType compute_dtype = backend.is_fp16() ? DType::Float16 : DType::Float32;
    std::size_t elem = dtype_size(compute_dtype);

    // Invalidate stale GPU mirrors from previous forward passes
    backend.begin_forward();

    // Initialize KV cache (idempotent); reset on recompute
    kv_cache.init(config.num_hidden_layers, kv_dim);
    if (pos == 0) kv_cache.reset();

    // Hidden state: [seq_len, hidden] — on backend's device, in compute dtype
    Tensor x({seq_len * hidden}, compute_dtype, dev);

    // 1. Embedding lookup (always FP32 output, then convert if needed)
    if (compute_dtype == DType::Float16)
    {
        Tensor x_fp32({seq_len * hidden}, DType::Float32, dev);
        for (std::size_t s = 0; s < seq_len; s++)
        {
            backend.embedding(x_fp32.data<float>() + s * hidden, embed_tokens, token_ids[s],
                              hidden, config.vocab_size);
        }
        // Convert FP32 → FP16
        backend.fp32_to_fp16(x.raw_data(), x_fp32.data<float>(), seq_len * hidden);
    }
    else
    {
        for (std::size_t s = 0; s < seq_len; s++)
        {
            backend.embedding(x.data<float>() + s * hidden, embed_tokens, token_ids[s], hidden,
                              config.vocab_size);
        }
    }

    // Scratch buffers on backend's device in compute dtype
    Tensor norm_out({seq_len * hidden}, compute_dtype, dev);
    Tensor q({seq_len * num_heads * head_dim}, compute_dtype, dev);
    Tensor k_new({seq_len * kv_dim}, compute_dtype, dev);
    Tensor v_new({seq_len * kv_dim}, compute_dtype, dev);
    Tensor attn_out({seq_len * hidden}, compute_dtype, dev);
    Tensor gate_up({seq_len * 2 * intermediate}, compute_dtype, dev);
    Tensor gate({seq_len * intermediate}, compute_dtype, dev);

    // 2. Transformer layers
    for (std::size_t layer_idx = 0; layer_idx < config.num_hidden_layers; layer_idx++)
    {
        auto& layer = layers[layer_idx];
        auto& attn = layer.attention;
        auto& mlp = layer.mlp;

        // --- Self-Attention ---

        // norm_out = rms_norm(x)
        rms_norm_seq(norm_out.at(0), x.at(0), layer.input_layernorm_weight,
                     seq_len, hidden, eps, backend, elem);

        // Q/K/V projections: batched over all tokens
        linear(q.at(0), norm_out.at(0), attn.q_proj_weight, q_bias(attn),
               seq_len, num_heads * head_dim, hidden, backend);
        linear(k_new.at(0), norm_out.at(0), attn.k_proj_weight, k_bias(attn),
               seq_len, kv_dim, hidden, backend);
        linear(v_new.at(0), norm_out.at(0), attn.v_proj_weight, v_bias(attn),
               seq_len, kv_dim, hidden, backend);

        // q, k_new = rope(q, k_new)
        backend.rope(q.at(0), k_new.at(0), seq_len, num_heads, num_kv_heads,
                     head_dim, pos, rope_theta);

        // Append K/V to cache (pointers are already on the right device)
        kv_cache.append(layer_idx, k_new.at(0), v_new.at(0), seq_len);
        const float* cache_k = kv_cache.k(layer_idx);
        const float* cache_v = kv_cache.v(layer_idx);

        // attn_out = softmax(q @ K^T / sqrt(head_dim)) @ V  (K, V from cache)
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (std::size_t s = 0; s < seq_len; s++)
        {
            backend.causal_attention(attn_out.at(s * hidden), q.at(0), s,
                                     cache_k, cache_v,
                                     pos + s + 1, num_heads, num_kv_heads, head_dim, scale);
        }

        // x = attn_out @ o_weight^T + x  (fused projection + residual add)
        linear_add(x.at(0), attn_out.at(0), attn.o_proj_weight,
                   nullptr, x.at(0), seq_len, hidden, hidden, backend);

        // --- MLP ---

        // norm_out = rms_norm(x)
        rms_norm_seq(norm_out.at(0), x.at(0),
                     layer.post_attention_layernorm_weight, seq_len, hidden, eps, backend, elem);

        // Fused gate+up projection: gate_up = norm_out @ gate_up_weight^T
        linear(gate_up.at(0), norm_out.at(0), mlp.gate_up_proj_weight, nullptr,
               seq_len, 2 * intermediate, hidden, backend);

        // silu_mul per token: gate_up is [seq_len, 2*intermediate] interleaved
        for (std::size_t s = 0; s < seq_len; s++)
        {
            float* gu = gate_up.at(s * 2 * intermediate);
            backend.silu_mul(gate.at(s * intermediate), gu,
                             gate_up.at(s * 2 * intermediate + intermediate), intermediate);
        }

        // x = gate @ down_weight^T + x  (fused projection + residual add)
        linear_add(x.at(0), gate.at(0), mlp.down_proj_weight, nullptr,
                   x.at(0), seq_len, hidden, intermediate, backend);
    }

    // Advance cache position after all layers processed
    kv_cache.advance(seq_len);

    // 3. final_out = rms_norm(x[-1])  (last token only)
    Tensor final_out({hidden}, compute_dtype, dev);
    backend.rms_norm(final_out.at(0), x.at((seq_len - 1) * hidden), final_norm_weight, hidden,
                     eps);

    // 4. logits = final_out * embed_tokens  (tied weights)
    // Logits always computed in FP32 for sampler precision
    Tensor logits({static_cast<std::size_t>(config.vocab_size)}, compute_dtype, dev);
    linear(logits.at(0), final_out.at(0), embed_tokens, nullptr, 1, config.vocab_size, hidden,
           backend);

    // Download logits to CPU as FP32
    std::vector<float> result(config.vocab_size);
    if (compute_dtype == DType::Float16 && dev == Device::CUDA)
    {
        // FP16 logits on GPU → convert to FP32 on GPU → download
        Tensor logits_fp32({static_cast<std::size_t>(config.vocab_size)}, DType::Float32, dev);
        backend.fp16_to_fp32(logits_fp32.data<float>(), logits.raw_data(), config.vocab_size);
        Tensor cpu_logits = logits_fp32.to(Device::CPU);
        std::memcpy(result.data(), cpu_logits.data<float>(), result.size() * sizeof(float));
    }
    else if (dev == Device::CUDA)
    {
        Tensor cpu_logits = logits.to(Device::CPU);
        std::memcpy(result.data(), cpu_logits.data<float>(), result.size() * sizeof(float));
    }
    else
    {
        std::memcpy(result.data(), logits.data<float>(), result.size() * sizeof(float));
    }

    return result;
}

}  // namespace vvllm
