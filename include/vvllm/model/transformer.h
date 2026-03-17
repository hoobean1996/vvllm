#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "vvllm/backend/backend.h"
#include "vvllm/config/config.h"
#include "vvllm/model/kv_cache.h"
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
void rms_norm_seq(float* out, const float* x, const float* weight, std::size_t seq_len,
                  std::size_t hidden, float eps, Backend& backend);

/// Shared forward pass for transformer models.
/// Attn must provide: q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight.
/// ADL free functions q_bias(Attn&), k_bias(Attn&), v_bias(Attn&) resolve bias.
template <typename Attn>
std::vector<float> transformer_forward(const std::vector<TransformerBlock<Attn>>& layers,
                                       const float* embed_tokens, const float* final_norm_weight,
                                       const ModelConfig& config, Backend& backend,
                                       const std::vector<int>& token_ids, std::size_t pos,
                                       KVCache& kv_cache)
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

    // Invalidate stale GPU mirrors from previous forward passes
    backend.begin_forward();

    // Initialize GPU KV cache (idempotent); reset on recompute
    backend.kv_cache_init(config.num_hidden_layers, kv_dim);
    if (pos == 0) backend.kv_cache_reset();

    const bool use_gpu_kv = (backend.kv_cache_k(0) != nullptr);

    // Hidden state: [seq_len, hidden]
    std::vector<float> x(seq_len * hidden);

    // 1. Embedding lookup
    for (std::size_t s = 0; s < seq_len; s++)
    {
        backend.embedding(x.data() + s * hidden, embed_tokens, token_ids[s], hidden);
    }

    // Scratch buffers (only for new tokens)
    std::vector<float> norm_out(seq_len * hidden);
    std::vector<float> q(seq_len * num_heads * head_dim);
    std::vector<float> k_new(seq_len * kv_dim);
    std::vector<float> v_new(seq_len * kv_dim);
    std::vector<float> attn_out(seq_len * hidden);
    std::vector<float> gate_up(seq_len * 2 * intermediate);
    std::vector<float> gate(seq_len * intermediate);

    // 2. Transformer layers
    for (std::size_t layer_idx = 0; layer_idx < config.num_hidden_layers; layer_idx++)
    {
        auto& layer = layers[layer_idx];
        auto& attn = layer.attention;
        auto& mlp = layer.mlp;

        // --- Self-Attention ---

        // norm_out = rms_norm(x)
        rms_norm_seq(norm_out.data(), x.data(), layer.input_layernorm_weight, seq_len, hidden, eps,
                     backend);

        // Q/K/V projections: batched over all tokens
        linear(q.data(), norm_out.data(), attn.q_proj_weight, q_bias(attn),
               seq_len, num_heads * head_dim, hidden, backend);
        linear(k_new.data(), norm_out.data(), attn.k_proj_weight, k_bias(attn),
               seq_len, kv_dim, hidden, backend);
        linear(v_new.data(), norm_out.data(), attn.v_proj_weight, v_bias(attn),
               seq_len, kv_dim, hidden, backend);

        // q, k_new = rope(q, k_new)
        backend.rope(q.data(), k_new.data(), seq_len, num_heads, num_kv_heads, head_dim, pos,
                     rope_theta);

        // Append new K/V and get cache pointers (GPU or CPU path)
        const float* cache_k;
        const float* cache_v;
        if (use_gpu_kv)
        {
            // GPU path: D2D copy new tokens to GPU KV buffer, no CPU involvement
            backend.kv_cache_append(layer_idx, k_new.data(), v_new.data(), seq_len);
            cache_k = backend.kv_cache_k(layer_idx);
            cache_v = backend.kv_cache_v(layer_idx);
        }
        else
        {
            // CPU path: flush GPU mirrors to CPU, append to CPU KV cache
            backend.flush(k_new.data(), k_new.size() * sizeof(float));
            backend.flush(v_new.data(), v_new.size() * sizeof(float));
            kv_cache.append(layer_idx, k_new.data(), v_new.data(), seq_len, kv_dim);
            cache_k = kv_cache.k_data(layer_idx);
            cache_v = kv_cache.v_data(layer_idx);
        }

        // attn_out = softmax(q @ K^T / sqrt(head_dim)) @ V  (K, V from cache)
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (std::size_t s = 0; s < seq_len; s++)
        {
            backend.causal_attention(attn_out.data() + s * hidden, q.data(), s,
                                     cache_k, cache_v,
                                     pos + s + 1, num_heads, num_kv_heads, head_dim, scale);
        }

        // x = attn_out @ o_weight^T + x  (fused projection + residual add)
        linear_add(x.data(), attn_out.data(), attn.o_proj_weight,
                   nullptr, x.data(), seq_len, hidden, hidden, backend);

        // --- MLP ---

        // norm_out = rms_norm(x)
        rms_norm_seq(norm_out.data(), x.data(), layer.post_attention_layernorm_weight, seq_len,
                     hidden, eps, backend);

        // Fused gate+up projection: gate_up = norm_out @ gate_up_weight^T
        linear(gate_up.data(), norm_out.data(), mlp.gate_up_proj_weight, nullptr,
               seq_len, 2 * intermediate, hidden, backend);

        // silu_mul per token: gate_up is [seq_len, 2*intermediate] interleaved
        for (std::size_t s = 0; s < seq_len; s++)
        {
            float* gu = gate_up.data() + s * 2 * intermediate;
            backend.silu_mul(gate.data() + s * intermediate, gu, gu + intermediate, intermediate);
        }

        // x = gate @ down_weight^T + x  (fused projection + residual add)
        linear_add(x.data(), gate.data(), mlp.down_proj_weight, nullptr,
                   x.data(), seq_len, hidden, intermediate, backend);
    }

    // Advance cache position after all layers processed
    kv_cache.advance(seq_len);
    backend.kv_cache_advance(seq_len);

    // 3. final_out = rms_norm(x[-1])  (last token only)
    std::vector<float> final_out(hidden);
    backend.rms_norm(final_out.data(), x.data() + (seq_len - 1) * hidden, final_norm_weight, hidden,
                     eps);

    // 4. logits = final_out * embed_tokens  (tied weights)
    std::vector<float> logits(config.vocab_size);
    linear(logits.data(), final_out.data(), embed_tokens, nullptr, 1, config.vocab_size, hidden,
           backend);

    backend.flush(logits.data(), logits.size() * sizeof(float));
    return logits;
}

}  // namespace vvllm
