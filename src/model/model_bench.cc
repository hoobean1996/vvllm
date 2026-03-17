#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"
#include "src/backend/backend_blas.h"
#include "src/backend/backend_naive.h"
#include "vvllm/config/config.h"
#include "vvllm/model/llama/model.h"
#include "vvllm/model/transformer.h"
#include "vvllm/quantize/quantize.h"

namespace
{

std::vector<float> random_floats(std::size_t n, unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
    std::vector<float> v(n);
    for (auto& x : v)
    {
        x = dist(rng);
    }
    return v;
}

// ---------------------------------------------------------------------------
// BM_Linear — matrix-vector product with varying hidden_size
// ---------------------------------------------------------------------------

template <typename BackendT>
void BM_Linear(benchmark::State& state)
{
    const std::size_t dim = static_cast<std::size_t>(state.range(0));
    auto weight = random_floats(dim * dim);
    auto input = random_floats(dim);
    std::vector<float> output(dim);
    BackendT backend;

    for (auto _ : state)
    {
        vvllm::linear(output.data(), input.data(), weight.data(), nullptr, 1, dim, dim, backend);
        benchmark::DoNotOptimize(output.data());
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_Linear<vvllm::BackendCPU>)->Arg(64)->Arg(128)->Arg(256)->Arg(576);
BENCHMARK(BM_Linear<vvllm::BackendBLAS>)->Arg(64)->Arg(128)->Arg(256)->Arg(576);

// ---------------------------------------------------------------------------
// BM_RmsNormSeq — rms_norm across a sequence, varying seq_len
// ---------------------------------------------------------------------------

void BM_RmsNormSeq(benchmark::State& state)
{
    const std::size_t seq_len = static_cast<std::size_t>(state.range(0));
    const std::size_t hidden = 256;
    vvllm::BackendCPU backend;
    auto x = random_floats(seq_len * hidden);
    auto weight = random_floats(hidden);
    std::vector<float> out(seq_len * hidden);

    for (auto _ : state)
    {
        vvllm::rms_norm_seq(out.data(), x.data(), weight.data(), seq_len, hidden, 1e-6f, backend);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(seq_len));
}

BENCHMARK(BM_RmsNormSeq)->Arg(1)->Arg(8)->Arg(32)->Arg(128);

// ---------------------------------------------------------------------------
// BM_CausalAttention — single query attention, varying attend_len (O(n))
// ---------------------------------------------------------------------------

template <typename BackendT>
void BM_CausalAttention(benchmark::State& state)
{
    const std::size_t attend_len = static_cast<std::size_t>(state.range(0));
    const std::size_t num_heads = 4;
    const std::size_t num_kv_heads = 4;
    const std::size_t head_dim = 64;
    const float scale = 1.0f / 8.0f;  // 1/sqrt(64)

    BackendT backend;
    auto q = random_floats(num_heads * head_dim);
    auto k = random_floats(attend_len * num_kv_heads * head_dim);
    auto v = random_floats(attend_len * num_kv_heads * head_dim);
    std::vector<float> out(num_heads * head_dim);

    for (auto _ : state)
    {
        backend.causal_attention(out.data(), q.data(), 0, k.data(), v.data(), attend_len, num_heads,
                                 num_kv_heads, head_dim, scale);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_CausalAttention<vvllm::BackendCPU>)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_CausalAttention<vvllm::BackendBLAS>)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);

// ---------------------------------------------------------------------------
// BM_CausalAttentionGQA — GQA config matching Qwen2.5-0.5B (14 heads, 2 KV)
// ---------------------------------------------------------------------------

template <typename BackendT>
void BM_CausalAttentionGQA(benchmark::State& state)
{
    const std::size_t attend_len = static_cast<std::size_t>(state.range(0));
    const std::size_t num_heads = 14;
    const std::size_t num_kv_heads = 2;
    const std::size_t head_dim = 64;
    const float scale = 1.0f / 8.0f;  // 1/sqrt(64)

    BackendT backend;
    auto q = random_floats(num_heads * head_dim);
    auto k = random_floats(attend_len * num_kv_heads * head_dim);
    auto v = random_floats(attend_len * num_kv_heads * head_dim);
    std::vector<float> out(num_heads * head_dim);

    for (auto _ : state)
    {
        backend.causal_attention(out.data(), q.data(), 0, k.data(), v.data(), attend_len, num_heads,
                                 num_kv_heads, head_dim, scale);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_CausalAttentionGQA<vvllm::BackendCPU>)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_CausalAttentionGQA<vvllm::BackendBLAS>)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);

// ---------------------------------------------------------------------------
// BM_KVCacheAppend — append tokens at different cache depths
// ---------------------------------------------------------------------------

void BM_KVCacheAppend(benchmark::State& state)
{
    const std::size_t cache_depth = static_cast<std::size_t>(state.range(0));
    const std::size_t kv_dim = 256;
    const std::size_t num_layers = 2;

    // Pre-fill cache to desired depth
    auto prefill_k = random_floats(cache_depth * kv_dim);
    auto prefill_v = random_floats(cache_depth * kv_dim);
    auto new_k = random_floats(kv_dim);
    auto new_v = random_floats(kv_dim);

    for (auto _ : state)
    {
        state.PauseTiming();
        vvllm::BackendCPU backend;
        backend.kv_cache_init(num_layers, kv_dim);
        for (std::size_t l = 0; l < num_layers; l++)
        {
            backend.kv_cache_append(l, prefill_k.data(), prefill_v.data(), cache_depth);
        }
        backend.kv_cache_advance(cache_depth);
        state.ResumeTiming();

        for (std::size_t l = 0; l < num_layers; l++)
        {
            backend.kv_cache_append(l, new_k.data(), new_v.data(), 1);
        }
        benchmark::DoNotOptimize(backend.kv_cache_k(0));
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_KVCacheAppend)->Arg(0)->Arg(64)->Arg(256)->Arg(1024);

// ---------------------------------------------------------------------------
// Shared tiny model setup for transformer benchmarks
// ---------------------------------------------------------------------------

struct TinyModel
{
    static constexpr std::size_t kHidden = 64;
    static constexpr std::size_t kIntermediate = 128;
    static constexpr std::size_t kNumHeads = 4;
    static constexpr std::size_t kNumKvHeads = 4;
    static constexpr std::size_t kHeadDim = kHidden / kNumHeads;  // 16
    static constexpr std::size_t kNumLayers = 2;
    static constexpr std::size_t kVocabSize = 32;
    static constexpr std::size_t kMaxPos = 2048;

    vvllm::ModelConfig config;
    std::vector<float> embed_tokens;
    std::vector<float> final_norm_weight;

    // Per-layer weight storage
    struct LayerWeights
    {
        std::vector<float> input_ln;
        std::vector<float> post_ln;
        std::vector<float> q_proj;
        std::vector<float> k_proj;
        std::vector<float> v_proj;
        std::vector<float> o_proj;
        std::vector<float> gate_up_proj;
        std::vector<float> down_proj;
    };
    std::vector<LayerWeights> layer_weights;
    std::vector<vvllm::TransformerBlock<vvllm::LlamaAttention>> layers;

    TinyModel()
    {
        config.model_type = "llama";
        config.vocab_size = kVocabSize;
        config.hidden_size = kHidden;
        config.intermediate_size = kIntermediate;
        config.num_hidden_layers = kNumLayers;
        config.num_attention_heads = kNumHeads;
        config.num_key_value_heads = kNumKvHeads;
        config.max_position_embeddings = kMaxPos;
        config.rms_norm_eps = 1e-6;
        config.rope_theta = 10000.0;
        config.bos_token_id = 0;
        config.eos_token_ids = {1};
        config.tie_word_embeddings = true;

        embed_tokens = random_floats(kVocabSize * kHidden, 1);
        final_norm_weight = random_floats(kHidden, 2);
        // Make norm weights positive (RMSNorm scaling)
        for (auto& w : final_norm_weight)
        {
            w = 1.0f + w;
        }

        const std::size_t kv_dim = kNumKvHeads * kHeadDim;

        layer_weights.resize(kNumLayers);
        layers.resize(kNumLayers);

        for (std::size_t i = 0; i < kNumLayers; i++)
        {
            auto& lw = layer_weights[i];
            unsigned seed_base = static_cast<unsigned>(100 + i * 10);

            lw.input_ln = random_floats(kHidden, seed_base);
            lw.post_ln = random_floats(kHidden, seed_base + 1);
            for (auto& w : lw.input_ln)
            {
                w = 1.0f + w;
            }
            for (auto& w : lw.post_ln)
            {
                w = 1.0f + w;
            }

            lw.q_proj = random_floats(kNumHeads * kHeadDim * kHidden, seed_base + 2);
            lw.k_proj = random_floats(kv_dim * kHidden, seed_base + 3);
            lw.v_proj = random_floats(kv_dim * kHidden, seed_base + 4);
            lw.o_proj = random_floats(kHidden * kHidden, seed_base + 5);
            lw.gate_up_proj = random_floats(2 * kIntermediate * kHidden, seed_base + 6);
            lw.down_proj = random_floats(kHidden * kIntermediate, seed_base + 8);

            layers[i].input_layernorm_weight = lw.input_ln.data();
            layers[i].post_attention_layernorm_weight = lw.post_ln.data();
            layers[i].attention.q_proj_weight = {lw.q_proj.data(), nullptr, nullptr};
            layers[i].attention.k_proj_weight = {lw.k_proj.data(), nullptr, nullptr};
            layers[i].attention.v_proj_weight = {lw.v_proj.data(), nullptr, nullptr};
            layers[i].attention.o_proj_weight = {lw.o_proj.data(), nullptr, nullptr};
            layers[i].mlp.gate_up_proj_weight = {lw.gate_up_proj.data(), nullptr, nullptr};
            layers[i].mlp.down_proj_weight = {lw.down_proj.data(), nullptr, nullptr};
        }
    }
};

// ---------------------------------------------------------------------------
// BM_Prefill — full transformer_forward for a batch of tokens
// ---------------------------------------------------------------------------

void BM_Prefill(benchmark::State& state)
{
    const std::size_t seq_len = static_cast<std::size_t>(state.range(0));
    TinyModel model;
    vvllm::BackendCPU backend;

    std::vector<int> token_ids(seq_len);
    std::iota(token_ids.begin(), token_ids.end(), 2);  // tokens 2,3,4,...
    for (auto& t : token_ids)
    {
        t = t % static_cast<int>(TinyModel::kVocabSize);
    }

    for (auto _ : state)
    {
        backend.kv_cache_reset();
        auto logits = vvllm::transformer_forward(model.layers, model.embed_tokens.data(),
                                                 model.final_norm_weight.data(), model.config,
                                                 backend, token_ids, 0);
        benchmark::DoNotOptimize(logits.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<long>(seq_len));
}

BENCHMARK(BM_Prefill)->Arg(4)->Arg(16)->Arg(64);

// ---------------------------------------------------------------------------
// BM_DecodeWithCache — single-token decode with pre-filled KV cache
// ---------------------------------------------------------------------------

void BM_DecodeWithCache(benchmark::State& state)
{
    const std::size_t context_len = static_cast<std::size_t>(state.range(0));
    TinyModel model;
    vvllm::BackendCPU backend;

    // Pre-fill the cache by running the prefill phase once
    std::vector<int> prefill_ids(context_len);
    std::iota(prefill_ids.begin(), prefill_ids.end(), 2);
    for (auto& t : prefill_ids)
    {
        t = t % static_cast<int>(TinyModel::kVocabSize);
    }

    vvllm::transformer_forward(model.layers, model.embed_tokens.data(),
                               model.final_norm_weight.data(), model.config, backend, prefill_ids,
                               0);

    // Decode a single token from the cached state
    std::vector<int> decode_id = {5};

    for (auto _ : state)
    {
        // Rewind cache to prefill state
        backend.kv_cache_truncate(context_len);
        auto logits = vvllm::transformer_forward(model.layers, model.embed_tokens.data(),
                                                 model.final_norm_weight.data(), model.config,
                                                 backend, decode_id, context_len);
        benchmark::DoNotOptimize(logits.data());
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_DecodeWithCache)->Arg(16)->Arg(64)->Arg(256);

// ---------------------------------------------------------------------------
// BM_DecodeNoCache — recompute full sequence each step (the slow path)
// ---------------------------------------------------------------------------

void BM_DecodeNoCache(benchmark::State& state)
{
    const std::size_t seq_len = static_cast<std::size_t>(state.range(0));
    TinyModel model;
    vvllm::BackendCPU backend;

    // Build token sequence
    std::vector<int> token_ids(seq_len);
    std::iota(token_ids.begin(), token_ids.end(), 2);
    for (auto& t : token_ids)
    {
        t = t % static_cast<int>(TinyModel::kVocabSize);
    }

    for (auto _ : state)
    {
        // Fresh cache each time — full recompute (pos=0 triggers reset)
        auto logits = vvllm::transformer_forward(model.layers, model.embed_tokens.data(),
                                                 model.final_norm_weight.data(), model.config,
                                                 backend, token_ids, 0);
        benchmark::DoNotOptimize(logits.data());
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_DecodeNoCache)->Arg(16)->Arg(64)->Arg(256);

// ---------------------------------------------------------------------------
// BM_LinearQ8 — int8 quantized linear vs fp32
// ---------------------------------------------------------------------------

template <typename BackendT>
void BM_LinearQ8(benchmark::State& state)
{
    const std::size_t dim = static_cast<std::size_t>(state.range(0));
    auto weight_fp32 = random_floats(dim * dim);
    auto input = random_floats(dim);
    auto qw = vvllm::quantize_per_channel(weight_fp32.data(), dim, dim);
    std::vector<float> output(dim);
    BackendT backend;

    for (auto _ : state)
    {
        backend.linear_q8(output.data(), input.data(), qw.data.data(), qw.scales.data(), nullptr, 1,
                          dim, dim);
        benchmark::DoNotOptimize(output.data());
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_LinearQ8<vvllm::BackendCPU>)->Arg(64)->Arg(128)->Arg(256)->Arg(576);

}  // namespace
