#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "src/backend/backend_blas.h"
#include "src/backend/backend_cuda.h"
#include "src/backend/backend_naive.h"
#include "src/kv_cache/kv_cache_cpu.h"
#include "src/kv_cache/kv_cache_cuda.h"
#include "vvllm/config/config.h"
#include "vvllm/ir/executable.h"
#include "vvllm/ir/graph_builder.h"
#include "vvllm/ir/optimize.h"
#include "vvllm/model/model.h"
#include "vvllm/quantize/quantize.h"
#include "vvllm/safetensors/safetensors.h"
#include "src/sampler/sampler_cpu.h"
#include "src/sampler/sampler_cuda.h"
#include "vvllm/sampler/sampler.h"
#include "vvllm/stats/stats.h"
#include "vvllm/tokenizer/tokenizer.h"

DEFINE_string(model, "", "Path to model directory");
DEFINE_string(prompt, "", "Input prompt");
DEFINE_int32(max_tokens, 50, "Maximum number of tokens to generate");
DEFINE_double(temperature, 0.7, "Sampling temperature (0 = greedy)");
DEFINE_double(top_p, 0.9, "Top-p (nucleus) sampling threshold");
DEFINE_uint64(seed, 42, "Random seed for sampling");
DEFINE_string(backend, "cpu", "Backend to use: cpu, blas, cuda");
DEFINE_string(quantize, "", "Weight quantization: int8 (empty for none)");
DEFINE_bool(fp16, false, "Use FP16 inference (CUDA only)");
DEFINE_bool(compile, false, "Use compiled execution (IR graph) instead of interpreted");

int main(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_model.empty() || FLAGS_prompt.empty())
    {
        std::cerr << "Usage: --model <path to model directory> --prompt <text>" << std::endl;
        return 1;
    }

    std::string model_dir = FLAGS_model;
    std::string config_path = model_dir + "/config.json";
    std::string weights_path = model_dir + "/model.safetensors";
    std::string tokenizer_path = model_dir + "/tokenizer.json";

    // Load config
    auto config = vvllm::load_config(config_path);
    std::cout << "Model: " << config.model_type << " (" << config.hidden_size << "d, "
              << config.num_hidden_layers << " layers)" << std::endl;

    // Load tokenizer
    vvllm::Tokenizer tokenizer(tokenizer_path);
    tokenizer.load();

    // Create backend
    std::unique_ptr<vvllm::Backend> backend_ptr;
    if (FLAGS_backend == "cuda")
    {
        backend_ptr = std::make_unique<vvllm::BackendCUDA>(FLAGS_fp16);
    }
    else if (FLAGS_backend == "blas")
    {
        backend_ptr = std::make_unique<vvllm::BackendBLAS>();
    }
    else if (FLAGS_backend == "cpu")
    {
        backend_ptr = std::make_unique<vvllm::BackendCPU>();
    }
    else
    {
        std::cerr << "Unknown backend: " << FLAGS_backend << " (use cpu, blas, or cuda)"
                  << std::endl;
        return 1;
    }
    auto& backend = *backend_ptr;
    std::cout << "Backend: " << FLAGS_backend << std::endl;

    // Load weights
    std::cout << "Loading weights..." << std::endl;
    vvllm::SafeTensorsLoader loader(weights_path);
    loader.parse();
    auto weights = loader.load_all();
    std::cout << "Loaded " << weights.size() << " tensors" << std::endl;

    // Parse quantization flag
    bool quantize = false;
    if (!FLAGS_quantize.empty())
    {
        if (FLAGS_quantize == "int8")
        {
            quantize = true;
        }
        else
        {
            std::cerr << "Unknown quantization: " << FLAGS_quantize << " (use int8)" << std::endl;
            return 1;
        }
    }

    // Tokenize prompt
    std::vector<int> token_ids = tokenizer.encode(FLAGS_prompt);
    std::cout << "Prompt: \"" << FLAGS_prompt << "\" (" << token_ids.size() << " tokens)"
              << std::endl;

    // EOS check helper
    auto is_eos = [&config](int token) {
        for (int eos : config.eos_token_ids)
        {
            if (token == eos) return true;
        }
        return false;
    };

    // Create sampler and KV cache matching the backend
    std::unique_ptr<vvllm::Sampler> sampler;
    std::unique_ptr<vvllm::KVCache> kv_cache;
    if (FLAGS_backend == "cuda")
    {
        sampler = std::make_unique<vvllm::SamplerCUDA>(
            static_cast<float>(FLAGS_temperature), static_cast<float>(FLAGS_top_p), FLAGS_seed);
        kv_cache = std::make_unique<vvllm::KVCacheCUDA>(FLAGS_fp16);
    }
    else
    {
        sampler = std::make_unique<vvllm::SamplerCPU>(
            static_cast<float>(FLAGS_temperature), static_cast<float>(FLAGS_top_p), FLAGS_seed);
        kv_cache = std::make_unique<vvllm::KVCacheCPU>();
    }

    vvllm::Stats stats;
    vvllm::Tensor logits;

    if (FLAGS_compile)
    {
        // ===== Compiled execution path =====
        std::cout << "Mode: compiled" << std::endl;

        // Build WeightMap: fuse gate+up, optionally quantize (same as interpreted path)
        vvllm::ir::WeightMap weight_map;
        // Storage for fused gate_up weights and quantized weights (must outlive executable)
        std::vector<std::vector<float>> fused_gate_up;
        std::vector<vvllm::QuantizedWeight> quant_store;

        auto add_weight = [&](const std::string& name, const float* fp32,
                              std::size_t N, std::size_t K) {
            if (quantize)
            {
                quant_store.push_back(vvllm::quantize_per_channel(fp32, N, K));
                auto& qw = quant_store.back();
                weight_map.emplace(name, vvllm::ir::WeightRef{nullptr, qw.data.data(),
                                                               qw.scales.data(), true});
            }
            else
            {
                weight_map.emplace(name, vvllm::ir::WeightRef{fp32, nullptr, nullptr, false});
            }
        };

        const std::size_t H = config.hidden_size;
        const std::size_t I = config.intermediate_size;
        const std::size_t head_dim = H / config.num_attention_heads;
        const std::size_t kv_dim = config.num_key_value_heads * head_dim;

        // Embedding + final norm (never quantized)
        weight_map.emplace("model.embed_tokens.weight",
                           vvllm::ir::WeightRef{weights.at("model.embed_tokens.weight").data<float>(),
                                                nullptr, nullptr, false});
        weight_map.emplace("model.norm.weight",
                           vvllm::ir::WeightRef{weights.at("model.norm.weight").data<float>(),
                                                nullptr, nullptr, false});

        bool has_qkv_bias = (config.model_type == "qwen2");

        for (std::size_t i = 0; i < config.num_hidden_layers; i++)
        {
            std::string p = "model.layers." + std::to_string(i) + ".";

            // Norm weights (never quantized)
            weight_map.emplace(p + "input_layernorm.weight",
                               vvllm::ir::WeightRef{weights.at(p + "input_layernorm.weight").data<float>(),
                                                    nullptr, nullptr, false});
            weight_map.emplace(p + "post_attention_layernorm.weight",
                               vvllm::ir::WeightRef{weights.at(p + "post_attention_layernorm.weight").data<float>(),
                                                    nullptr, nullptr, false});

            // QKV + O projections
            add_weight(p + "self_attn.q_proj.weight",
                       weights.at(p + "self_attn.q_proj.weight").data<float>(),
                       config.num_attention_heads * head_dim, H);
            add_weight(p + "self_attn.k_proj.weight",
                       weights.at(p + "self_attn.k_proj.weight").data<float>(),
                       kv_dim, H);
            add_weight(p + "self_attn.v_proj.weight",
                       weights.at(p + "self_attn.v_proj.weight").data<float>(),
                       kv_dim, H);
            add_weight(p + "self_attn.o_proj.weight",
                       weights.at(p + "self_attn.o_proj.weight").data<float>(),
                       H, H);

            // Bias (if present, never quantized)
            if (has_qkv_bias)
            {
                weight_map.emplace(p + "self_attn.q_proj.bias",
                                   vvllm::ir::WeightRef{weights.at(p + "self_attn.q_proj.bias").data<float>(),
                                                        nullptr, nullptr, false});
                weight_map.emplace(p + "self_attn.k_proj.bias",
                                   vvllm::ir::WeightRef{weights.at(p + "self_attn.k_proj.bias").data<float>(),
                                                        nullptr, nullptr, false});
                weight_map.emplace(p + "self_attn.v_proj.bias",
                                   vvllm::ir::WeightRef{weights.at(p + "self_attn.v_proj.bias").data<float>(),
                                                        nullptr, nullptr, false});
            }

            // Fuse gate + up into single weight (same as Qwen2Model::load_weights)
            const float* gate_fp32 = weights.at(p + "mlp.gate_proj.weight").data<float>();
            const float* up_fp32 = weights.at(p + "mlp.up_proj.weight").data<float>();
            std::size_t half_size = I * H;
            fused_gate_up.emplace_back(2 * half_size);
            auto& fused = fused_gate_up.back();
            std::memcpy(fused.data(), gate_fp32, half_size * sizeof(float));
            std::memcpy(fused.data() + half_size, up_fp32, half_size * sizeof(float));
            add_weight(p + "mlp.gate_up_proj.weight", fused.data(), 2 * I, H);

            add_weight(p + "mlp.down_proj.weight",
                       weights.at(p + "mlp.down_proj.weight").data<float>(),
                       H, I);
        }

        // Build graph (decode mode, seq_len=1 — we'll reuse for prefill too)
        vvllm::ir::GraphBuildOptions prefill_opts;
        prefill_opts.mode = vvllm::ir::GraphMode::Prefill;
        prefill_opts.seq_len = token_ids.size();
        prefill_opts.has_qkv_bias = has_qkv_bias;
        auto prefill_graph = vvllm::ir::build_transformer_graph(config, prefill_opts);
        vvllm::ir::optimize(prefill_graph.get());

        vvllm::ir::GraphBuildOptions decode_opts;
        decode_opts.mode = vvllm::ir::GraphMode::Decode;
        decode_opts.seq_len = 1;
        decode_opts.has_qkv_bias = has_qkv_bias;
        auto decode_graph = vvllm::ir::build_transformer_graph(config, decode_opts);
        vvllm::ir::optimize(decode_graph.get());

        std::cout << "Compiled: prefill graph " << prefill_graph->num_nodes() << " ops, "
                  << "decode graph " << decode_graph->num_nodes() << " ops" << std::endl;

        vvllm::ir::Executable prefill_exe(prefill_graph.get(), weight_map, backend, config);
        vvllm::ir::Executable decode_exe(decode_graph.get(), weight_map, backend, config);

        std::cout << std::endl;

        // Prefill
        stats.begin_prefill();
        logits = prefill_exe.run(token_ids, 0, *kv_cache);
        stats.end_prefill(token_ids.size());

        // Decode
        std::cout << FLAGS_prompt << std::flush;

        for (int step = 0; step < FLAGS_max_tokens; step++)
        {
            int next_token = sampler->sample(logits, step);
            if (is_eos(next_token)) break;

            std::cout << tokenizer.decode({next_token}) << std::flush;
            token_ids.push_back(next_token);

            stats.begin_step();
            std::size_t pos = token_ids.size() - 1;
            logits = decode_exe.run({next_token}, pos, *kv_cache);
            stats.end_step();
        }
    }
    else
    {
        // ===== Interpreted execution path (original) =====
        std::cout << "Mode: interpreted" << std::endl;

        auto model = vvllm::create_model(config, backend);
        vvllm::load_weights(model, weights, quantize);
        std::cout << "Model initialized";
        if (quantize)
        {
            std::cout << " (int8 quantized)";
        }
        std::cout << std::endl << std::endl;

        // Prefill
        stats.begin_prefill();
        logits = vvllm::forward(model, token_ids, 0, *kv_cache);
        stats.end_prefill(token_ids.size());

        // Decode
        std::cout << FLAGS_prompt << std::flush;

        for (int step = 0; step < FLAGS_max_tokens; step++)
        {
            int next_token = sampler->sample(logits, step);
            if (is_eos(next_token)) break;

            std::cout << tokenizer.decode({next_token}) << std::flush;
            token_ids.push_back(next_token);

            stats.begin_step();
            std::size_t pos = token_ids.size() - 1;
            logits = vvllm::forward(model, {next_token}, pos, *kv_cache);
            stats.end_step();
        }
    }

    std::cout << "\n\n";
    stats.print(quantize);

    return 0;
}
