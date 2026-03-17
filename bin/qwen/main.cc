#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "src/backend/backend_blas.h"
#include "src/backend/backend_cuda.h"
#include "src/backend/backend_naive.h"
#include "vvllm/config/config.h"
#include "vvllm/model/kv_cache.h"
#include "vvllm/model/model.h"
#include "vvllm/safetensors/safetensors.h"
#include "vvllm/sampler/sampler.h"
#include "vvllm/tokenizer/tokenizer.h"

DEFINE_string(model, "", "Path to model directory");
DEFINE_string(prompt, "", "Input prompt");
DEFINE_int32(max_tokens, 50, "Maximum number of tokens to generate");
DEFINE_double(temperature, 0.7, "Sampling temperature (0 = greedy)");
DEFINE_double(top_p, 0.9, "Top-p (nucleus) sampling threshold");
DEFINE_uint64(seed, 42, "Random seed for sampling");
DEFINE_bool(kv_cache, true, "Enable KV cache (disable to recompute full sequence each step)");
DEFINE_string(backend, "cpu", "Backend to use: cpu, blas, cuda");
DEFINE_string(quantize, "", "Weight quantization: int8 (empty for none)");
DEFINE_int32(best_of, 1, "Best-of-N sampling: generate N candidates, output the best");

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
        backend_ptr = std::make_unique<vvllm::BackendCUDA>();
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
    auto weights = loader.load_all(backend);
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

    // Create model
    auto model = vvllm::create_model(config, backend);
    vvllm::load_weights(model, weights, quantize);
    std::cout << "Model initialized";
    if (quantize)
    {
        std::cout << " (int8 quantized)";
    }
    std::cout << std::endl;

    // Tokenize prompt
    std::vector<int> token_ids = tokenizer.encode(FLAGS_prompt);
    std::cout << "Prompt: \"" << FLAGS_prompt << "\" (" << token_ids.size() << " tokens)"
              << std::endl;
    std::cout << std::endl;

    // EOS check helper
    auto is_eos = [&config](int token) {
        for (int eos : config.eos_token_ids)
        {
            if (token == eos) return true;
        }
        return false;
    };

    // Create sampler
    vvllm::Sampler sampler(static_cast<float>(FLAGS_temperature), static_cast<float>(FLAGS_top_p),
                           FLAGS_seed);

    // Compute log-probability of a token from raw logits (length-normalized scoring)
    auto log_prob = [](const std::vector<float>& logits, int token) -> float {
        float mx = *std::max_element(logits.begin(), logits.end());
        float sum = 0;
        for (float l : logits) sum += std::exp(l - mx);
        return (logits[token] - mx) - std::log(sum);
    };

    using Clock = std::chrono::steady_clock;
    auto to_ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };

    vvllm::KVCache kv_cache(config.num_hidden_layers);
    std::size_t prompt_tokens = token_ids.size();

    // Prefill: process the full prompt, populating the KV cache
    auto t_prefill_start = Clock::now();
    auto prefill_logits = vvllm::forward(model, token_ids, 0, kv_cache);
    auto t_prefill_end = Clock::now();

    int best_of = FLAGS_best_of;
    if (best_of < 1) best_of = 1;

    if (best_of > 1 && FLAGS_temperature == 0.0)
    {
        std::cerr << "Warning: --temperature 0 with --best_of " << best_of
                  << " will produce identical candidates. Consider setting temperature > 0."
                  << std::endl;
    }

    // ============================================================
    // best_of == 1: original streaming path (unchanged)
    // ============================================================
    if (best_of == 1)
    {
        std::cout << FLAGS_prompt << std::flush;
        auto logits = prefill_logits;

        int generated = 0;
        std::vector<double> token_latencies_ms;

        for (int step = 0; step < FLAGS_max_tokens; step++)
        {
            int next_token = sampler.sample(logits);

            if (is_eos(next_token)) break;

            std::string text = tokenizer.decode({next_token});
            std::cout << text << std::flush;

            token_ids.push_back(next_token);
            generated++;

            auto t_step_start = Clock::now();

            if (FLAGS_kv_cache)
            {
                std::size_t pos = token_ids.size() - 1;
                logits = vvllm::forward(model, {next_token}, pos, kv_cache);
            }
            else
            {
                kv_cache.reset();
                logits = vvllm::forward(model, token_ids, 0, kv_cache);
            }

            token_latencies_ms.push_back(to_ms(Clock::now() - t_step_start));
        }

        std::cout << "\n\n";

        double prefill_ms = to_ms(t_prefill_end - t_prefill_start);
        double total_decode_ms =
            std::accumulate(token_latencies_ms.begin(), token_latencies_ms.end(), 0.0);

        std::printf("=== Performance ===\n");
        std::printf("KV cache:    %s\n", FLAGS_kv_cache ? "enabled" : "disabled");
        std::printf("Quantize:    %s\n", quantize ? "int8" : "none");
        std::printf("Prompt:      %zu tokens\n", prompt_tokens);
        std::printf("Generated:   %d tokens\n\n", generated);

        std::printf("Prefill\n");
        std::printf("  Total:       %8.1f ms\n", prefill_ms);
        std::printf("  Throughput:  %8.2f tok/s\n", prompt_tokens / (prefill_ms / 1000.0));
        std::printf("  Per token:   %8.1f ms\n\n", prefill_ms / prompt_tokens);

        if (!token_latencies_ms.empty())
        {
            std::sort(token_latencies_ms.begin(), token_latencies_ms.end());
            double min_lat = token_latencies_ms.front();
            double max_lat = token_latencies_ms.back();
            double avg = total_decode_ms / generated;
            auto percentile = [&](double p) {
                std::size_t idx = std::min(
                    static_cast<std::size_t>(p / 100.0 * generated),
                    token_latencies_ms.size() - 1);
                return token_latencies_ms[idx];
            };

            std::printf("Decode\n");
            std::printf("  Total:       %8.1f ms\n", total_decode_ms);
            std::printf("  Throughput:  %8.2f tok/s\n\n",
                        generated / (total_decode_ms / 1000.0));

            std::printf("  Latency per token (ms)\n");
            std::printf("    Min:       %8.1f\n", min_lat);
            std::printf("    Avg:       %8.1f\n", avg);
            std::printf("    P50:       %8.1f\n", percentile(50));
            std::printf("    P90:       %8.1f\n", percentile(90));
            std::printf("    P99:       %8.1f\n", percentile(99));
            std::printf("    Max:       %8.1f\n", max_lat);
        }
    }
    // ============================================================
    // best_of > 1: generate N candidates, pick the best by log-prob
    // ============================================================
    else
    {
        std::size_t prefill_seq_len = token_ids.size();

        struct Candidate
        {
            std::vector<int> tokens;
            float score;
        };
        std::vector<Candidate> candidates;

        auto t_decode_start = Clock::now();

        for (int ci = 0; ci < best_of; ci++)
        {
            // Rewind KV cache and token_ids to prefill state
            kv_cache.truncate(prefill_seq_len);
            backend.kv_cache_truncate(prefill_seq_len);
            token_ids.resize(prefill_seq_len);

            auto logits = prefill_logits;
            std::vector<int> candidate_tokens;
            float total_log_prob = 0;

            for (int step = 0; step < FLAGS_max_tokens; step++)
            {
                int next_token = sampler.sample(logits);

                total_log_prob += log_prob(logits, next_token);

                if (is_eos(next_token)) break;

                candidate_tokens.push_back(next_token);
                token_ids.push_back(next_token);

                if (FLAGS_kv_cache)
                {
                    std::size_t pos = token_ids.size() - 1;
                    logits = vvllm::forward(model, {next_token}, pos, kv_cache);
                }
                else
                {
                    kv_cache.reset();
                    logits = vvllm::forward(model, token_ids, 0, kv_cache);
                }
            }

            float score = candidate_tokens.empty()
                              ? -std::numeric_limits<float>::infinity()
                              : total_log_prob / static_cast<float>(candidate_tokens.size());

            candidates.push_back({std::move(candidate_tokens), score});
        }

        auto t_decode_end = Clock::now();

        // Find best candidate
        int best_idx = 0;
        for (int i = 1; i < static_cast<int>(candidates.size()); i++)
        {
            if (candidates[i].score > candidates[best_idx].score)
            {
                best_idx = i;
            }
        }

        // Print all candidates with scores
        std::printf("=== Best-of-%d Candidates ===\n\n", best_of);
        for (int i = 0; i < static_cast<int>(candidates.size()); i++)
        {
            std::string text = tokenizer.decode(candidates[i].tokens);
            std::printf("Candidate %d (score: %.4f, %zu tokens)%s:\n", i + 1,
                        candidates[i].score, candidates[i].tokens.size(),
                        i == best_idx ? " [BEST]" : "");
            std::cout << "  " << FLAGS_prompt << text << "\n\n";
        }

        // Performance stats
        double prefill_ms = to_ms(t_prefill_end - t_prefill_start);
        double decode_ms = to_ms(t_decode_end - t_decode_start);
        int total_tokens = 0;
        for (const auto& c : candidates) total_tokens += static_cast<int>(c.tokens.size());

        std::printf("=== Performance ===\n");
        std::printf("KV cache:    %s\n", FLAGS_kv_cache ? "enabled" : "disabled");
        std::printf("Quantize:    %s\n", quantize ? "int8" : "none");
        std::printf("Prompt:      %zu tokens\n", prompt_tokens);
        std::printf("Candidates:  %d\n", best_of);
        std::printf("Total generated: %d tokens\n\n", total_tokens);

        std::printf("Prefill\n");
        std::printf("  Total:       %8.1f ms\n", prefill_ms);
        std::printf("  Throughput:  %8.2f tok/s\n\n", prompt_tokens / (prefill_ms / 1000.0));

        std::printf("Decode (all candidates)\n");
        std::printf("  Total:       %8.1f ms\n", decode_ms);
        std::printf("  Throughput:  %8.2f tok/s\n", total_tokens / (decode_ms / 1000.0));
    }

    return 0;
}
