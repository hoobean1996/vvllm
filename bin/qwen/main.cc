#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "src/backend/backend_blas.h"
#include "src/backend/backend_cuda.h"
#include "src/backend/backend_naive.h"
#include "vvllm/config/config.h"
#include "vvllm/model/model.h"
#include "vvllm/safetensors/safetensors.h"
#include "vvllm/sampler/sampler.h"
#include "vvllm/stats/stats.h"
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

    vvllm::Stats stats;

    // Prefill: process the full prompt, populating the KV cache
    stats.begin_prefill();
    auto logits = vvllm::forward(model, token_ids, 0);
    stats.end_prefill(token_ids.size());

    // Decode: stream tokens one at a time
    std::cout << FLAGS_prompt << std::flush;

    for (int step = 0; step < FLAGS_max_tokens; step++)
    {
        int next_token = sampler.sample(logits);

        if (is_eos(next_token)) break;

        std::cout << tokenizer.decode({next_token}) << std::flush;
        token_ids.push_back(next_token);

        stats.begin_step();

        if (FLAGS_kv_cache)
        {
            std::size_t pos = token_ids.size() - 1;
            logits = vvllm::forward(model, {next_token}, pos);
        }
        else
        {
            logits = vvllm::forward(model, token_ids, 0);
        }

        stats.end_step();
    }

    std::cout << "\n\n";
    stats.print(FLAGS_kv_cache, quantize);

    return 0;
}
