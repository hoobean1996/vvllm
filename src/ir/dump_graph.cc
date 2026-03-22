#include <iostream>
#include <string>

#include "gflags/gflags.h"
#include "vvllm/config/config.h"
#include "vvllm/ir/graph_builder.h"

DEFINE_string(model, "", "Path to model directory (needs config.json)");
DEFINE_uint64(seq_len, 1, "Sequence length (1 for decode, >1 for prefill)");

int main(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_model.empty())
    {
        std::cerr << "Usage: --model <path to model directory>" << std::endl;
        return 1;
    }

    auto config = vvllm::load_config(FLAGS_model + "/config.json");

    bool has_qkv_bias = (config.model_type == "qwen2");
    auto mode = (FLAGS_seq_len > 1) ? vvllm::ir::GraphMode::Prefill
                                    : vvllm::ir::GraphMode::Decode;

    vvllm::ir::GraphBuildOptions opts;
    opts.mode = mode;
    opts.seq_len = FLAGS_seq_len;
    opts.has_qkv_bias = has_qkv_bias;

    auto graph = vvllm::ir::build_transformer_graph(config, opts);

    std::cout << graph->dump();
    std::cout << "// Total: " << graph->num_nodes() << " ops, "
              << graph->num_values() << " values\n";

    return 0;
}
