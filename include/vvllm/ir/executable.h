#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vvllm/backend/backend.h"
#include "vvllm/config/config.h"
#include "vvllm/ir/graph.h"
#include "vvllm/kv_cache/kv_cache.h"
#include "vvllm/tensor/tensor.h"

namespace vvllm
{
namespace ir
{

/// A weight map resolves IR weight names to raw pointers + metadata.
struct WeightRef
{
    const float* fp32 = nullptr;
    const int8_t* int8 = nullptr;
    const float* scales = nullptr;
    bool quantized = false;
};

using WeightMap = std::unordered_map<std::string, WeightRef>;

/// Memory plan: maps each Value id to a pre-allocated buffer slot.
struct MemoryPlan
{
    std::unordered_map<uint32_t, uint32_t> slot_index;
    std::vector<std::size_t> slot_sizes;
    std::unordered_set<uint32_t> inplace_values;
    std::size_t num_slots() const { return slot_sizes.size(); }
};

/// Analyze value lifetimes and compute a buffer reuse plan.
MemoryPlan plan_memory(const Graph& graph, const ModelConfig& config, std::size_t seq_len);

/// Pre-compiled instruction — all pointers and parameters resolved at construction time.
/// No hash lookups, no variant gets, no string comparisons at run time.
struct ResolvedOp
{
    OpType op;

    // Buffer slot indices (into pool_). UINT32_MAX = unused.
    uint32_t out_slot;
    uint32_t out_slot2;     // RoPE second output
    uint32_t in_slots[4];   // up to 4 inputs
    uint8_t num_inputs;

    // Pre-resolved weight pointers
    const float* weight_fp32;
    const int8_t* weight_int8;
    const float* weight_scales;
    const float* bias;
    bool weight_quantized;

    // Pre-resolved residual slot (for LinearAdd)
    uint32_t residual_slot;

    // Op-specific params (pre-extracted, no Attributes lookup)
    std::size_t N;      // Linear/LinearAdd output dim
    std::size_t K;      // Linear/LinearAdd input dim
    float eps;          // RMSNorm
    std::size_t hidden; // RMSNorm hidden dim
    std::size_t nh;     // num_heads (RoPE, Attention)
    std::size_t nkv;    // num_kv_heads
    std::size_t hd;     // head_dim
    float theta;        // RoPE theta
    std::size_t layer;  // Attention layer index
    std::size_t intermediate; // SiLUMul
    bool is_logits;     // Linear: final logits projection
    bool rope_q_inplace;
    bool rope_k_inplace;
};

/// Compiled execution plan with pre-resolved instruction list.
class Executable
{
public:
    Executable(Graph* graph, const WeightMap& weights, Backend& backend,
               const ModelConfig& config);

    /// Run forward pass. Returns logits tensor (FP32).
    Tensor run(const std::vector<int>& token_ids, std::size_t pos, KVCache& kv_cache);

private:
    Graph* graph_;
    const WeightMap& weights_;
    Backend& backend_;
    const ModelConfig& config_;

    /// Memory plan and buffer pool.
    MemoryPlan plan_;
    mutable std::vector<Tensor> pool_;

    /// Pre-compiled instruction list (built once at construction).
    std::vector<ResolvedOp> ops_;

    /// Logits output slot index.
    uint32_t logits_slot_;

    /// Cached seq_len that ops_ and pool_ were compiled for.
    std::size_t compiled_seq_len_ = 0;

    /// Pre-allocated logits tensor (reused across runs, avoids alloc+copy).
    mutable Tensor logits_buf_;
    mutable Tensor logits_fp32_buf_;

    /// Build the pre-compiled instruction list.
    void compile(std::size_t seq_len);

    /// Allocate pool and compile instructions for a given seq_len.
    void allocate_pool(std::size_t seq_len);

    /// Direct buffer access via slot index (no hash lookup).
    float* slot_buf(uint32_t slot) const { return pool_[slot].data<float>(); }
};

}  // namespace ir
}  // namespace vvllm
