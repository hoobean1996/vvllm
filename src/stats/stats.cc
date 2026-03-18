#include <algorithm>
#include <cstdio>
#include <numeric>

#include "vvllm/stats/stats.h"

namespace vvllm
{

void Stats::begin_prefill() { t_prefill_start_ = Clock::now(); }

void Stats::end_prefill(std::size_t prompt_tokens)
{
    prefill_ms_ = to_ms(Clock::now() - t_prefill_start_);
    prompt_tokens_ = prompt_tokens;
}

void Stats::begin_step() { t_step_start_ = Clock::now(); }

void Stats::end_step()
{
    decode_latencies_ms_.push_back(to_ms(Clock::now() - t_step_start_));
    generated_++;
}

void Stats::print(bool quantize) const
{
    std::printf("=== Performance ===\n");
    std::printf("Quantize:    %s\n", quantize ? "int8" : "none");
    std::printf("Prompt:      %zu tokens\n", prompt_tokens_);
    std::printf("Generated:   %d tokens\n\n", generated_);

    std::printf("Prefill\n");
    std::printf("  Total:       %8.1f ms\n", prefill_ms_);
    std::printf("  Throughput:  %8.2f tok/s\n", prompt_tokens_ / (prefill_ms_ / 1000.0));
    std::printf("  Per token:   %8.1f ms\n\n", prefill_ms_ / prompt_tokens_);

    if (decode_latencies_ms_.empty()) return;

    auto sorted = decode_latencies_ms_;
    std::sort(sorted.begin(), sorted.end());
    double total_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    double avg = total_ms / generated_;
    auto pct = [&](double p) {
        std::size_t idx =
            std::min(static_cast<std::size_t>(p / 100.0 * generated_), sorted.size() - 1);
        return sorted[idx];
    };

    std::printf("Decode\n");
    std::printf("  Total:       %8.1f ms\n", total_ms);
    std::printf("  Throughput:  %8.2f tok/s\n\n", generated_ / (total_ms / 1000.0));

    std::printf("  Latency per token (ms)\n");
    std::printf("    Min:       %8.1f\n", sorted.front());
    std::printf("    Avg:       %8.1f\n", avg);
    std::printf("    P50:       %8.1f\n", pct(50));
    std::printf("    P90:       %8.1f\n", pct(90));
    std::printf("    P99:       %8.1f\n", pct(99));
    std::printf("    Max:       %8.1f\n", sorted.back());
}

double Stats::to_ms(Clock::duration d)
{
    return std::chrono::duration<double, std::milli>(d).count();
}

}  // namespace vvllm
