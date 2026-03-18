#pragma once

#include <chrono>
#include <cstddef>
#include <vector>

namespace vvllm
{

class Stats
{
public:
    using Clock = std::chrono::steady_clock;

    void begin_prefill();
    void end_prefill(std::size_t prompt_tokens);

    void begin_step();
    void end_step();

    void print(bool quantize) const;

private:
    static double to_ms(Clock::duration d);

    Clock::time_point t_prefill_start_;
    Clock::time_point t_step_start_;
    double prefill_ms_ = 0;
    std::size_t prompt_tokens_ = 0;
    int generated_ = 0;
    std::vector<double> decode_latencies_ms_;
};

}  // namespace vvllm
