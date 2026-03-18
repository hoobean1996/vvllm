#include "src/sampler/sampler_cuda.h"

#include "src/backend/cuda_kernels.h"

namespace vvllm
{

SamplerCUDA::SamplerCUDA(float temperature, float top_p, std::uint64_t seed, Backend& backend)
    : temperature_(temperature), top_p_(top_p), seed_(seed), backend_(backend)
{
}

int SamplerCUDA::sample(const std::vector<float>& logits, int step)
{
    const void* d_logits = backend_.device_ptr(logits.data());
    if (!d_logits) return -1;

    int fp16 = backend_.is_fp16() ? 1 : 0;
    return cuda_sample(d_logits, logits.size(), temperature_, top_p_, seed_, step, fp16);
}

}  // namespace vvllm
