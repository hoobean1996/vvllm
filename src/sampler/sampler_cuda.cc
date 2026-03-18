#include "src/sampler/sampler_cuda.h"

#include "src/backend/cuda_kernels.h"

namespace vvllm
{

SamplerCUDA::SamplerCUDA(float temperature, float top_p, std::uint64_t seed)
    : temperature_(temperature), top_p_(top_p), seed_(seed)
{
}

int SamplerCUDA::sample(const Tensor& logits, int step)
{
    // Logits are FP32 on GPU — sample directly, no CPU roundtrip
    return cuda_sample(logits.raw_data(), logits.size(), temperature_, top_p_, seed_, step,
                       /*fp16=*/0);
}

}  // namespace vvllm
