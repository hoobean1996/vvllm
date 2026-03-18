#include "src/sampler/sampler_cuda.h"

#include "src/backend/cuda_kernels.h"

namespace vvllm
{

SamplerCUDA::SamplerCUDA(float temperature, float top_p, std::uint64_t seed)
    : temperature_(temperature), top_p_(top_p), seed_(seed)
{
}

SamplerCUDA::~SamplerCUDA()
{
    if (d_logits_) cuda_free(d_logits_);
}

int SamplerCUDA::sample(const std::vector<float>& logits, int step)
{
    std::size_t bytes = logits.size() * sizeof(float);
    if (bytes > d_logits_bytes_)
    {
        if (d_logits_) cuda_free(d_logits_);
        d_logits_ = cuda_malloc(bytes);
        d_logits_bytes_ = bytes;
    }
    cuda_memcpy_h2d(d_logits_, logits.data(), bytes);

    // Logits are always FP32 (converted in transformer_forward before download)
    return cuda_sample(d_logits_, logits.size(), temperature_, top_p_, seed_, step, /*fp16=*/0);
}

}  // namespace vvllm
