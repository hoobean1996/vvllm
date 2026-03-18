#include "vvllm/model/transformer.h"

namespace vvllm
{

void linear(float* out, const float* inp, const float* weight, const float* bias,
            std::size_t M, std::size_t out_dim, std::size_t in_dim, Backend& backend)
{
    backend.linear(out, inp, weight, bias, M, out_dim, in_dim);
}

void linear(float* out, const float* inp, const LinearWeight& weight, const float* bias,
            std::size_t M, std::size_t out_dim, std::size_t in_dim, Backend& backend)
{
    if (weight.is_quantized())
    {
        backend.linear_q8(out, inp, weight.int8, weight.scales, bias, M, out_dim, in_dim);
    }
    else
    {
        backend.linear(out, inp, weight.fp32, bias, M, out_dim, in_dim);
    }
}

void linear_add(float* out, const float* inp, const LinearWeight& weight, const float* bias,
                const float* residual, std::size_t M, std::size_t out_dim, std::size_t in_dim,
                Backend& backend)
{
    if (weight.is_quantized())
    {
        backend.linear_q8_add(out, inp, weight.int8, weight.scales, bias, residual, M, out_dim,
                              in_dim);
    }
    else
    {
        backend.linear_add(out, inp, weight.fp32, bias, residual, M, out_dim, in_dim);
    }
}

void rms_norm_seq(float* out, const float* x, const float* weight, std::size_t seq_len,
                  std::size_t hidden, float eps, Backend& backend, std::size_t elem_size)
{
    for (std::size_t s = 0; s < seq_len; s++)
    {
        std::size_t byte_offset = s * hidden * elem_size;
        auto* out_s = reinterpret_cast<float*>(reinterpret_cast<char*>(out) + byte_offset);
        auto* x_s = reinterpret_cast<const float*>(reinterpret_cast<const char*>(x) + byte_offset);
        backend.rms_norm(out_s, x_s, weight, hidden, eps);
    }
}

}  // namespace vvllm
