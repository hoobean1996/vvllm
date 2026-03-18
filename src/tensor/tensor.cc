#include "vvllm/tensor/tensor.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

// CUDA memory functions (defined in cuda_kernels.cu, linked when CUDA is available)
extern "C" {
void* cuda_malloc(size_t bytes);
void cuda_free(void* ptr);
void cuda_memcpy_h2d(void* dst, const void* src, size_t bytes);
void cuda_memcpy_d2h(void* dst, const void* src, size_t bytes);
void cuda_memcpy_d2d(void* dst, const void* src, size_t bytes);
}

namespace vvllm
{

// ============================================================
// CUDA caching allocator
// ============================================================
//
// nsys profiling revealed cudaMalloc + cudaFree consumed 44% of CUDA API time
// (266ms out of 600ms) because each call synchronizes the GPU. The forward pass
// creates ~12 temporary Tensors per call, and the sizes repeat every decode step.
//
// Fix: cache freed GPU buffers in a free list keyed by byte size. Same-size
// allocations are O(1) map lookups instead of device-synchronizing cudaMalloc.
// Buffers are only truly freed at process exit.

// Free list: byte_size → [ptr, ptr, ...] (multiple same-size buffers possible)
static std::multimap<std::size_t, void*> cuda_free_list_;

// Live allocations: ptr → byte_size (needed to return buffer to correct bucket)
static std::unordered_map<void*, std::size_t> cuda_alloc_sizes_;

static void* cuda_pool_alloc(std::size_t bytes)
{
    auto it = cuda_free_list_.find(bytes);
    if (it != cuda_free_list_.end())
    {
        void* ptr = it->second;
        cuda_free_list_.erase(it);
        cuda_alloc_sizes_.emplace(ptr, bytes);
        return ptr;
    }
    void* ptr = cuda_malloc(bytes);
    cuda_alloc_sizes_.emplace(ptr, bytes);
    return ptr;
}

static void cuda_pool_free(void* ptr)
{
    auto it = cuda_alloc_sizes_.find(ptr);
    if (it != cuda_alloc_sizes_.end())
    {
        cuda_free_list_.emplace(it->second, ptr);
        cuda_alloc_sizes_.erase(it);
    }
    else
    {
        // Not from our pool (e.g. from BackendCUDA weight cache) — real free
        cuda_free(ptr);
    }
}

// --- Device memory management ---

void* device_allocate(Device dev, std::size_t bytes)
{
    if (bytes == 0) return nullptr;
    if (dev == Device::CUDA)
    {
        return cuda_pool_alloc(bytes);
    }
    void* p = std::malloc(bytes);
    if (!p) throw std::bad_alloc();
    return p;
}

void device_deallocate(Device dev, void* ptr)
{
    if (!ptr) return;
    if (dev == Device::CUDA)
    {
        cuda_pool_free(ptr);
    }
    else
    {
        std::free(ptr);
    }
}

void device_copy(void* dst, Device dst_dev, const void* src, Device src_dev, std::size_t bytes)
{
    if (bytes == 0) return;
    if (src_dev == Device::CPU && dst_dev == Device::CPU)
    {
        std::memcpy(dst, src, bytes);
    }
    else if (src_dev == Device::CPU && dst_dev == Device::CUDA)
    {
        cuda_memcpy_h2d(dst, src, bytes);
    }
    else if (src_dev == Device::CUDA && dst_dev == Device::CPU)
    {
        cuda_memcpy_d2h(dst, src, bytes);
    }
    else
    {
        cuda_memcpy_d2d(dst, src, bytes);
    }
}

// --- Tensor ---

static std::size_t compute_size(const std::vector<std::size_t>& shape)
{
    if (shape.empty()) return 0;
    std::size_t s = 1;
    for (auto d : shape) s *= d;
    return s;
}

Tensor::Tensor() = default;

Tensor::Tensor(std::vector<std::size_t> shape, DType dtype, Device device)
    : shape_(std::move(shape)), dtype_(dtype), device_(device), owning_(true)
{
    size_ = compute_size(shape_);
    data_ = device_allocate(device_, bytes());
}

Tensor::Tensor(std::initializer_list<std::size_t> shape, DType dtype, Device device)
    : Tensor(std::vector<std::size_t>(shape), dtype, device)
{
}

Tensor::~Tensor()
{
    if (owning_ && data_)
    {
        device_deallocate(device_, data_);
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      size_(other.size_),
      dtype_(other.dtype_),
      device_(other.device_),
      owning_(other.owning_)
{
    other.data_ = nullptr;
    other.size_ = 0;
    other.owning_ = true;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this != &other)
    {
        if (owning_ && data_)
        {
            device_deallocate(device_, data_);
        }
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        owning_ = other.owning_;

        other.data_ = nullptr;
        other.size_ = 0;
        other.owning_ = true;
    }
    return *this;
}

Tensor Tensor::to(Device target) const
{
    if (target == device_) return Tensor(); // no-op, caller should check
    Tensor result(std::vector<std::size_t>(shape_), dtype_, target);
    device_copy(result.data_, target, data_, device_, bytes());
    return result;
}

Tensor Tensor::view(std::size_t byte_offset, std::vector<std::size_t> new_shape) const
{
    return Tensor(static_cast<char*>(data_) + byte_offset, std::move(new_shape),
                  compute_size(new_shape), dtype_, device_);
}

// Private view constructor
Tensor::Tensor(void* data, std::vector<std::size_t> shape, std::size_t size, DType dtype,
               Device device)
    : data_(data), shape_(std::move(shape)), size_(size), dtype_(dtype), device_(device),
      owning_(false)
{
}

}  // namespace vvllm
