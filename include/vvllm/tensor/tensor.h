#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace vvllm
{

enum class Device
{
    CPU,
    CUDA
};

enum class DType
{
    Float32,
    Float16,
    Int8
};

inline std::size_t dtype_size(DType dt)
{
    switch (dt)
    {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::Int8: return 1;
    }
    return 0;
}

class Tensor
{
public:
    Tensor();
    Tensor(std::vector<std::size_t> shape, DType dtype, Device device);
    Tensor(std::initializer_list<std::size_t> shape, DType dtype, Device device);
    ~Tensor();

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    template <typename T>
    T* data()
    {
        return static_cast<T*>(data_);
    }
    template <typename T>
    const T* data() const
    {
        return static_cast<const T*>(data_);
    }

    void* raw_data() { return data_; }
    const void* raw_data() const { return data_; }

    const std::vector<std::size_t>& shape() const { return shape_; }
    std::size_t size() const { return size_; }
    std::size_t bytes() const { return size_ * dtype_size(dtype_); }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    bool empty() const { return data_ == nullptr; }

    /// Deep copy to target device.
    Tensor to(Device target) const;

    /// Non-owning view into this tensor at a byte offset.
    Tensor view(std::size_t byte_offset, std::vector<std::size_t> new_shape) const;

private:
    // Private constructor for views (non-owning)
    Tensor(void* data, std::vector<std::size_t> shape, std::size_t size, DType dtype,
           Device device);

    void* data_ = nullptr;
    std::vector<std::size_t> shape_;
    std::size_t size_ = 0;
    DType dtype_ = DType::Float32;
    Device device_ = Device::CPU;
    bool owning_ = true;
};

// Device memory management (implemented in tensor.cc, CUDA via cuda_kernels.h)
void* device_allocate(Device dev, std::size_t bytes);
void device_deallocate(Device dev, void* ptr);
void device_copy(void* dst, Device dst_dev, const void* src, Device src_dev, std::size_t bytes);

}  // namespace vvllm
