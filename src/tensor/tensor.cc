#include "vvllm/tensor/tensor.h"

namespace vvllm
{

template <typename T>
Tensor<T>::Tensor(Backend& backend, std::initializer_list<std::size_t> shape)
    : backend_(&backend), data_(nullptr), shape_(shape), size_(1)
{
    for (auto dim : shape_)
    {
        size_ *= dim;
    }

    data_ = static_cast<T*>(backend_->allocate(bytes()));
}

template <typename T>
Tensor<T>::Tensor(Backend& backend, std::vector<std::size_t> shape)
    : backend_(&backend), data_(nullptr), shape_(std::move(shape)), size_(1)
{
    for (auto dim : shape_)
    {
        size_ *= dim;
    }

    data_ = static_cast<T*>(backend_->allocate(bytes()));
}

template <typename T>
Tensor<T>::~Tensor()
{
    if (data_)
    {
        backend_->deallocate(data_);
    }
}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : backend_(other.backend_),
      data_(other.data_),
      shape_(std::move(other.shape_)),
      size_(other.size_)
{
    other.data_ = nullptr;
    other.size_ = 0;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& other) noexcept
{
    if (this != &other)
    {
        if (data_)
        {
            backend_->deallocate(data_);
        }

        backend_ = other.backend_;
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        size_ = other.size_;

        other.data_ = nullptr;
        other.size_ = 0;
    }

    return *this;
}

template <typename T>
T* Tensor<T>::data()
{
    return data_;
}

template <typename T>
const T* Tensor<T>::data() const
{
    return data_;
}

template <typename T>
const std::vector<std::size_t>& Tensor<T>::shape() const
{
    return shape_;
}

template <typename T>
std::size_t Tensor<T>::size() const
{
    return size_;
}

template <typename T>
std::size_t Tensor<T>::bytes() const
{
    return size_ * sizeof(T);
}

template class Tensor<float>;
template class Tensor<double>;

}  // namespace vvllm
