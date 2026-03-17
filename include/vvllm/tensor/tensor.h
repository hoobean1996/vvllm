#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

#include "vvllm/backend/backend.h"

namespace vvllm
{

template <typename T>
class Tensor
{
public:
    Tensor(Backend& backend, std::initializer_list<std::size_t> shape);
    Tensor(Backend& backend, std::vector<std::size_t> shape);
    ~Tensor();

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    T* data();
    const T* data() const;

    const std::vector<std::size_t>& shape() const;
    std::size_t size() const;
    std::size_t bytes() const;

private:
    Backend* backend_;
    // data_ points to the allocated memory from the Backend.
    T* data_;
    std::vector<std::size_t> shape_;
    // size_t is the total elements of our Tensor
    std::size_t size_;
};

}  // namespace vvllm
