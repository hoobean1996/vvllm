#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "vvllm/backend/backend.h"
#include "vvllm/tensor/tensor.h"

namespace vvllm
{

struct TensorInfo
{
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;
    std::size_t offset_begin;
    std::size_t offset_end;
};

class SafeTensorsLoader
{
public:
    explicit SafeTensorsLoader(std::string filepath);
    ~SafeTensorsLoader();

    SafeTensorsLoader(const SafeTensorsLoader&) = delete;
    SafeTensorsLoader& operator=(const SafeTensorsLoader&) = delete;

    /// Parse the file header and populate tensor metadata.
    void parse();

    /// Get metadata for all tensors found in the file.
    const std::unordered_map<std::string, TensorInfo>& tensor_infos() const;

    /// Load a single tensor by name, converting to float.
    Tensor<float> load_tensor(const std::string& name, Backend& backend);

    /// Load all tensors, converting to float.
    std::unordered_map<std::string, Tensor<float>> load_all(Backend& backend);

private:
    std::string filepath_;
    std::size_t header_size_;
    std::size_t data_offset_;
    std::unordered_map<std::string, TensorInfo> tensor_infos_;
    bool parsed_;

    int fd_;
    void* mapped_;
    std::size_t file_size_;
    const uint8_t* data_start_;  // points to start of tensor data in mapped region

    /// Convert bfloat16 to float32.
    static float bf16_to_f32(uint16_t bf16);
};

}  // namespace vvllm
