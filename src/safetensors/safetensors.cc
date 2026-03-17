#include <fcntl.h>     // open
#include <sys/mman.h>  // mmap, munmap
#include <sys/stat.h>  // fstat
#include <unistd.h>    // close

#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

#include "vvllm/safetensors/safetensors.h"

namespace vvllm
{

SafeTensorsLoader::SafeTensorsLoader(std::string filepath)
    : filepath_(std::move(filepath)),
      header_size_(0),
      data_offset_(0),
      parsed_(false),
      fd_(-1),
      mapped_(nullptr),
      file_size_(0),
      data_start_(nullptr)
{
}

SafeTensorsLoader::~SafeTensorsLoader()
{
    if (mapped_ && mapped_ != MAP_FAILED)
    {
        munmap(mapped_, file_size_);
    }
    if (fd_ >= 0)
    {
        close(fd_);
    }
}

/// SafeTensor Format
/// 8 bytes header_size
/// JSON header
/// raw tensor data
void SafeTensorsLoader::parse()
{
    fd_ = open(filepath_.c_str(), O_RDONLY);
    if (fd_ < 0)
    {
        throw std::runtime_error("failed to open file: " + filepath_);
    }
    struct stat st;
    fstat(fd_, &st);
    file_size_ = st.st_size;

    mapped_ = mmap(nullptr,  // address decided by OS
                   file_size_,
                   PROT_READ,    // read_only
                   MAP_PRIVATE,  // private mapping
                   fd_, 0);
    if (mapped_ == MAP_FAILED)
    {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("failed to mmap file: " + filepath_);
    }
    auto* base = static_cast<const uint8_t*>(mapped_);

    uint64_t header_size = *reinterpret_cast<const uint64_t*>(base);

    const char* header_json_base = reinterpret_cast<const char*>(base + 8);
    std::string header_json_str(header_json_base, header_size);
    auto json = nlohmann::json::parse(header_json_str);

    for (auto& [name, value] : json.items())
    {
        if (name == "__metadata__")
        {
            continue;
        }
        TensorInfo info;
        info.name = name;
        info.dtype = value["dtype"].get<std::string>();
        info.shape = value["shape"].get<std::vector<std::size_t>>();
        info.offset_begin = value["data_offsets"][0].get<std::size_t>();
        info.offset_end = value["data_offsets"][1].get<std::size_t>();
        tensor_infos_.emplace(name, std::move(info));
    }

    header_size_ = header_size;
    data_offset_ = 8 + header_size;
    data_start_ = base + data_offset_;
    parsed_ = true;
}

const std::unordered_map<std::string, TensorInfo>& SafeTensorsLoader::tensor_infos() const
{
    return tensor_infos_;
}

Tensor<float> SafeTensorsLoader::load_tensor(const std::string& name, Backend& backend)
{
    if (!parsed_)
    {
        throw std::runtime_error("must call parse() before load_tensor()");
    }
    // [] operator is non-const, it would automatically insert when the name is not existed.
    // update to .at
    auto& info = tensor_infos_.at(name);

    const uint8_t* tensor_bytes = data_start_ + info.offset_begin;
    size_t num_bytes = info.offset_end - info.offset_begin;

    Tensor<float> tensor(backend, info.shape);
    if (info.dtype == "F32")
    {
        std::memcpy(tensor.data(), tensor_bytes, num_bytes);
    }
    else if (info.dtype == "BF16")
    {
        auto* src = reinterpret_cast<const uint16_t*>(tensor_bytes);
        for (size_t i = 0; i < tensor.size(); i++)
        {
            tensor.data()[i] = bf16_to_f32(src[i]);
        }
    }
    else
    {
        throw std::runtime_error("failed to load tensor, unsupported data type.");
    }
    return tensor;
}

std::unordered_map<std::string, Tensor<float>> SafeTensorsLoader::load_all(Backend& backend)
{
    if (!parsed_)
    {
        throw std::runtime_error("must call parse before call load_all");
    }
    std::unordered_map<std::string, Tensor<float>> tensors;
    for (auto& [name, info] : tensor_infos_)
    {
        tensors.emplace(name, load_tensor(name, backend));
    }
    return tensors;
}

float SafeTensorsLoader::bf16_to_f32(uint16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

}  // namespace vvllm
