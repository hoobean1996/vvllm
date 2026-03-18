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

    mapped_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
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
        if (name == "__metadata__") continue;
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

Tensor SafeTensorsLoader::load_tensor(const std::string& name)
{
    if (!parsed_)
    {
        throw std::runtime_error("must call parse() before load_tensor()");
    }
    auto& info = tensor_infos_.at(name);

    const uint8_t* tensor_bytes = data_start_ + info.offset_begin;
    size_t num_bytes = info.offset_end - info.offset_begin;

    Tensor tensor(info.shape, DType::Float32, Device::CPU);
    if (info.dtype == "F32")
    {
        std::memcpy(tensor.data<float>(), tensor_bytes, num_bytes);
    }
    else if (info.dtype == "BF16")
    {
        auto* src = reinterpret_cast<const uint16_t*>(tensor_bytes);
        float* dst = tensor.data<float>();
        for (size_t i = 0; i < tensor.size(); i++)
        {
            dst[i] = bf16_to_f32(src[i]);
        }
    }
    else
    {
        throw std::runtime_error("failed to load tensor, unsupported data type.");
    }
    return tensor;
}

std::unordered_map<std::string, Tensor> SafeTensorsLoader::load_all()
{
    if (!parsed_)
    {
        throw std::runtime_error("must call parse before call load_all");
    }
    std::unordered_map<std::string, Tensor> tensors;
    for (auto& [name, info] : tensor_infos_)
    {
        tensors.emplace(name, load_tensor(name));
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
