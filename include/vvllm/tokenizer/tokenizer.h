#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace vvllm
{

class Tokenizer
{
public:
    explicit Tokenizer(std::string filepath);
    ~Tokenizer();

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    /// Parse tokenizer.json and build vocabulary + merge rules.
    void load();

    /// Encode text into token IDs.
    std::vector<int> encode(const std::string& text) const;

    /// Decode token IDs back into text.
    std::string decode(const std::vector<int>& token_ids) const;

    /// Look up the ID for a token string. Throws if not found.
    int token_id(const std::string& token) const;

    /// Get vocabulary size.
    std::size_t vocab_size() const;

private:
    std::string filepath_;
    bool loaded_;

    /// Token string → token ID
    std::unordered_map<std::string, int> token_to_id_;

    /// Token ID → token string
    std::unordered_map<int, std::string> id_to_token_;

    /// BPE merge rules in priority order.
    /// Each pair is (first, second) → merged token.
    std::vector<std::pair<std::string, std::string>> merges_;

    /// Fast lookup: "first second" → merge rank (index in merges_)
    std::unordered_map<std::string, int> merge_ranks_;
};

}  // namespace vvllm
