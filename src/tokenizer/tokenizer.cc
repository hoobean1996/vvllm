#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "vvllm/tokenizer/tokenizer.h"

namespace vvllm
{

Tokenizer::Tokenizer(std::string filepath) : filepath_(std::move(filepath)), loaded_(false) {}

Tokenizer::~Tokenizer() = default;

void Tokenizer::load()
{
    std::ifstream file(filepath_);
    if (!file.is_open())
    {
        throw std::runtime_error("failed to open tokenizer: " + filepath_);
    }
    auto json = nlohmann::json::parse(file);
    auto& vocab = json["model"]["vocab"];
    for (auto& [token, id] : vocab.items())
    {
        int token_id = id.get<int>();
        token_to_id_.emplace(token, token_id);
        id_to_token_.emplace(token_id, token);
    }

    for (auto& merge : json["model"]["merges"])
    {
        std::string merge_str = merge.get<std::string>();
        size_t space = merge_str.find(' ');
        std::string first = merge_str.substr(0, space);
        std::string second = merge_str.substr(space + 1);
        int rank = static_cast<int>(merges_.size());
        merges_.emplace_back(first, second);
        merge_ranks_.emplace(merge_str, rank);
    }

    for (auto& added : json["added_tokens"])
    {
        std::string token = added["content"].get<std::string>();
        int id = added["id"].get<int>();
        token_to_id_.emplace(token, id);
        id_to_token_.emplace(id, token);
    }

    loaded_ = true;
}

std::vector<int> Tokenizer::encode(const std::string& text) const
{
    // Step 1: Pre-tokenize — split by whitespace, prefix with Ġ for spaces
    std::vector<std::string> words;
    std::string current;
    for (size_t i = 0; i < text.size(); i++)
    {
        char c = text[i];
        if (c == ' ')
        {
            if (!current.empty())
            {
                words.push_back(current);
            }
            current = "\xC4\xA0";  // Ġ in UTF-8
        }
        else if (ispunct(static_cast<unsigned char>(c)))
        {
            if (!current.empty())
            {
                words.push_back(current);
                current.clear();
            }
            words.push_back(std::string(1, c));
        }
        else
        {
            current += c;
        }
    }
    if (!current.empty())
    {
        words.push_back(current);
    }

    std::vector<int> token_ids;

    for (auto& word : words)
    {
        // Step 2: Split word into UTF-8 characters
        std::vector<std::string> tokens;
        size_t i = 0;
        while (i < word.size())
        {
            auto c = static_cast<unsigned char>(word[i]);
            size_t char_len = 1;
            if ((c & 0x80) == 0)
                char_len = 1;
            else if ((c & 0xE0) == 0xC0)
                char_len = 2;
            else if ((c & 0xF0) == 0xE0)
                char_len = 3;
            else if ((c & 0xF8) == 0xF0)
                char_len = 4;
            tokens.push_back(word.substr(i, char_len));
            i += char_len;
        }

        // Step 3: Apply BPE merges
        while (tokens.size() > 1)
        {
            // Find the pair with the lowest merge rank
            int best_rank = -1;
            size_t best_idx = 0;
            for (size_t j = 0; j < tokens.size() - 1; j++)
            {
                std::string key = tokens[j] + " " + tokens[j + 1];
                auto it = merge_ranks_.find(key);
                if (it != merge_ranks_.end())
                {
                    if (best_rank < 0 || it->second < best_rank)
                    {
                        best_rank = it->second;
                        best_idx = j;
                    }
                }
            }

            if (best_rank < 0) break;  // no more merges

            // Merge the best pair
            tokens[best_idx] = tokens[best_idx] + tokens[best_idx + 1];
            tokens.erase(tokens.begin() + best_idx + 1);
        }

        // Step 4: Look up token IDs
        for (auto& token : tokens)
        {
            auto it = token_to_id_.find(token);
            if (it != token_to_id_.end())
            {
                token_ids.push_back(it->second);
            }
        }
    }

    return token_ids;
}

int Tokenizer::token_id(const std::string& token) const
{
    auto it = token_to_id_.find(token);
    if (it == token_to_id_.end())
    {
        throw std::runtime_error("unknown token: " + token);
    }
    return it->second;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const
{
    std::string result;
    for (int id : token_ids)
    {
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end())
        {
            result += it->second;
        }
    }

    // Convert BPE byte tokens back to raw bytes.
    // BPE encodes byte N as chr(256 + N). In UTF-8:
    //   chr(256..319)  = 0xC4 0x80..0xBF  (byte 0x00..0x3F)
    //   chr(320..383)  = 0xC5 0x80..0xBF  (byte 0x40..0x7F)
    //   chr(384..511)  = 0xC6..0xC7 ...   (byte 0x80..0xFF)
    // Common examples: Ġ = chr(288) → space, Ċ = chr(266) → newline
    std::string output;
    size_t i = 0;
    while (i < result.size())
    {
        auto c0 = static_cast<unsigned char>(result[i]);
        if (i + 1 < result.size() && (c0 >= 0xC4 && c0 <= 0xC7))
        {
            auto c1 = static_cast<unsigned char>(result[i + 1]);
            if ((c1 & 0xC0) == 0x80)  // valid 2-byte UTF-8 continuation
            {
                // Decode UTF-8 codepoint
                unsigned int cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
                if (cp >= 256 && cp < 512)
                {
                    // BPE byte token: chr(256 + N) → raw byte N
                    output += static_cast<char>(cp - 256);
                    i += 2;
                    continue;
                }
            }
        }
        output += result[i];
        i++;
    }

    return output;
}

std::size_t Tokenizer::vocab_size() const { return token_to_id_.size(); }

}  // namespace vvllm
