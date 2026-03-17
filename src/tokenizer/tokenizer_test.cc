#include "gtest/gtest.h"
#include "vvllm/tokenizer/tokenizer.h"

namespace vvllm
{
namespace
{

// Path to real tokenizer — tests require the model to be downloaded
const char* TOKENIZER_PATH = "models/Qwen2.5-0.5B/tokenizer.json";

class TokenizerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        tokenizer_ = std::make_unique<Tokenizer>(TOKENIZER_PATH);
        tokenizer_->load();
    }

    std::unique_ptr<Tokenizer> tokenizer_;
};

TEST_F(TokenizerTest, LoadVocabSize)
{
    // 151643 base vocab + 22 added tokens
    EXPECT_EQ(tokenizer_->vocab_size(), 151665);
}

TEST_F(TokenizerTest, DecodeKnownTokens)
{
    // Single token decode
    auto result = tokenizer_->decode({151643});
    EXPECT_EQ(result, "<|endoftext|>");
}

TEST_F(TokenizerTest, EncodeDecodeRoundtrip)
{
    std::vector<std::string> texts = {
        "Hello",
        "Hello world",
        "I am fine",
        "The cat sat on the mat",
    };

    for (auto& text : texts)
    {
        auto ids = tokenizer_->encode(text);
        EXPECT_FALSE(ids.empty()) << "encode returned empty for: " << text;
        auto decoded = tokenizer_->decode(ids);
        EXPECT_EQ(decoded, text) << "roundtrip failed for: " << text;
    }
}

TEST_F(TokenizerTest, EncodeProducesNonEmpty)
{
    auto ids = tokenizer_->encode("test");
    EXPECT_GT(ids.size(), 0);
}

TEST_F(TokenizerTest, EncodeHandlesPunctuation)
{
    auto ids = tokenizer_->encode("Hello, world!");
    EXPECT_FALSE(ids.empty());
    auto decoded = tokenizer_->decode(ids);
    EXPECT_EQ(decoded, "Hello, world!");
}

}  // namespace
}  // namespace vvllm
