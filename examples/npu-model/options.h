#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace npu_model {

// Parser/llama_pos representability only. The selected provider enforces capacity.
constexpr uint32_t max_context = (std::numeric_limits<int32_t>::max)();
constexpr size_t max_prompt_bytes = 1024 * 1024;

struct options {
    std::string model;
    std::string weights;
    std::string prompt;
    uint32_t predict = 32;
    uint32_t context = 4096;
    bool help = false;
};

inline uint32_t number(const std::string & text, uint32_t limit, const std::string & name) {
    if (text.empty()) { throw std::invalid_argument(name + ": expected unsigned decimal integer"); }
    uint32_t result = 0;
    for (const char c : text) {
        if (c < '0' || c > '9' || result > limit / 10 ||
            (result == limit / 10 && uint32_t(c - '0') > limit % 10)) {
            throw std::invalid_argument(name + ": invalid or out-of-range integer");
        }
        result = result * 10 + uint32_t(c - '0');
    }
    return result;
}

inline options parse(int argc, const char * const * argv) {
    options result;
    if (argc == 2 && std::string(argv[1]) == "--help") { result.help = true; return result; }
    std::unordered_set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key != "--model" && key != "--weights" && key != "--prompt" &&
            key != "--predict" && key != "--ctx-size") {
            throw std::invalid_argument("unknown argument: " + key + " (--help must be used alone)");
        }
        if (!seen.insert(key).second) { throw std::invalid_argument("duplicate argument: " + key); }
        if (++i == argc) { throw std::invalid_argument("missing value for " + key); }
        const std::string value = argv[i];
        if (key == "--model") { result.model = value; }
        if (key == "--weights") { result.weights = value; }
        if (key == "--prompt") { result.prompt = value; }
        if (key == "--predict") { result.predict = number(value, max_context, key); }
        if (key == "--ctx-size") { result.context = number(value, max_context, key); }
    }
    if (result.model.empty() || result.weights.empty() || !seen.count("--prompt")) {
        throw std::invalid_argument("required: --model <tokenizer GGUF> --weights <packed package directory> --prompt <text>");
    }
    if (!result.context || result.predict > result.context) {
        throw std::invalid_argument("--ctx-size must be 1..2147483647 and --predict must not exceed it");
    }
    if (result.prompt.size() > max_prompt_bytes) { throw std::invalid_argument("--prompt exceeds 1 MiB"); }
    return result;
}

inline void check_budget(size_t prompt_tokens, const options & opts) {
    // Conservative: reserve space for every requested output, including the last
    // token (which need not be decoded). Never truncate or shift the prompt.
    if (!prompt_tokens || prompt_tokens > opts.context || opts.predict > opts.context - prompt_tokens) {
        throw std::runtime_error("prompt tokens + --predict exceed --ctx-size (or prompt has no tokens)");
    }
}

} // namespace npu_model