// Define NPU_MODEL_PARSER_ONLY for an isolated parser test without linking a
// possibly concurrently rebuilt llama library. The normal CMake target always
// checks the public ABI defaults as well.
#ifndef NPU_MODEL_PARSER_ONLY
#include "llama.h"
#endif
#include "../examples/npu-model/options.h"

#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

static npu_model::options parse(std::initializer_list<const char *> args) {
    std::vector<const char *> argv = { "llama-npu-model" };
    argv.insert(argv.end(), args);
    return npu_model::parse(static_cast<int>(argv.size()), argv.data());
}

static void invalid(std::initializer_list<const char *> args) {
    try { (void) parse(args); }
    catch (const std::invalid_argument &) { return; }
    throw std::runtime_error("invalid arguments were accepted");
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--skip-full") {
        std::puts("SKIP: full-model fixture not explicitly configured; no numerical/model verification performed");
        return 77;
    }
    try {
        require(argc == 1, "unexpected test arguments");
        // Public ABI, safe on shared Windows builds: no internal llama_npu symbols.
    #ifndef NPU_MODEL_PARSER_ONLY
        const auto cp = llama_context_default_params();
        const auto mp = llama_model_default_params();
        require(cp.npu_model_path == nullptr, "normal contexts must not opt in to NPU by default");
        require(!mp.vocab_only, "ordinary model loading default changed");
    #endif
        require(parse({"--help"}).help, "help must not need assets");
        const auto opts = parse({"--model", "tokenizer.gguf", "--weights", "packed directory",
                                 "--prompt", "hello", "--predict", "0", "--ctx-size", "65792"});
        require(opts.predict == 0 && opts.context == 65792 && opts.weights == "packed directory", "valid bounds/paths rejected");
        invalid({});
        invalid({"--model"});
        invalid({"--help", "--predict", "1"});
        invalid({"--model", "a", "--weights", "b"});
        invalid({"--model", "a", "--weights", "b", "--prompt", "x", "--model", "c"});
        require(parse({"--model", "a", "--weights", "b", "--prompt", "x", "--predict", "0",
                   "--ctx-size", "2147483647"}).context == npu_model::max_context, "generic syntax upper bound");
        require(parse({"--model", "a", "--weights", "b", "--prompt", "x", "--predict", "0",
                   "--ctx-size", "65793"}).context == 65793, "parser must not impose Qwen capacity");
        for (const char * value : {"-1", "+1", "1x", " 1", "", "2147483648", "429496729600"}) {
            invalid({"--model", "a", "--weights", "b", "--prompt", "x", "--predict", value});
        }
        invalid({"--model", "a", "--weights", "b", "--prompt", "x", "--ctx-size", "0"});
        invalid({"--model", "a", "--weights", "b", "--prompt", "x", "--ctx-size", "1", "--predict", "2"});
        invalid({"--device", "NPU"});
        const std::string oversized(npu_model::max_prompt_bytes + 1, 'x');
        invalid({"--model", "a", "--weights", "b", "--prompt", oversized.c_str()});
        npu_model::options budget;
        budget.context = 8; budget.predict = 3;
        npu_model::check_budget(5, budget);
        for (size_t count : {size_t(0), size_t(6), size_t(9), (size_t(-1))}) {
            bool rejected = false;
            try { npu_model::check_budget(count, budget); }
            catch (const std::runtime_error &) { rejected = true; }
            require(rejected, "context budget accepted empty/overflowing prompt");
        }
    #ifndef NPU_MODEL_PARSER_ONLY
        require(llama_context_default_params().npu_model_path == nullptr, "parsing changed public defaults");
        std::puts("PASS: public opt-in defaults");
    #endif
        std::puts("PASS: CLI parsing/bounds and context budget (no runtime/model needed)");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}