#include "llama.h"
#include "options.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

struct backend_guard {
    backend_guard() { llama_backend_init(); }
    ~backend_guard() { llama_backend_free(); }
    backend_guard(const backend_guard &) = delete;
    backend_guard & operator=(const backend_guard &) = delete;
};

void usage() {
    std::puts("Usage: llama-npu-model --model <tokenizer GGUF> --weights <packed package directory>\n"
              "                       --prompt <text> [--predict <count>] [--ctx-size <tokens>]\n"
              "       llama-npu-model --help\n"
              "In-process complete-model NPU bridge; NOT ggml graph offload.\n"
              "Defaults: --predict 32, --ctx-size 4096; capacity is provider-defined.\n"
              "Numeric syntax limit: 2147483647 (not a model capacity guarantee).\n"
              "Single sequence, CPU greedy sampling, no chat template, no CPU model fallback.\n"
              "Requires a bound tokenizer GGUF and strict identity.txt package; no conversion.\n"
              "Exit: 0 success/help, 2 arguments, 1 model/package/runtime/output failure.");
}

void decode(llama_context * ctx, llama_token * tokens, int32_t count, llama_pos start) {
    std::vector<llama_pos> positions(count);
    std::vector<int8_t> logits(count, 0);
    for (int32_t i = 0; i < count; ++i) { positions[i] = start + i; }
    logits.back() = 1;
    llama_batch batch = {};
    batch.n_tokens = count;
    batch.token = tokens;
    batch.pos = positions.data();
    // Null sequence arrays mean sequence 0, not multiple sequences.
    batch.logits = logits.data();
    const int rc = llama_decode(ctx, batch);
    if (rc != 0) { throw std::runtime_error("llama_decode failed (code " + std::to_string(rc) + "); no retry/fallback"); }
}

void print_piece(const llama_vocab * vocab, llama_token token) {
    char small[256];
    int32_t n = llama_token_to_piece(vocab, token, small, sizeof(small), 0, true);
    const char * data = small;
    std::vector<char> large;
    if (n < 0) {
        if (n == (std::numeric_limits<int32_t>::min)()) { throw std::runtime_error("token piece size overflow"); }
        large.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(vocab, token, large.data(), static_cast<int32_t>(large.size()), 0, true);
        data = large.data();
        if (n < 0 || static_cast<size_t>(n) > large.size()) { throw std::runtime_error("token piece conversion failed"); }
    }
    if ((n && std::fwrite(data, 1, n, stdout) != static_cast<size_t>(n)) || std::fflush(stdout) != 0) {
        throw std::runtime_error("token output failed");
    }
}

int run(const npu_model::options & opts) {
    using clock = std::chrono::steady_clock;
    const auto request_start = clock::now();
    const auto seconds = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    if (!std::filesystem::is_regular_file(opts.model)) { throw std::runtime_error("--model is not a regular tokenizer GGUF file"); }
    if (!std::filesystem::is_directory(opts.weights)) { throw std::runtime_error("--weights is not a packed package directory"); }
#ifdef _WIN32
    // Preserve token bytes (especially LF) rather than CRT text translation.
    if (_setmode(_fileno(stdout), _O_BINARY) == -1) { throw std::runtime_error("cannot set binary token output"); }
#endif
    backend_guard backend;
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(opts.model.c_str(), mp), llama_model_free);
    if (!model) { throw std::runtime_error("cannot load tokenizer GGUF (vocab_only=true)"); }
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    // Probe the tokenizer rather than allocating from an unchecked context
    // option. Model capacity is validated separately by the selected provider.
    const int32_t probe = llama_tokenize(vocab, opts.prompt.data(),
        static_cast<int32_t>(opts.prompt.size()), nullptr, 0, true, true);
    if (probe == (std::numeric_limits<int32_t>::min)()) {
        throw std::runtime_error("prompt token count overflow");
    }
    const size_t required = probe < 0 ? size_t(-probe) : size_t(probe);
    npu_model::check_budget(required, opts);
    std::vector<llama_token> prompt(required);
    const int32_t count = llama_tokenize(vocab, opts.prompt.data(), static_cast<int32_t>(opts.prompt.size()),
                                         prompt.data(), static_cast<int32_t>(prompt.size()), true, true);
    if (count < 0) { throw std::runtime_error("prompt tokenization failed or exceeds --ctx-size"); }
    prompt.resize(count);
    npu_model::check_budget(prompt.size(), opts);

    auto cp = llama_context_default_params();
    cp.npu_model_path = opts.weights.c_str();
    cp.n_ctx = opts.context;
    cp.n_seq_max = 1;
    cp.n_batch = std::min<uint32_t>(opts.context, 512);
    cp.n_ubatch = cp.n_batch;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) {
        throw std::runtime_error("NPU context rejected: check bridge build, architecture/ABI, tokenizer identity, package hashes and runtime diagnostics above; no CPU fallback");
    }
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
    if (!sampler) { throw std::runtime_error("cannot create greedy sampler"); }
    const auto prefill_start = clock::now();
    clock::time_point first_output{}, last_output{};
    uint32_t emitted = 0;
    uint32_t decode_calls = 0;
    double decode_seconds = 0;
    // Even predict=0 creates/verifies the NPU context and evaluates the prompt.
    for (size_t offset = 0; offset < prompt.size();) {
        const auto n = static_cast<int32_t>(std::min<size_t>(cp.n_batch, prompt.size() - offset));
        decode(ctx.get(), prompt.data() + offset, n, static_cast<llama_pos>(offset));
        offset += n;
    }
    for (auto token : prompt) { llama_sampler_accept(sampler.get(), token); }
    auto position = static_cast<llama_pos>(prompt.size());
    for (uint32_t i = 0; i < opts.predict; ++i) {
        // llama_sampler_sample calls llama_sampler_accept internally. Do not
        // accept generated tokens twice; no sampler is attached to the backend.
        llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        if (token < 0 || token >= llama_vocab_n_tokens(vocab)) { throw std::runtime_error("sampler returned invalid token"); }
        if (llama_vocab_is_eog(vocab, token)) { break; }
        print_piece(vocab, token);
        last_output = clock::now();
        if (emitted++ == 0) { first_output = last_output; }
        if (i + 1 < opts.predict) {
            const auto start = clock::now();
            decode(ctx.get(), &token, 1, position++);
            decode_seconds += seconds(start, clock::now());
            ++decode_calls;
        }
    }
    std::fprintf(stderr, "NPU_PERF prompt_tokens=%zu emitted_tokens=%u setup_s=%.6f decode_calls=%u decode_s=%.6f decode_tok_s=%.6f\n",
                 prompt.size(), emitted, seconds(request_start, prefill_start), decode_calls, decode_seconds,
                 decode_seconds > 0 ? decode_calls / decode_seconds : 0);
    if (emitted) {
        std::fprintf(stderr, "NPU_PERF cold_ttft_s=%.6f ready_ttft_s=%.6f intertoken_tok_s=%.6f\n",
                     seconds(request_start, first_output), seconds(prefill_start, first_output),
                     emitted > 1 ? (emitted - 1) / seconds(first_output, last_output) : 0);
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    npu_model::options opts;
    try { opts = npu_model::parse(argc, argv); }
    catch (const std::invalid_argument & e) {
        std::fprintf(stderr, "argument error: %s\nUse --help for usage.\n", e.what());
        return 2;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "llama-npu-model: argument allocation failed: %s\n", e.what());
        return 1;
    }
    try {
        if (opts.help) { usage(); return 0; }
        return run(opts);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "llama-npu-model: %s\n", e.what());
        return 1;
    }
}