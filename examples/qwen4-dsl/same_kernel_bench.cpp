#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int main(int argc, char ** argv) try {
    if (argc != 6) throw std::runtime_error("usage: same_kernel_bench DLL GGUF PROMPT PACK QSA_OVERLAY");
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::ifstream in(argv[3], std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamoff>(in.tellg());
    if (bytes <= 0 || bytes % 4 || bytes / 4 + 64 > 32768) throw std::runtime_error("invalid prompt");
    std::vector<llama_token> ids(static_cast<size_t>(bytes / 4));
    in.seekg(0);
    if (!in.read(reinterpret_cast<char *>(ids.data()), bytes)) throw std::runtime_error("short prompt read");

    llama_backend_init();
    const auto reg = ggml_backend_load(argv[1]);
    if (!reg) throw std::runtime_error("could not load same-kernel backend");
    const auto prepare = reinterpret_cast<int (*)(const char *, const char *)>(ggml_backend_reg_get_proc_address(reg, "same_kernel_prepare"));
    const auto release = reinterpret_cast<void (*)()>(ggml_backend_reg_get_proc_address(reg, "same_kernel_release"));
    const auto count = reinterpret_cast<uint32_t (*)()>(ggml_backend_reg_get_proc_address(reg, "same_kernel_eval_count"));
    ggml_backend_dev_t devices[] = {ggml_backend_reg_dev_get(reg, 0), nullptr};
    if (!devices[0] || !prepare || !release || !count) throw std::runtime_error("incomplete adapter");

    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    const auto setup = Clock::now();
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[2], mp), llama_model_free);
    if (!model) throw std::runtime_error("GGUF load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 32768;
    cp.n_batch = cp.n_ubatch = 4096;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx || llama_n_ctx(ctx.get()) != 32768 || llama_n_batch(ctx.get()) != 4096 || llama_n_ubatch(ctx.get()) != 4096)
        throw std::runtime_error("context creation or requested sizing failed");
    if (prepare(argv[4], argv[5])) throw std::runtime_error("NPU session creation failed");
    struct Release { void (*fn)(); ~Release() { fn(); } } guard{release};
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
    const bool directGreedy = std::getenv("QWEN4_RAW_GREEDY") != nullptr;
    const bool profile = std::getenv("LLAMA_PROFILE_UBATCH") != nullptr;
    const int vocabSize = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    llama_batch batch = llama_batch_init(4096, 0, 1);
    struct BatchGuard { llama_batch value; ~BatchGuard() { llama_batch_free(value); } } batch_guard{batch};
    uint32_t expected = 0, position = 0;
    auto eval = [&](const llama_token * tokens, int n) {
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = position + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == n - 1;
        }
        if (llama_decode(ctx.get(), batch) != 0) throw std::runtime_error("llama_decode failed");
        const auto syncStart = profile ? Clock::now() : Clock::time_point{};
        llama_synchronize(ctx.get());
        if (profile) std::fprintf(stderr, "bench_sync_profile n=%d sync_ms=%.3f\n", n, elapsed(syncStart));
        if (count() != ++expected) throw std::runtime_error("llama batch did not produce exactly one NPU takeover");
        position += n;
    };
    auto sample = [&] {
        const auto sampleStart = profile ? Clock::now() : Clock::time_point{};
        llama_token token;
        if (directGreedy) {
            // This benchmark has one full-vocabulary row and no sampler chain.
            // Keep the first maximum, as in llama_sampler_init_greedy().
            const float * logits = llama_get_logits_ith(ctx.get(), -1);
            if (!logits || vocabSize <= 0) throw std::runtime_error("missing logits row");
            token = static_cast<llama_token>(std::max_element(logits, logits + vocabSize) - logits);
        } else {
            token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        }
        if (profile) std::fprintf(stderr, "bench_sample_profile sample_ms=%.3f\n", elapsed(sampleStart));
        return token;
    };
    std::printf("setup_ms %.3f\n", elapsed(setup));
    auto start = Clock::now();
    for (size_t s = 0; s < ids.size(); s += 4096) eval(ids.data() + s, static_cast<int>(std::min<size_t>(4096, ids.size() - s)));
    auto token = sample();
    std::printf("prefill total %zu tokens ms %.3f\n", ids.size(), elapsed(start));
    std::vector<llama_token> result{token};
    start = Clock::now();
    for (int i = 0; i < 63; ++i) {
        const auto step = Clock::now();
        eval(&token, 1);
        token = sample();
        if (directGreedy) {
            char line[96];
            const int len = std::snprintf(line, sizeof(line), "token %d step_ms %.3f\n", token, elapsed(step));
            if (len < 0 || len >= sizeof(line) || std::fwrite(line, 1, len, stdout) != len)
                throw std::runtime_error("token log write failed");
        } else {
            std::printf("token %d step_ms %.3f\n", token, elapsed(step));
        }
        result.push_back(token);
    }
    std::printf("decode total 63 tokens ms %.3f\nids ", elapsed(start));
    for (size_t i = 0; i < result.size(); ++i) std::printf("%s%d", i ? "," : "", result[i]);
    std::puts("");
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}