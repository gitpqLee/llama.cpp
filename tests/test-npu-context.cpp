// MOCK HOST BRIDGE CONTRACT ONLY. No model math, real weights, or hardware.
// Link llama normally: these definitions satisfy every qwen36_session symbol
// before the linker's archive scan. Its single implementation object stays out.
#include "llama.h"
#include "gguf.h"
#include "qwen36_contract.h"
#include "model_api.h"
#include "qwen36_session.h"
extern "C" {
#include "hash/sha256/sha256.h"
}

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace llama_npu_contract = qwen36_contract; // Qwen-specific test fixture only
constexpr const char * mock_abi = "MOCK-qwen-bridge-v2";

namespace {
namespace fs = std::filesystem;
constexpr int32_t n_vocab = 248320;
int checks = 0;
void check(bool value, const std::string & message) {
    ++checks;
    if (!value) { throw std::runtime_error(message); }
}

struct call {
    std::vector<uint32_t> tokens;
    size_t position;
    size_t vocabulary;
    bool has_logits;
};
struct mock_observer {
    int constructed = 0;
    int destroyed = 0;
    int resets = 0;
    bool throw_next = false;
    std::string directory;
    std::vector<call> calls;
} mock;

// Exactly representable float tags, not plausible model outputs. Checking every
// element detects truncation, wrong row/stride, and stale output publication.
float tag(size_t position, size_t index) {
    return static_cast<float>(position * n_vocab + index);
}
}

struct qwen36_session::impl {
    size_t position = 0;
    bool poisoned = false;
    submission_counts counts;
};

qwen36_session::qwen36_session(const options & opts) : pimpl_(new impl) {
    ++mock.constructed;
    mock.directory = opts.weights_dir;
    check(opts.n_resident == (std::numeric_limits<int>::max)() &&
          opts.use_single_whole_model_kernels, "bridge changed session defaults");
    // Deliberately no package/weight-size validation here. The real bridge
    // verifies hashes; these tiny blobs are NOT a numerical weight package.
}
qwen36_session::~qwen36_session() { ++mock.destroyed; }
void qwen36_session::evaluate(const uint32_t * tokens, size_t count,
                            size_t pos, float * logits, size_t vocabulary) {
    mock.calls.push_back({std::vector<uint32_t>(tokens, tokens + count), pos, vocabulary, logits != nullptr});
    check(!pimpl_->poisoned && count && pos == pimpl_->position && logits &&
          vocabulary == vocabulary_size, "invalid arguments reached mocked session");
    if (count > 1) { ++pimpl_->counts.prefill; }
    else { ++pimpl_->counts.decode; }
    ++pimpl_->counts.kernels;
    if (mock.throw_next) {
        mock.throw_next = false;
        // A device can modify output before throwing. None may be published.
        std::fill_n(logits, vocabulary, -123.0f);
        pimpl_->poisoned = true;
        throw std::runtime_error("mock device submission failure (no hardware)");
    }
    pimpl_->position += count;
    for (size_t i = 0; i < vocabulary; ++i) { logits[i] = tag(pos, i); }
}
void qwen36_session::reset() { ++mock.resets; pimpl_->position = 0; }
size_t qwen36_session::position() const noexcept { return pimpl_->position; }
qwen36_session::submission_counts qwen36_session::submissions() const noexcept { return pimpl_->counts; }
bool qwen36_session::poisoned() const noexcept { return pimpl_->poisoned; }

namespace {
using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

struct owned_temp {
    fs::path root;
    owned_temp() {
        // Only a directory successfully created by this instance is owned.
        for (int i = 0; i < 100; ++i) {
            const auto candidate = fs::temp_directory_path() /
                ("llama-npu-context-mock-" + std::to_string(std::random_device{}()));
            if (fs::create_directory(candidate)) { root = candidate; return; }
        }
        throw std::runtime_error("cannot create exclusive temporary fixture");
    }
    ~owned_temp() {
        if (!root.empty()) { std::error_code ec; fs::remove_all(root, ec); }
    }
    owned_temp(const owned_temp &) = delete;
    owned_temp & operator=(const owned_temp &) = delete;
};

void write(const fs::path & path, const std::string & text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(text.data(), text.size());
    out.close();
    check(bool(out), "fixture write: " + path.string());
}
std::string sha(const std::string & text) {
    sha256_t state;
    unsigned char digest[32];
    sha256_init(&state);
    sha256_update(&state, reinterpret_cast<const unsigned char *>(text.data()), text.size());
    sha256_final(&state, digest);
    std::string result;
    const char * hex = "0123456789abcdef";
    for (auto c : digest) { result += hex[c >> 4]; result += hex[c & 15]; }
    return result;
}
model_ptr load(const fs::path & path) {
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    model_ptr result(llama_model_load_from_file(path.string().c_str(), params), llama_model_free);
    check(bool(result), "vocab-only model load: " + path.string());
    check(llama_vocab_n_tokens(llama_model_get_vocab(result.get())) == n_vocab, "fixture vocabulary size");
    return result;
}

struct fixture {
    owned_temp temp;
    gguf_ptr kv{gguf_init_empty(), gguf_free};
    fs::path bound = temp.root / "bound.gguf";
    std::string directory = temp.root.string();
    std::string identity;
    std::string tokenizer;

    fixture() {
        check(bool(kv), "gguf_init_empty");
        gguf_set_val_str(kv.get(), "general.architecture", "qwen35moe");
        // Exact scalar ABI from the DSL Qwen provider contract. No tensor data.
        const std::pair<const char *, uint32_t> dims[] = {
            {"embedding_length", 2048}, {"block_count", 40},
            {"attention.head_count", 16}, {"attention.head_count_kv", 2},
            {"attention.key_length", 256}, {"attention.value_length", 256},
            {"expert_count", 256}, {"expert_used_count", 8},
            {"expert_feed_forward_length", 512}, {"expert_shared_feed_forward_length", 512},
            {"ssm.conv_kernel", 4}, {"ssm.inner_size", 4096}, {"ssm.state_size", 128},
            {"ssm.time_step_rank", 32}, {"ssm.group_count", 16},
            {"full_attention_interval", 4}, {"rope.dimension_count", 64},
        };
        for (const auto & dim : dims) {
            gguf_set_val_u32(kv.get(), (std::string("qwen35moe.") + dim.first).c_str(), dim.second);
        }
        gguf_set_val_f32(kv.get(), "qwen35moe.rope.freq_base", 10000000.0f);
        gguf_set_val_f32(kv.get(), "qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);
        gguf_set_val_str(kv.get(), "tokenizer.ggml.model", "gpt2");
        gguf_set_val_str(kv.get(), "tokenizer.ggml.pre", "gpt-2");
        std::vector<std::string> tokens;
        tokens.reserve(n_vocab);
        for (int i = 0; i < n_vocab - 1; ++i) { tokens.push_back("mock_token_" + std::to_string(i)); }
        tokens.emplace_back("<|endoftext|>");
        std::vector<const char *> strings;
        for (const auto & token : tokens) { strings.push_back(token.c_str()); }
        std::vector<float> scores(n_vocab, 0.0f);
        std::vector<int32_t> types(n_vocab, 1);
        types.back() = 3;
        gguf_set_arr_str(kv.get(), "tokenizer.ggml.tokens", strings.data(), strings.size());
        gguf_set_arr_str(kv.get(), "tokenizer.ggml.merges", nullptr, 0);
        gguf_set_arr_data(kv.get(), "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
        gguf_set_arr_data(kv.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
        gguf_set_val_u32(kv.get(), "tokenizer.ggml.bos_token_id", n_vocab - 1);
        gguf_set_val_u32(kv.get(), "tokenizer.ggml.eos_token_id", n_vocab - 1);
        gguf_set_val_bool(kv.get(), "tokenizer.ggml.add_bos_token", false);
        gguf_set_val_bool(kv.get(), "tokenizer.ggml.add_eos_token", false);
        const auto source = temp.root / "source.gguf";
        save(source);
        auto source_model = load(source);
        char digest[65] = {};
        check(llama_npu_tokenizer_sha256(source_model.get(), digest, sizeof(digest)), "exported tokenizer SHA");
        tokenizer = digest;
        check(tokenizer.size() == 64, "tokenizer digest size");
        identity = "MOCK-qwen-host-v2\ntokenizer " + tokenizer + "\n";
        std::vector<std::pair<std::string, std::string>> files = {
            {"manifest.txt", ""},
            {"layer_types.txt", ""}, {"globals.bin", "mock globals"},
        };
        for (int i = 0; i < 40; ++i) {
            files[1].second += (i + 1) % 4 ? "linear_attention\n" : "full_attention\n";
            char name[32]; std::snprintf(name, sizeof(name), "layer_%02d.bin", i);
            files.emplace_back(name, "mock layer " + std::to_string(i));
        }
        check(files.size() == 43, "package file count");
        for (size_t i = 2; i < files.size(); ++i) {
            files[0].second += files[i].first + " MOCK-blob 0 " + std::to_string(files[i].second.size()) + "\n";
        }
        for (const auto & file : files) {
            write(temp.root / file.first, file.second);
            identity += sha(file.second) + " " + file.first + "\n";
        }
        write(temp.root / "identity.txt", identity);
        gguf_set_val_str(kv.get(), "npu.dsl.abi", mock_abi);
        gguf_set_val_str(kv.get(), "npu.dsl.tokenizer_sha256", tokenizer.c_str());
        gguf_set_val_str(kv.get(), "npu.dsl.package_id", sha(identity).c_str());
        save(bound);
    }
    void save(const fs::path & path) {
        check(gguf_get_n_tensors(kv.get()) == 0, "fixture must remain metadata-only");
        check(gguf_write_to_file(kv.get(), path.string().c_str(), true), "write metadata GGUF");
    }
    llama_context_params params() const {
        auto cp = llama_context_default_params();
        cp.npu_model_path = directory.c_str();
        cp.n_ctx = 128;
        cp.n_batch = 128;
        cp.n_ubatch = 128;
        return cp;
    }
};

context_ptr context(llama_model * model, llama_context_params cp) {
    const int before = mock.constructed;
    context_ptr result(llama_init_from_model(model, cp), llama_free);
    check(bool(result), "valid bridge context construction");
    check(mock.constructed == before + 1, "exactly one mock session constructed (link-time replacement)");
    check(llama_n_ctx(result.get()) == cp.n_ctx, "context size");
    check(llama_get_memory(result.get()) != nullptr, "bridge exposes public memory");
    return result;
}
void no_logits(llama_context * ctx) {
    check(!llama_get_logits(ctx), "no raw logits after failure/no-output");
    for (int i : {-2, -1, 0, 1, 2, 127}) {
        check(!llama_get_logits_ith(ctx, i), "no indexed logits after failure/no-output");
    }
}
void logits(llama_context * ctx, int last, size_t pos) {
    auto * raw = llama_get_logits(ctx);
    auto * row = llama_get_logits_ith(ctx, last);
    check(raw && row == raw && llama_get_logits_ith(ctx, -1) == row, "last and -1 row mapping");
    check(!llama_get_logits_ith(ctx, -2) && !llama_get_logits_ith(ctx, last + 1), "invalid logit rows");
    for (int i = 0; i < last; ++i) { check(!llama_get_logits_ith(ctx, i), "non-output row must be absent"); }
    for (int i = 0; i < n_vocab; ++i) {
        if (row[i] != tag(pos, i)) { check(false, "full-vocabulary mock logit mapping at " + std::to_string(i)); }
    }
    check(true, "full-vocabulary logit mapping");
}
void position(llama_context * ctx, llama_pos next) {
    const auto mem = llama_get_memory(ctx);
    check(llama_memory_seq_pos_max(mem, 0) == next - 1, "public memory max position");
    check(llama_memory_seq_pos_min(mem, 0) == (next ? 0 : -1), "public memory min position");
    check(llama_memory_seq_pos_max(mem, 1) == -1, "nonexistent sequence position");
}
void received(const std::vector<uint32_t> & tokens, size_t pos) {
    check(!mock.calls.empty(), "expected a mock evaluation");
    const auto & c = mock.calls.back();
    check(c.tokens == tokens && c.position == pos && c.vocabulary == n_vocab && c.has_logits,
          "bridge must forward exact tokens, start position, full vocabulary, and output buffer");
}

struct batch {
    llama_batch value;
    explicit batch(std::initializer_list<llama_token> tokens, llama_pos start) :
        value(llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1)) {
        value.n_tokens = static_cast<int32_t>(tokens.size());
        int i = 0;
        for (auto token : tokens) {
            value.token[i] = token;
            value.pos[i] = start + i;
            value.n_seq_id[i] = 1;
            value.seq_id[i][0] = 0;
            value.logits[i] = i + 1 == value.n_tokens;
            ++i;
        }
    }
    ~batch() { llama_batch_free(value); }
    batch(const batch &) = delete;
    batch & operator=(const batch &) = delete;
};
void rejected(llama_context * ctx, llama_batch value, llama_pos next, const char * what) {
    const auto before = mock.calls.size();
    const auto resets = mock.resets;
    check(llama_decode(ctx, value) != 0, std::string("reject ") + what);
    check(mock.calls.size() == before && mock.resets == resets, std::string("reject before mock: ") + what);
    position(ctx, next);
    no_logits(ctx);
}

void test_decode(llama_model * model, fixture & f) {
    auto ctx = context(model, f.params());
    check(fs::path(mock.directory) == fs::canonical(f.directory), "forwarded canonical package directory");
    no_logits(ctx.get());
    position(ctx.get(), 0);
    batch prefill({10, 11, 12}, 0);
    // Reject earlier requested output before evaluating any token, then retry
    // at exactly the same position. Argument errors must not poison the bridge.
    prefill.value.logits[0] = 1;
    rejected(ctx.get(), prefill.value, 0, "earlier output");
    prefill.value.logits[0] = 0;
    auto before = mock.calls.size();
    check(llama_decode(ctx.get(), prefill.value) == 0, "multi-token decode");
    check(mock.calls.size() == before + 1, "prefill is one session call, not token loop");
    received({10, 11, 12}, 0);
    position(ctx.get(), 3);
    logits(ctx.get(), 2, 0);

    batch next({13}, 3);
    before = mock.calls.size();
    check(llama_decode(ctx.get(), next.value) == 0, "single-token decode");
    check(mock.calls.size() == before + 1, "single decode one call");
    received({13}, 3);
    logits(ctx.get(), 0, 3);
    position(ctx.get(), 4);

    batch silent({14, 15}, 4);
    std::fill_n(silent.value.logits, 2, 0);
    before = mock.calls.size();
    check(llama_decode(ctx.get(), silent.value) == 0, "no-output decode still evaluates");
    check(mock.calls.size() == before + 1, "no-output batch one call");
    received({14, 15}, 4);
    position(ctx.get(), 6);
    no_logits(ctx.get());

    llama_token implicit[] = {16, 17};
    before = mock.calls.size();
    check(llama_decode(ctx.get(), llama_batch_get_one(implicit, 2)) == 0, "implicit-position compat batch");
    check(mock.calls.size() == before + 1, "implicit batch one call");
    received({16, 17}, 6);
    position(ctx.get(), 8);
    logits(ctx.get(), 1, 6);

    batch bad({18, 19}, 8);
    bad.value.pos[1] = 10;
    rejected(ctx.get(), bad.value, 8, "nonsequential position");
    bad.value.pos[1] = 9;
    bad.value.token[1] = n_vocab;
    rejected(ctx.get(), bad.value, 8, "out-of-range token");
    bad.value.token[1] = -1;
    rejected(ctx.get(), bad.value, 8, "negative token");
    bad.value.token[1] = 19;
    bad.value.seq_id[0][0] = 1;
    rejected(ctx.get(), bad.value, 8, "nonzero sequence");
    bad.value.seq_id[0][0] = 0;
    bad.value.n_seq_id[0] = 2;
    rejected(ctx.get(), bad.value, 8, "multiple sequences");
    bad.value.n_seq_id[0] = 1;
    float embedding = 0;
    auto emb = bad.value;
    emb.embd = &embedding;
    rejected(ctx.get(), emb, 8, "tokens plus embedding");
    emb.token = nullptr;
    rejected(ctx.get(), emb, 8, "embedding-only input");
    bad.value.logits[0] = 1;
    rejected(ctx.get(), bad.value, 8, "extra output");
    bad.value.logits[0] = 0;
    auto empty = bad.value;
    empty.n_tokens = 0;
    rejected(ctx.get(), empty, 8, "empty batch");
    before = mock.calls.size();
    check(llama_decode(ctx.get(), bad.value) == 0, "argument rejections allow retry at same position");
    check(mock.calls.size() == before + 1, "retry one call");
    received({18, 19}, 8);
    logits(ctx.get(), 1, 8);
    position(ctx.get(), 10);

    const auto mem = llama_get_memory(ctx.get());
    check(!llama_memory_can_shift(mem), "bridge cannot shift");
    const int resets = mock.resets;
    before = mock.calls.size();
    check(!llama_memory_seq_rm(mem, 0, 1, -1), "partial seq_rm refused");
    check(!llama_memory_seq_rm(mem, 0, 0, 9), "prefix seq_rm refused");
    check(!llama_memory_seq_rm(mem, 1, 0, -1), "other seq_rm refused");
    check(mock.resets == resets && mock.calls.size() == before, "partial removal no mock mutation");
    position(ctx.get(), 10);
    logits(ctx.get(), 1, 8);
    batch after_partial({20}, 10);
    check(llama_decode(ctx.get(), after_partial.value) == 0, "decode after partial removal refusal");
    received({20}, 10);
    for (bool data : {false, true}) {
        const auto n = mock.resets;
        llama_memory_clear(mem, data);
        check(mock.resets == n + 1, "memory clear invokes reset once");
        position(ctx.get(), 0);
        batch restart({21}, 0);
        check(llama_decode(ctx.get(), restart.value) == 0, "decode after clear starts at zero");
        received({21}, 0);
    }
    const auto n = mock.resets;
    check(llama_memory_seq_rm(mem, 0, 0, -1), "whole sequence removal supported");
    check(mock.resets == n + 1, "whole removal resets session");
    position(ctx.get(), 0);

    std::array<uint8_t, 64> state;
    state.fill(0xa5);
    before = mock.calls.size();
    check(llama_state_get_size(ctx.get()) == 0, "state size unsupported");
    check(llama_state_get_data(ctx.get(), state.data(), state.size()) == 0, "state save unsupported");
    check(llama_state_set_data(ctx.get(), state.data(), state.size()) == 0, "state load unsupported");
    check(llama_state_seq_get_size(ctx.get(), 0) == 0, "sequence state size unsupported");
    check(llama_state_seq_get_data(ctx.get(), state.data(), state.size(), 0) == 0, "sequence state save unsupported");
    check(llama_state_seq_set_data(ctx.get(), state.data(), state.size(), 0) == 0, "sequence state load unsupported");
    check(std::all_of(state.begin(), state.end(), [](uint8_t b) { return b == 0xa5; }), "state buffer untouched");
    check(mock.calls.size() == before, "state APIs do not evaluate");
    position(ctx.get(), 0);

    // Shift is a void public API: it rejects by latching the bridge unusable.
    batch seed({22}, 0);
    check(llama_decode(ctx.get(), seed.value) == 0, "seed before shift");
    llama_memory_seq_add(mem, 0, 0, -1, 1);
    position(ctx.get(), 1);
    batch shifted({23}, 1);
    rejected(ctx.get(), shifted.value, 1, "latched seq_shift");
    llama_memory_clear(mem, true);
    shifted.value.pos[0] = 0;
    rejected(ctx.get(), shifted.value, 0, "clear cannot undo unsupported-operation latch");
}

void test_limit_and_failure(llama_model * model, fixture & f) {
    auto cp = f.params();
    cp.n_ctx = 3;
    auto ctx = context(model, cp);
    batch seed({1, 2}, 0);
    check(llama_decode(ctx.get(), seed.value) == 0, "small-context prefill");
    batch overflow({3, 4}, 2);
    rejected(ctx.get(), overflow.value, 2, "context limit");
    overflow.value.n_tokens = 1;
    overflow.value.logits[0] = 1;
    check(llama_decode(ctx.get(), overflow.value) == 0, "last context slot remains usable");
    received({3}, 2);
    position(ctx.get(), 3);
    batch full({4}, 3);
    rejected(ctx.get(), full.value, 3, "full context");

    auto failing = context(model, f.params());
    batch first({31}, 0);
    check(llama_decode(failing.get(), first.value) == 0, "publish output before device failure");
    logits(failing.get(), 0, 0);
    batch failure({32}, 1);
    const auto before = mock.calls.size();
    mock.throw_next = true;
    check(llama_decode(failing.get(), failure.value) != 0, "device exception must fail decode, not CPU retry");
    check(mock.calls.size() == before + 1 && !mock.throw_next, "device failure submitted exactly once");
    received({32}, 1);
    no_logits(failing.get());
    position(failing.get(), 1);
    rejected(failing.get(), failure.value, 1, "poisoned session subsequent decode");
    llama_memory_clear(llama_get_memory(failing.get()), true);
    failure.value.pos[0] = 0;
    rejected(failing.get(), failure.value, 0, "reset cannot unpoison device failure");
}

void reject_context(llama_model * model, llama_context_params cp, const char * what) {
    const int before = mock.constructed;
    const auto calls = mock.calls.size();
    context_ptr ctx(llama_init_from_model(model, cp), llama_free);
    check(!ctx, std::string("reject context: ") + what);
    check(mock.constructed == before && mock.calls.size() == calls,
          std::string("reject before mock construction: ") + what);
}

void test_trunk_metadata(fixture & f) {
    auto verify = [&](uint32_t blocks, const char * nextn, bool accepted) {
        gguf_set_val_u32(f.kv.get(), "qwen35moe.block_count", blocks);
        gguf_remove_key(f.kv.get(), "qwen35moe.nextn_predict_layers");
        if (nextn) { gguf_set_val_str(f.kv.get(), "qwen35moe.nextn_predict_layers", nextn); }
        const auto path = f.temp.root / "trunk.gguf";
        f.save(path);
        auto model = load(path);
        if (accepted) {
            auto ctx = context(model.get(), f.params());
            batch token({42}, 0);
            const auto before = mock.calls.size();
            check(llama_decode(ctx.get(), token.value) == 0, "40 trunk layers decode with optional MTP excluded");
            check(mock.calls.size() == before + 1, "trunk decode reaches mock exactly once");
            received({42}, 0);
            logits(ctx.get(), 0, 0);
            auto cp = f.params();
            cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            reject_context(model.get(), cp, "MTP context unsupported");
        } else {
            reject_context(model.get(), f.params(), "incompatible trunk/MTP metadata");
        }
    };
    // Fail first on the actual source GGUF layout: 40 trunk + one MTP block.
    verify(41, "1", true);
    verify(40, nullptr, true);
    verify(40, "0", true);
    verify(40, "1", false);
    verify(41, nullptr, false);
    verify(41, "0", false);
    verify(42, "2", false);
    for (const auto & malformed : {std::string(""), std::string("junk"), std::string("1x"),
                                  std::string("-1"), std::string("01"), std::string(" 1"),
                                  std::string("18446744073709551616"), std::string(257, '0')}) {
        verify(40, malformed.c_str(), false);
        verify(41, malformed.c_str(), false);
    }
    gguf_remove_key(f.kv.get(), "qwen35moe.nextn_predict_layers");
    gguf_set_val_u32(f.kv.get(), "qwen35moe.block_count", 40);
}

void test_construction(llama_model * model, fixture & f) {
    auto cp = f.params();
    cp.n_seq_max = 2;
    reject_context(model, cp, "multiple sequences");
    cp = f.params(); cp.embeddings = true;
    reject_context(model, cp, "embeddings");
    cp = f.params(); cp.n_ctx = qwen36_session::context_size + 1;
    reject_context(model, cp, "context exceeds ABI");
    cp = f.params(); cp.n_batch = 0;
    reject_context(model, cp, "zero batch");

    // No implicit opt-in even when the GGUF has every NPU binding field.
    const int before = mock.constructed;
    auto normal = llama_context_default_params();
    check(normal.npu_model_path == nullptr, "NPU path is opt-in by default");
    normal.n_ctx = 128;
    context_ptr plain(llama_init_from_model(model, normal), llama_free);
    check(bool(plain), "ordinary vocab-only context remains constructible");
    check(mock.constructed == before, "ordinary context must not construct NPU session");
    check(llama_get_memory(plain.get()) == nullptr, "ordinary vocab-only context has no NPU memory");

    auto corrupt_model = [&](const char * what) {
        const auto path = f.temp.root / "corrupt.gguf";
        f.save(path);
        auto bad = load(path);
        reject_context(bad.get(), f.params(), what);
    };
    gguf_set_val_u32(f.kv.get(), "qwen35moe.embedding_length", 2049);
    corrupt_model("dimension metadata");
    gguf_set_val_u32(f.kv.get(), "qwen35moe.embedding_length", 2048);
    gguf_set_val_str(f.kv.get(), "npu.dsl.abi", "wrong-abi");
    corrupt_model("ABI metadata");
    gguf_set_val_str(f.kv.get(), "npu.dsl.abi", mock_abi);
    gguf_set_val_bool(f.kv.get(), "tokenizer.ggml.add_bos_token", true);
    corrupt_model("actual tokenizer corruption with unchanged binding");
    gguf_set_val_bool(f.kv.get(), "tokenizer.ggml.add_bos_token", false);
    gguf_set_val_str(f.kv.get(), "npu.dsl.tokenizer_sha256", std::string(64, '0').c_str());
    corrupt_model("tokenizer binding corruption");
    gguf_set_val_str(f.kv.get(), "npu.dsl.tokenizer_sha256", f.tokenizer.c_str());
    gguf_set_val_str(f.kv.get(), "npu.dsl.package_id", std::string(64, '0').c_str());
    corrupt_model("package binding corruption");
    gguf_set_val_str(f.kv.get(), "npu.dsl.package_id", sha(f.identity).c_str());

    write(f.temp.root / "identity.txt", f.identity + "corruption\n");
    reject_context(model, f.params(), "identity hash corruption");
    write(f.temp.root / "identity.txt", f.identity);
    write(f.temp.root / "layer_17.bin", "modified mock blob");
    reject_context(model, f.params(), "weight blob hash corruption");
    write(f.temp.root / "layer_17.bin", "mock layer 17");
    // Prove restored fixtures still construct, so negative cases aren't merely
    // failing due to a permanently invalid synthetic package.
    auto restored = context(model, f.params());
}
}

int main() {
    // Explicit host-only provider schema: tiny test blobs must never masquerade
    // as validated Qwen numerical assets. The session remains link-time mocked.
    auto mock_provider = npudsl_model::providers().resolve(llama_npu_contract::abi);
    mock_provider.abi = mock_abi;
    mock_provider.package.identity_header = "MOCK-qwen-host-v2";
    mock_provider.format = {"MOCK-tags", "MOCK-none", "MOCK-tags", "MOCK-tags", "MOCK-qwen-host-v2", 1};
    mock_provider.package.tensor_alignment = 1;
    mock_provider.package.tensors.clear();
    mock_provider.package.tensors["globals.bin"]["MOCK-blob"] = std::string("mock globals").size();
    for (int i = 0; i < 40; ++i) {
        char name[32]; std::snprintf(name, sizeof(name), "layer_%02d.bin", i);
        mock_provider.package.tensors[name]["MOCK-blob"] = ("mock layer " + std::to_string(i)).size();
    }
    npudsl_model::use_legacy_package(mock_provider);
    npudsl_model::providers().add(std::move(mock_provider));
    std::cout << "MOCK HOST BRIDGE CONTRACT: real llama context, link-time mock qwen36_session; "
                 "NO hardware, model math, or numerical validation\n";
    try {
        fixture f;
        test_trunk_metadata(f);
        auto model = load(f.bound);
        char digest[65] = {};
        check(llama_npu_tokenizer_sha256(model.get(), digest, sizeof(digest)) && digest == f.tokenizer,
              "binding metadata must not change actual tokenizer fingerprint");
        test_decode(model.get(), f);
        test_limit_and_failure(model.get(), f);
        test_construction(model.get(), f);
        check(mock.constructed > 0 && mock.constructed == mock.destroyed, "all mock sessions destroyed");
        check(!mock.calls.empty(), "real llama decode reached link-time mock");
        std::cout << "PASS mock host bridge contract: " << checks << " checks, "
                  << mock.constructed << " sessions, " << mock.calls.size() << " evaluations, "
                  << mock.resets << " resets; no device execution\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL mock host bridge contract: " << e.what() << '\n';
        return 1;
    }
}