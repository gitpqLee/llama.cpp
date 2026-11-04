#include "llama-npu.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef LLAMA_NPU_MODEL
#include "model_api.h"
extern "C" {
#include "hash/sha256/sha256.h"
}

namespace {

struct npu_hash {
    sha256_t state;
    npu_hash() { sha256_init(&state); }
    void bytes(const void * data, size_t n) {
        sha256_update(&state, static_cast<const unsigned char *>(data), n);
    }
    void integer(uint64_t n) {
        unsigned char le[8];
        for (int i = 0; i < 8; ++i) { le[i] = n & 255; n >>= 8; }
        bytes(le, sizeof(le));
    }
    void field(const std::string & s) { integer(s.size()); bytes(s.data(), s.size()); }
    std::string finish() {
        unsigned char digest[32];
        sha256_final(&state, digest);
        static const char hex[] = "0123456789abcdef";
        std::string result;
        for (auto c : digest) { result += hex[c >> 4]; result += hex[c & 15]; }
        return result;
    }
};

std::string npu_meta(const llama_model & model, const char * key) {
    const auto it = model.gguf_kv.find(key);
    if (it == model.gguf_kv.end() || it->second.size() > 256) {
        throw std::runtime_error(std::string("NPU: missing or oversized metadata: ") + key);
    }
    return it->second;
}

bool npu_digest(const std::string & s) {
    return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos;
}

// v2 fingerprints original metadata AND effective vocabulary settings after
// loader defaults/kv_overrides. Keep the binder on this same implementation.
// Fields are length-prefixed bytes; every integer is little-endian u64.
std::string npu_tokenizer_hash(const llama_model & model) {
    npu_hash hash;
    hash.field("llama-npu-tokenizer-v2");
    std::map<std::string, std::string> scalars;
    for (const auto & kv : model.gguf_kv) {
        if (kv.first.compare(0, 10, "tokenizer.") == 0) { scalars.emplace(kv); }
    }
    hash.integer(scalars.size());
    for (const auto & kv : scalars) { hash.field(kv.first); hash.field(kv.second); }
    const auto & vocab = model.vocab;
    hash.field(vocab.get_tokenizer_model());
    hash.field(vocab.get_tokenizer_pre());
    hash.integer(vocab.get_type());
    hash.integer(vocab.get_pre_type());
    hash.integer(vocab.n_token_types());
    hash.integer(vocab.max_token_len());
    // prefix/middle/suffix are aliases of fim_pre/fim_mid/fim_suf.
    for (auto id : {vocab.token_bos(), vocab.token_eos(), vocab.token_eot(), vocab.token_eom(),
                    vocab.token_unk(), vocab.token_sep(), vocab.token_nl(), vocab.token_pad(),
                    vocab.token_mask(), vocab.token_fim_pre(), vocab.token_fim_suf(),
                    vocab.token_fim_mid(), vocab.token_fim_pad(), vocab.token_fim_rep(), vocab.token_fim_sep()}) {
        hash.integer(static_cast<uint32_t>(id));
    }
    for (bool flag : {vocab.get_add_space_prefix(), vocab.get_add_bos(), vocab.get_add_eos(),
                      vocab.get_add_sep(), vocab.get_dna_auto_tags(), vocab.get_ignore_merges(),
                      vocab.get_clean_spaces(), vocab.get_remove_extra_whitespaces(),
                      vocab.get_escape_whitespaces(), vocab.get_treat_whitespace_as_suffix()}) {
        hash.integer(flag);
    }
    const auto & normalizer = vocab.get_normalizer_opts();
    hash.integer(normalizer.nfc);
    hash.integer(normalizer.lowercase);
    hash.integer(normalizer.strip_accents);
    hash.integer(vocab.n_tokens());
    for (uint32_t i = 0; i < vocab.n_tokens(); ++i) {
        const auto & token = vocab.get_token_data(i);
        uint32_t score;
        static_assert(sizeof(score) == sizeof(token.score), "float32 required");
        std::memcpy(&score, &token.score, sizeof(score));
        hash.field(token.text); hash.integer(score); hash.integer(token.attr);
        hash.integer(vocab.is_eog(i));
    }
    const auto merges = vocab.get_bpe_merges();
    hash.integer(merges.size());
    for (const auto & merge : merges) { hash.field(merge); }
    const auto charsmap = vocab.get_precompiled_charsmap();
    hash.integer(charsmap.size()); hash.bytes(charsmap.data(), charsmap.size());
    const auto & suppress = vocab.get_suppress_tokens();
    hash.integer(suppress.size());
    for (auto id : suppress) { hash.integer(static_cast<uint32_t>(id)); }
    return hash.finish();
}

std::filesystem::path npu_file(const std::filesystem::path & root, const std::string & name) {
    auto file = std::filesystem::canonical(root / name);
    if (file.parent_path() != root || !std::filesystem::is_regular_file(file)) {
        throw std::runtime_error("NPU: package file escapes directory: " + name);
    }
    return file;
}

npudsl_model::prepared_handle npu_verify_package(const llama_model & model, const char * directory,
                        const npudsl_model::provider & provider) {
    const auto & schema = provider.assets;
    namespace fs = std::filesystem;
    const auto root = fs::canonical(directory);
    const auto identity = npu_file(root, "identity.txt");
    const auto size = fs::file_size(identity);
    if (size == 0 || size > 8192) { throw std::runtime_error("NPU: invalid identity.txt size"); }
    std::string text(static_cast<size_t>(size), '\0');
    std::ifstream input(identity, std::ios::binary);
    if (!input.read(&text[0], text.size()) || input.peek() != EOF) {
        throw std::runtime_error("NPU: cannot read identity.txt");
    }
    npu_hash hash;
    hash.bytes(text.data(), text.size());
    const auto package = npu_meta(model, "npu.dsl.package_id");
    if (!npu_digest(package) || hash.finish() != package) {
        throw std::runtime_error("NPU: package identity mismatch");
    }
    const auto tokenizer = npu_meta(model, "npu.dsl.tokenizer_sha256");
    if (!npu_digest(tokenizer) || npu_tokenizer_hash(model) != tokenizer) {
        throw std::runtime_error("NPU: tokenizer content mismatch");
    }
    // Exact, ordered manifest. No user-controlled file names or unbounded line parsing.
    std::istringstream lines(text);
    std::string line;
    if (!std::getline(lines, line) || line != schema.identity_header ||
        !std::getline(lines, line) || line != "tokenizer " + tokenizer) {
        throw std::runtime_error("NPU: unsupported identity contract");
    }
    std::vector<char> buffer(1024 * 1024);
    std::map<std::string, uint64_t> sizes;
    for (const auto & entry : schema.files) {
        const auto & name = entry.name;
        if (!std::getline(lines, line) || line.size() != 65 + name.size() ||
            line.substr(64) != " " + name || !npu_digest(line.substr(0, 64))) {
            throw std::runtime_error("NPU: malformed identity file entry: " + name);
        }
        const auto file = npu_file(root, name);
        const auto length = fs::file_size(file);
        sizes[name] = length;
        if (!length || length > entry.max_bytes) { throw std::runtime_error("NPU: invalid file size: " + name); }
        std::ifstream data(file, std::ios::binary);
        npu_hash content;
        uint64_t remaining = length;
        while (remaining) {
            const auto count = static_cast<std::streamsize>(std::min<uint64_t>(remaining, buffer.size()));
            if (!data.read(buffer.data(), count)) { throw std::runtime_error("NPU: package read failed: " + name); }
            content.bytes(buffer.data(), static_cast<size_t>(count)); remaining -= count;
        }
        if (data.peek() != EOF || content.finish() != line.substr(0, 64)) {
            throw std::runtime_error("NPU: weight/metadata hash mismatch: " + name);
        }
    }
    if (std::getline(lines, line) || text.back() != '\n') {
        throw std::runtime_error("NPU: trailing data or missing newline in identity.txt");
    }
    return npudsl_model::prepare_assets(provider, npudsl_model::directory_assets(schema, root.string()),
        {package, tokenizer, true});
}

const npudsl_model::provider & npu_validate(const llama_model & model, const llama_context_params & p,
                                         npudsl_model::prepared_handle & assets, npudsl_model::session_options & request) {
    if (int(p.type_k) < 0 || int(p.type_k) >= GGML_TYPE_COUNT ||
        int(p.type_v) < 0 || int(p.type_v) >= GGML_TYPE_COUNT) {
        throw std::runtime_error("NPU: invalid K/V cache type enum");
    }
    if (!model.hparams.vocab_only) { throw std::runtime_error("NPU: GGUF must be loaded with vocab_only=true"); }
    const auto & provider = npudsl_model::providers().resolve(npu_meta(model, "npu.dsl.abi"));
    const npudsl_model::metadata get = [&](const char * key) -> std::optional<std::string> {
        const auto it = model.gguf_kv.find(key);
        if (it == model.gguf_kv.end()) { return std::nullopt; }
        return it->second;
    };
    // Reject known metadata and request mismatches before streaming large assets.
    // Missing format is allowed only after successful validated legacy parsing.
    auto declared = npudsl_model::validate_model_contract(provider, get, model.vocab.n_tokens());
    const char * format_fields[] = {"weights", "scales", "dense", "head", "layout", "group_size"};
    const bool absent_format = std::all_of(std::begin(format_fields), std::end(format_fields), [&](const char * field) {
        return model.gguf_kv.count(std::string("npu.dsl.format.") + field) == 0;
    });
    if (absent_format) {
        declared.errors.erase(std::remove_if(declared.errors.begin(), declared.errors.end(), [](const npudsl_model::diagnostic & e) {
            return e.field.compare(0, 15, "npu.dsl.format.") == 0 && e.actual == "<missing>";
        }), declared.errors.end());
    }
    declared.require();
    request.context = p.n_ctx; request.batch = p.n_batch; request.sequences = p.n_seq_max;
    // Match the public logical/physical normalization; excessive raw logical
    // requests still reject before hashing. A large ubatch is merely clamped.
    request.physical_batch = (std::min)(p.n_batch, p.n_ubatch ? p.n_ubatch : p.n_batch);
    request.cache_k = p.type_k == GGML_TYPE_F16 ? "f16" : ggml_type_name(p.type_k);
    request.cache_v = p.type_v == GGML_TYPE_F16 ? "f16" : ggml_type_name(p.type_v);
    request = npudsl_model::normalize_options(provider, request);
    const auto defaults = llama_context_default_params();
    if (!p.npu_model_path || !*p.npu_model_path || p.n_seq_max != 1 || p.n_rs_seq != 0 ||
        !p.n_batch ||
        p.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || p.embeddings || p.ctx_other ||
        (p.attention_type != LLAMA_ATTENTION_TYPE_UNSPECIFIED && p.attention_type != LLAMA_ATTENTION_TYPE_CAUSAL) ||
        (p.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && p.pooling_type != LLAMA_POOLING_TYPE_NONE) ||
        p.rope_scaling_type != defaults.rope_scaling_type || p.rope_freq_base != defaults.rope_freq_base ||
        p.rope_freq_scale != defaults.rope_freq_scale || p.yarn_ext_factor != defaults.yarn_ext_factor ||
        p.yarn_attn_factor != defaults.yarn_attn_factor || p.yarn_beta_fast != defaults.yarn_beta_fast ||
        p.yarn_beta_slow != defaults.yarn_beta_slow || p.yarn_orig_ctx != defaults.yarn_orig_ctx ||
        p.flash_attn_type != defaults.flash_attn_type || p.moe_cache_size ||
        p.cb_eval || p.abort_callback || p.samplers || p.n_samplers) {
        throw std::runtime_error("NPU: unsupported context options for fixed model ABI");
    }
    assets = npu_verify_package(model, p.npu_model_path, provider);
    npudsl_model::validate_model_contract(provider, get, model.vocab.n_tokens(), &assets->format()).require();
    return provider;
}

} // namespace

struct llama_npu::impl {
    npudsl_model::model_info info;
    std::unique_ptr<npudsl_model::session> session;
    impl(const npudsl_model::provider & provider, npudsl_model::prepared_handle assets, const npudsl_model::session_options & options)
        : info(provider.info), session(npudsl_model::create_session(provider, std::move(assets), options)) {
        const auto & c = session->configuration();
        info.context = c.options.context;
        LLAMA_LOG_INFO("NPU: effective context=%u batch=%u physical=%u allocated_context=%u fixed_allocation=%d state=append/reset\n",
            c.options.context, c.options.batch, c.options.physical_batch, c.allocated_context, int(c.fixed_allocation));
    }
};
#else
struct llama_npu::impl {};
#endif

bool llama_npu_tokenizer_sha256(const llama_model * model, char * output, size_t size) {
    if (output && size) { output[0] = '\0'; }
#ifdef LLAMA_NPU_MODEL
    if (!model || !output || size < 65) { return false; }
    try {
        const auto digest = npu_tokenizer_hash(*model);
        std::memcpy(output, digest.c_str(), 65);
        return true;
    } catch (...) { return false; }
#else
    (void) model;
    return false;
#endif
}

llama_npu::llama_npu(const llama_model & model, const llama_context_params & params) {
#ifdef LLAMA_NPU_MODEL
    npudsl_model::prepared_handle assets;
    npudsl_model::session_options options;
    const auto & provider = npu_validate(model, params, assets, options);
    pimpl = std::make_unique<impl>(provider, std::move(assets), options);
#else
    (void) model; (void) params;
    throw std::runtime_error("NPU: model bridge requires GGML_NPU=ON and NPU_DSL_ROOT");
#endif
}

llama_npu::~llama_npu() = default;

uint32_t llama_npu::vocabulary_size() const {
#ifdef LLAMA_NPU_MODEL
    return pimpl->info.vocabulary;
#else
    return 0;
#endif
}
uint32_t llama_npu::context_size() const {
#ifdef LLAMA_NPU_MODEL
    return pimpl->info.context;
#else
    return 0;
#endif
}
uint32_t llama_npu::layer_count() const {
#ifdef LLAMA_NPU_MODEL
    return pimpl->info.layers;
#else
    return 0;
#endif
}

size_t llama_npu::position() const {
#ifdef LLAMA_NPU_MODEL
    return pimpl->session->position();
#else
    return 0;
#endif
}

bool llama_npu::usable() const {
#ifdef LLAMA_NPU_MODEL
    return !rejected && !pimpl->session->poisoned();
#else
    return false;
#endif
}

void llama_npu::evaluate(const llama_batch_ext & batch, uint32_t n_ctx, float * logits) {
    if (!usable() || batch.tokens.empty() || !batch.embd.empty() || !batch.state.empty() || batch.n_embd ||
        position() > n_ctx || batch.tokens.size() > n_ctx - position()) {
        throw std::runtime_error("NPU: invalid batch, context limit, or unusable session");
    }
    std::vector<uint32_t> tokens;
    tokens.reserve(batch.tokens.size());
    for (size_t i = 0; i < batch.tokens.size(); ++i) {
        const auto & t = batch.tokens[i];
        if (t.has_embd || t.has_state || t.decision_order || t.id < 0 || uint32_t(t.id) >= vocabulary_size() ||
            t.pos[0] != static_cast<llama_pos>(position() + i) || t.pos[1] || t.pos[2] || t.pos[3] ||
            t.seq_ids.size() != 1 || !t.seq_ids.count(0) || (t.output && i + 1 != batch.tokens.size())) {
            throw std::runtime_error("NPU: only sequential tokens, seq0, and last-only logits are supported");
        }
        tokens.push_back(t.id);
    }
#ifdef LLAMA_NPU_MODEL
    const auto & config = pimpl->session->configuration();
    if (n_ctx != config.options.context || tokens.size() > config.options.batch || tokens.size() > config.options.physical_batch ||
        !(config.state & npudsl_model::state_append)) { throw std::runtime_error("NPU: effective session limits/state mismatch"); }
    pimpl->session->evaluate(tokens.data(), tokens.size(), position(), logits, vocabulary_size());
    const auto counts = pimpl->session->submissions();
    LLAMA_LOG_INFO("NPU: complete-model evaluate tokens=%zu position=%zu submissions prefill=%llu decode=%llu kernels=%llu\n",
                   tokens.size(), position(),
                   static_cast<unsigned long long>(counts.prefill),
                   static_cast<unsigned long long>(counts.decode),
                   static_cast<unsigned long long>(counts.kernels));
#else
    (void) logits;
#endif
}

void llama_npu::reject(const char * operation) {
    rejected = true;
    LLAMA_LOG_ERROR("NPU: unsupported memory operation: %s; context disabled\n", operation);
}

void llama_npu::clear(bool) {
#ifdef LLAMA_NPU_MODEL
    if (!(pimpl->session->configuration().state & npudsl_model::state_reset)) { reject("reset unsupported"); return; }
    try { pimpl->session->reset(); }
    catch (const std::exception & e) { reject(e.what()); }
#endif
}

bool llama_npu::seq_rm(llama_seq_id seq, llama_pos p0, llama_pos p1) {
    if ((seq != -1 && seq != 0) || p0 > 0 || (p1 >= 0 && size_t(p1) < position())) { return false; }
    clear(true);
    return usable() && position() == 0;
}

void llama_npu::seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) { reject("seq_cp"); }
void llama_npu::seq_keep(llama_seq_id) { reject("seq_keep"); }
void llama_npu::seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) { reject("seq_add"); }
void llama_npu::seq_div(llama_seq_id, llama_pos, llama_pos, int) { reject("seq_div"); }
llama_pos llama_npu::seq_pos_min(llama_seq_id seq) const { return seq == 0 && position() ? 0 : -1; }
llama_pos llama_npu::seq_pos_max(llama_seq_id seq) const { return seq == 0 && position() ? llama_pos(position() - 1) : -1; }
llama_memory_context_ptr llama_npu::init_batch(llama_batch_allocr &, uint32_t, bool) { reject("init_batch"); return {}; }
llama_memory_context_ptr llama_npu::init_full() { reject("init_full"); return {}; }
llama_memory_context_ptr llama_npu::init_update(llama_context *, bool) { reject("init_update"); return {}; }
void llama_npu::state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const {
    throw std::runtime_error("NPU: state save is not supported");
}
void llama_npu::state_read(llama_io_read_i &, llama_seq_id, llama_state_seq_flags) {
    throw std::runtime_error("NPU: state load is not supported");
}