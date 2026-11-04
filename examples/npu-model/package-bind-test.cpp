#define NPU_PACKAGE_BIND_TEST
#include "package-bind.cpp"
#include "qwen36_contract.h"

namespace llama_npu_contract = qwen36_contract;
// Existing Qwen integrity regressions explicitly select the Qwen provider.
namespace npu_bind {
const npudsl_model::provider & fixture_provider() { return npudsl_model::providers().resolve(qwen36_contract::abi); }
auto files() { return files(fixture_provider()); }
auto expected_tensors() { return fixture_provider().package.tensors; }
void validate_layers(const std::string & text) {
    fixture_provider().package.validate_metadata({{"layer_types.txt", text}});
}
void validate_manifest(const std::string & text, const std::map<std::string, uint64_t> & sizes) {
    validate_manifest(fixture_provider(), text, sizes);
}
void validate_file_size(const std::string & name, uint64_t size) { validate_file_size(fixture_provider(), name, size); }
auto package_file(const fs::path & root, const std::string & name) { return package_file(fixture_provider(), root, name); }
void validate_package(const fs::path & root) { validate_package(fixture_provider(), root); }
auto identity_text(const fs::path & root, const std::string & hash) { return identity_text(fixture_provider(), root, hash); }
}

namespace {

void check(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

template<class F>
void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception &) { rejected = true; }
    check(rejected, "expected rejection");
}

void write(const npu_bind::fs::path & path, const std::string & text) {
    std::ofstream out(path, std::ios::binary);
    out << text; out.close();
    check(bool(out), "fixture write failed");
}

void test_manifest(const npu_bind::fs::path & directory) {
    using namespace npu_bind;
    // Text-only layout descriptions, never fabricated numerical weight blobs.
    std::map<std::string, uint64_t> sizes;
    std::string manifest;
    for (const auto & file : expected_tensors()) {
        uint64_t end = 0;
        for (const auto & tensor : file.second) {
            const auto offset = (end + 4095) / 4096 * 4096;
            manifest += file.first + " " + tensor.first + " " + std::to_string(offset) + " " + std::to_string(tensor.second) + "\n";
            end = offset + tensor.second;
        }
        sizes[file.first] = end;
    }
    validate_manifest(manifest, sizes);
    std::string crlf_manifest;
    for (char c : manifest) { crlf_manifest += c == '\n' ? "\r\n" : std::string(1, c); }
    validate_manifest(crlf_manifest, sizes);
    rejects([&] { validate_manifest("", sizes); });
    rejects([&] { validate_manifest(manifest + manifest.substr(0, manifest.find('\n') + 1), sizes); });
    rejects([&] { validate_manifest("../outside.bin X 0 1\n", sizes); });
    rejects([&] { validate_manifest("globals.bin X 0 1 extra\n", sizes); });
    const auto first_space = manifest.find(' ', manifest.find(' ') + 1);
    const auto length_space = manifest.find(' ', first_space + 1);
    auto overflow = manifest;
    overflow.replace(first_space + 1, length_space - first_space - 1, "18446744073709551615");
    rejects([&] { validate_manifest(overflow, sizes); });
    auto unaligned = manifest;
    unaligned.replace(first_space + 1, length_space - first_space - 1, "1");
    rejects([&] { validate_manifest(unaligned, sizes); });
    auto wrong_size = sizes;
    --wrong_size.begin()->second;
    rejects([&] { validate_manifest(manifest, wrong_size); });
    ++wrong_size.begin()->second; ++wrong_size.begin()->second;
    rejects([&] { validate_manifest(manifest, wrong_size); });
    check(number("18446744073709551615") == (std::numeric_limits<uint64_t>::max)(), "uint64 boundary");
    for (const std::string text : {"", "-1", "+1", "1x", "18446744073709551616"}) {
        rejects([&] { number(text); });
    }
    std::string layers;
    for (int i = 0; i < 40; ++i) { layers += (i + 1) % 4 ? "linear_attention\n" : "full_attention\n"; }
    validate_layers(layers);
    std::string crlf;
    for (char c : layers) { crlf += c == '\n' ? "\r\n" : std::string(1, c); }
    validate_layers(crlf);
    // Both production callers use this same parser. Binary streams reproduce
    // POSIX getline semantics even when the tests run on Windows.
    const auto lf_file = directory / "layers-lf", crlf_file = directory / "layers-crlf";
    write(lf_file, layers); write(crlf_file, crlf);
    for (const auto & file : {lf_file, crlf_file}) {
        std::ifstream input(file, std::ios::binary);
        llama_npu_contract::validate_layers(input);
        validate_layers(read_text(file));
    }
    check(read_text(crlf_file) == crlf, "CRLF raw bytes changed");
    check(hash_file(lf_file) != hash_file(crlf_file), "line endings normalized before hashing");
    rejects([&] { validate_layers("linear_attention\r\r\n" + layers.substr(layers.find('\n') + 1)); });
    rejects([&] { validate_layers("linear_attention \n" + layers.substr(layers.find('\n') + 1)); });
    rejects([&] { validate_layers("linear_\rattention\n" + layers.substr(layers.find('\n') + 1)); });
    rejects([&] { validate_layers(crlf + "\r\n"); });
    rejects([&] { validate_layers(layers.substr(0, layers.size() - 1) + "\r"); });
    rejects([&] { validate_layers(layers + "linear_attention\n"); });
    rejects([&] { validate_layers("full_attention\n" + layers); });
    rejects([&] { validate_layers(layers.substr(0, layers.find_last_of('\n', layers.size() - 2) + 1)); });
}

void test_tokenizer(const npu_bind::fs::path & directory) {
    using namespace npu_bind;
    gguf_ptr kv(gguf_init_empty(), gguf_free);
    gguf_set_val_str(kv.get(), "general.architecture", "llama");
    gguf_set_val_str(kv.get(), "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(kv.get(), "tokenizer.ggml.pre", "gpt-2");
    const char * tokens[] = {"a", "b", "<|endoftext|>"};
    const float scores[] = {0, 0, 0};
    const int32_t types[] = {1, 1, 3};
    gguf_set_arr_str(kv.get(), "tokenizer.ggml.tokens", tokens, 3);
    gguf_set_arr_str(kv.get(), "tokenizer.ggml.merges", nullptr, 0);
    gguf_set_arr_data(kv.get(), "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores, 3);
    gguf_set_arr_data(kv.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types, 3);
    gguf_set_val_u32(kv.get(), "tokenizer.ggml.bos_token_id", 2);
    gguf_set_val_u32(kv.get(), "tokenizer.ggml.eos_token_id", 2);
    // Explicit baseline values keep these fixtures independent of defaults.
    for (const char * key : {"add_bos_token", "add_eos_token", "add_sep_token", "add_space_prefix",
                             "remove_extra_whitespaces", "clean_spaces", "dna_auto_tags",
                             "normalizer.nfc", "normalizer.lowercase", "normalizer.strip_accents"}) {
        gguf_set_val_bool(kv.get(), (std::string("tokenizer.ggml.") + key).c_str(), false);
    }
    const auto source = directory / "tiny-tokenizer.gguf";
    check(gguf_write_to_file(kv.get(), source.string().c_str(), true), "write tokenizer");
    auto model = load_tokenizer(source);
    const auto digest = fingerprint(model.get());
    check(digest.size() == 64, "fingerprint length");
    // Real vocab-only loads of the same tiny, metadata-only GGUF. No external
    // model/device needed, and no independent fingerprint algorithm in tests.
    auto overridden = [&](const char * key, llama_model_kv_override value) {
        llama_model_kv_override overrides[2] = {};
        overrides[0] = value;
        std::snprintf(overrides[0].key, sizeof(overrides[0].key), "%s", key);
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        params.kv_overrides = overrides;
        model_ptr result(llama_model_load_from_file(source.string().c_str(), params), llama_model_free);
        check(bool(result), "override fixture load failed");
        char before[256] = {}, after[256] = {};
        check(llama_model_meta_val_str(model.get(), key, before, sizeof(before)) ==
              llama_model_meta_val_str(result.get(), key, after, sizeof(after)) &&
              std::string(before) == after, "override unexpectedly changed original metadata");
        return result;
    };
    // Use exported APIs only: the tests must also link with a shared llama DLL.
    struct bool_case { const char * key; bool (*get)(const llama_vocab *); };
    const bool_case flags[] = {
        {"tokenizer.ggml.add_bos_token", llama_vocab_get_add_bos},
        {"tokenizer.ggml.add_eos_token", llama_vocab_get_add_eos},
        {"tokenizer.ggml.add_sep_token", llama_vocab_get_add_sep},
        {"tokenizer.ggml.add_space_prefix", nullptr},
        {"tokenizer.ggml.remove_extra_whitespaces", nullptr},
        {"tokenizer.ggml.clean_spaces", nullptr},
        {"tokenizer.ggml.dna_auto_tags", nullptr},
    };
    const auto * vocab = llama_model_get_vocab(model.get());
    for (const auto & flag : flags) {
        llama_model_kv_override value = {};
        value.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        value.val_bool = true;
        const auto changed = overridden(flag.key, value);
        if (flag.get) {
            check(!flag.get(vocab) && flag.get(llama_model_get_vocab(changed.get())), "effective flag unchanged");
        }
        check(fingerprint(changed.get()) != digest, "effective flag not fingerprinted");
        value.val_bool = !value.val_bool;
        const auto unchanged = overridden(flag.key, value);
        check(fingerprint(unchanged.get()) == digest, "no-op override changed fingerprint");
    }
    struct id_case { const char * key; llama_token (*get)(const llama_vocab *); };
    const id_case ids[] = {
        {"tokenizer.ggml.bos_token_id", llama_vocab_bos},
        {"tokenizer.ggml.eos_token_id", llama_vocab_eos},
        {"tokenizer.ggml.eot_token_id", llama_vocab_eot},
        {"tokenizer.ggml.eom_token_id", nullptr},
        {"tokenizer.ggml.unknown_token_id", nullptr},
        {"tokenizer.ggml.seperator_token_id", llama_vocab_sep},
        {"tokenizer.ggml.padding_token_id", llama_vocab_pad},
        {"tokenizer.ggml.mask_token_id", llama_vocab_mask},
        {"tokenizer.ggml.fim_pre_token_id", llama_vocab_fim_pre},
        {"tokenizer.ggml.fim_suf_token_id", llama_vocab_fim_suf},
        {"tokenizer.ggml.fim_mid_token_id", llama_vocab_fim_mid},
        {"tokenizer.ggml.fim_pad_token_id", llama_vocab_fim_pad},
        {"tokenizer.ggml.fim_rep_token_id", llama_vocab_fim_rep},
        {"tokenizer.ggml.fim_sep_token_id", llama_vocab_fim_sep},
    };
    for (const auto & id : ids) {
        llama_model_kv_override value = {};
        value.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        value.val_i64 = 0;
        const auto changed = overridden(id.key, value);
        if (id.get) {
            check(id.get(vocab) != 0 && id.get(llama_model_get_vocab(changed.get())) == 0, "effective special ID unchanged");
        }
        check(fingerprint(changed.get()) != digest, "effective special ID not fingerprinted");
    }
    for (const char * key : {"tokenizer.ggml.normalizer.nfc", "tokenizer.ggml.normalizer.lowercase",
                             "tokenizer.ggml.normalizer.strip_accents"}) {
        llama_model_kv_override value = {};
        value.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        value.val_bool = true;
        const auto changed = overridden(key, value);
        check(fingerprint(changed.get()) != digest, "effective normalizer not fingerprinted");
    }
    llama_model_kv_override pre = {};
    pre.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
    std::snprintf(pre.val_str, sizeof(pre.val_str), "%s", "llama3");
    const auto changed_pre = overridden("tokenizer.ggml.pre", pre);
        // llama3 selects a different pretype and enables ignore_merges internally.
    check(fingerprint(changed_pre.get()) != digest, "effective pretokenizer not fingerprinted");
        llama_model_kv_override token_types = {};
        token_types.tag = LLAMA_KV_OVERRIDE_TYPE_INT; token_types.val_i64 = 2;
        const auto changed_types = overridden("tokenizer.ggml.token_type_count", token_types);
        check(fingerprint(changed_types.get()) != digest, "effective token type count not fingerprinted");
    auto bos = llama_model_kv_override{};
        bos.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL; bos.val_bool = true;
    const auto changed_bos = overridden("tokenizer.ggml.add_bos_token", bos);
        llama_token before_tokens[4] = {}, after_tokens[4] = {};
        check(llama_tokenize(vocab, "a", 1, before_tokens, 4, true, false) == 1 &&
            llama_tokenize(llama_model_get_vocab(changed_bos.get()), "a", 1, after_tokens, 4, true, false) == 2 &&
            after_tokens[0] == llama_vocab_bos(vocab) && after_tokens[1] == before_tokens[0],
          "BOS override did not change actual tokenization");
    char small[64] = "sentinel";
    check(!llama_npu_tokenizer_sha256(model.get(), small, sizeof(small)) && !small[0], "short buffer");
    char output[65] = "sentinel";
    check(!llama_npu_tokenizer_sha256(nullptr, output, sizeof(output)) && !output[0], "null model");
    check(!llama_npu_tokenizer_sha256(model.get(), nullptr, 65), "null buffer");
    rejects([&] { validate_model(model.get()); });
    gguf_ptr clone(gguf_init_empty(), gguf_free);
    gguf_set_kv(clone.get(), kv.get());
    gguf_set_val_str(clone.get(), "npu.dsl.source_id", "unverified-test-revision");
    const auto copy = directory / "clone.gguf";
    check(gguf_write_to_file(clone.get(), copy.string().c_str(), true), "write clone");
    auto loaded_copy = load_tokenizer(copy);
    check(fingerprint(loaded_copy.get()) == digest, "KV clone changed fingerprint");
    gguf_set_val_bool(clone.get(), "tokenizer.ggml.add_bos_token", true);
    check(gguf_write_to_file(clone.get(), copy.string().c_str(), true), "write changed tokenizer");
    loaded_copy = load_tokenizer(copy);
    check(fingerprint(loaded_copy.get()) != digest, "tokenizer scalar change not fingerprinted");
}

} // namespace

int main() {
    using namespace npu_bind;
    try {
        staged_file scratch(fs::temp_directory_path() / ("npu-bind-test-" + std::to_string(std::random_device{}())));
        const auto root = fs::canonical(scratch.directory);
        test_manifest(root);
        test_tokenizer(root);
        for (const auto & file : files()) {
            const uint64_t limit = file == "manifest.txt" || file == "layer_types.txt" ? 16ull << 20 : 64ull << 30;
            validate_file_size(file, limit);
            rejects([&] { validate_file_size(file, 0); });
            rejects([&] { validate_file_size(file, limit + 1); });
        }
        const char * args[] = {"bind", "--tokenizer", "t", "--weights", "w", "--output", "o", "--source-id", "repo@revision", "--acknowledge-external-weights"};
        check(parse(10, args).source_id == "repo@revision", "parse valid arguments");
        rejects([&] { parse(9, args); });
        args[8] = "bad source";
        rejects([&] { parse(10, args); });
        const auto long_id = std::string(257, 'a');
        args[8] = long_id.c_str();
        rejects([&] { parse(10, args); });
        args[8] = "repo@revision";
        const auto empty = root / "empty"; write(empty, "");
        rejects([&] { package_file(root, "empty"); });
        rejects([&] { validate_package(root); });
        const auto data = root / "hash-input"; write(data, "abc");
        check(hash_file(data) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 known vector");
        rejects([&] { identity_text(root, std::string(64, 'G')); });
        write(root / "manifest.txt", "invalid assets, not numerical weights");
        fs::resize_file(root / "manifest.txt", (16ull << 20) + 1);
        rejects([&] { package_file(root, "manifest.txt"); });
        require_new(root / "output");
        {
            staged_file first(root / "output"), second(root / "identity.txt");
            write(first.temporary, "metadata"); write(second.temporary, "identity");
            second.publish();
            write(first.destination, "racing writer");
            rejects([&] { first.publish(); });
        }
        check(read_text(root / "output") == "racing writer", "no-overwrite publication");
        check(!fs::exists(root / "identity.txt"), "failed publication did not roll back identity");
        rejects([&] { require_new(root / "output"); });
        options opts = parse(10, args);
        opts.tokenizer = (root / "tiny-tokenizer.gguf").string();
        opts.weights = root.string(); opts.output = (root / "new.gguf").string();
        write(root / "identity.txt", "existing identity");
        rejects([&] { bind(opts); });
        check(!fs::exists(opts.output) && read_text(root / "identity.txt") == "existing identity", "existing identity modified");
        fs::remove(root / "identity.txt");
        rejects([&] { bind(opts); });
        check(!fs::exists(opts.output) && !fs::exists(root / "identity.txt"), "invalid assets mutated outputs");
        std::cout << "npu-package-bind contract tests passed (no numerical package or device execution)\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "npu-package-bind tests: " << e.what() << "\n";
        return 1;
    }
}