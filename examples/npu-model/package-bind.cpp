#include "llama.h"
#include "gguf.h"
#include "model_api.h"

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npu_bind {
namespace fs = std::filesystem;

struct options {
    std::string tokenizer, weights, output, source_id;
    bool help = false;
};

options parse(int argc, const char * const * argv) {
    options result;
    if (argc == 2 && std::string(argv[1]) == "--help") { result.help = true; return result; }
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key != "--tokenizer" && key != "--weights" && key != "--output" &&
            key != "--source-id" && key != "--acknowledge-external-weights") {
            throw std::invalid_argument("unknown argument: " + key + " (--help must be used alone)");
        }
        if (!seen.insert(key).second) { throw std::invalid_argument("duplicate argument: " + key); }
        if (key == "--acknowledge-external-weights") { continue; }
        if (++i == argc || !*argv[i]) { throw std::invalid_argument("missing value for " + key); }
        if (key == "--tokenizer") { result.tokenizer = argv[i]; }
        if (key == "--weights") { result.weights = argv[i]; }
        if (key == "--output") { result.output = argv[i]; }
        if (key == "--source-id") { result.source_id = argv[i]; }
    }
    if (result.tokenizer.empty() || result.weights.empty() || result.output.empty() || result.source_id.empty()) {
        throw std::invalid_argument("required: --tokenizer --weights --output --source-id");
    }
    if (!seen.count("--acknowledge-external-weights")) {
        throw std::invalid_argument("--acknowledge-external-weights is required: caller must establish checkpoint equivalence");
    }
    if (result.source_id.size() > 256 || result.source_id.find_first_not_of(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-/:@+") != std::string::npos) {
        throw std::invalid_argument("--source-id must be 1..256 ASCII characters from A-Z a-z 0-9 . _ - / : @ +");
    }
    return result;
}

void require_new(const fs::path & path) {
    // symlink_status also rejects dangling symlinks.
    if (fs::exists(fs::symlink_status(path))) {
        throw std::runtime_error("refusing existing output: " + path.string());
    }
}

std::vector<std::string> files(const npudsl_model::provider & provider) {
    std::vector<std::string> result;
    for (const auto & file : provider.assets.files) { result.push_back(file.name); }
    return result;
}

void validate_file_size(const npudsl_model::provider & provider, const std::string & name, uint64_t length) {
    for (const auto & file : provider.assets.files) {
        if (file.name == name && length && length <= file.max_bytes) { return; }
    }
    throw std::runtime_error("invalid package file or size: " + name);
}

fs::path package_file(const npudsl_model::provider & provider, const fs::path & root, const std::string & name) {
    const auto path = fs::canonical(root / name);
    if (path.parent_path() != root || !fs::is_regular_file(path)) {
        throw std::runtime_error("package file escapes directory or is not regular: " + name);
    }
    validate_file_size(provider, name, fs::file_size(path));
    return path;
}

std::string read_text(const fs::path & path) {
    const auto size = fs::file_size(path);
    if (!size || size > 16ull << 20) { throw std::runtime_error("invalid text file size"); }
    std::string result(static_cast<size_t>(size), '\0');
    std::ifstream input(path, std::ios::binary);
    if (!input.read(&result[0], result.size()) || input.peek() != EOF) {
        throw std::runtime_error("cannot read metadata: " + path.string());
    }
    return result;
}

uint64_t number(const std::string & text) {
    uint64_t result = 0;
    if (text.empty()) { throw std::runtime_error("empty manifest integer"); }
    for (char c : text) {
        if (c < '0' || c > '9' || result > ((std::numeric_limits<uint64_t>::max)() - (c - '0')) / 10) {
            throw std::runtime_error("invalid or overflowing manifest integer");
        }
        result = result * 10 + (c - '0');
    }
    return result;
}

void validate_manifest(const npudsl_model::provider & provider, const std::string & text,
                       const std::map<std::string, uint64_t> & sizes) {
    npudsl_model::validate_manifest_contract(provider, text, sizes);
}

npudsl_model::execution_format validate_package(const npudsl_model::provider & provider, const fs::path & root) {
    return npudsl_model::prepare_assets(provider, npudsl_model::directory_assets(provider.assets, root.string()))->format();
}

struct hash {
    sha256_t state;
    hash() { sha256_init(&state); }
    void bytes(const void * data, size_t size) { sha256_update(&state, static_cast<const unsigned char *>(data), size); }
    std::string finish() {
        unsigned char digest[32]; sha256_final(&state, digest);
        static const char hex[] = "0123456789abcdef";
        std::string result;
        for (auto c : digest) { result += hex[c >> 4]; result += hex[c & 15]; }
        return result;
    }
};

std::string hash_file(const fs::path & path) {
    auto remaining = fs::file_size(path);
    std::ifstream input(path, std::ios::binary);
    std::vector<char> buffer(1024 * 1024);
    hash content;
    while (remaining) {
        const auto count = static_cast<std::streamsize>((std::min<uint64_t>)(remaining, buffer.size()));
        if (!input.read(buffer.data(), count)) { throw std::runtime_error("package read failed: " + path.string()); }
        content.bytes(buffer.data(), static_cast<size_t>(count));
        remaining -= count;
    }
    if (input.peek() != EOF) { throw std::runtime_error("package changed during hashing"); }
    return content.finish();
}

std::string identity_text(const npudsl_model::provider & provider, const fs::path & root, const std::string & tokenizer) {
    if (tokenizer.size() != 64 || tokenizer.find_first_not_of("0123456789abcdef") != std::string::npos) {
        throw std::runtime_error("invalid tokenizer SHA-256");
    }
    std::string text = provider.assets.identity_header + "\ntokenizer " + tokenizer + "\n";
    for (const auto & file : files(provider)) { text += hash_file(package_file(provider, root, file)) + " " + file + "\n"; }
    if (text.size() > 8192) { throw std::runtime_error("identity exceeds contract bound"); }
    return text;
}

// Hard-link publication is atomic and never replaces an existing destination, including on POSIX.
// Each staging directory is on its destination filesystem. Failure rolls back only our own files.
struct staged_file {
    fs::path destination, directory, temporary;
    bool published = false, committed = false;
    explicit staged_file(const fs::path & path) : destination(path) {
        require_new(destination);
        std::random_device random;
        for (int i = 0; i < 32; ++i) {
            const auto candidate = path.parent_path() / (".npu-bind-" + std::to_string(random()) + "-" + std::to_string(random()));
            if (fs::create_directory(candidate)) {
                directory = candidate;
                temporary = directory / "data";
                return;
            }
        }
        throw std::runtime_error("cannot reserve staging directory");
    }
    staged_file(const staged_file &) = delete;
    staged_file & operator=(const staged_file &) = delete;
    ~staged_file() {
        std::error_code error;
        if (published && !committed) { fs::remove(destination, error); }
        fs::remove_all(directory, error);
    }
    void publish() { fs::create_hard_link(temporary, destination); published = true; }
};

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

model_ptr load_tokenizer(const fs::path & path) {
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    model_ptr model(llama_model_load_from_file(path.string().c_str(), params), llama_model_free);
    if (!model) { throw std::runtime_error("cannot load tokenizer GGUF"); }
    return model;
}

std::string fingerprint(const llama_model * model) {
    char digest[65];
    if (!llama_npu_tokenizer_sha256(model, digest, sizeof(digest))) {
        throw std::runtime_error("tokenizer fingerprint unavailable; complete-model NPU bridge must be enabled");
    }
    return digest;
}

npudsl_model::metadata model_metadata(const llama_model * model) {
    return [model](const char * key) -> std::optional<std::string> {
        char value[257];
        const auto size = llama_model_meta_val_str(model, key, value, sizeof(value));
        if (size == -1) { return std::nullopt; }
        if (size < -1 || size > 256) { throw std::runtime_error(std::string("oversized or unreadable metadata: ") + key); }
        return std::string(value, size);
    };
}
const npudsl_model::provider & validate_model(const llama_model * model, const npudsl_model::package_probe & probe = {}) {
    const auto get = model_metadata(model);
    const auto vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model));
    return npudsl_model::providers().resolve_model(get, vocabulary, probe);
}

void bind(const options & opts) {
    const auto root = fs::canonical(opts.weights);
    const auto source = fs::canonical(opts.tokenizer);
    const auto requested = fs::absolute(opts.output).lexically_normal();
    const auto output = fs::canonical(requested.parent_path()) / requested.filename();
    const auto identity = root / "identity.txt";
    require_new(output);
    require_new(identity);
    if (output == identity) { throw std::runtime_error("output must not be identity.txt"); }
    auto model = load_tokenizer(source);
    const auto & provider = validate_model(model.get(), [&](const npudsl_model::provider & p) { return validate_package(p, root); });
    const auto format = validate_package(provider, root);
    const auto tokenizer = fingerprint(model.get());
    const auto text = identity_text(provider, root, tokenizer);
    hash package_hash; package_hash.bytes(text.data(), text.size());
    const auto package_id = package_hash.finish();

    gguf_ptr input(gguf_init_from_file(source.string().c_str(), {true, nullptr}), gguf_free);
    gguf_ptr metadata(gguf_init_empty(), gguf_free);
    if (!input || !metadata) { throw std::runtime_error("cannot read GGUF metadata"); }
    gguf_set_kv(metadata.get(), input.get());
    // The result is a standalone tokenizer, not a shard of the original tensor GGUF.
    for (const char * key : {"split.no", "split.count", "split.tensors.count"}) { gguf_remove_key(metadata.get(), key); }
    gguf_set_val_str(metadata.get(), "npu.dsl.abi", provider.abi.c_str());
    gguf_set_val_str(metadata.get(), "npu.dsl.package_id", package_id.c_str());
    gguf_set_val_str(metadata.get(), "npu.dsl.tokenizer_sha256", tokenizer.c_str());
    gguf_set_val_str(metadata.get(), "npu.dsl.source_id", opts.source_id.c_str());
    // Never derive format from provider selection or original GGUF quantization.
    // This evidence came from the validated external execution package.
    gguf_set_val_str(metadata.get(), "npu.dsl.format.weights", format.weights.c_str());
    gguf_set_val_str(metadata.get(), "npu.dsl.format.scales", format.scales.c_str());
    gguf_set_val_str(metadata.get(), "npu.dsl.format.dense", format.dense.c_str());
    gguf_set_val_str(metadata.get(), "npu.dsl.format.head", format.head.c_str());
    gguf_set_val_str(metadata.get(), "npu.dsl.format.layout", format.layout.c_str());
    gguf_set_val_u32(metadata.get(), "npu.dsl.format.group_size", format.group_size);

    staged_file out(output), id(identity);
    if (!gguf_write_to_file(metadata.get(), out.temporary.string().c_str(), true)) { throw std::runtime_error("cannot write output GGUF"); }
    {
        std::ofstream stream(id.temporary, std::ios::binary);
        stream.write(text.data(), text.size()); stream.close();
        if (!stream) { throw std::runtime_error("cannot write identity.txt"); }
    }
    auto reloaded = load_tokenizer(out.temporary);
    validate_model(reloaded.get());
    if (fingerprint(reloaded.get()) != tokenizer) { throw std::runtime_error("output tokenizer fingerprint changed"); }
    // Recheck structure before publication. Keep the source package immutable throughout binding and use.
    validate_package(provider, root);
    require_new(output); require_new(identity);
    id.publish();
    out.publish();
    id.committed = out.committed = true;
    std::cerr << "Bound metadata-only GGUF: " << output << "\nPackage ID: " << package_id
              << "\nTokenizer SHA-256: " << tokenizer << "\nSource ID (caller assertion): " << opts.source_id << "\n";
}

} // namespace npu_bind

#ifndef NPU_PACKAGE_BIND_TEST
int main(int argc, char ** argv) {
    npu_bind::options opts;
    try { opts = npu_bind::parse(argc, argv); }
    catch (const std::invalid_argument & e) {
        std::cerr << "npu-package-bind: " << e.what() << "\n";
        return 2;
    }
    try {
        if (opts.help) {
            std::cout << "npu-package-bind --tokenizer <source-matching GGUF> --weights <packed dir>\n"
                         "  --output <NEW metadata-only GGUF> --source-id <checkpoint revision ID>\n"
                         "  --acknowledge-external-weights\n"
                         "Does not transform GGUF weights or establish checkpoint equivalence.\n"
                         "The caller must independently verify that tokenizer and packed weights match.\n"
                         "source-id is unverified provenance, not part of the v1 identity contract.\n";
            return 0;
        }
        std::cerr << "Binding does not transform GGUF weights or establish checkpoint equivalence.\n"
                     "The caller is responsible for matching external weights and tokenizer to the stated source revision.\n";
        npu_bind::bind(opts);
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "npu-package-bind: " << e.what() << "\n";
        return 1;
    }
}
#endif