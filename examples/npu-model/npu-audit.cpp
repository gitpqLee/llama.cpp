#include "llama.h"
#include "options.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
namespace fs = std::filesystem;
constexpr uint32_t max_tokens = 4096;
constexpr uint64_t max_binary_bytes = 512ull << 20;
using file_ptr = std::unique_ptr<FILE, decltype(&std::fclose)>;

struct options {
    std::string model, weights, tokens, output, reference;
    uint32_t prompt = 0;
    bool help = false;
};

void usage() {
    std::puts("Usage: llama-npu-audit --model BOUND.gguf --weights PACKAGE --tokens IDS.txt\n"
              "                       --prompt-count N --output PREFIX [--reference REF_PREFIX]\n"
              "IDS.txt: whitespace-separated decimal uint32 IDs, prompt followed by teacher-forced IDs.\n"
              "Limits: 1..4096 total IDs, 1..total prompt IDs, paths <=240 bytes, <=512 MiB per binary.\n"
              "One prompt llama_decode, then one llama_decode per remaining ID; last-only full logits.\n"
              "Writes PREFIX.meta.txt, .audit.txt, .logits.f32, .replay.f32, .sequential.f32, and .split.f32 (prompt>2).\n"
              "reset_replay: bit-exact. split_continuation/independent_reference: golden criteria.\n"
              "Golden: per-row relL2 mean<=0.04, max<=0.10; exact argmax OR reference chosen-token gap<=0.5.\n"
              "No tolerance option. Nonfinite logits always fail; bit/argmax differences are always reported.\n"
              "Optional independent reference: REF_PREFIX.meta.txt (exact schema/identity/tokens) and\n"
              "REF_PREFIX.logits.f32 (little-endian float32 rows). No inference subprocess is launched.\n"
              "Without --reference, success validates state only, NOT independent model numerics.\n"
              "Existing artifacts are never overwritten. Partial artifacts remain on failure.\n"
              "Exit: 0 completed/help, 2 arguments/input, 1 runtime/numerical/state/output failure.");
}

options parse(int argc, char ** argv) {
    options o;
    if (argc == 2 && std::string(argv[1]) == "--help") { o.help = true; return o; }
    std::unordered_set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key != "--model" && key != "--weights" && key != "--tokens" &&
            key != "--prompt-count" && key != "--output" && key != "--reference") {
            throw std::invalid_argument("unknown argument: " + key);
        }
        if (!seen.insert(key).second) { throw std::invalid_argument("duplicate argument: " + key); }
        if (++i == argc) { throw std::invalid_argument("missing value: " + key); }
        const std::string value = argv[i];
        if (value.empty() || value.size() > 240) { throw std::invalid_argument("empty/oversized value: " + key); }
        if (key == "--model") { o.model = value; }
        if (key == "--weights") { o.weights = value; }
        if (key == "--tokens") { o.tokens = value; }
        if (key == "--output") { o.output = value; }
        if (key == "--reference") { o.reference = value; }
        if (key == "--prompt-count") { o.prompt = npu_model::number(value, max_tokens, key); }
    }
    if (o.model.empty() || o.weights.empty() || o.tokens.empty() || o.output.empty() || !o.prompt) {
        throw std::invalid_argument("required: --model --weights --tokens --prompt-count (1..4096) --output");
    }
    return o;
}

std::string read_text(const fs::path & path, uint64_t limit) {
    if (!fs::is_regular_file(path) || fs::file_size(path) > limit) {
        throw std::invalid_argument("missing/oversized input: " + path.string());
    }
    std::ifstream in(path, std::ios::binary);
    std::string text(static_cast<size_t>(fs::file_size(path)), '\0');
    if (!in.read(text.data(), static_cast<std::streamsize>(text.size())) || in.peek() != EOF) {
        throw std::invalid_argument("cannot read input: " + path.string());
    }
    return text;
}

bool space(unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f'; }

std::vector<llama_token> read_tokens(const options & o) {
    const auto text = read_text(o.tokens, 65536);
    std::vector<llama_token> tokens;
    size_t i = 0;
    while (i < text.size()) {
        if (space(text[i])) { ++i; continue; }
        const auto start = i;
        while (i < text.size() && !space(text[i])) { ++i; }
        if (i - start > 10 || tokens.size() == max_tokens) { throw std::invalid_argument("oversized token ID/sequence"); }
        const auto id = npu_model::number(text.substr(start, i - start), UINT32_MAX, "token ID");
        if (id > static_cast<uint32_t>((std::numeric_limits<llama_token>::max)())) {
            throw std::invalid_argument("token ID exceeds llama_token range");
        }
        tokens.push_back(static_cast<llama_token>(id));
    }
    if (tokens.size() < o.prompt) { throw std::invalid_argument("prompt-count exceeds input token count"); }
    return tokens;
}

file_ptr exclusive(const std::string & path) {
    file_ptr file(std::fopen(path.c_str(), "wbx"), std::fclose);
    if (!file) { throw std::runtime_error("cannot exclusively create output: " + path); }
    return file;
}

void write(FILE * file, const void * data, size_t bytes) {
    if (std::fwrite(data, 1, bytes, file) != bytes || std::fflush(file)) {
        throw std::runtime_error("artifact write/flush failed");
    }
}

void close_file(file_ptr & file) {
    if (file && std::fclose(file.release())) { throw std::runtime_error("artifact close failed"); }
}

void log(FILE * file, const std::string & message) {
    const auto line = message + '\n';
    write(file, line.data(), line.size());
    std::fputs(line.c_str(), stderr);
}

struct backend_guard {
    backend_guard() { llama_backend_init(); }
    ~backend_guard() { llama_backend_free(); }
};

std::string metadata(llama_model * model, const std::vector<llama_token> & tokens, uint32_t prompt, size_t vocab) {
    std::ostringstream text;
    text << "llama-npu-audit-v1\ndtype float32\nendian little\nlayout row-major\nrows "
         << tokens.size() - prompt + 1 << "\nvocab " << vocab << "\nprompt_count " << prompt << '\n';
    for (const char * key : {"npu.dsl.abi", "npu.dsl.package_id", "npu.dsl.tokenizer_sha256"}) {
        char value[257] = {};
        const auto n = llama_model_meta_val_str(model, key, value, sizeof(value));
        if (n <= 0 || n > 256 || std::strchr(value, '\n') || std::strchr(value, '\r')) {
            throw std::runtime_error(std::string("missing/invalid binding metadata: ") + key);
        }
        text << key << ' ' << value << '\n';
    }
    text << "tokens";
    for (auto token : tokens) { text << ' ' << token; }
    text << "\nrow_end_positions";
    for (size_t pos = prompt - 1; pos < tokens.size(); ++pos) { text << ' ' << pos; }
    text << "\n";
    return text.str();
}

void position(llama_context * ctx, size_t next) {
    const auto mem = llama_get_memory(ctx);
    if (!mem || llama_memory_seq_pos_max(mem, 0) != static_cast<llama_pos>(next) - 1 ||
        llama_memory_seq_pos_min(mem, 0) != (next ? 0 : -1)) {
        throw std::runtime_error("public memory position mismatch at " + std::to_string(next));
    }
}

const float * decode(llama_context * ctx, std::vector<llama_token> & tokens, size_t start, size_t count) {
    position(ctx, start);
    std::vector<llama_pos> positions(count);
    std::vector<int8_t> outputs(count, 0);
    for (size_t i = 0; i < count; ++i) { positions[i] = static_cast<llama_pos>(start + i); }
    outputs.back() = 1;
    llama_batch batch = {};
    batch.n_tokens = static_cast<int32_t>(count);
    batch.token = tokens.data() + start;
    batch.pos = positions.data();
    batch.logits = outputs.data();
    const int rc = llama_decode(ctx, batch);
    if (rc) { throw std::runtime_error("llama_decode failed at " + std::to_string(start) + ": " + std::to_string(rc)); }
    position(ctx, start + count);
    const auto * row = llama_get_logits_ith(ctx, -1);
    if (!row || row != llama_get_logits_ith(ctx, static_cast<int32_t>(count - 1)) ||
        llama_get_logits_ith(ctx, -2)) { throw std::runtime_error("invalid last-only logit mapping"); }
    for (size_t i = 0; i + 1 < count; ++i) {
        if (llama_get_logits_ith(ctx, static_cast<int32_t>(i))) { throw std::runtime_error("unexpected earlier logits"); }
    }
    return row;
}

size_t argmax(const float * row, size_t vocab) {
    if (!vocab) { throw std::runtime_error("empty logit row"); }
    size_t best = 0;
    for (size_t i = 0; i < vocab; ++i) {
        if (!std::isfinite(row[i])) { throw std::runtime_error("nonfinite logit at vocabulary index " + std::to_string(i)); }
        if (row[i] > row[best]) { best = i; }
    }
    return best;
}

struct differences {
    // Fixed audit policy, not CLI options. The original golden_check.py defines
        // TAU=.5, MEAN_TOL=.04, MAX_TOL=.10; near ties use the chosen-token gap.
    static constexpr double tau = 0.5, mean_tol = 0.04, max_tol = 0.10;
    uint64_t rows = 0, mismatches = 0, mismatched_bits = 0, argmax_mismatches = 0;
    uint64_t near_ties = 0, far_argmax_mismatches = 0;
    double max_abs = 0, max_rel = 0, rel_l2_sum = 0, rel_l2_max = 0;

    struct row_difference {
        double rel_l2, reference_top2_margin;
        bool argmax_equal;
    };

    row_difference compare(const float * actual, const float * expected, size_t vocab) {
        // Validate both entire rows before accumulating. First index wins ties.
        const auto actual_best = argmax(actual, vocab);
        const auto expected_best = argmax(expected, vocab);
        double second = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < vocab; ++i) {
            if (i != expected_best) { second = std::max(second, double(expected[i])); }
        }
        const double margin = double(expected[expected_best]) - second;
        const double chosen_gap = double(expected[expected_best]) - double(expected[actual_best]);
        const bool equal = actual_best == expected_best;
        argmax_mismatches += !equal;
        near_ties += !equal && chosen_gap <= tau;
        far_argmax_mismatches += !equal && chosen_gap > tau;
        double delta_squared = 0, reference_squared = 0;
        for (size_t i = 0; i < vocab; ++i) {
            uint32_t a, b;
            std::memcpy(&a, actual + i, sizeof(a));
            std::memcpy(&b, expected + i, sizeof(b));
            uint32_t bits = a ^ b;
            mismatches += bits != 0;
            while (bits) { ++mismatched_bits; bits &= bits - 1; }
            const double delta = std::abs(double(actual[i]) - double(expected[i]));
            max_abs = std::max(max_abs, delta);
            max_rel = std::max(max_rel, delta / std::max(std::abs(double(expected[i])), 1e-30));
            // Finite float32 squares (including subnormals) and all bounded audit
            // rows fit in double. Never subtract/square/sum in float32.
            delta_squared += delta * delta;
            reference_squared += double(expected[i]) * double(expected[i]);
        }
        const double rel = reference_squared > 0 ? std::sqrt(delta_squared) / std::sqrt(reference_squared) :
                           delta_squared == 0 ? 0 : std::numeric_limits<double>::infinity();
        ++rows;
        rel_l2_sum += rel;
        rel_l2_max = std::max(rel_l2_max, rel);
        return {rel, margin, equal};
    }

    bool report(FILE * file, const char * label) const {
        // Labels are fixed internal policy keys, not caller-configurable knobs.
        const bool bitexact = std::strcmp(label, "reset_replay") == 0;
        if (!bitexact && std::strcmp(label, "split_continuation") != 0 &&
            std::strcmp(label, "sequential_prompt") != 0 &&
            std::strcmp(label, "independent_reference") != 0) {
            throw std::runtime_error("unknown audit comparison policy");
        }
        const double mean = rows ? rel_l2_sum / double(rows) : std::numeric_limits<double>::infinity();
        const bool golden = rows && std::isfinite(mean) && std::isfinite(rel_l2_max) &&
                            mean <= mean_tol && rel_l2_max <= max_tol && !far_argmax_mismatches;
        const bool passed = bitexact ? rows && !mismatches : golden;
        std::ostringstream text;
        text << std::setprecision(17) << label
                         << " criterion=" << (bitexact ? "bitexact" : "golden_relL2_mean<=0.04_max<=0.10_and_(argmax_exact_OR_reference_chosen_gap<=0.5)")
             << " rows=" << rows << " mismatched_elements=" << mismatches << " mismatched_bits=" << mismatched_bits
             << " argmax_mismatches=" << argmax_mismatches << " max_abs=" << max_abs << " max_rel=" << max_rel;
        text << " rel_l2_mean=" << mean << " rel_l2_max=" << rel_l2_max
             << " near_ties=" << near_ties << " far_argmax_mismatches=" << far_argmax_mismatches
             << " result=" << (passed ? "PASS" : "FAIL");
        log(file, text.str());
        return passed;
    }
};

void report_row(FILE * file, const char * label, size_t index, const differences::row_difference & d) {
    std::ostringstream text;
    text << std::setprecision(17) << label << " comparison_row=" << index << " rel_l2=" << d.rel_l2
         << " argmax_equal=" << d.argmax_equal << " reference_top2_margin=" << d.reference_top2_margin;
    log(file, text.str());
}

void read_row(std::ifstream & file, std::vector<float> & row) {
    if (!file.read(reinterpret_cast<char *>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)))) {
        throw std::runtime_error("reference/baseline row read failed");
    }
}

int run(const options & o, std::vector<llama_token> tokens) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE float32 required");
    const uint32_t endian = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian) != 1) { throw std::runtime_error("little-endian host required"); }
    std::vector<std::string> suffixes = {".audit.txt", ".meta.txt", ".logits.f32", ".replay.f32"};
    if (o.prompt > 2) { suffixes.push_back(".split.f32"); }
    suffixes.push_back(".sequential.f32");
    for (const auto & suffix : suffixes) {
        if (fs::exists(fs::symlink_status(o.output + suffix))) {
            throw std::runtime_error("output already exists: " + o.output + suffix);
        }
    }
    auto audit = exclusive(o.output + suffixes[0]);
    try {
        auto meta = exclusive(o.output + suffixes[1]);
        auto baseline = exclusive(o.output + suffixes[2]);
        auto replay = exclusive(o.output + suffixes[3]);
        file_ptr split(nullptr, std::fclose);
        if (o.prompt > 2) { split = exclusive(o.output + suffixes[4]); }
        auto sequential = exclusive(o.output + ".sequential.f32");
        log(audit.get(), "START real llama_decode full-model NPU audit; no CPU fallback");
        log(audit.get(), "POLICY reset_replay=bitexact; split/sequential/independent_reference=golden: "
                         "per-row relL2 mean<=0.04 max<=0.10 AND (argmax exact OR reference chosen-token gap<=0.5); "
                 "finite required; no tolerance knob; all bit/argmax differences reported");
        log(audit.get(), "model=" + o.model + " weights=" + o.weights + " tokens=" + o.tokens);
        backend_guard backend;
        auto mp = llama_model_default_params();
        mp.vocab_only = true;
        mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(o.model.c_str(), mp), llama_model_free);
        if (!model) { throw std::runtime_error("bound GGUF load failed"); }
        const auto vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        if (vocab <= 0 || uint64_t(tokens.size() - o.prompt + 1) * uint64_t(vocab) * sizeof(float) > max_binary_bytes) {
            throw std::runtime_error("invalid vocabulary or output exceeds 512 MiB per binary");
        }
        for (auto token : tokens) {
            if (token >= vocab) { throw std::runtime_error("token ID outside model vocabulary"); }
        }
        const auto description = metadata(model.get(), tokens, o.prompt, vocab);
        write(meta.get(), description.data(), description.size());
        close_file(meta);
        std::ifstream reference;
        if (!o.reference.empty()) {
            if (read_text(o.reference + ".meta.txt", 131072) != description) {
                throw std::runtime_error("independent reference metadata/identity/token sequence mismatch");
            }
            const auto path = o.reference + ".logits.f32";
            const uint64_t bytes = uint64_t(tokens.size() - o.prompt + 1) * vocab * sizeof(float);
            if (!fs::is_regular_file(path) || fs::file_size(path) != bytes) {
                throw std::runtime_error("independent reference binary size mismatch");
            }
            reference.open(path, std::ios::binary);
            if (!reference) { throw std::runtime_error("cannot open independent reference"); }
            log(audit.get(), "reference=" + o.reference + "; caller must ensure independent same-checkpoint provenance");
        } else {
            log(audit.get(), "REFERENCE_NOT_SUPPLIED: independent numerical correctness is NOT validated");
        }
        auto cp = llama_context_default_params();
        cp.npu_model_path = o.weights.c_str();
        cp.n_ctx = static_cast<uint32_t>(tokens.size());
        cp.n_seq_max = 1;
        cp.n_batch = o.prompt;
        cp.n_ubatch = o.prompt;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) { throw std::runtime_error("real NPU context construction failed; no retry/fallback"); }
        bool passed = true;
        std::vector<float> expected(vocab);
        differences independent;
        for (int pass = 0; pass < 4; ++pass) {
            if (pass == 2 && o.prompt <= 2) { continue; }
            const char * label = pass == 0 ? "baseline" : pass == 1 ? "reset_replay" :
                                pass == 2 ? "split_continuation" : "sequential_prompt";
            if (pass) {
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                position(ctx.get(), 0);
            }
            std::ifstream original;
            if (pass) {
                original.open(o.output + ".logits.f32", std::ios::binary);
                if (!original) { throw std::runtime_error("cannot reopen baseline"); }
            }
            FILE * output = pass == 0 ? baseline.get() : pass == 1 ? replay.get() :
                            pass == 2 ? split.get() : sequential.get();
            differences state;
            size_t row_index = 0;
            if (pass == 3) {
                for (size_t i = 0; i + 1 < o.prompt; ++i) {
                    argmax(decode(ctx.get(), tokens, i, 1), vocab);
                }
            }
            for (size_t start = pass == 3 ? o.prompt - 1 : 0; start < tokens.size();) {
                size_t count = start == 0 ? o.prompt : 1;
                if (pass == 2 && start == 0) {
                    const size_t first = o.prompt / 2;
                    const auto * discarded = decode(ctx.get(), tokens, 0, first);
                    argmax(discarded, vocab);
                    start = first;
                    count = o.prompt - first;
                    log(audit.get(), "split_boundary=" + std::to_string(first));
                }
                const auto * row = decode(ctx.get(), tokens, start, count);
                const auto best = argmax(row, vocab);
                write(output, row, static_cast<size_t>(vocab) * sizeof(float));
                if (pass) {
                    read_row(original, expected);
                    report_row(audit.get(), label, row_index, state.compare(row, expected.data(), vocab));
                } else if (reference.is_open()) {
                    read_row(reference, expected);
                    report_row(audit.get(), "independent_reference", row_index,
                               independent.compare(row, expected.data(), vocab));
                }
                log(audit.get(), std::string(label) + " row=" + std::to_string(row_index++) +
                    " end_position=" + std::to_string(start + count - 1) + " finite=1 argmax=" + std::to_string(best));
                start += count;
            }
            if (pass) {
                if (original.peek() != EOF) { throw std::runtime_error("baseline has trailing bytes"); }
                passed = state.report(audit.get(), label) && passed;
            } else {
                close_file(baseline);
                if (reference.is_open()) {
                    if (reference.peek() != EOF) { throw std::runtime_error("reference has trailing bytes"); }
                    passed = independent.report(audit.get(), "independent_reference") && passed;
                }
            }
        }
        if (o.prompt <= 2) { log(audit.get(), "split_continuation SKIPPED: prompt_count<=2"); }
        close_file(replay);
        close_file(split);
        close_file(sequential);
        if (!passed) { throw std::runtime_error("full-logit audit comparison failed; see each reported fixed policy criterion/result"); }
        log(audit.get(), o.reference.empty() ? "STATE_PASS; INDEPENDENT_NUMERICS_NOT_TESTED" : "PASS state and supplied independent reference");
        close_file(audit);
        return 0;
    } catch (const std::exception & e) {
        if (audit) { log(audit.get(), std::string("FAIL: ") + e.what()); }
        throw;
    }
}
} // namespace

#ifndef NPU_AUDIT_TEST
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    try {
        const auto o = parse(argc, argv);
        if (o.help) { usage(); return 0; }
        auto tokens = read_tokens(o);
        if (!fs::is_regular_file(o.model) || !fs::is_directory(o.weights)) {
            throw std::invalid_argument("--model must be a regular GGUF; --weights must be a package directory");
        }
        return run(o, std::move(tokens));
    } catch (const std::invalid_argument & e) {
        std::fprintf(stderr, "llama-npu-audit: argument/input error: %s\n", e.what());
        return 2;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "llama-npu-audit: %s\n", e.what());
        return 1;
    }
}
#endif