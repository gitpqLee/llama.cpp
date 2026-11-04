// ggml-npu end to end through the public llama API, with the DSL runner on its mock NPU runtime.
//
// A real llama architecture (qwen35moe, tiny dimensions, random weights) loads onto the NPU device with no llama
// change. The model package is generated from the model's own tensor list -- the backend and runner know nothing of
// it. A mock "kernel" computes logits from the weights it reaches through the layer table and the global params, so
// the test checks that every GGUF tensor arrived in its slot, that token ids and positions are read from the graph,
// and that the logits reach llama_get_logits. NOT numerics: the mock is not the model.
//
//   test-npu-backend             positive run
//   test-npu-backend --missing   the package omits output.weight: llama keeps it on the CPU, the graph splits,
//                                and decoding must fail instead of returning CPU logits of an unrun graph
int test_llama_archs_main(int argc, char ** argv);
#define main test_llama_archs_main
#include "test-llama-archs.cpp"
#undef main

#include "ggml-npu.h"

#include <npu_runtime.hpp>   // the mock runtime: mock_npu::on_submit stands in for the kernels

#include <fstream>
#include <map>
#include <sstream>

namespace {

int failures = 0;

void check(bool ok, const std::string & what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct recorded {
    ggml_type type;
    std::vector<int64_t> ne;
    std::vector<uint8_t> bytes;
};

std::map<std::string, recorded> g_tensors;
std::set<std::string> g_referenced;   // weights the model's graph actually reads

bool collect_weights(ggml_tensor * t, bool ask, void *) {
    if (ask) {
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (t->src[s] && g_tensors.count(t->src[s]->name)) {
                g_referenced.insert(t->src[s]->name);
            }
        }
    }
    return false;
}

// test-llama-archs' deterministic filler, also recording what each tensor holds
void record_tensor_data(ggml_tensor * tensor, void * userdata) {
    const auto & params = *static_cast<const tensor_data_params *>(userdata);
    tensor_filler fill = fill_normal;
    for (const auto & f : TENSOR_FILLERS) {
        if (strstr(tensor->name, f.match)) {
            fill = f.fill;
            break;
        }
    }
    std::vector<float> tmp(ggml_nelements(tensor));
    fill(tmp, tensor->name, params);
    recorded r{tensor->type, {}, std::vector<uint8_t>(ggml_nbytes(tensor))};
    for (int i = 0; i < ggml_n_dims(tensor); ++i) {
        r.ne.push_back(tensor->ne[i]);
    }
    if (tensor->type == GGML_TYPE_F32) {
        memcpy(r.bytes.data(), tmp.data(), r.bytes.size());
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(tmp.size());
        for (size_t i = 0; i < tmp.size(); ++i) {
            h[i] = ggml_fp32_to_fp16(tmp[i]);
        }
        memcpy(r.bytes.data(), h.data(), r.bytes.size());
    } else {
        GGML_ABORT("unexpected tensor type");
    }
    ggml_backend_tensor_set(tensor, r.bytes.data(), 0, r.bytes.size());
    g_tensors[tensor->name] = std::move(r);
}

llama_model_ptr load(gguf_context * gguf, const std::vector<ggml_backend_dev_t> & devs) {
    auto mp = llama_model_default_params();
    mp.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> d = devs;
    d.push_back(nullptr);
    mp.devices = d.data();
    static tensor_data_params params = {42, 0.1f};
    llama_model_ptr model(llama_model_init_from_user(gguf, record_tensor_data, &params, mp));
    if (!model) {
        throw std::runtime_error("model load failed");
    }
    return model;
}

// ---- the model package, written from the model's tensor list ----

struct package {
    std::string path;
    std::vector<std::string> globals;                  // in decode param order
    std::vector<std::string> columns;                  // table column names
    std::vector<int> kind_of_layer;
    std::vector<std::set<std::string>> kinds;
    uint32_t n_layers = 0, n_vocab = 0, n_embd = 0;
    size_t decode_params = 0, prefill_params = 0;
};

std::string json_list(const std::vector<int64_t> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        s += (i ? ", " : "") + std::to_string(v[i]);
    }
    return s + "]";
}

package write_package(const std::string & path, bool omit_output) {
    package p;
    p.path = path;
    std::map<int, std::set<std::string>> per_layer;
    for (const auto & t : g_tensors) {
        if (!g_referenced.count(t.first)) {
            continue;   // in the GGUF but unused by the graph (e.g. optional activation scales)
        }
        int layer = -1;
        char rest[256];
        if (sscanf(t.first.c_str(), "blk.%d.%255s", &layer, rest) == 2) {
            per_layer[layer].insert(rest);
        } else if (!(omit_output && t.first == "output.weight")) {
            p.globals.push_back(t.first);
        }
    }
    p.n_layers = uint32_t(per_layer.size());
    p.n_vocab = uint32_t(g_tensors.at("token_embd.weight").ne[1]);
    p.n_embd = uint32_t(g_tensors.at("token_embd.weight").ne[0]);
    std::set<std::string> all;
    for (const auto & l : per_layer) {
        auto it = std::find(p.kinds.begin(), p.kinds.end(), l.second);
        if (it == p.kinds.end()) {
            p.kinds.push_back(l.second);
            it = p.kinds.end() - 1;
        }
        p.kind_of_layer.push_back(int(it - p.kinds.begin()));
        all.insert(l.second.begin(), l.second.end());
    }
    p.columns.assign(all.begin(), all.end());

    auto bytes = [](const recorded & r) { return std::to_string(r.bytes.size()); };
    auto type = [](const recorded & r) { return std::string(ggml_type_name(r.type)); };
    auto layer_tensor = [&](const std::string & col) -> const recorded & {
        for (size_t l = 0; l < p.n_layers; ++l) {
            const auto it = g_tensors.find("blk." + std::to_string(l) + "." + col);
            if (it != g_tensors.end()) {
                return it->second;
            }
        }
        throw std::logic_error(col);
    };
    std::ostringstream j;
    j << R"({"format": "npudsl-model", "version": 1, "name": "test-qwen35moe", "arch": "qwen35moe",
 "target": {"device_key": "5010", "tiles": 3, "shaves_per_tile": 2, "workers": 6},
 "status": {"slot_bytes": 32, "profile_bytes": 0, "ok": 1, "cmx_error": 3},
 "limits": {"vocab": )" << p.n_vocab << R"(, "max_seq": 256, "max_tokens": 64},
 "table": {"header_words": 2, "rows": )" << p.n_layers << R"(, "columns": [)";
    for (size_t c = 0; c < p.columns.size(); ++c) {
        const auto & r = layer_tensor(p.columns[c]);
        j << (c ? ", " : "") << R"({"name": ")" << p.columns[c] << R"(", "role": "weight", "bytes": )" << bytes(r) << "}";
    }
    j << "]},\n \"kinds\": [";
    for (size_t k = 0; k < p.kinds.size(); ++k) {
        j << (k ? ", " : "") << R"({"name": "k)" << k << R"(", "id": )" << k << R"(, "columns": [)";
        size_t i = 0;
        for (const auto & c : p.kinds[k]) {
            j << (i++ ? ", " : "") << '"' << c << '"';
        }
        j << "]}";
    }
    j << "],\n \"layers\": [";
    for (size_t l = 0; l < p.n_layers; ++l) {
        j << (l ? ", " : "") << "\"k" << p.kind_of_layer[l] << '"';
    }
    const uint64_t table_bytes = uint64_t(p.n_layers) * (2 + 2 * p.columns.size()) * 4;
    auto globals = [&] {
        std::string s;
        for (const auto & g : p.globals) {
            s += R"(, {"name": ")" + g + R"(", "role": "global", "bytes": )" + bytes(g_tensors.at(g)) + "}";
        }
        return s;
    };
    const std::string logits = R"(, {"name": "Logits", "role": "io", "bind": "logits", "dtype": "f32", "bytes": )" +
                               std::to_string(uint64_t(p.n_vocab) * 4) + R"(}, {"name": "TokenOut", "role": "io", "bind": "token_out", "dtype": "u32", "bytes": 4})";
    j << "],\n \"kernels\": {\"decode\": {\"name\": \"test_decode\", \"config_bytes\": 64, \"config\": ["
      << R"({"name": "n_layers", "offset": 0, "bytes": 4, "bind": "n_layers"}, {"name": "pos", "offset": 4, "bytes": 4, "bind": "position"}, {"name": "token", "offset": 8, "bytes": 4, "bind": "token"}],)"
      << R"( "params": [{"name": "LTab", "role": "table", "bytes": )" << table_bytes << "}" << globals() << logits << "]},\n"
      << R"(  "prefill": {"name": "test_prefill", "config_bytes": 64, "config": [{"name": "n_layers", "offset": 0, "bytes": 4, "bind": "n_layers"}, {"name": "n", "offset": 4, "bytes": 4, "bind": "n_tokens"}, {"name": "pos0", "offset": 8, "bytes": 4, "bind": "position"}],)"
      << R"( "params": [{"name": "LTab", "role": "table", "bytes": )" << table_bytes
      << R"(}, {"name": "Ids", "role": "io", "bind": "tokens", "dtype": "u32", "bytes": 1024})" << globals() << logits << "]}},\n";
    j << R"( "sources": {"format": "gguf", "arch": "qwen35moe", "tokens": "token_embd.weight", "logits": "result_output", "positions": "rope", "tensors": [)";
    size_t n = 0;
    for (const auto & g : p.globals) {
        const auto & r = g_tensors.at(g);
        j << (n++ ? ",\n  " : "\n  ") << R"({"src": ")" << g << R"(", "dst": ")" << g << R"(", "types": [")" << type(r)
          << R"("], "ne": )" << json_list(r.ne) << R"(, "xform": "copy"})";
    }
    for (const auto & c : p.columns) {
        const auto & r = layer_tensor(c);
        j << ",\n  " << R"({"src": "blk.{L}.)" << c << R"(", "dst": ")" << c << R"(", "types": [")" << type(r)
          << R"("], "ne": )" << json_list(r.ne) << R"(, "xform": "copy"})";
    }
    j << "]}}\n";
    std::ofstream(path) << j.str();
    p.decode_params = 1 + 1 + p.globals.size() + 2 + 1;
    p.prefill_params = p.decode_params + 1;
    return p;
}

// ---- the mock kernels ----

float first_value(const void * data, ggml_type type) {
    if (type == GGML_TYPE_F32) {
        return *static_cast<const float *>(data);
    }
    return ggml_fp16_to_fp32(*static_cast<const ggml_fp16_t *>(data));
}

float embedding(const void * table, ggml_type type, uint32_t token, uint32_t n_embd) {
    const size_t elem = type == GGML_TYPE_F32 ? 4 : 2;
    return first_value(static_cast<const uint8_t *>(table) + size_t(token) * n_embd * elem, type);
}

// logits[v] = embedding(last token)[0] * 100 + sum over layers of column 0 [0] + position of the last token + v/1000
struct kernels {
    explicit kernels(const package & pkg) : p(pkg) {}
    const package & p;
    size_t launches = 0;
    std::vector<std::pair<uint32_t, uint32_t>> calls;   // (n_tokens, first position)
    std::vector<uint32_t> last_ids;

    void operator()(const std::vector<void *> & params) {
        ++launches;
        const bool decode = params.size() == p.decode_params;
        const auto * cfg = static_cast<const uint32_t *>(params[0]);
        const auto * table = static_cast<const uint32_t *>(params[1]);
        const size_t first_global = decode ? 2 : 3;
        uint32_t token, pos, n;
        if (decode) {
            token = cfg[2];
            pos = cfg[1];
            n = 1;
            last_ids = {token};
        } else {
            const auto * ids = static_cast<const uint32_t *>(params[2]);
            n = cfg[1];
            pos = cfg[2] + n - 1;
            token = ids[n - 1];
            last_ids.assign(ids, ids + n);
        }
        calls.push_back({n, decode ? cfg[1] : cfg[2]});
        const auto g = std::find(p.globals.begin(), p.globals.end(), "token_embd.weight") - p.globals.begin();
        float value = 100.0f * embedding(params[first_global + size_t(g)], g_tensors.at("token_embd.weight").type, token, p.n_embd);
        const size_t words = 2 + 2 * p.columns.size();
        for (uint32_t l = 0; l < cfg[0]; ++l) {
            const auto * row = table + l * words;
            for (size_t c = 0; c < p.columns.size(); ++c) {
                const uint64_t a = row[2 + 2 * c] | (uint64_t(row[3 + 2 * c]) << 32);
                if (a) {
                    value += first_value(reinterpret_cast<const void *>(a), g_tensors.at("blk." + std::to_string(l) + "." + p.columns[c]).type);
                    break;
                }
            }
        }
        auto * logits = static_cast<float *>(params[params.size() - 3]);
        for (uint32_t v = 0; v < p.n_vocab; ++v) {
            logits[v] = value + float(pos) + float(v) / 1000.0f;
        }
        *static_cast<uint32_t *>(params[params.size() - 2]) = p.n_vocab - 1;
    }
};

// the same function from the recorded GGUF data
std::vector<float> expected(const package & p, uint32_t token, uint32_t pos) {
    const auto & emb = g_tensors.at("token_embd.weight");
    float value = 100.0f * embedding(emb.bytes.data(), emb.type, token, p.n_embd);
    for (uint32_t l = 0; l < p.n_layers; ++l) {
        for (const auto & c : p.columns) {
            const auto it = g_tensors.find("blk." + std::to_string(l) + "." + c);
            if (it != g_tensors.end()) {
                value += first_value(it->second.bytes.data(), it->second.type);
                break;
            }
        }
    }
    std::vector<float> out(p.n_vocab);
    for (uint32_t v = 0; v < p.n_vocab; ++v) {
        out[v] = value + float(pos) + float(v) / 1000.0f;
    }
    return out;
}

int decode(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos pos0) {
    llama_batch batch = llama_batch_init(int32_t(tokens.size()), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos0 + llama_pos(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i + 1 == tokens.size();
    }
    batch.n_tokens = int32_t(tokens.size());
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc;
}

bool same(const float * a, const std::vector<float> & b) {
    if (!a) {
        fprintf(stderr, "  no logits\n");
        return false;
    }
    for (size_t i = 0; i < b.size(); ++i) {
        if (a[i] != b[i]) {
            fprintf(stderr, "  logit %zu: %f, expected %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int run(bool missing) {
    ggml_backend_dev_t npu = ggml_backend_dev_by_name("NPU");
    check(npu != nullptr, "the NPU device is registered (mock runtime)");
    if (!npu) {
        return 1;
    }
    init_normal_pool(42, 0.1f);
    auto gguf = get_gguf_ctx(LLM_ARCH_QWEN35MOE, true);

    // 1. the reference run on the CPU records the tensors the model is made of and which of them its graph reads
    const std::vector<llama_token> prompt = get_tokens(10, 128, 7);
    {
        auto cpu_model = load(gguf.get(), {});
        auto cp = llama_context_default_params();
        cp.n_ctx = 256;
        cp.cb_eval = collect_weights;
        llama_context_ptr cpu_ctx(llama_init_from_model(cpu_model.get(), cp));
        check(cpu_ctx && decode(cpu_ctx.get(), prompt, 0) == 0, "CPU reference decode");
        check(g_referenced.count("token_embd.weight") && g_referenced.count("output.weight"), "the graph reads the embedding and the head");
    }
    const auto path = (std::filesystem::temp_directory_path() / (missing ? "test-npu-missing.json" : "test-npu-model.json")).string();
    const package p = write_package(path, missing);
    check(ggml_backend_npu_add_model(path.c_str()), "the generated package is accepted");

    kernels k(p);
    mock_npu::on_submit = std::ref(k);

    // 2. the same model on the NPU device
    auto model = load(gguf.get(), {npu});
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 128;
    cp.n_ubatch = 64;
    cp.n_seq_max = 1;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    check(ctx != nullptr, "context on the NPU");
    if (!ctx) {
        return 1;
    }

    if (missing) {
        check(decode(ctx.get(), prompt, 0) != 0 && k.launches == 0, "a split graph (output.weight on the CPU) fails before any launch");
        return failures ? 1 : 0;
    }

    check(decode(ctx.get(), prompt, 0) == 0, "prefill of 10 tokens");
    check(k.calls.size() == 1 && k.calls.back() == std::make_pair(10u, 0u), "one prefill launch: 10 tokens at 0");
    check(std::equal(prompt.begin(), prompt.end(), k.last_ids.begin()), "token ids read from the graph");
    check(same(llama_get_logits_ith(ctx.get(), -1), expected(p, uint32_t(prompt.back()), 9)), "prefill logits through llama");

    llama_pos pos = 10;
    for (llama_token t : {3, 17, 99}) {
        check(decode(ctx.get(), {t}, pos) == 0, "decode at " + std::to_string(pos));
        check(k.calls.back() == std::make_pair(1u, uint32_t(pos)), "decode launch position " + std::to_string(pos));
        check(same(llama_get_logits_ith(ctx.get(), -1), expected(p, uint32_t(t), uint32_t(pos))), "decode logits at " + std::to_string(pos));
        ++pos;
    }

    // a prompt longer than n_ubatch: two graph splits of one batch, positions continue
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    const auto launches = k.calls.size();
    const std::vector<llama_token> longer = get_tokens(100, p.n_vocab, 9);
    check(decode(ctx.get(), longer, 0) == 0, "restart at 0 with a 100-token prompt");
    check(k.calls.size() == launches + 2 && k.calls[launches] == std::make_pair(64u, 0u) && k.calls[launches + 1] == std::make_pair(36u, 64u),
          "two ubatches: 64 tokens at 0, 36 at 64 (the restart reset the kernels' state)");
    check(same(llama_get_logits_ith(ctx.get(), -1), expected(p, uint32_t(longer.back()), 99)), "logits of the last ubatch");


    mock_npu::on_submit = nullptr;
    return failures ? 1 : 0;
}

} // namespace

int main(int argc, char ** argv) {
    common_init();
    llama_backend_init();
    const bool missing = argc > 1 && std::string(argv[1]) == "--missing";
    int rc = 1;
    try {
        rc = run(missing);
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
    }
    llama_backend_free();
    if (rc == 0) {
        printf("test-npu-backend%s: all checks passed\n", missing ? " --missing" : "");
    }
    return rc;
}
