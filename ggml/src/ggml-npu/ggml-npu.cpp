// Intel NPU backend: whole-model DSL kernels behind the standard ggml backend interface.
//
// The backend knows no model. A model package (model.json, libraries.npu.dsl model_runner) lists the GGUF tensors
// it consumes and the graph conventions it relies on (`sources`). Then:
//   load   weights stay where llama loads them on the host (mmap'd GGUF); this backend never stores them.
//   split  supports_op claims an op when its weights are sources of a package consistent with every weight claimed
//          so far, and the weightless ops once such a package exists. Ops of a model no package lists stay on the
//          other backends; this backend's buffers are host memory, so the CPU runs them in place (KV included).
//   run    graph_compute finds the package whose every source the graph references, converts those tensors into
//          the package layout through the runner (again whenever their data changes), reads the token ids (the
//          row lookup of the package's token source) and the positions (rope), evaluates the whole model in one
//          kernel launch and writes the last token's logits into the package's logits node. The split's other
//          nodes are not executed.
// The package kernels own the model state. A sequence restarting at position 0 resets it; any other jump fails.
// One NPU model per process: the binding follows the most recently computed graph.
#include "ggml-npu.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "npudsl_runner.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

namespace {

// ---------------------------------------------------------------------------------------------------------------
// device state

struct npu_package {
    std::string path;
    npudsl_model * model = nullptr;
    npudsl_model_info info = {};
    ~npu_package() { npudsl_model_free(model); }
};

struct npu_device_context {
    std::recursive_mutex mutex;
    bool probed = false, present = false;
    std::string key, description;
    uint32_t tiles = 0;

    bool catalog_loaded = false;
    std::vector<std::string> added;   // ggml_backend_npu_add_model paths
    std::vector<std::unique_ptr<npu_package>> catalog;
    // what loading the catalog found, logged again by every backend init: the first load can happen while llama
    // silences logging (its memory-fit dry run)
    std::vector<std::pair<ggml_log_level, std::string>> catalog_log;
    std::vector<bool> consistent;     // per package: accepts every weight claimed so far
    bool matched = false;             // the last weight queried belongs to a consistent package
    std::string rejected;             // why the first rejected weight was refused (bind's error names it)

    const npu_package * bound = nullptr;
    npudsl_runner * runner = nullptr;
    std::map<std::string, const void *> imported;   // source name -> data it was converted from
    std::vector<float> logits;

    void unbind() {
        if (runner) {
            npudsl_runner_free(runner);
        }
        runner = nullptr;
        bound = nullptr;
        imported.clear();
    }
};

npu_device_context & device_context() {
    static npu_device_context ctx;
    return ctx;
}

void probe(npu_device_context & ctx) {
    if (ctx.probed) {
        return;
    }
    ctx.probed = true;
    char key[64];
    char err[512];
    ctx.present = npudsl_device_probe(key, sizeof(key), &ctx.tiles, err, sizeof(err)) == NPUDSL_OK;
    if (!ctx.present) {
        GGML_LOG_DEBUG("%s: no NPU: %s\n", __func__, err);
        return;
    }
    ctx.key = key;
    ctx.description = "Intel NPU " + ctx.key + " (" + std::to_string(ctx.tiles) + " tiles), whole-model DSL kernels";
}

void note(npu_device_context & ctx, ggml_log_level level, const std::string & message) {
    ctx.catalog_log.emplace_back(level, message);
    ggml_log_internal(level, "%s\n", message.c_str());
}

bool load_package(npu_device_context & ctx, const std::string & path) {
    for (const auto & p : ctx.catalog) {
        if (p->path == path) {
            return true;
        }
    }
    auto pkg = std::make_unique<npu_package>();
    pkg->path = path;
    char err[1024];
    if (npudsl_model_load(path.c_str(), &pkg->model, err, sizeof(err)) != NPUDSL_OK) {
        note(ctx, GGML_LOG_LEVEL_WARN, "NPU: ignoring model package " + path + ": " + err);
        return false;
    }
    pkg->info.struct_size = sizeof(pkg->info);
    npudsl_model_get_info(pkg->model, &pkg->info);
    const auto & info = pkg->info;
    if (!info.n_sources || std::string(info.source_format) != "gguf") {
        note(ctx, GGML_LOG_LEVEL_WARN, "NPU: ignoring model package " + path + ": it has no GGUF sources");
        return false;
    }
    if (ctx.present && (info.device_key != ctx.key || info.tiles != ctx.tiles)) {
        note(ctx, GGML_LOG_LEVEL_WARN, "NPU: ignoring model package " + path + ": built for " + info.device_key + "/" +
                                           std::to_string(info.tiles) + " tiles, device is " + ctx.key + "/" +
                                           std::to_string(ctx.tiles) + " tiles");
        return false;
    }
    note(ctx, GGML_LOG_LEVEL_INFO, std::string("NPU: model package ") + info.name + " (" + info.source_arch + ", " +
                                   std::to_string(info.n_layers) + " layers, " + std::to_string(info.n_sources) +
                                   " source tensors) from " + path);
    ctx.catalog.push_back(std::move(pkg));
    ctx.consistent.push_back(true);
    return true;
}

void load_path(npu_device_context & ctx, const std::string & path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::is_regular_file(path, ec)) {
        load_package(ctx, path);
        return;
    }
    if (!fs::is_directory(path, ec)) {
        note(ctx, GGML_LOG_LEVEL_WARN, "NPU: model package path " + path + " does not exist");
        return;
    }
    if (fs::is_regular_file(fs::path(path) / "model.json", ec)) {
        load_package(ctx, (fs::path(path) / "model.json").string());
    }
    std::vector<std::string> nested;
    for (const auto & entry : fs::directory_iterator(path, ec)) {
        if (entry.is_directory(ec) && fs::is_regular_file(entry.path() / "model.json", ec)) {
            nested.push_back((entry.path() / "model.json").string());
        }
    }
    std::sort(nested.begin(), nested.end());
    for (const auto & p : nested) {
        load_package(ctx, p);
    }
}

void load_catalog(npu_device_context & ctx) {
    if (ctx.catalog_loaded) {
        return;
    }
    ctx.catalog_loaded = true;
#if defined(_WIN32)
    const char sep = ';';
#else
    const char sep = ':';
#endif
    if (const char * env = getenv("GGML_NPU_MODELS")) {
        const std::string list = env;
        size_t start = 0;
        while (start <= list.size()) {
            const size_t end = std::min(list.find(sep, start), list.size());
            if (end > start) {
                load_path(ctx, list.substr(start, end - start));
            }
            start = end + 1;
        }
    }
#ifdef GGML_NPU_MODELS_DIR
    load_path(ctx, GGML_NPU_MODELS_DIR);
#endif
    for (const auto & p : ctx.added) {
        load_path(ctx, p);
    }
}

bool accepts(const npu_package & pkg, const ggml_tensor * t) {
    const int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    return npudsl_model_source_accepts(pkg.model, t->name, ggml_type_name(t->type), ne, 4) == 1;
}

enum class weight_kind { source, rejected, other };

std::string shape_text(const int64_t * ne, uint32_t n) {
    std::string s = "[";
    for (uint32_t d = 0; d < n; ++d) {
        s += (d ? ", " : "") + std::to_string(ne[d]);
    }
    return s + "]";
}

// "blk.0.x.weight is q4_0 [64, 32]; qwen36 takes f32/f16/bf16 [64, 32]"
std::string rejection(const npu_device_context & ctx, const ggml_tensor * w) {
    std::string s = std::string(w->name) + " is " + ggml_type_name(w->type) + " " + shape_text(w->ne, uint32_t(ggml_n_dims(w)));
    for (const auto & pkg : ctx.catalog) {
        const int64_t i = npudsl_model_find_source(pkg->model, w->name);
        npudsl_source_info si = {};
        si.struct_size = sizeof(si);
        if (i < 0 || npudsl_model_source(pkg->model, uint32_t(i), &si) != NPUDSL_OK) {
            continue;
        }
        s += std::string("; ") + pkg->info.name + " takes ";
        for (uint32_t t = 0; t < si.n_types; ++t) {
            s += (t ? "/" : "") + std::string(si.types[t]);
        }
        s += " " + shape_text(si.ne, si.n_dims);
    }
    return s;
}

// A weight is a source of a package that also accepted every source claimed before it, a source name no such
// package accepts (another model: its graph stays on the other backends), or no package's source at all (the KV and
// recurrent state llama flags as weights). The scheduler queries a graph's ops in order, so a model's first weight
// (its token lookup) decides `matched` for the state ops of its graph.
weight_kind classify_weight(npu_device_context & ctx, const ggml_tensor * w) {
    load_catalog(ctx);
    std::vector<bool> ok(ctx.catalog.size());
    bool named = false, any = false;
    for (size_t i = 0; i < ctx.catalog.size(); ++i) {
        named = named || npudsl_model_find_source(ctx.catalog[i]->model, w->name) >= 0;
        ok[i] = ctx.consistent[i] && accepts(*ctx.catalog[i], w);
        any = any || ok[i];
    }
    if (!named) {
        return weight_kind::other;
    }
    if (!any) {
        ctx.matched = false;
        if (ctx.rejected.empty()) {
            ctx.rejected = rejection(ctx, w);
        }
        return weight_kind::rejected;
    }
    for (size_t i = 0; i < ctx.catalog.size(); ++i) {
        ctx.consistent[i] = ok[i];
    }
    ctx.matched = true;
    return weight_kind::source;
}

// ---------------------------------------------------------------------------------------------------------------
// graph execution

// graphs carry alternative branches (ggml_build_forward_select); only nodes flagged for computation are live
bool live(const ggml_tensor * node) {
    return (node->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
}

bool host_weight(const ggml_tensor * t) {
    return t && t->buffer && t->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && ggml_backend_buffer_is_host(t->buffer);
}

std::vector<int32_t> read_i32(const ggml_tensor * t, int64_t n) {
    if (t->type != GGML_TYPE_I32 || !ggml_is_contiguous(t) || ggml_nelements(t) < n) {
        GGML_ABORT("NPU: %s is not a contiguous I32 tensor of at least %" PRId64 " elements", t->name, n);
    }
    std::vector<int32_t> v(static_cast<size_t>(n));
    ggml_backend_tensor_get(t, v.data(), 0, size_t(n) * sizeof(int32_t));
    return v;
}

// the package whose every source this graph references; its runner, with the sources converted from their data
const ggml_tensor * base_of(const ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t;
}

bool writes_view(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return false;
        default:
            return node->view_src != nullptr;   // e.g. SET_ROWS / CPY into the KV or recurrent state
    }
}

bool bind(npu_device_context & ctx, ggml_cgraph * graph) {
    std::map<std::string, const ggml_tensor *> weights;   // every host weight the graph reads, by name
    std::set<const ggml_tensor *> written;                // state: weight-flagged tensors the graph writes
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (!live(node)) {
            continue;
        }
        if (writes_view(node) && host_weight(base_of(node))) {
            written.insert(base_of(node));
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * t = base_of(node->src[s]);
            if (host_weight(t)) {
                weights.emplace(t->name, t);
            }
        }
    }
    auto complete = [&](const npu_package & pkg, std::string * missing) {
        npudsl_source_info si = {};
        si.struct_size = sizeof(si);
        for (uint32_t i = 0; i < pkg.info.n_sources; ++i) {
            npudsl_model_source(pkg.model, i, &si);
            const auto it = weights.find(si.name);
            if (it == weights.end() || !accepts(pkg, it->second)) {
                if (missing) {
                    *missing = si.name;
                }
                return false;
            }
        }
        return true;
    };
    // Every weight the graph reads must be consumed by the package or be state the graph writes; anything else
    // (an incomplete package, a LoRA adapter, a control vector) would silently not be applied.
    auto foreign = [&](const npu_package & pkg) -> const ggml_tensor * {
        for (const auto & w : weights) {
            if (npudsl_model_find_source(pkg.model, w.first.c_str()) < 0 && !written.count(w.second)) {
                return w.second;
            }
        }
        return nullptr;
    };
    // still bound, sources unchanged: nothing to do
    if (ctx.bound && complete(*ctx.bound, nullptr) && !foreign(*ctx.bound) &&
        std::all_of(ctx.imported.begin(), ctx.imported.end(), [&](const auto & e) { return weights.at(e.first)->data == e.second; })) {
        return true;
    }
    const npu_package * pick = nullptr;
    std::string missing;
    for (size_t i = 0; i < ctx.catalog.size() && !pick; ++i) {
        if (ctx.consistent[i] && complete(*ctx.catalog[i], &missing)) {
            pick = ctx.catalog[i].get();
        }
    }
    if (!pick) {
        if (!ctx.rejected.empty()) {   // a weight of the model was refused, so its ops (and the split) went elsewhere
            GGML_LOG_ERROR("NPU: the GGUF does not fit the model package: %s\n", ctx.rejected.c_str());
        } else {
            GGML_LOG_ERROR("NPU: the graph references no complete model package (%s is missing or different)\n",
                           missing.empty() ? "every source" : missing.c_str());
        }
        return false;
    }
    if (const ggml_tensor * w = foreign(*pick)) {
        GGML_LOG_ERROR("NPU: the graph reads %s, which model package %s does not consume (an adapter, a control vector "
                       "or an incomplete package); refusing to run without it\n", w->name, pick->info.name);
        return false;
    }
    ctx.unbind();
    char err[1024];
    npudsl_runner_options options = {};
    options.struct_size = sizeof(options);
    if (npudsl_runner_create(pick->model, &options, &ctx.runner, err, sizeof(err)) != NPUDSL_OK) {
        GGML_LOG_ERROR("NPU: cannot create the runner of %s: %s\n", pick->path.c_str(), err);
        return false;
    }
    npudsl_source_info si = {};
    si.struct_size = sizeof(si);
    for (uint32_t i = 0; i < pick->info.n_sources; ++i) {
        npudsl_model_source(pick->model, i, &si);
        const ggml_tensor * t = weights.at(si.name);
        const int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
        if (npudsl_runner_import(ctx.runner, t->name, ggml_type_name(t->type), ne, 4, t->data, ggml_nbytes(t)) != NPUDSL_OK) {
            GGML_LOG_ERROR("NPU: cannot import %s: %s\n", t->name, npudsl_runner_error(ctx.runner));
            ctx.unbind();
            return false;
        }
        ctx.imported[t->name] = t->data;
    }
    if (npudsl_runner_prepare(ctx.runner) != NPUDSL_OK) {
        GGML_LOG_ERROR("NPU: %s\n", npudsl_runner_error(ctx.runner));
        ctx.unbind();
        return false;
    }
    ctx.bound = pick;
    ctx.logits.assign(pick->info.vocab, 0.0f);
    npudsl_setup_stats st = {};
    st.struct_size = sizeof(st);
    npudsl_runner_setup_stats(ctx.runner, &st);
    GGML_LOG_INFO("NPU: running %s from %s (%u source tensors converted, max_seq %u)\n", pick->info.name,
                  pick->path.c_str(), pick->info.n_sources, pick->info.max_seq);
    GGML_LOG_INFO("NPU: setup %.1f s: create %.1f s, %u imports of %.1f GB in %.1f s on %u threads, prepare %.1f s "
                  "(zero-filling state and buffers %.1f s of it)\n", (st.create_us + st.import_us + st.prepare_us) / 1e6,
                  st.create_us / 1e6, st.import_calls, st.import_bytes / 1e9, st.import_us / 1e6, st.threads,
                  st.prepare_us / 1e6, st.zero_us / 1e6);
    return true;
}

ggml_status evaluate(npu_device_context & ctx, const ggml_tensor * ids, ggml_cgraph * graph) {
    const int64_t n = ids->ne[0];
    if (n <= 0 || ggml_nrows(ids) != 1) {
        GGML_LOG_ERROR("NPU: token ids %s must be one row of tokens\n", ids->name);
        return GGML_STATUS_FAILED;
    }
    const auto tokens = read_i32(ids, n);
    size_t pos0 = npudsl_runner_position(ctx.runner);
    if (std::string(ctx.bound->info.source_positions) == "rope") {
        const ggml_tensor * pos = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(graph) && !pos; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_ROPE && node->src[1] && live(node)) {
                pos = node->src[1];
            }
        }
        if (!pos) {
            GGML_LOG_ERROR("NPU: %s takes positions from rope, but the graph split has no rope\n", ctx.bound->info.name);
            return GGML_STATUS_FAILED;
        }
        const auto p = read_i32(pos, n);
        for (int64_t i = 0; i < n; ++i) {
            if (p[size_t(i)] != p[0] + i || p[0] < 0) {
                GGML_LOG_ERROR("NPU: positions must be one ascending run (position %" PRId64 " is %d after %d)\n", i,
                               p[size_t(i)], p[0]);
                return GGML_STATUS_FAILED;
            }
        }
        pos0 = size_t(p[0]);
    }
    if (pos0 == 0 && npudsl_runner_position(ctx.runner) != 0) {
        if (npudsl_runner_reset(ctx.runner) != NPUDSL_OK) {
            GGML_LOG_ERROR("NPU: reset failed: %s\n", npudsl_runner_error(ctx.runner));
            return GGML_STATUS_FAILED;
        }
    }
    if (pos0 != npudsl_runner_position(ctx.runner)) {
        GGML_LOG_ERROR("NPU: the model state is at position %zu, the batch starts at %zu; the package kernels keep the "
                       "state and support only appending or restarting a sequence at 0\n",
                       npudsl_runner_position(ctx.runner), pos0);
        return GGML_STATUS_FAILED;
    }
    std::vector<uint32_t> u(tokens.begin(), tokens.end());
    if (npudsl_runner_eval(ctx.runner, u.data(), u.size(), pos0, ctx.logits.data(), ctx.logits.size()) != NPUDSL_OK) {
        GGML_LOG_ERROR("NPU: %s\n", npudsl_runner_error(ctx.runner));
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

ggml_status publish(npu_device_context & ctx, ggml_tensor * out, ggml_cgraph * graph, int64_t n_tokens) {
    const int64_t n_out = ggml_nrows(out);
    if (n_out == 0) {
        return GGML_STATUS_SUCCESS;
    }
    if (out->type != GGML_TYPE_F32 || out->ne[0] != int64_t(ctx.logits.size()) || n_out != 1 || !ggml_is_contiguous(out)) {
        GGML_LOG_ERROR("NPU: %s gives the last token's logits only; %" PRId64 " output rows of %" PRId64 " were requested\n",
                       ctx.bound->info.name, n_out, out->ne[0]);
        return GGML_STATUS_FAILED;
    }
    // the one output row must be the last token of the batch
    for (int i = 0; i < ggml_graph_n_nodes(graph) && n_tokens > 1; ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_GET_ROWS && live(node) && node->src[1] && strstr(node->src[1]->name, "inp_out_ids") &&
            node->src[1]->ne[0] == 1) {
            if (read_i32(node->src[1], 1)[0] != n_tokens - 1) {
                GGML_LOG_ERROR("NPU: %s gives the last token's logits only\n", ctx.bound->info.name);
                return GGML_STATUS_FAILED;
            }
            break;
        }
    }
    ggml_backend_tensor_set(out, ctx.logits.data(), 0, ggml_nbytes(out));
    return GGML_STATUS_SUCCESS;
}

ggml_status npu_graph_compute(ggml_backend_t, ggml_cgraph * graph) {
    auto & ctx = device_context();
    std::lock_guard<std::recursive_mutex> lock(ctx.mutex);
    if (!bind(ctx, graph)) {
        return GGML_STATUS_FAILED;
    }
    const ggml_tensor * ids = nullptr;
    const ggml_tensor * lookup = nullptr;
    ggml_tensor * out = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (!live(node)) {
            continue;
        }
        if (node->op == GGML_OP_GET_ROWS && node->src[0] && strcmp(node->src[0]->name, ctx.bound->info.source_tokens) == 0) {
            lookup = node;
            ids = node->src[1];
        }
        if (strcmp(node->name, ctx.bound->info.source_logits) == 0) {
            out = node;
        }
    }
    // token rows scattered among embedding rows (multimodal input): the kernels take token ids only
    for (int i = 0; i < ggml_graph_n_nodes(graph) && lookup; ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (live(node) && node->op == GGML_OP_SET_ROWS && node->src[1] == lookup) {
            GGML_LOG_ERROR("NPU: %s takes token ids only; batches mixing embeddings are not supported\n", ctx.bound->info.name);
            return GGML_STATUS_FAILED;
        }
    }
    // The whole model is one kernel launch: the token lookup and the logits node must be in this one split. Any other
    // backend computing part of the graph would compute from tensors this backend never writes.
    if (!ids || !out) {
        GGML_LOG_ERROR("NPU: %s is not in the NPU graph split; %s runs only as a whole model on token ids -- offload "
                       "every layer, do not override tensor placement, and pass tokens rather than embeddings\n",
                       ids ? ctx.bound->info.source_logits : "the token lookup", ctx.bound->info.name);
        return GGML_STATUS_FAILED;
    }
    const ggml_status st = evaluate(ctx, ids, graph);
    return st == GGML_STATUS_SUCCESS ? publish(ctx, out, graph, ids->ne[0]) : st;
}

const char * npu_backend_get_name(ggml_backend_t) {
    return "NPU";
}

void npu_backend_free(ggml_backend_t backend) {
    delete backend;
}

const ggml_backend_i npu_backend_iface = {
    /* .get_name            = */ npu_backend_get_name,
    /* .free                = */ npu_backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ npu_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

ggml_guid_t npu_guid() {
    static ggml_guid guid = {0x92, 0x0d, 0xc2, 0xf0, 0xc6, 0x82, 0x4c, 0x12, 0x91, 0xbc, 0x6a, 0x63, 0x93, 0xe7, 0x2a, 0x15};
    return &guid;
}

// ---------------------------------------------------------------------------------------------------------------
// device

const char * npu_dev_get_name(ggml_backend_dev_t) {
    return "NPU";
}

const char * npu_dev_get_description(ggml_backend_dev_t) {
    return device_context().description.c_str();
}

void npu_dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    // NPU memory is host memory shared with the CPU
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = size_t(status.ullTotalPhys);
    *free = size_t(status.ullAvailPhys);
#else
    const size_t ps = size_t(sysconf(_SC_PAGESIZE));
    *total = size_t(sysconf(_SC_PHYS_PAGES)) * ps;
    *free = size_t(sysconf(_SC_AVPHYS_PAGES)) * ps;
#endif
}

enum ggml_backend_dev_type npu_dev_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_IGPU;
}

void npu_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    *props = {};
    props->name = npu_dev_get_name(dev);
    props->description = npu_dev_get_description(dev);
    props->type = npu_dev_get_type(dev);
    npu_dev_get_memory(dev, &props->memory_free, &props->memory_total);
}

ggml_backend_t npu_dev_init_backend(ggml_backend_dev_t dev, const char *) {
    auto & ctx = device_context();
    {
        std::lock_guard<std::recursive_mutex> lock(ctx.mutex);
        load_catalog(ctx);
        for (const auto & m : ctx.catalog_log) {
            ggml_log_internal(m.first, "%s\n", m.second.c_str());
        }
        if (ctx.catalog.empty()) {
            GGML_LOG_INFO("NPU: no usable model package (GGML_NPU_MODELS); every model runs on the other backends\n");
        }
    }
    return new ggml_backend{npu_guid(), npu_backend_iface, dev, nullptr};
}

bool npu_dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    auto & ctx = device_context();
    std::lock_guard<std::recursive_mutex> lock(ctx.mutex);
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * t = op->src[s];
        if (!t || !t->buffer || t->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t && t->buffer && ggml_backend_buffer_get_size(t->buffer) == 0) {
                return classify_weight(ctx, t) == weight_kind::source;   // llama's weight placement probe (a 0-byte buffer)
            }
            continue;   // activations and inputs
        }
        if (!ggml_backend_buffer_is_host(t->buffer)) {
            return false;
        }
        // a package source, or (within a matched model's graph) the KV/recurrent state llama flags as weights
        const weight_kind kind = classify_weight(ctx, t);
        if (kind == weight_kind::rejected || (kind == weight_kind::other && !ctx.matched)) {
            return false;
        }
    }
    return ctx.matched;
}

bool npu_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    // host tensors (weights, KV, inputs) are read in place, never copied in
    return ggml_backend_buft_is_host(buft);
}

ggml_backend_buffer_type_t npu_dev_get_buffer_type(ggml_backend_dev_t) {
    // NPU memory is host memory: the device works in CPU buffers (as BLAS does), so the CPU also runs, in place,
    // every model no package lists, and llama keeps its KV and recurrent state there
    return ggml_backend_cpu_buffer_type();
}

const ggml_backend_device_i npu_device_iface = {
    /* .get_name             = */ npu_dev_get_name,
    /* .get_description      = */ npu_dev_get_description,
    /* .get_memory           = */ npu_dev_get_memory,
    /* .get_type             = */ npu_dev_get_type,
    /* .get_props            = */ npu_dev_get_props,
    /* .init_backend         = */ npu_dev_init_backend,
    /* .get_buffer_type      = */ npu_dev_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ npu_dev_supports_op,
    /* .supports_buft        = */ npu_dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

const char * npu_reg_get_name(ggml_backend_reg_t) {
    return "NPU";
}

size_t npu_reg_get_device_count(ggml_backend_reg_t) {
    auto & ctx = device_context();
    std::lock_guard<std::recursive_mutex> lock(ctx.mutex);
    probe(ctx);
    return ctx.present ? 1 : 0;
}

ggml_backend_dev_t npu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index >= npu_reg_get_device_count(reg)) {
        return nullptr;
    }
    static ggml_backend_device device = {npu_device_iface, reg, nullptr};
    return &device;
}

void * npu_reg_get_proc_address(ggml_backend_reg_t, const char * name) {
    if (strcmp(name, "ggml_backend_npu_add_model") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_add_model);
    }
    return nullptr;
}

} // namespace

ggml_backend_reg_t ggml_backend_npu_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ {
            /* .get_name         = */ npu_reg_get_name,
            /* .get_device_count = */ npu_reg_get_device_count,
            /* .get_device       = */ npu_reg_get_device,
            /* .get_proc_address = */ npu_reg_get_proc_address,
        },
        /* .context = */ nullptr,
    };
    return &reg;
}

ggml_backend_t ggml_backend_npu_init(void) {
    ggml_backend_dev_t dev = npu_reg_get_device(ggml_backend_npu_reg(), 0);
    return dev ? npu_dev_init_backend(dev, nullptr) : nullptr;
}

bool ggml_backend_is_npu(ggml_backend_t backend) {
    return backend && ggml_guid_matches(backend->guid, npu_guid());
}

bool ggml_backend_npu_add_model(const char * manifest_path) {
    if (!manifest_path) {
        return false;
    }
    auto & ctx = device_context();
    std::lock_guard<std::recursive_mutex> lock(ctx.mutex);
    probe(ctx);
    ctx.added.push_back(manifest_path);
    if (!ctx.catalog_loaded) {
        load_catalog(ctx);
        return std::any_of(ctx.catalog.begin(), ctx.catalog.end(), [&](const auto & p) { return p->path == manifest_path; });
    }
    return load_package(ctx, manifest_path);
}

GGML_BACKEND_DL_IMPL(ggml_backend_npu_reg)
