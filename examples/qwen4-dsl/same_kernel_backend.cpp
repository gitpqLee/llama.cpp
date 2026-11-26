#include "ggml-backend-impl.h"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <direct_host_extract.inc>

namespace {
struct Request {
    const int32_t * tokens = nullptr;
    uint32_t n = 0, pos = 0;
    float * logits = nullptr;
    double ms = 0;
    bool profile = false;
    double copy_ms = 0, convert_ms = 0, check_ms = 0;
};

// Device, buffers and stream stay in the original host's stack frame on one thread.
class Session {
    std::mutex mutex;
    std::condition_variable cv;
    Request * pending = nullptr;
    bool ready = false, stop = false;
    std::exception_ptr error;
    std::thread worker;
public:
    Session(std::string weights, std::string overlay) {
        worker = std::thread([this, weights, overlay] {
            try {
                with_direct_host(weights, overlay, [&](auto eval) {
                    std::unique_lock<std::mutex> lock(mutex);
                    ready = true;
                    cv.notify_all();
                    while (true) {
                        cv.wait(lock, [&] { return stop || pending; });
                        if (stop) break;
                        Request * r = pending;
                        lock.unlock();
                        eval(*r);
                        lock.lock();
                        pending = nullptr;
                        cv.notify_all();
                    }
                });
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                error = std::current_exception();
                cv.notify_all();
            }
        });
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return ready || error; });
        if (error) {
            auto failure = error;
            lock.unlock();
            worker.join();
            std::rethrow_exception(failure);
        }
    }
    ~Session() {
        { std::lock_guard<std::mutex> lock(mutex); stop = true; }
        cv.notify_all();
        worker.join();
    }
    void eval(Request & r) {
        std::unique_lock<std::mutex> lock(mutex);
        if (error) std::rethrow_exception(error);
        pending = &r;
        cv.notify_all();
        cv.wait(lock, [&] { return !pending || error; });
        if (error) std::rethrow_exception(error);
    }
};

std::unique_ptr<Session> session;
uint32_t position = 0;
uint32_t evaluations = 0;

int prepare(const char * weights, const char * overlay) {
    try {
        if (session) throw std::runtime_error("session already active");
        session = std::make_unique<Session>(weights, overlay);
        position = evaluations = 0;
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "same-kernel prepare failed: %s\n", e.what());
        return 1;
    }
}
void release() { session.reset(); }
uint32_t eval_count() { return evaluations; }

ggml_status graph_compute(ggml_backend_t, ggml_cgraph * graph) try {
    static const bool profile = std::getenv("LLAMA_PROFILE_UBATCH") != nullptr;
    using Clock = std::chrono::steady_clock;
    const auto start = profile ? Clock::now() : Clock::time_point{};
    const int nodes = ggml_graph_n_nodes(graph);
    const ggml_tensor * ids = nullptr, * pos = nullptr, * out_ids = nullptr;
    ggml_tensor * out = nullptr;
    bool rope = false, indexer = false;
    for (int i = 0; i < nodes; ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE)) continue;
        if (node->op == GGML_OP_GET_ROWS && node->src[0] &&
            std::strcmp(node->src[0]->name, "token_embd.weight") == 0) ids = node->src[1];
        if (node->op == GGML_OP_ROPE && node->src[1] &&
            std::strcmp(node->src[1]->name, "inp_pos") == 0) {
            pos = node->src[1];
            rope = true;
        }
        if (node->op == GGML_OP_GET_ROWS && node->src[1] &&
            std::strcmp(node->src[1]->name, "out_ids") == 0) out_ids = node->src[1];
        if (node->op == GGML_OP_LIGHTNING_INDEXER) indexer = true;
        if (std::strcmp(node->name, "result_output") == 0) out = node;
    }
    if (!session || nodes < 1000 || !ids || !rope || !indexer || !out ||
        ids->type != GGML_TYPE_I32 || ggml_nrows(ids) != 1 || !ggml_is_contiguous(ids) ||
        pos->type != GGML_TYPE_I32 || !ggml_is_contiguous(pos) ||
        out->type != GGML_TYPE_F32 || out->ne[0] != kVocab || ggml_nrows(out) != 1 ||
        !ggml_is_contiguous(out)) {
        std::fprintf(stderr, "graph shape: nodes=%d ids=%d pos=%d indexer=%d output=%d rows=%lld\n",
                     nodes, !!ids, !!pos, indexer, !!out, (long long)(out ? ggml_nrows(out) : -1));
        throw std::runtime_error("not the complete single-output Qwen4 graph");
    }
    const auto n = ids->ne[0];
    if (n < 1 || n > 4096 || position + n > kMaxSeq || ggml_nelements(pos) < n)
        throw std::runtime_error("graph batch outside session bounds");
    if (n > 1 && (!out_ids || out_ids->type != GGML_TYPE_I32 || ggml_nelements(out_ids) != 1))
        throw std::runtime_error("graph did not select a single output");
    if (n > 1) {
        int32_t row = -1;
        ggml_backend_tensor_get(out_ids, &row, 0, sizeof(row));
        if (row != n - 1) throw std::runtime_error("graph output must be the last token");
    }
    std::vector<int32_t> tokens(n), positions(n);
    ggml_backend_tensor_get(ids, tokens.data(), 0, n * sizeof(int32_t));
    ggml_backend_tensor_get(pos, positions.data(), 0, n * sizeof(int32_t));
    for (int64_t i = 0; i < n; ++i) {
        if (positions[i] != position + i || tokens[i] < 0 || tokens[i] >= kVocab)
            throw std::runtime_error("only valid tokens and contiguous append positions are supported");
    }
    if (n > 1 && position % kMaxTokens) throw std::runtime_error("unaligned prefill");
    Request request{tokens.data(), uint32_t(n), position, static_cast<float *>(out->data)};
    if (!request.logits || !out->buffer || !ggml_backend_buffer_is_host(out->buffer))
        throw std::runtime_error("logits must be host-visible");
    request.profile = profile;
    const auto dispatch = profile ? Clock::now() : Clock::time_point{};
    session->eval(request);
    const auto completed = profile ? Clock::now() : Clock::time_point{};
    ++evaluations;
    if (std::getenv("QWEN4_STAGE_LOGITS")) {
        // stdout is unbuffered; format locally to avoid multiple console/file writes.
        char line[128];
        const int len = std::snprintf(line, sizeof(line), "NPU_TAKEOVER pos=%u n=%u nodes=%d launch_ms=%.3f\n",
                position, request.n, nodes, request.ms);
        if (len < 0 || len >= sizeof(line) || std::fwrite(line, 1, len, stdout) != len)
            throw std::runtime_error("takeover log write failed");
    } else {
        std::printf("NPU_TAKEOVER pos=%u n=%u nodes=%d launch_ms=%.3f\n", position, request.n, nodes, request.ms);
    }
    if (profile) {
        using Ms = std::chrono::duration<double, std::milli>;
        const double worker = Ms(completed - dispatch).count();
        std::fprintf(stderr, "adapter_profile n=%u graph_io_ms=%.3f copy_ms=%.3f convert_ms=%.3f check_ms=%.3f worker_other_ms=%.3f log_ms=%.3f\n",
                request.n, Ms(dispatch - start).count(), request.copy_ms, request.convert_ms, request.check_ms,
                worker - request.ms - request.copy_ms - request.convert_ms - request.check_ms, Ms(Clock::now() - completed).count());
    }
    position += request.n;
    return GGML_STATUS_SUCCESS;
} catch (const std::exception & e) {
    std::fprintf(stderr, "same-kernel graph rejected: %s\n", e.what());
    return GGML_STATUS_FAILED;
}

const char * name(ggml_backend_t) { return "NPU-SAME-KERNEL"; }
void free_backend(ggml_backend_t b) { delete b; }
ggml_guid guid{0x79, 0x51, 0x45, 0x6e, 0x03, 0x2b, 0x4c, 0x07, 0x99, 0x3e, 0xa2, 0x8e, 0x88, 0x0d, 0x7d, 0x12};
ggml_backend_t init(ggml_backend_dev_t dev, const char *) {
    ggml_backend_i iface{};
    iface.get_name = name;
    iface.free = free_backend;
    iface.graph_compute = graph_compute;
    return new ggml_backend{&guid, iface, dev, nullptr};
}
const char * dev_name(ggml_backend_dev_t) { return "NPU-SAME-KERNEL"; }
void memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = *total = 0; }
enum ggml_backend_dev_type type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_IGPU; }
void props(ggml_backend_dev_t dev, ggml_backend_dev_props * p) {
    *p = {};
    p->name = p->description = dev_name(dev);
    p->type = type(dev);
    p->caps.mmap_support = p->caps.buffer_from_host_ptr = true;
}
ggml_backend_buffer_type_t buft(ggml_backend_dev_t) { return ggml_backend_cpu_buffer_type(); }
ggml_backend_buffer_t from_ptr(ggml_backend_dev_t, void * ptr, size_t n, size_t) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, n);
}
bool supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t b) { return ggml_backend_buft_is_host(b); }
bool supports_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }
const char * reg_name(ggml_backend_reg_t) { return "NPU-SAME-KERNEL"; }
size_t dev_count(ggml_backend_reg_t) { return 1; }
ggml_backend_dev_t get_dev(ggml_backend_reg_t reg, size_t i) {
    static ggml_backend_device dev{{dev_name, dev_name, memory, type, props, init, buft, nullptr,
                                    from_ptr, supports_op, supports_buft}, reg, nullptr};
    return i == 0 ? &dev : nullptr;
}
void * get_proc(ggml_backend_reg_t, const char * symbol) {
    if (std::strcmp(symbol, "same_kernel_prepare") == 0) return reinterpret_cast<void *>(prepare);
    if (std::strcmp(symbol, "same_kernel_release") == 0) return reinterpret_cast<void *>(release);
    if (std::strcmp(symbol, "same_kernel_eval_count") == 0) return reinterpret_cast<void *>(eval_count);
    return nullptr;
}
ggml_backend_reg_t registry() {
    static ggml_backend_reg reg{GGML_BACKEND_API_VERSION, {reg_name, dev_count, get_dev, get_proc}, nullptr};
    return &reg;
}
} // namespace

extern "C" __declspec(dllexport) ggml_backend_reg_t ggml_backend_init() { return registry(); }