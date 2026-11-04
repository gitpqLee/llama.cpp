#include "ggml-npu.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(int argc, char ** argv) {
#ifdef GGML_NPU_NATIVE_TEST
    ggml_backend_reg_t reg;
    if (argc == 3 && std::strcmp(argv[1], "--discover") == 0) {
        ggml_backend_load_all_from_path(argv[2]);
        reg = ggml_backend_reg_by_name("NPU");
    } else {
        reg = argc > 1 ? ggml_backend_load(argv[1]) : ggml_backend_reg_by_name("NPU");
    }
    CHECK(reg);
    CHECK(ggml_backend_reg_by_name("NPU") == reg);
#else
    CHECK(argc == 2);
    ggml_backend_reg_t reg = ggml_backend_npu_reg();
    CHECK(reg == ggml_backend_npu_reg());
    CHECK(!ggml_backend_is_npu(nullptr));
#endif
    CHECK(std::strcmp(ggml_backend_reg_name(reg), "NPU") == 0);
    const size_t count = ggml_backend_reg_dev_count(reg);
    CHECK(count <= 1);
    CHECK(count == ggml_backend_reg_dev_count(reg));
#ifndef GGML_NPU_NATIVE_TEST
    CHECK(count == static_cast<size_t>(std::atoi(argv[1])));
#endif
    std::printf("NPU registered: %zu device(s)\n", count);
    CHECK(!ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    if (count == 0) {
#ifndef GGML_NPU_NATIVE_TEST
        CHECK(!ggml_backend_npu_init());
#endif
        return 0;
    }

    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    CHECK(dev);
    CHECK(dev == ggml_backend_reg_dev_get(reg, 0));
    CHECK(ggml_backend_dev_backend_reg(dev) == reg);
    CHECK(std::strcmp(ggml_backend_dev_name(dev), "NPU") == 0);
    CHECK(std::strcmp(ggml_backend_dev_description(dev), "Intel NPU complete DSL model") == 0);
    CHECK(ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL);
    ggml_backend_dev_props props;
    std::memset(&props, 0xff, sizeof(props));
    ggml_backend_dev_get_props(dev, &props);
    CHECK(props.type == GGML_BACKEND_DEVICE_TYPE_ACCEL);
    CHECK(std::strcmp(props.name, "NPU") == 0);
    CHECK(std::strcmp(props.description, "Intel NPU complete DSL model") == 0);
    CHECK(!props.device_id);
    CHECK(props.memory_free == 0 && props.memory_total == 0);
    CHECK(!props.caps.async && !props.caps.host_buffer && !props.caps.buffer_from_host_ptr);
    CHECK(!props.caps.events && !props.caps.mmap_support);
    size_t memory_free = 1, memory_total = 1;
    ggml_backend_dev_memory(dev, &memory_free, &memory_total);
    CHECK(memory_free == 0 && memory_total == 0);
    CHECK(!ggml_backend_dev_host_buffer_type(dev));
    CHECK(ggml_backend_dev_buffer_type(dev) == ggml_backend_cpu_buffer_type());
    CHECK(ggml_backend_dev_supports_buft(dev, ggml_backend_cpu_buffer_type()));
    CHECK(!ggml_backend_dev_supports_buft(dev, nullptr));
    CHECK(!ggml_backend_dev_init(dev, "unsupported"));

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    CHECK(backend);
    CHECK(ggml_backend_get_device(backend) == dev);
    CHECK(std::strcmp(ggml_backend_name(backend), "NPU") == 0);
#ifndef GGML_NPU_NATIVE_TEST
    CHECK(ggml_backend_is_npu(backend));
    ggml_backend_t other = ggml_backend_npu_init();
    CHECK(other && other != backend && ggml_backend_is_npu(other));
    ggml_backend_free(other);
#endif

    ggml_init_params params = { ggml_graph_overhead_custom(8, false) + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_graph_compute(backend, nullptr) == GGML_STATUS_FAILED);

    float sentinel = 123.0f;
    ggml_tensor tensor = {};
    tensor.type = GGML_TYPE_F32;
    tensor.data = &sentinel;
    ggml_graph_add_node(graph, &tensor);
    // Even metadata nodes are rejected: no successful nonempty graph is advertised.
    for (int op = GGML_OP_NONE; op < GGML_OP_COUNT; ++op) {
        tensor.op = static_cast<ggml_op>(op);
        CHECK(!ggml_backend_dev_supports_op(dev, &tensor));
        CHECK(!ggml_backend_dev_offload_op(dev, &tensor));
        CHECK(!ggml_backend_supports_op(backend, &tensor));
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
        CHECK(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_FAILED);
        CHECK(sentinel == 123.0f);
    }
    ggml_backend_synchronize(backend);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::puts("NPU contract passed: no GGML ops, no numeric graph execution");
    return 0;
}