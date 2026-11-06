// mmap loading needs both truthful device capabilities and a non-owning host
// buffer view. No model, kernels or large NPU allocations are needed here.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-npu.h"

#include <cstdio>

int main() {
    ggml_backend_reg_t reg = ggml_backend_npu_reg();
    if (ggml_backend_reg_dev_count(reg) == 0) {
        std::puts("test-npu-mmap: SKIPPED (no NPU device)");
        return 77;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.mmap_support || !props.caps.buffer_from_host_ptr) {
        std::fputs("FAIL: NPU must advertise mmap and host-pointer buffer support\n", stderr);
        return 1;
    }

    // Page alignment satisfies the same alignment required of mapped GGUF data.
    alignas(4096) float source[16]{};
    source[0] = 3.25f;
    ggml_backend_buffer_t buffer = ggml_backend_dev_buffer_from_host_ptr(dev, source, sizeof(source), sizeof(source));
    if (!buffer) {
        std::fputs("FAIL: cannot wrap an external host buffer\n", stderr);
        return 1;
    }
    bool ok = ggml_backend_buffer_is_host(buffer) &&
              ggml_backend_buffer_get_base(buffer) == source &&
              ggml_backend_buffer_get_size(buffer) == sizeof(source) &&
              ggml_backend_dev_supports_buft(dev, ggml_backend_buffer_get_type(buffer));

    ggml_init_params params{ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_buffer_free(buffer);
        std::fputs("FAIL: cannot create the tensor context\n", stderr);
        return 1;
    }
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ok = ok && ggml_backend_tensor_alloc(buffer, tensor, source) == GGML_STATUS_SUCCESS;
    if (ok) {
        float value = 0;
        ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
        ok = value == 3.25f;
        // A caller-side update must be visible without a hidden source copy.
        source[0] = 7.5f;
        ggml_backend_tensor_get(tensor, &value, 0, sizeof(value));
        ok = ok && value == 7.5f;
    }
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    // The view does not own source; freeing it must leave the caller's data intact.
    ok = ok && source[0] == 7.5f;
    if (!ok) {
        std::fputs("FAIL: host-buffer view must alias data and leave ownership with the caller\n", stderr);
        return 1;
    }
    std::puts("test-npu-mmap: capabilities, zero-copy view and caller ownership passed");
    return 0;
}