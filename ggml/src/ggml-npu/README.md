# Intel NPU whole-model registration

`GGML_NPU=ON` registers the `NPU` backend using the standard static or dynamic backend mechanism. It describes an `ACCEL` device as `Intel NPU complete DSL model`. It is not a tensor-operation backend: every `supports_op` and `offload_op` query returns false, and every nonempty graph (including `NONE` metadata) fails with `GGML_STATUS_FAILED`. Empty graphs succeed without work.

The default buffer type is the existing CPU buffer for host staging only. No NPU tensor storage, pinned host buffer, host-pointer import, mmap, async execution, events, or device-memory capacity is advertised. The whole-model session owns its model buffers and execution resources outside GGML graph execution.

## Runtime and session integration

Set `NPU_DSL_ROOT` to the `libraries.npu.dsl` checkout and `GENERIC_COMPUTE_ROOT` to the `frameworks.ai.npu.generic_compute` checkout. Environment variables of the same names are accepted. The external runtime may also require `MV_TOOLS_PATH` and exported sources selected by `QWEN36_GEN`.

The backend adds the DSL `model_api` project unless `npudsl_model_api` already exists. The parent model bridge links this versioned provider API, not a model-specific session. The DSL project internally builds the current Qwen provider, its `qwen36_session` and exported kernels together; additional static providers are registered through its factory extension mechanism. GGML registration links only `npu_runtime`. The registration layer does not replace the kernel registry or modify the external build.

If an existing `npu_runtime` target is provided, registration can use it without adding the session project. Without any runtime target, the backend still registers but exposes zero devices and `ggml_backend_npu_init()` returns null. With the runtime linked, detection constructs and releases an `npu::Device` and checks its device key, catching initialization failures. This runtime has no enumeration API and selects only one NPU, so registration exposes at most one device. The result is cached on first discovery; device presence does not imply a particular model or exported kernel is compatible.

## Contract tests

Enable `GGML_NPU_TESTS=ON` and CTest at the top level, or configure the standalone harness in `tests/standalone` into a separate build directory. Tests cover no-runtime discovery, mocked runtime success and failures (including unknown exceptions and empty device keys), native static registration or dynamic loading and discovery, all GGML op support queries, empty graph success, and nonempty graph rejection without output modification. These are registration tests, not model inference or numeric conformance tests.