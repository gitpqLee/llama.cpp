#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Intel NPU backend running whole-model DSL kernels (libraries.npu.dsl, model_runner).
//
// A model package is a manifest (model.json) whose `sources` section maps GGUF tensors onto the kernels' weights.
// Packages are found through GGML_NPU_MODELS (model.json files or directories, ':'-separated, ';' on Windows) or
// ggml_backend_npu_add_model. While a GGUF loads, the backend accepts exactly the weights a package lists and
// converts them into the package's layout; the graph split it receives is then executed by the package's kernels.
// Ops the backend cannot place on a package report unsupported, so other models run on the other backends.
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_npu_reg(void);
GGML_BACKEND_API ggml_backend_t ggml_backend_npu_init(void);
GGML_BACKEND_API bool ggml_backend_is_npu(ggml_backend_t backend);
// Register a model package (model.json path) in addition to GGML_NPU_MODELS. Returns false if it cannot be used.
GGML_BACKEND_API bool ggml_backend_npu_add_model(const char * manifest_path);

#ifdef __cplusplus
}
#endif
