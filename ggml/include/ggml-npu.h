#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Whole-model registration only. No GGML operations are supported.
// Returns NULL when the DSL runtime is not linked or no usable NPU is found.
GGML_BACKEND_API ggml_backend_t ggml_backend_npu_init(void);
GGML_BACKEND_API bool ggml_backend_is_npu(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_npu_reg(void);

#ifdef __cplusplus
}
#endif