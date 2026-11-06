ggml_add_backend_library(ggml-npu ggml-npu.cpp)
target_compile_features(ggml-npu PRIVATE cxx_std_17)

# The generic whole-model runner of libraries.npu.dsl. It names no model: models are directories (model.json plus the
# model's kernels) found at run time -- by default those the DSL build installs into <NPU_DSL_ROOT>/build/npu_models
# (python model_runner/tools/build_models.py), or a GGML_NPU_MODELS override.
option(GGML_NPU_MOCK_RUNTIME "ggml-npu: link the runner against the mock NPU runtime (host tests, no device)" OFF)

# the runner resolves GENERIC_COMPUTE_ROOT (cache or environment) itself
if (NOT NPU_DSL_ROOT AND DEFINED ENV{NPU_DSL_ROOT})
    set(NPU_DSL_ROOT "$ENV{NPU_DSL_ROOT}")
endif()
set(NPU_DSL_ROOT "${NPU_DSL_ROOT}" CACHE PATH "ggml-npu: path to the libraries.npu.dsl checkout")
if (NOT NPU_DSL_ROOT)
    message(FATAL_ERROR "GGML_NPU requires NPU_DSL_ROOT (a libraries.npu.dsl checkout)")
endif()
file(TO_CMAKE_PATH "${NPU_DSL_ROOT}/build/npu_models" _ggml_npu_models_default)
set(GGML_NPU_MODELS_DIR "${_ggml_npu_models_default}" CACHE PATH
    "ggml-npu: where model directories are looked for when GGML_NPU_MODELS is not set (default: the DSL build's)")
if (NOT TARGET npudsl_runner)
    if (NOT EXISTS "${NPU_DSL_ROOT}/model_runner/CMakeLists.txt")
        message(FATAL_ERROR "NPU_DSL_ROOT must contain model_runner/CMakeLists.txt")
    endif()
    set(NPUDSL_RUNNER_MOCK ${GGML_NPU_MOCK_RUNTIME} CACHE BOOL "" FORCE)
    add_subdirectory("${NPU_DSL_ROOT}/model_runner" "${CMAKE_CURRENT_BINARY_DIR}/model_runner")
endif()
if (NOT TARGET npudsl_runner)
    message(FATAL_ERROR "GGML_NPU: no NPU runtime -- set GENERIC_COMPUTE_ROOT, or GGML_NPU_MOCK_RUNTIME=ON for host tests")
endif()
target_link_libraries(ggml-npu PRIVATE npudsl_runner)
if (GGML_NPU_MODELS_DIR)
    file(TO_CMAKE_PATH "${GGML_NPU_MODELS_DIR}" _ggml_npu_models_dir)   # forward slashes: a C string literal
    target_compile_definitions(ggml-npu PRIVATE GGML_NPU_MODELS_DIR="${_ggml_npu_models_dir}")
    message(STATUS "ggml-npu: model directories are looked for in ${_ggml_npu_models_dir}")
endif()
