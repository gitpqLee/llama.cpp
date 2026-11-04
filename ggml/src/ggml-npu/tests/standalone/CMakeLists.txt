cmake_minimum_required(VERSION 3.20)
project(ggml_npu_registration_tests LANGUAGES C CXX)

enable_testing()
set(GGML_NPU ON CACHE BOOL "Enable NPU registration")
set(GGML_NPU_TESTS ON CACHE BOOL "Enable registration contract tests")
set(GGML_BUILD_TESTS OFF CACHE BOOL "Disable unrelated ggml tests")
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Disable ggml examples")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../../.." "${CMAKE_CURRENT_BINARY_DIR}/ggml")