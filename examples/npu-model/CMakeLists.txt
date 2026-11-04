# The parent adds this directory under GGML_NPU, independently of
# LLAMA_BUILD_EXAMPLES. No changes to src or ggml are needed in this directory.
add_executable(llama-npu-model npu-model.cpp options.h)
target_compile_features(llama-npu-model PRIVATE cxx_std_17)
target_link_libraries(llama-npu-model PRIVATE llama)
install(TARGETS llama-npu-model RUNTIME)

if (TARGET npudsl_model_api)
	add_executable(llama-npu-audit npu-audit.cpp)
	target_compile_features(llama-npu-audit PRIVATE cxx_std_17)
	target_link_libraries(llama-npu-audit PRIVATE llama)
	install(TARGETS llama-npu-audit RUNTIME)

	add_executable(npu-package-bind package-bind.cpp ../../vendor/hash/sha256/sha256.c)
	target_compile_features(npu-package-bind PRIVATE cxx_std_17)
	target_include_directories(npu-package-bind PRIVATE ../../vendor ../../vendor/hash)
	target_link_libraries(npu-package-bind PRIVATE llama npudsl_model_api)
	install(TARGETS npu-package-bind RUNTIME)

	if (LLAMA_BUILD_TESTS)
			# Executable-side registration shares the bridge registry only in
			# static hosts. Shared hosts use build-time provider factories.
			if (NOT BUILD_SHARED_LIBS)
		add_executable(test-npu-provider provider-test.cpp ../../vendor/hash/sha256/sha256.c)
		target_compile_features(test-npu-provider PRIVATE cxx_std_17)
		target_include_directories(test-npu-provider PRIVATE ../../vendor ../../vendor/hash)
		target_link_libraries(test-npu-provider PRIVATE llama npudsl_model_api)
				if (NPUDSL_MODEL_QWEN36)
					target_compile_definitions(test-npu-provider PRIVATE NPUDSL_TEST_QWEN36)
				endif()
		cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
			add_test NAME test-npu-provider COMMAND test-npu-provider)
		cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
			set_tests_properties test-npu-provider PROPERTIES LABELS "npu;contract;mock;host;no-model-math" TIMEOUT 60)
	    endif()
		add_executable(test-npu-audit npu-audit-test.cpp)
		target_compile_features(test-npu-audit PRIVATE cxx_std_17)
		target_link_libraries(test-npu-audit PRIVATE llama)
		cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
			add_test NAME test-npu-audit COMMAND test-npu-audit)
		cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
			set_tests_properties test-npu-audit PROPERTIES LABELS "npu;contract" TIMEOUT 60)

				if (NPUDSL_MODEL_QWEN36)
				add_executable(test-npu-package-bind package-bind-test.cpp ../../vendor/hash/sha256/sha256.c)
		target_compile_features(test-npu-package-bind PRIVATE cxx_std_17)
		target_include_directories(test-npu-package-bind PRIVATE ../../vendor ../../vendor/hash)
		target_link_libraries(test-npu-package-bind PRIVATE llama npudsl_model_api)
		# This directory is reached through src before the parent enables testing.
		cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
			add_test NAME test-npu-package-bind COMMAND test-npu-package-bind)
		cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
			set_tests_properties test-npu-package-bind PROPERTIES LABELS "npu;contract" TIMEOUT 60)
		endif()
	endif()
endif()