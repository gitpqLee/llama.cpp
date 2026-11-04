# Intel NPU backend (whole-model DSL kernels)

`ggml-npu` runs models on the Intel NPU through whole-model kernels written in the NPU DSL
([libraries.npu.dsl](../../../libraries.npu.dsl), `model_runner/`). It is a regular ggml backend: llama.cpp is not
modified, models are ordinary GGUF files, and the device is selected like any other (`-dev NPU`).

## How it works

A **model package** is a `model.json` written by the DSL export plus the kernels linked into the runner. Its
`sources` section lists the GGUF tensors the kernels consume (name, type, shape, conversion) and the graph
conventions they rely on (the token lookup, the logits node, rope positions).

- **Load.** Weights stay where llama loads them in host memory (mmap'd GGUF). The device works in CPU buffers.
- **Split.** `supports_op` claims an op when its weights are sources of a package, and the remaining ops of that
  model's graph (including its KV/recurrent state ops). Ops of a model no package lists are not claimed: the CPU
  runs them, in place, as if the NPU were absent.
- **Run.** On the first compute the backend picks the package whose every source the graph reads, converts those
  tensors into the package layout (again whenever their data changes), then evaluates each ubatch as one kernel
  launch: token ids from the graph's token lookup, positions from rope, and the last token's logits written into
  `result_output`. The split's other nodes are not executed.

Adding a model is DSL-side only (DSL source, `ModelSpec`, export). The backend contains no model name, tensor name
or layer kind.

## Build

```sh
cmake -B build -DGGML_NPU=ON -DNPU_DSL_ROOT=<libraries.npu.dsl> -DGENERIC_COMPUTE_ROOT=<generic_compute> \
      -DNPUDSL_RUNNER_MODEL_DIRS=../kernels_poc/qwen36_dsl_poc
cmake --build build --config Release
```

`NPUDSL_RUNNER_MODEL_DIRS` (relative to `model_runner/`) lists the DSL directories whose CMake builds and registers
model kernels. `GGML_NPU_MODELS_DIR` sets a default package directory.

## Run

```sh
GGML_NPU_MODELS=<dir with model.json, or model.json paths separated by ':' (';' on Windows)> \
  ./build/bin/llama-cli -m model.gguf -dev NPU -c 4096
```

The log names the package found (`NPU: model package ...`) and bound (`NPU: running ...`).

Hardware validation steps (build on Windows, compare the generic runner with `qwen36_run --single` on the device,
check this backend's plumbing) are in [model_runner/HOW_TO_RUN.md](../../../libraries.npu.dsl/model_runner/HOW_TO_RUN.md).

## Limits (fail with an error, never wrong output)

- One NPU model per process; one sequence (`-np 1`), token ids only (no embedding / mixed multimodal batches).
- Logits of the last token of each ubatch only (no perplexity / all-logits / embeddings outputs).
- The package kernels own the state: a sequence may append or restart at position 0; other jumps (partial
  `seq_rm`, context shift) fail. State save/load does not include it.
- The whole model must be on the NPU: a graph reading weights the package does not consume (LoRA adapters,
  control vectors, an incomplete package) or split across backends is refused before any launch.
- Tensor split mode (`-sm tensor`) is not supported: the device works in host memory.

## Tests

`-DGGML_NPU_MOCK_RUNTIME=ON -DLLAMA_BUILD_TESTS=ON` (static build) builds `test-npu-backend`: a tiny qwen35moe GGUF
through the public llama API against the runner's mock NPU runtime, with a package generated from the model's own
graph. It checks weights, token ids, positions, ubatch splitting, restart, and the refusal of a split graph. It does
not check numerics. `test-llama-archs` with the NPU device registered checks that models without a package run
unchanged.
