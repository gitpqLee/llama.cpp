# Intel NPU backend (whole-model DSL kernels)

`ggml-npu` runs models on the Intel NPU through whole-model kernels written in the NPU DSL
([libraries.npu.dsl](../../../libraries.npu.dsl), `model_runner/`). It is a regular ggml backend: llama.cpp is not
modified, models are ordinary GGUF files, and the device is selected like any other (`-dev NPU`).

## Supported models and devices

| Model | Source checkpoint (HF) | GGUF | NPU |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B (qwen35moe) | `Twu31/Qwen3.6-35B-A3B-GPTQ-INT4-W4A16-LowLatency` | `hf_to_gguf_lossless.py` | PTL (NPU 5010, 3 tiles) |
| Qwen3-4B (qwen3) | `RedHatAI/Qwen3-4B-quantized.w4a16` | `hf_to_gguf_lossless.py --q8-rows lm_head.weight` | PTL (NPU 5010, 3 tiles) |

A model runs on the NPU when the DSL repository has its kernels (`kernels_poc/<model>/generated/model.json`); every other
GGUF runs on the other backends, unchanged.

## 1. Prerequisites

- An Intel NPU of a supported generation with its driver installed.
- Windows: Visual Studio 2026 with the C++ workload (and its CMake); Linux: a C++17 compiler and CMake >= 3.20.
- Python >= 3.10 (the build script uses only the standard library; converting a checkpoint to GGUF needs llama.cpp's
  `requirements/requirements-convert_hf_to_gguf.txt`).
- The repositories, side by side:
  - [libraries.npu.dsl](../../../libraries.npu.dsl) (the NPU DSL, its models and the runner);
  - `frameworks.ai.npu.generic_compute` at the revision of `libraries.npu.dsl/process/generic_compute.pin`, cloned with
    its submodules (`git clone --recursive -b dc/dsl_barrier <remote>`; see "generic_compute setup" in the DSL README);
  - moviTools (the NPU kernel compiler), needed only to build the models' kernels.

## 2. Build (each once)

Windows (PowerShell):

```powershell
$env:GENERIC_COMPUTE_ROOT = "C:\path\to\frameworks.ai.npu.generic_compute"
$env:MV_TOOLS_PATH        = "C:\path\to\moviTools"
$DSL   = "C:\path\to\libraries.npu.dsl"
$LLAMA = "C:\path\to\llama.cpp"
# a. the models: every model's kernels, installed as model directories into $DSL\build\npu_models
python "$DSL\model_runner\tools\build_models.py" --generator "Visual Studio 18 2026" --platform x64
# b. llama.cpp, with no model in it
cmake -G "Visual Studio 18 2026" -A x64 -S $LLAMA -B "$LLAMA\build" "-DGGML_NPU=ON" "-DNPU_DSL_ROOT=$DSL" `
      "-DGENERIC_COMPUTE_ROOT=$env:GENERIC_COMPUTE_ROOT" "-DBUILD_SHARED_LIBS=OFF"
cmake --build "$LLAMA\build" --config Release
```

Linux:

```sh
export GENERIC_COMPUTE_ROOT=<generic_compute> MV_TOOLS_PATH=<moviTools>
python <libraries.npu.dsl>/model_runner/tools/build_models.py
cmake -B build -DGGML_NPU=ON -DNPU_DSL_ROOT=<libraries.npu.dsl> -DGENERIC_COMPUTE_ROOT=$GENERIC_COMPUTE_ROOT
cmake --build build --config Release
```

Step a writes `build/npu_models/<model>/` in the DSL repository for each model -- a **model directory**: `model.json`
plus the model's kernels `npu<device>_<kernel>.elf`. Step b's configure prints `ggml-npu: model directories are looked
for in <DSL>/build/npu_models`. After a `git pull` of the DSL repository rerun step a only; llama.cpp does not change.
Quote every `-D...` argument in PowerShell (unquoted, `-DNAME=../path` is split and CMake ignores the path).

## 3. Get a GGUF

The NPU kernels keep their int4 symmetric group-128 weights, so the GGUF is written from the int4 checkpoint directly,
the integers and scales kept exact (Q4_0 blocks whose four 32-blocks per group share the scale; an ordinary Q4_0 file
for every other backend). `llama-quantize` does not write such files (a scale per 32 values), and a GGUF it writes is
refused by the NPU with the tensor's name.

```sh
python -m pip install -r <llama.cpp>/requirements/requirements-convert_hf_to_gguf.txt
python <libraries.npu.dsl>/model_runner/tools/hf_to_gguf_lossless.py --llama <llama.cpp> --model <checkpoint dir> \
       --outfile model.gguf [--q8-rows lm_head.weight]
```

`--q8-rows lm_head.weight` (Qwen3-4B) stores the checkpoint's unquantized lm_head as per-row int8, the format its
kernels read; every backend then uses the same integers. The conversion is deterministic (the same GGUF on Windows and
Linux).

## 4. Run

```sh
llama-completion -m model.gguf -dev NPU -c 4096 -p "The capital of France is" -n 64 --temp 0 -no-cnv
llama-cli        -m model.gguf -dev NPU -c 4096
llama-server     -m model.gguf -dev NPU -c 4096 -np 1 --no-cache-prompt -ctxcp 0 --cache-ram 0
```

Nothing has to be set: the backend reads the model directories it finds and binds the one whose tensors the GGUF has.
The log shows `NPU: model package <model> ...` (found), `NPU: running <model> ...` (bound, on the first evaluation,
after converting the GGUF's weights into the kernels' layout) and `NPU: setup ...` (where that time went). A GGUF no
model directory matches runs on the other backends. For `llama-server` the flags above keep it inside the backend's
limits (one sequence, no prompt-cache reuse, no state checkpoints); it is not yet validated on hardware.

Where model directories are looked for, first match wins:

1. `GGML_NPU_MODELS`, if set -- a model directory, a directory of model directories, or model.json paths separated by
   `;` (Windows) / `:` -- and nothing else;
2. otherwise the build's default, `<DSL>/build/npu_models` (`-DGGML_NPU_MODELS_DIR=<dir>` changes it), and
   `npu_models/` next to the executable.

## 5. Deploy to another machine

The default location is the DSL repository of the build machine. To ship a build: copy the executables, and into
`npu_models\` next to them the model directories (`<DSL>\build\npu_models\<model>`, or
`model_runner/tools/make_model_dir.py --model <model.json> --kernels <kernel build> --out npu_models/<model>`), plus
the GGUFs. The target machine needs the NPU driver only -- no DSL repository, moviTools or Python.

## 6. Troubleshooting

| Message | Meaning / action |
| --- | --- |
| `NPU: no usable model directory (searched ...)` | no model directory where the backend looked: run step 2a, or set `GGML_NPU_MODELS` |
| `NPU: ignoring model package ...: built for 5010/3 tiles, device is ...` | the kernels are for another NPU generation |
| `NPU: the GGUF does not fit the model package: <tensor> is <type> [...]; <model> takes ...` | a GGUF in another format (e.g. from `llama-quantize`): convert with step 3 |
| `group G of 128 values has 32-blocks with different scales` | a Q4_0 with per-32 scales: convert with step 3 |
| `... not found in kernel registry (kernel file: no ...\npu<key>_<kernel>.elf)` | the model directory lacks a kernel: rerun step 2a |
| `NPU: the model state is at position ..., the batch starts at ...` | a sequence jump the kernels cannot do (prompt-cache reuse, context shift): see the llama-server flags |
| first evaluation slow | the GGUF's weights are converted once per process (`NPU: setup ...` shows the time) |
| `NPU: ... GB of weights, twice resident while converted, against ... GB of memory` | the GGUF and its converted copy do not both fit in memory: the GGUF's pages are released as they are converted (`GGML_NPU_RELEASE_SOURCES=1` / `0` forces it on / off) |

## Limits (fail with an error, never wrong output)

- One NPU model per process; one sequence (`-np 1`), token ids only (no embedding / mixed multimodal batches).
- Logits of the last token of each ubatch only (no perplexity / all-logits / embeddings outputs).
- The package kernels own the state: a sequence may append or restart at position 0; other jumps (partial
  `seq_rm`, context shift) fail. State save/load does not include it.
- The whole model must be on the NPU: a graph reading weights the package does not consume (LoRA adapters,
  control vectors, an incomplete package) or split across backends is refused before any launch.
- Tensor split mode (`-sm tensor`) is not supported: the device works in host memory.

## How it works

A **model directory** is a `model.json` written by the DSL export plus the model's kernels. Its `sources` section lists
the GGUF tensors the kernels consume (name, type, shape, conversion) and the graph conventions they rely on (the token
lookup, the logits node, rope positions).

- **Load.** Weights stay where llama loads them in host memory (mmap'd GGUF). The device works in CPU buffers.
- **Split.** `supports_op` claims an op when its weights are sources of a model directory, and the remaining ops of
  that model's graph (including its KV/recurrent state ops). Ops of a model no directory lists are not claimed: the CPU
  runs them, in place, as if the NPU were absent.
- **Run.** On the first compute the backend picks the model whose every source the graph reads, loads its kernels into
  the NPU runtime's kernel registry, converts those tensors into the kernels' layout (again whenever their data changes),
  then evaluates each ubatch as one kernel launch: token ids from the graph's token lookup, positions from rope, and the
  last token's logits written into `result_output`. The split's other nodes are not executed.

Adding a model is DSL-side only (DSL source, `ModelSpec`, export). The backend contains no model name, tensor name or
layer kind. Hardware validation steps are in
[model_runner/HOW_TO_RUN.md](../../../libraries.npu.dsl/model_runner/HOW_TO_RUN.md).

## Tests

`-DGGML_NPU_MOCK_RUNTIME=ON -DLLAMA_BUILD_TESTS=ON` (static build) builds `test-npu-backend`: a tiny qwen35moe GGUF
through the public llama API against the runner's mock NPU runtime, with a package generated from the model's own
graph. It checks weights, token ids, positions, ubatch splitting, restart, and the refusal of a split graph. It does
not check numerics. `test-llama-archs` with the NPU device registered checks that models without a package run
unchanged.
