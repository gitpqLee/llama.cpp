# Deploying the complete Qwen DSL model on another Windows machine

## Scope and current delivery status

The host bridge and binder now use the DSL version-1 generic provider boundary
(`npudsl_model_api`, DSL model_api directory). Deliver that directory as part of
the DSL source snapshot. Qwen compilation and generated ABI handling are internal
to the DSL entry point. Future static providers register their own model metadata,
schema and session factory without changing llama bridge code. This delivery still
supports **only the real Qwen checkpoint below**; the small MOCK provider is test
code only. Existing bound package/GGUF bytes and fingerprints need no changes.

This is a custom llama.cpp fork with an **in-process whole-model bridge**.
It is not upstream llama.cpp support, arbitrary GGUF offload, or an inference
subprocess wrapper. The supported checkpoint is
`Twu31/Qwen3.6-35B-A3B-GPTQ-INT4-W4A16-LowLatency`, revision
`ca91a480f397ff995ef767acde2ef5d2ee7f9b80`, not Qwen3.8 Q4_K_M.

The implementation currently exists as **uncommitted local changes in two
repositories**. Cloning the current remote branches alone will NOT reproduce
it. Before handing this to another person, deliver reviewed source snapshots
including new/untracked implementation files, or commit/publish the intended
changes in BOTH repositories after authorization. Do not copy just a git diff
without checking for untracked files. Do not deliver machine-specific build
directories as a substitute for sources. The runtime checkout is unmodified.

No second-machine reproduction has yet been performed. See
[the measured acceptance and limitations](NPU-validation.md).

## 1. Target machine prerequisites

- Windows x64. The tested host build uses Visual Studio 18 2026 with MSVC19.50,
  Windows SDK, CMake and the Desktop C++ workload.
- **NPU5010, three tiles, two SHAVEs per tile / six workers.** The current
  session explicitly checks this topology. A generic Intel NPU, another
  generation, GPU or CPU is not an interchangeable target.
- A compatible Intel NPU driver exposing the Level Zero graph/native-kernel
  functionality used by the generic-compute runtime, plus a compatible
  `ze_loader.dll`. The test machine has an internal development driver;
  ordinary consumer-driver compatibility is NOT established. Obtain the
  approved driver/loader installation from the platform owner. Copying an EXE
  cannot install a driver or add missing graph extensions.
  The measured driver version is **32.0.300.23847**, device description
  `npu-driver-user-user_vpuxjenkins_npu_cid_update_20261022_022130-23847 RelWithDebInfo`.
  This is an environment baseline, not a claim that arbitrary versions work.
- For a source build: MoviTools **26.05.0-internal** and access to the
  generic-compute runtime repository and its required third-party headers/libs.
  These are not installed by cloning llama.cpp. Access and redistribution rights
  for internal toolchains, runtime, driver and model must be checked separately.
- RAM: tested on32GiB; use at least this as a starting configuration and verify
  NPU-visible allocation succeeds. The resident package is approximately18.93GB
  plus state/scratch and host overhead. Avoid simultaneous CPU reference runs.
- Disk: approximately19GB for the bound model package. Source export additionally
  needs21.55GB for the original checkpoint, Python dependencies, sources and build
  output; allow at least55GB free for that workflow, more for retained artifacts.

## 2. Choose a delivery mode

### A. Prebuilt binary + bound assets (simplest for an end user)

On a compatible platform, deliver:

```text
release/
  bin/llama-npu-model.exe
  bin/llama-npu-audit.exe         # optional numerical/state verification
  model/qwen36-ca91a480-bound.gguf
  model/qwen36-ca91a480-npu/
    identity.txt
    manifest.txt
    layer_types.txt
    globals.bin
    layer_00.bin ... layer_39.bin
  provenance/                    # source commits/snapshot IDs and checksums
```

The packed directory contains44 required files totaling18,930,641,789 bytes in
the tested export. Copy all of them **byte-for-byte**, including text files.
Do not normalize line endings, regenerate weights, or rebind an already-bound
package merely because its directory moved. Absolute paths are supplied at run
time, not embedded in the identity. Treat assets as immutable during verification
and inference. The bound GGUF and its package must travel as a matched pair.

The tested static-host executable still imports `ze_loader.dll`, VC++ runtime
(`MSVCP140`, `VCRUNTIME140`, `VCRUNTIME140_1`), `VCOMP140` OpenMP runtime and Windows
system DLLs. Install the appropriate official x64 VC runtime and approved Level
Zero components; do not copy random DLLs from another PC. Embedded kernels are
linked through the generated registry; recipients of this prebuilt artifact do
not need Python, PyTorch or MoviTools just to invoke it, but driver/runtime
compatibility must still be reproduced. Validate redistribution permissions.

Publish checksums for the EXEs, bound GGUF and package identity through a trusted
channel. The existing identity validates content/binding but is not a signature.
Keep the original checkpoint provenance and its license/usage requirements.

### B. Build from the delivered sources

Example layout (choose any absolute root):

```text
D:/npu-deploy/
  llama.cpp/                         # delivered modified fork
  libraries.npu.dsl/                 # delivered modified DSL checkout
  frameworks.ai.npu.generic_compute/ # pinned unmodified runtime
  models/
```

The tested base revisions, BEFORE local modifications, are:

| Repository | Base revision |
| --- | --- |
| llama.cpp | `785322d17f2d51e5185cf996a1f4e46d87640841` |
| libraries.npu.dsl | `3c7e0ac412440b16d77577e1de8a1a1e55accdf5` |
| frameworks.ai.npu.generic_compute | `6bb585cb85a9777dbd8c0a9136b721f4e9010065` |

Base revisions alone do not include this integration: record final delivered
commit IDs or authenticated snapshot hashes, not just these bases. The DSL
snapshot must include the shared session, common headers/ASM and generated
Qwen36 sources/ABI that were built together. Do not regenerate kernels using
unrelated defaults or substitute another `QWEN36_GEN`.

PowerShell, using the tested configuration:

```powershell
$root = 'D:/npu-deploy'
cmake -S "$root/llama.cpp" -B "$root/build" -G 'Visual Studio 18 2026' -A x64 `
  -DGGML_NPU=ON -DBUILD_SHARED_LIBS=OFF `
  "-DNPU_DSL_ROOT=$root/libraries.npu.dsl" `
  "-DGENERIC_COMPUTE_ROOT=$root/frameworks.ai.npu.generic_compute" `
  '-DMV_TOOLS_PATH=C:/npu/moviTools_26.05.0-internal/26.05.0-internal' `
  -DQWEN36_DEVICES=5010 -DQWEN36_GEN=generated -DQWEN36_DEQUANT_ASM=ON `
  -DQWEN36_PROFILE=OFF -DQWEN36_TRACE=OFF `
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF `
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF `
  -DLLAMA_BUILD_TESTS=ON -DGGML_NPU_TESTS=ON -DQWEN36_SESSION_TESTS=ON
if ($LASTEXITCODE -ne 0) { throw 'Configure failed' }
cmake --build "$root/build" --config Release --parallel 4 --target `
  llama-npu-model llama-npu-audit npu-package-bind test-npu-audit `
  test-npu-model-contract test-npu-context test-npu-package-bind `
  test-npu-native test-npu-unavailable test-npu-mock `
  qwen36_session_contracts qwen36_session_api test-npu-provider
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
```

If the matched prepacked assets were delivered, skip the next section.

## 3. Optional: produce assets from the pinned original checkpoint

This is producer work, not required on each end-user machine. Download the exact
revision above, all indexed shards plus INT8-head shard, config and tokenizer
files; verify upstream file sizes and LFS SHA256. A mirror is a transport option,
not permission to change revisions. Do not use the unrelated Q4_K_M GGUF as
weight input or borrow its tokenizer.

The export uses CPU Python3.11, NumPy, CPU PyTorch, safetensors. Official tokenizer
conversion additionally needs transformers, tokenizers, sentencepiece and the
fork's gguf Python package. Tested versions include torch2.15.0+cpu,
safetensors0.8.0, transformers5.22.1 and tokenizers0.23.3. Set up a dedicated
environment and install requirements consistent with the delivered conversion
code. CPU reference acceptance needs the same environment, but inference does not.

With that environment active, choose NEW output paths:

```powershell
$root = 'D:/npu-deploy'
$env:QWEN36_CKPT = "$root/models/Qwen3.6-35B-A3B-GPTQ-INT4-W4A16-LowLatency"
Set-Location "$root/libraries.npu.dsl"
python -u -c "from pathlib import Path; import torch; torch.set_num_threads(8); from qwen36_e2e.export import write_weights; p=Path(r'D:/npu-deploy/models/qwen36-ca91a480-npu'); assert not p.exists(); write_weights(p)"
if ($LASTEXITCODE -ne 0) { throw 'Export failed; do not bind partial output' }
python "$root/llama.cpp/convert_hf_to_gguf.py" "$env:QWEN36_CKPT" --vocab-only `
  --outfile "$root/models/qwen36-ca91a480-tokenizer.gguf"
if ($LASTEXITCODE -ne 0) { throw 'Tokenizer conversion failed' }
& "$root/build/bin/Release/npu-package-bind.exe" `
  --tokenizer "$root/models/qwen36-ca91a480-tokenizer.gguf" `
  --weights "$root/models/qwen36-ca91a480-npu" `
  --output "$root/models/qwen36-ca91a480-bound.gguf" `
  --source-id 'Twu31/Qwen3.6-35B-A3B-GPTQ-INT4-W4A16-LowLatency@ca91a480f397ff995ef767acde2ef5d2ee7f9b80' `
  --acknowledge-external-weights
if ($LASTEXITCODE -ne 0) { throw 'Binding failed' }
```

The binder refuses existing identity/output paths. Do not delete identity to
silence a mismatch; diagnose checkpoint, tokenizer, ABI and file integrity.
See [binding details](../../examples/npu-model/README.bind.md).
Expected identities for the tested export:

- Package: `a538eeb9f0bc150aec7fd1763b8e18e7f0524e536540a792d0ae108d98434ef9`.
- Tokenizer: `7db36422d6384bf4344c6d47448ae4bde24829d59481498eda90a4256e5a51af`.

An exporter/converter version difference may change hashes; investigate it and
repeat independent validation rather than assuming matching model names suffice.

## 4. Run on the target machine

For source-build layout (change binary location to release/bin for mode A):

```powershell
$root = 'D:/npu-deploy'
& "$root/build/bin/Release/llama-npu-model.exe" `
  --model "$root/models/qwen36-ca91a480-bound.gguf" `
  --weights "$root/models/qwen36-ca91a480-npu" `
  --prompt 'The capital of France is' --predict 3 --ctx-size 64
if ($LASTEXITCODE -ne 0) { throw 'NPU inference failed; no CPU fallback' }
```

This tested prompt produced ` Paris.`. Output token text is stdout; diagnostics,
real whole-model submission counters and `NPU_PERF` timing are stderr. Do not
mistake `--help`, native device registration or package binding for inference.
The measured15-token prompt had ready TTFT457ms and output8.81tokens/s on the
test machine; fresh-process setup was78.69s. These are not target-machine speed
guarantees and filesystem caches were not forced cold.

The tool accepts raw text, does NOT apply a chat template automatically, and is
single-sequence with contiguous positions and last-token logits. MTP, vision,
KV shifting, speculative decoding and state serialization are not supported.
Long context is not numerically/performance certified by the short fixtures.
The fixed approximately10-second submission wait can matter for long prefill.

## 5. Acceptance on another machine

1. Match the delivered hardware/driver/toolchain/source/asset manifest.
2. Verify trusted checksums; run actual short-prompt inference with exit0,
   positive prefill/decode/kernel counters and no device/timeout errors.
3. Build the listed host tests and run the selected suite:

```powershell
ctest --test-dir "$root/build" -C Release -R 'npu|qwen36_session' `
  -E '^test-npu-model-full$' --output-on-failure
```

4. Enable the real smoke fixture explicitly (otherwise this test is skipped):

```powershell
cmake -S "$root/llama.cpp" -B "$root/build" `
  "-DLLAMA_NPU_MODEL_TOKENIZER=$root/models/qwen36-ca91a480-bound.gguf" `
  "-DLLAMA_NPU_MODEL_WEIGHTS=$root/models/qwen36-ca91a480-npu"
ctest --test-dir "$root/build" -C Release -R '^test-npu-model-full$' --output-on-failure
```

5. For numerical acceptance, deliver the independently generated reference
   metadata/tokens/full-logit files and use the
   [audit protocol](../../examples/npu-model/README.audit.md). Require finite
   complete logits, fixed golden L2/chosen-token criteria, bit-exact reset,
   split continuation and sequential prompt checks. Reference provenance must
   remain the original checkpoint, not outputs copied from the tested NPU.
6. Preserve complete commands, stdout/stderr, exit codes, exact identities,
   driver versions and device topology. Do not declare success based solely on
   text that happens to look plausible.

## 6. Embedding into another llama application

The demonstrated entry point is `llama-npu-model`, **not** an unchanged
`llama-cli -ngl ...` or `llama-server`. Application integration requires the
modified public header/library: load the bound GGUF with `vocab_only=true`, set
`llama_context_params.npu_model_path` to the immutable package directory before
context construction, then use ordinary tokenization, supported `llama_decode`
batches, last-token logits and CPU sampling. Set single sequence0 and matching
context/batch budgets. Never fall back to normal model compute on rejection.

Use the full working example in
[npu-model.cpp](../../examples/npu-model/npu-model.cpp). A server integration,
multi-user scheduling, session persistence and general upstream CLI wiring have
not been delivered or validated. Do not advertise them as already supported.