# Qwen4 DSL whole-model adapter (reference only)

llama.cpp supplies token IDs and positions; the plugin runs the whole Qwen4 graph on the DSL NPU backend and returns the last row of logits to llama.cpp for sampling. This is execution dispatch, not a GGUF weight converter.

- Start with `same_kernel_backend.cpp`: `prepare()` starts a worker that owns the DSL session; `graph_compute()` validates the active graph, copies IDs and positions, dispatches, and publishes logits. `release()` tears it down. The plugin advertises all ops solely to obtain a single split; it cannot execute arbitrary ops, models or partial graphs. Use only with the dedicated driver, one append-only sequence, one session, and last-token output.
- `same_kernel_bench.cpp` shows dynamic plugin loading, explicit device selection, GGUF loading, and one real `llama_decode()` call per batch. Compare output IDs against the direct DSL run before comparing times.
- `extract.ps1` reuses the matching direct DSL host setup and launch logic, including persistent KV buffers. It depends on that host's source layout, not a stable DSL API. For another framework, replace the ggml graph inspection with a call from its model executor: pass IDs, position and count; keep the session for the sequence; return logits.

## Build and run

Windows x64 MSVC Release, Intel NPU 5010. First build the Qwen4 direct DSL example (including its generated ABI, packed weights and kernel registry) and a shared-library Release Ninja build of this llama.cpp fork. The reference used generic_compute `507ecf284` and moviTools `26.08.2-internal`. Supply the matching Qwen4 DSL host snapshot; do not mix kernels, ABI headers and packs from different exports. Third-party source, weights, driver and binaries are not included.

In an x64 Visual Studio developer PowerShell, run `build.ps1 -DslRoot <DSL source> -DirectBuild <DSL Qwen4 VS build> -LlamaBuild <llama Ninja build> -Out <absolute output directory>`. Put the llama/ggml runtime DLLs from that same llama build on PATH. No change to the main llama.cpp build or its legacy NPU backend is required.

From the output directory in **cmd.exe**:

```bat
set QWEN4_STAGE_LOGITS=1
set QWEN4_RAW_GREEDY=1
same_kernel_bench.exe .\same_kernel_npu.dll MODEL.gguf PROMPT.u32le PACK_DIR ""
```

The GGUF supplies the llama graph. The separate DSL pack must represent the same checkpoint in device layout; a complete pack needs no overlay (the final empty argument). Only historical incomplete packs need the matching indexer overlay and overlay-capable DSL host. The prompt is raw little-endian uint32 token IDs, no header. A 1024-ID prompt generates one prefill and 63 decode launches for 64 output tokens. Check `NPU_TAKEOVER` and the final IDs. Timings exclude model/session setup and are not HTTP latency. Staging copies FP16 logits to ordinary host memory before conversion; it does not skip logits readback. This example is not the streaming server plugin or the legacy manifest runner.