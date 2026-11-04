# Full-model NPU numerical/state audit

`llama-npu-audit` uses the real in-process `llama` library and the resolved DSL
provider session (currently only Qwen has real numerical support). The bridge
includes no Qwen headers; Qwen session adaptation is internal to the DSL provider.
It is built including when `LLAMA_BUILD_EXAMPLES=OFF`. It does not launch another
inference process, substitute a session, modify runtime code, or sample tokens.

## Invocation

```text
llama-npu-audit --model BOUND.gguf --weights PACKED_DIRECTORY --tokens IDS.txt --prompt-count N --output PREFIX [--reference REF_PREFIX]
```

The token input is ASCII whitespace-separated unsigned decimal IDs: the complete reference prompt followed by every generated token to feed back. No tokenizer, BOS insertion, chat template, or EOS stopping is applied. The prompt must contain at least one ID. The sequence is limited to 4096 IDs and 65536 input bytes; CLI paths are limited to 240 bytes. Each binary is limited to 512 MiB. The input sequence must fit these limits without truncation.

One `llama_decode` call evaluates the entire prompt with last-only output. Each subsequent teacher-forced ID gets its own decode call. Row 0 is the prediction after the last prompt token; row 1 is after consuming the first supplied continuation ID, and so on. There is also a row after the final supplied ID, even if it is EOS. There are `total_ids - prompt_count + 1` rows, not one row per prompt token.

The executable checks public sequence positions on every call and requests only the final token's logits. It checks every vocabulary element for finiteness, reports each row's argmax (first index wins ties), and writes all logits before another decode can replace them.

## Artifacts and state checks

- `PREFIX.meta.txt`: canonical LF text schema, rows, vocabulary, row-major float32/little-endian format, prompt count, ABI, package ID, tokenizer digest, exact input IDs, and zero-based inclusive row-end positions.
- `PREFIX.logits.f32`: initial single-prefill run, no binary header or padding.
- `PREFIX.replay.f32`: identical sequence after public `llama_memory_clear(memory, true)`.
- `PREFIX.sequential.f32`: clear again, evaluate every prompt token using one-token decode, then feed the same continuation IDs. Only aligned final-prompt and continuation logits are stored; every intermediate row is checked for finiteness.
- `PREFIX.split.f32`: for prompts longer than two tokens, another clear, then prompt chunks of `floor(N/2)` and `N-floor(N/2)` followed by the same continuation IDs. The intermediate chunk's full logits are checked for finiteness but not stored; final prompt and continuation rows align with the initial run.
- `PREFIX.audit.txt`: row finiteness/argmax, per-row relative L2 and reference top-two margin, full-vocabulary mismatched element/bit counts, exact argmax mismatch counts (including accepted near ties), maximum absolute/relative differences, relative L2 mean/max, near-tie/far-margin counts, policy criterion and PASS/FAIL for each comparison, and failure or completion status. Runtime/session diagnostics continue to use their existing stdout/stderr logs; capture both streams externally. Public decode calls are not claimed as device submission counters.

## Fixed comparison policy (policy correction)

The original blanket bit-exact policy was inappropriate for split-prefill rounding and independent CPU/NPU arithmetic. The corrected policy is fixed in code, with **no tolerance knob**, retry, or CPU fallback:

| Comparison | Acceptance criterion |
| --- | --- |
| `reset_replay` | Every float32 bit must match the initial run, including signed zero. Any bit difference fails, even with identical argmax. |
| `split_continuation` | Golden criteria below, with the initial unsplit NPU rows as reference. |
| `sequential_prompt` | Same golden criteria, with the initial unsplit NPU rows as reference. |
| `independent_reference` | Same golden criteria, with the supplied independent rows as reference. |

For each complete vocabulary row, compute `relL2 = norm(actual - reference) / norm(reference)`. Accept only if the **mean across rows <= 0.04 AND maximum row <= 0.10**, and **each row has exact argmax equality OR the reference maximum minus the reference logit at the NPU-chosen token <= 0.5**. All inequalities are inclusive. This is the existing chosen-token-gap policy and `TAU=0.5`, `MEAN_TOL=0.04`, `MAX_TOL=0.10` from the DSL golden checker, not newly fitted tolerances. The separately reported `reference_top2_margin` is diagnostic only: a close reference runner-up cannot excuse choosing a distant third token. Host regression tests explicitly reject that case.

First index wins argmax ties. Equal reference maxima give top-two margin zero; a single-entry vocabulary has infinite top-two margin but cannot have different argmax indices. Norm subtraction, squaring, accumulation and division use doubles, avoiding float32 overflow/underflow. A zero reference norm gives relative L2 zero only for an all-zero difference; otherwise it is infinite and fails. Empty comparisons fail. Nonfinite logits in **either** row always fail, including identical infinities/NaNs; there is no near-tie exemption from finiteness or L2 checks.

Bit differences, maximum absolute differences and exact argmax differences are always reported under both policies, even when the golden criterion passes. `max_rel` is only an elementwise diagnostic using `max(abs(reference), 1e-30)` as denominator; it is **not** the relative L2 criterion. Full artifacts are retained after numerical mismatch; runtime failures can leave partial files. A completion status in the audit log, not mere file existence, indicates success. Output files are exclusively created, existing paths are rejected, and successful writes and closes are checked.

### Original failure evidence (unchanged)

The original full-model hardware audit at `C:\intel\pengqiang\poc\downloads\qwen36-npu-audit.audit.txt` **failed the old zero-tolerance policy**; it did not pass overall. It recorded three baseline/reset/split rows, bit-exact reset (`mismatched_elements=0`, `argmax_mismatches=0`), and split rounding differences (`mismatched_elements=734842`, `max_abs=0.39837646484375`, `argmax_mismatches=0`), followed by `FAIL: full-logit bit-exact comparison failed; tolerance unchanged (zero)`. Its `REFERENCE_NOT_SUPPLIED` status means it did not validate independent numerics. The log and all original binary artifacts remain untouched. This policy correction does not relabel that failed run as passed and does not by itself establish golden acceptance of those saved rows. Use a fresh prefix for any future execution; no hardware or Python reference run is required for host policy tests.

## Independent same-checkpoint reference

`--reference REF_PREFIX` consumes `REF_PREFIX.meta.txt` and `REF_PREFIX.logits.f32`. Supply independently computed **full logits**, not greedy IDs, from the exact source checkpoint used to pack the bound package, using the same teacher-forced input IDs and row convention. The metadata must byte-match the canonical schema above, including the GGUF's package/tokenizer identity and exact token sequence; the binary must have exactly `rows * vocab * 4` bytes. A reference producer can use the audit metadata as the run specification, but must independently compute the binary. Export float32 little-endian contiguous rows without a NumPy/header wrapper. The caller owns reference provenance: a metadata match cannot prove which implementation produced a binary.

The executable compares all supplied reference rows with the initial NPU run using the fixed golden criterion above, not bit equality. It still reports every aggregate bit/argmax difference, per-row relative L2 and reference top-two margin, and the criterion/result. Without `--reference`, the tool exports logits and checks reset/continuation, but explicitly reports `STATE_PASS; INDEPENDENT_NUMERICS_NOT_TESTED`. Such a run is **not** evidence of independent full-model numerical correctness. A subsequent run with a reference needs a fresh output prefix.

Exit codes: 0 completion/help, 2 invalid arguments/input, 1 model/package/runtime/output/state/numerical failure. Model/package identity validation is performed by the existing NPU bridge before inference. The hardware audit is manually invoked; the separate full-model smoke CTest is enabled only with explicit fixture paths. `test-npu-audit` tests only parser bounds, token input, fixed golden L2 mean/max gates, far-margin/near-tie argmax cases, bit-exact reset (including single ULP/signed zero), double norms, finite rejection, report criteria and exclusive output creation; it does not initialize a model or submit device work.

## Build and host-only tests (PowerShell)

```powershell
cmake --build C:\intel\pengqiang\poc\build-llama-npu-model --config Release --target llama-npu-audit test-npu-audit --parallel 4
ctest --test-dir C:\intel\pengqiang\poc\build-llama-npu-model -C Release -V -R '^test-npu-audit$'
```

The CLI is `C:\intel\pengqiang\poc\build-llama-npu-model\bin\Release\llama-npu-audit.exe` with the invocation shown above. `--help` is host-only. There are no `--tau`, `--mean-tol`, `--max-tol`, or other tolerance overrides. Building may compile linked NPU dependencies, but these tests do not execute device work or the Python reference.