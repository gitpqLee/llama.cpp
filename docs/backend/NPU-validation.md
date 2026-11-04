# Complete-model bridge: validation status

## B-route interface v3 verification — November 3, 2026

This is the latest architecture and regression result; older v2 sections below
are historical evidence. Sources and implementation details are described in
[NPU-model-contract.md](NPU-model-contract.md) and the sibling DSL model API README.
The model-independent host core, optional Qwen factory archive, provider-owned
asset preparation, checked Session options/effective configuration, and opaque
append/reset state are implemented. The generic bridge and binder no longer
require a manifest/blob tensor schema; a test-only opaque-record provider covers
actual binding and public context execution without one.

Review additionally repaired Qwen-OFF runtime/test coupling and bound prepared
handles to a private shared immutable preparation/creation implementation.
Changing either callback gives a new identity; tests cover descriptor copies,
registry snapshots, wrong callbacks, source/callback-capture lifetime, unverified
handles, invalid requests and inconsistent effective configuration.

All final evidence below uses the unique sibling downloads prefix
**npu-v3-fix-20261103-141712**:

- `-core-exit.log` and `-extension-exit.log`: standalone Qwen-OFF core and
  independently linked static factory configurations/builds/tests all exit 0.
- `-host-configure-final.log`, `-host-build-final.log`,
  `-host-ctest-final.log`, `-host-final-exit.log`: actual llama host build with
  Qwen OFF, no generic-compute runtime, exit 0. **17 tests passed, 1 intentional
  full-model fixture skip**. Generic provider tests report **124 checks** and
  core tests **44 checks**. This is host-contract evidence, not device execution.
- `-build-final.log` / `-build-final-exit.log`: all **13** requested Release
  targets built, exit 0. Existing MSVC same-name-object warning remains; no
  warning-free build is claimed.
- `-ctest-final.log` / `-ctest-final-exit.log`: default Qwen-enabled selection
  **21/21 passed, no skips**, exit 0, 99.17 seconds. Record provider tests report
  **126 checks**, mocked Qwen public bridge **654 checks**, and the real full-model
  smoke passes. Earlier non-final/interrupted logs are not final acceptance.
- `-real-audit-console.log`, `-real-audit-exit.log`, and
  `-real-audit.audit.txt`: freshly built real `llama-npu-audit` executed with the
  unchanged bound GGUF/external Qwen package, supplied independent reference,
  five prompt tokens and two continuation inputs, exit 0. Three complete
  248320-element FP32 rows: **mean relative L2 0.0065263350589211148, maximum
  0.011090676171006746, 3/3 exact argmax**. Finite outputs; reset/replay is
  bit-exact. Split continuation passes (mean 0.01884174154476043, maximum
  0.022874723770587708); sequential prompt passes (mean 0.0059349564395640613,
  maximum 0.012671023907800257). The latter two comparisons are to the device
  baseline, not additional direct CPU-reference comparisons. All original golden
  criteria remain unchanged. Raw logits/replay/split/sequential outputs and meta
  accompany the audit under the same prefix.
- `-protected-check.log`: **643 protected files, zero mismatches** against the
  retained v2 baseline. `-integrity.log`: both changed repositories and runtime
  pass `git diff --check`; runtime working tree is clean. No asset rewrite,
  kernel/packer/runtime change, destructive Git operation or commit was performed.

Only Qwen has real numerical support. This rerun does not certify long contexts,
new models, arbitrary GGUF weight loading, dynamic plugins, a stable C ABI, or
performance improvements. Qwen still allocates its fixed 65792-position capacity;
requested logical limits do not imply smaller physical allocations. A second
real DSL model remains the strongest next generality acceptance test.

## Contract review follow-up — November 3, 2026

Fresh verification of the interface-version-2 contract fixes, Windows / MSVC
18 2026 Release. This section supersedes earlier contract-test counts below;
historical hardware evidence remains historical, not additional new acceptance.

- Public `llama_init_from_model` and bridge validation bound K/V type enums
  before asserting GGML helpers. Negative, `GGML_TYPE_COUNT`, above-count and
  INT32_MAX inputs are rejected for both cache fields, with and without NPU
  opt-in, before provider creation.
- B-route supports unequal logical/physical caps but each decode call must fit
  both reported caps. Zero `n_ubatch` selects the normalized logical cap; there
  is no silent microbatch splitting. The small provider tests logical9/physical2,
  boundary acceptance, oversize rejection/state preservation/healthy retry, and
  the extended decode entry point. Existing CLI/audit equal-cap requests remain
  valid. This is a host scheduling contract, not long-prefill numerical evidence.
- Partial six-field format declarations are rejected by field in preflight,
  before package path resolution or hashing. Tests omit each field in turn and
  use a nonexistent package path to verify the metadata diagnostic wins. Only
  all-six-absent metadata may defer to subsequent validated package evidence;
  its unchanged bound legacy package is positively tested.
- Bounded metadata failures in `validate_model_contract` return structured
  field diagnostics, including oversized architecture/semantic/format values,
  callback-only values and throwing readers. Summaries are bounded; throwing
  `require()` retains the diagnostic fields.

All fresh artifacts below are in the sibling downloads directory, under the
unique prefix **npu-contract-review-20261103-122314-d088830c**:

- `-build-v2.log` / `-build-v2-exit.log`: all **13** requested Release targets
  completed, exit0. The initial `-build.log` was interrupted and has no completed
  exit evidence; it is retained and is not used to claim build success.
- `-ctest.log` / `-ctest-exit.log`: `ctest -C Release -V -R
  'npu|qwen36_session'`, **21/21 passed**, no skips, exit0. The test-only small
  provider reports **125 checks**; the existing mocked Qwen bridge reports
  **654 checks**. The real full-model smoke test passed (85.05 seconds, including
  setup; not an isolated prefill timing or independent numerical comparison).
- `-audit-command.log`, `-audit-help.log`, `-audit-stdout.log`,
  `-audit-stderr.log`, `-audit-exit.log`: actual freshly built
  `build-llama-npu-model/bin/Release/llama-npu-audit.exe` invocation, exit0,
  completed November3 at12:29:40 -08:00. It used the unchanged
  `models/qwen36-ca91a480-bound.gguf`, `models/qwen36-ca91a480-npu`,
  `downloads/qwen36-reference.tokens.txt`, prompt-count5 and the supplied
  independent `downloads/qwen36-reference` prefix. No mock or inference
  subprocess replaced this hardware execution.
- `-audit.audit.txt`, `-audit.meta.txt`, `-audit.logits.f32`,
  `-audit.replay.f32`, `-audit.split.f32`, `-audit.sequential.f32`: three complete
  248320-element FP32 rows (end positions4/5/6). Baseline versus the original
  independent FP32 checkpoint reference: **mean relative L2 0.0065263350589211148,
  maximum 0.011090676171006746, 3/3 exact argmax**, finite outputs, unchanged
  golden acceptance. Reset/replay has **zero mismatched elements/bits**. Split
  continuation versus the device baseline passes (mean0.01884174154476043,
  max0.022874723770587708); sequential prompt versus that baseline also passes
  (mean0.0059349564395640613, max0.012671023907800257). Those two in-executable
  comparisons are to the device baseline, not additional direct CPU-reference
  comparisons. Final counters:4 prefill,13 decode,17 whole-model submissions.
  The run reports no CPU fallback, timeout or nonfinite output. This rerun covers
  the five-token Paris prompt and two continuation inputs only.
- `-protected-baseline.clixml` / `-preservation.log`: **643 protected files,
  zero hash changes** across runtime files, tracked/nonignored kernel/session
  sources, packer/exporter and the selected bound/tokenizer/reference assets.
  `-diff-check.log`: all three repositories' `git diff --check` exit0;
  runtime working tree remains clean. Existing unrelated changes are preserved;
  no asset rewrite, kernel/packer edit, destructive Git operation or commit.

## Generic provider refactor verification

The llama bridge and package binder now consume the version-2 DSL model API.
Qwen architecture/dimension checks, generated tensor schema, package layout and
session adaptation live in the DSL Qwen provider. Context capacity, vocabulary
size and executed layer count come from the selected provider. Existing Qwen
ABI, identity bytes, tokenizer fingerprint and bound assets remain unchanged.

- All thirteen selected Release targets rebuilt successfully:
  sibling downloads/npu-provider-verified-build.log.
- The complete `npu|qwen36_session` CTest selection passed 21/21, including real
  Qwen inference: npu-provider-verified-ctest.log.
- The test-only second provider uses seven vocabulary entries, nine positions,
  two layers and different package filenames/alignment. Its 52 checks cover
  generic binding and public llama context evaluation, registry rejection,
  reset, output invalidation and permanent poisoning. It is not a second real
  numerical model and is never registered in production.
- A fresh real Qwen audit against the original independent CPU reference passed:
  npu-provider-audit-20261103-075922.audit.txt. Three full 248320-element rows
  had mean relative L2 0.0065263351, maximum 0.0110906762 and 3/3 exact argmax.
  Reset was bit-exact; split continuation and sequential prompt checks passed
  the unchanged golden criteria. This rerun covers the five-token Paris prompt
  and two continuation inputs, not new long-context acceptance.
- Both repositories passed `git diff --check`; the runtime remains unmodified.

Final CLI review replaced allocation from the unchecked context option with
token-count probing. A context of 2147483647 with a short prompt is rejected by
the Qwen provider without allocating a context-sized token vector (exit 1;
npu-provider-context-limit-negative.log). The final build and complete 21/21
CTest rerun passed again (npu-provider-final-build.log and
npu-provider-final-ctest.log). Executable-side registry extension tests require
a static llama host; shared hosts must register providers through build-time
factories inside the bridge's library. Shared-host execution was not validated
in this run.

This is a static C++17 provider boundary, not a stable C ABI or dynamic-plugin
loader. Adding a provider can require relinking but must not require changes to
the generic llama bridge. Only Qwen has real numerical acceptance so far.

## Scope and verdict

The fork now contains an explicit in-process model path:

llama tokenizer → `llama_decode` → generic provider session → existing complete DSL
prefill/decode → FP32 logits → llama sampling.

**Bounded real-hardware end-to-end execution and independent numerical/state
acceptance now PASS for the two fixtures below.** This is not a long-context,
throughput, arbitrary-checkpoint or production-readiness certification.
Successful compilation, hardware discovery and mock tests alone are not numerical evidence.
The previous external integration's real GET_ROWS-only results do not validate
this new whole-model path.

## Implemented

- Native `GGML_NPU` registration, restricted to the exported NPU5010 topology.
  No numerical GGML operator is advertised as supported.
- Explicit `llama_context_params.npu_model_path` selection; normal contexts remain
  unchanged. The model path requires vocab-only GGUF loading and never loads GGUF
  compute weights or executes a CPU transformer fallback.
- Shared host implementation with the original DSL runner, complete prefill and
  decode kernels, single-sequence contiguous positions, last-token logits, KV/GDN
  reset, poisoned sessions and strict unsupported-operation rejection.
- Package/ABI validation, per-file SHA256, effective tokenizer fingerprint v2,
  explicit metadata-only binding tool, no implicit quantization or substitution.
- Failed-wait quarantine: if completion cannot be established, the entire runtime
  and buffers are intentionally retained until process exit, and new sessions
  are rejected. This avoids freeing in-flight resources or calling the runtime's
  throwing stream destructor. It is not device recovery or cancellation.

## Executed checks

Windows / MSVC 18 2026 Release, runtime revision
`6bb585cb85a9777dbd8c0a9136b721f4e9010065` (unmodified).
The fork starts at `785322d17f2d51e5185cf996a1f4e46d87640841`; new work is
uncommitted. DSL session changes are also uncommitted dependencies.

- Native static build: complete-model executable, binding tool, session library,
  generated device kernels and contract tests build successfully.
- Initial root NPU/session suite: **18 passed, 1 explicitly skipped** before
  compatible assets were downloaded. The real model fixture is now enabled and
  passes; final host/session regressions are **19/19**, with no skips, plus the
  passing actual full-model CTest. The label-filtered suite is **10/10** including
  that real model test (overlaps the 19 host regressions; do not add both counts).
- Real llama-context / link-time mock session: **654 checks**, including normal
  decode/output mapping, default opt-in behavior, option and batch rejection,
  state/reset, corrupt assets, poisoning and optional MTP metadata. These tests
  do not execute device model mathematics.
- Session host contracts include nonfinite logits, unsupported device topology,
  partial-submit errors and isolated persistent-timeout/fatal-destructor controls.
- Binding tests include effective tokenizer overrides and LF/CRLF metadata.
- Native registration runs against the actual installed NPU runtime. This is
  discovery/capability evidence only, not a model execution test.
- `GGML_NPU=OFF`, shared-library CPU build and public opt-in-default test pass.

Saved build/test logs are in the sibling build directory
`build-llama-npu-model`, with prefix `npu-final-`. CPU-only logs are in the
sibling downloads directory with prefix `npu-off-`. MTP regressions also have
logs with prefix `npu-mtp-contract-`.

## Exact source and bound assets

Downloaded through hf-mirror.com, retaining the pinned checkpoint identity:

- Source: `Twu31/Qwen3.6-35B-A3B-GPTQ-INT4-W4A16-LowLatency`.
- Revision: `ca91a480f397ff995ef767acde2ef5d2ee7f9b80`.
- Download: 21 files, 21,549,143,954 bytes; file lengths and LFS SHA256 verified.
- Source directory: sibling models directory, Qwen3.6-35B-A3B-GPTQ-INT4-W4A16-LowLatency.
- Packed package: sibling models directory, qwen36-ca91a480-npu; all 40 trunk
  layers plus globals, original group128 signed INT4 and INT8 head layouts.
  Existing export logic performs layout repacking and BF16-to-FP16 conversion,
  not hidden requantization. MTP and vision tensors are not executed.
- Tokenizer: official same-directory `--vocab-only` conversion, explicitly bound
  into qwen36-ca91a480-bound.gguf with the package binder.
- Package ID: `a538eeb9f0bc150aec7fd1763b8e18e7f0524e536540a792d0ae108d98434ef9`.
- Effective tokenizer SHA256: `7db36422d6384bf4344c6d47448ae4bde24829d59481498eda90a4256e5a51af`.

The previously incompatible GGUF below remains untouched and was not substituted.

The available Qwen3.8 GGUF uses Q4_K/Q6_K/F32 weights, including Q6_K output
weights. Existing complete DSL kernels require symmetric signed group-128 INT4,
group-major FP16 scales, FP16 dense tensors/embedding and an INT8 output head.
The binding tool **does not** convert between these representations.

Earlier main-site TLS failures were bypassed using the user-provided mirror.
Compatible weights are now downloaded, packed and bound as recorded above; the
previous asset blocker is resolved.

The real GGUF declares 41 total blocks and one NextN/MTP block: the ordinary
generation trunk is 40 blocks. The contract handles this explicitly without
changing source metadata. Running the new executable on this unbound GGUF
fails with exit 1 for missing `npu.dsl.abi`, before constructing a model session;
this is an intentional negative test, not successful inference.

The unchanged runtime still imposes a fixed ten-second wait per submission.
Large single-kernel prefill may exceed that limit. Quarantine contains persistent
failure but does not extend the timeout. No long-prefill performance or timeout
adequacy is claimed.

## Real hardware numerical/state acceptance

Ordinary llama tokenizer, `llama_decode`, in-process full-model session and llama
greedy sampling produced ` Paris.` for `The capital of France is`, exit 0.
Elapsed 81.99 seconds includes package hashing/loading and teardown, and must
not be presented as prefill latency or tokens/second.

The separate real-linked `llama-npu-audit` exports every vocabulary element,
checks public memory positions/output mapping, resets/replays, splits prefill,
and evaluates the prompt again using sequential one-token decode. It does not
launch the standalone DSL inference runner or replace the session with a mock.

Independent oracle: the existing DSL `Qwen36Ref` FP32 CPU implementation reads
the exact original checkpoint, independently dequantizes the original values and
scales, uses an expert cache of 64, and teacher-forces the same token IDs. CPU
PyTorch 2.15.0, safetensors 0.8.0, tokenizers 0.23.3 were used. The reference
process finishes before the resident device process, avoiding RAM contention.

Acceptance uses the pre-existing golden criteria: full-row relative L2 mean
<=0.04, maximum <=0.10, exact argmax or reference chosen-token gap <=0.5.
Reset/replay additionally requires every FP32 bit identical. No threshold was
fit to these outputs. Every measured argmax below was exact; no near-tie
exception was needed.

| Fixture / path versus independent FP32 | Rows x vocabulary | Mean relative L2 | Maximum relative L2 | Exact argmax |
| --- | --- | --- | --- | --- |
| Paris, 5-token prefill + 2 decode, whole prefill | 3 x 248320 | 0.006526 | 0.011091 | 3/3 |
| Paris, split 2+3 prefill + same decode | 3 x 248320 | 0.019406 | 0.024252 | 3/3 |
| Paris, one-token prompt/decode | 3 x 248320 | 0.009324 | 0.012852 | 3/3 |
| Sky explanation, 15-token prefill + 8 decode | 9 x 248320 | 0.024849 | 0.045773 | 9/9 |
| Sky explanation, split 7+8 prefill + same decode | 9 x 248320 | 0.031927 | 0.082290 | 9/9 |
| Sky explanation, one-token prompt/decode | 9 x 248320 | 0.022365 | 0.045786 | 9/9 |

Both reset/replay runs are bit-identical to their own baselines, with zero
mismatched elements/bits. All logits are finite. The sky fixture uses the raw
prompt `Explain in one short sentence why the sky appears blue during the day.`
without a chat template; it is a numerical fixture, not a chat-quality benchmark.

Completed device counters in the final Paris audit are 4 prefill submissions,
13 decode submissions and 17 whole-model kernel submissions, all waited
successfully. The sky audit has 4 prefill, 47 decode and 51 kernel submissions.
Counts include all replay/split/sequential passes, not just the baseline.
No device error, timeout, nonfinite output or CPU model fallback occurred.

The first audit incorrectly required split calculations to match bit-for-bit;
its exit-1 log and binaries are retained unchanged under qwen36-npu-audit.
The audit policy was corrected to the existing golden numerical criterion while
keeping bit-exact reset. Fresh final hardware runs passed. An intermediate
audit temporarily used top-two-margin near ties; final code/tests use the
original chosen-token gap, with a negative test preventing distant-token
acceptance. All observed argmax values were exact under either policy.

### Evidence and reproduction

In the sibling downloads directory:

- qwen36-pack-ca91a480-retry.stdout.log: complete 40-layer export.
- qwen36-tokenizer-convert.log and qwen36-bind.log: conversion and identity.
- qwen36-npu-smoke.stdout.log / stderr.log: ordinary tokenizer/sampler smoke.
- qwen36-reference.json / tokens.txt / logits.f32: first independent reference.
- qwen36-reference-blue.json / tokens.txt / logits.f32: second reference.
- qwen36-npu-final.audit.txt / meta.txt / logits.f32 / replay.f32 / split.f32 /
  sequential.f32 and stderr.log: direct-reference hardware acceptance and counters.
- qwen36-npu-blue.*: second hardware run, including reset/continuation results.
  Its audit deliberately reports independent numerics not supplied in-process;
  the independent comparisons were subsequently performed on saved full logits.
- qwen36-paris-final-comparison.json and qwen36-blue-final-comparison.json:
  independent comparisons for every path, including exact token/package checks.
- qwen36-final-contracts.log: 19/19 host/session regressions.
- qwen36-final-ctest.log: 10/10 label-filtered suite, real full-model test included.
- qwen36-wrong-reference-negative.log: deliberately mismatched fixture rejected,
  exit 1, without publishing a comparison report. CPU-only shared opt-in-default
  regression also re-passed after the final changes; runtime Git status is clean.

The build now has `LLAMA_NPU_MODEL_TOKENIZER` and `LLAMA_NPU_MODEL_WEIGHTS`
explicitly configured to the bound assets. See the
[audit protocol](../../examples/npu-model/README.audit.md) for executable
arguments and artifact schema. The DSL reference exporter/comparator is
qwen36_e2e.reference.audit. Always choose fresh output prefixes; artifacts are
never silently overwritten.

### Measured single-request latency and token rate

One additional real `llama-npu-model` invocation used the raw sky prompt above,
15 prompt tokens, context128, greedy sampling, predict limit32. It emitted28
non-EOG tokens and stopped on EOG, exit0. Monotonic host timestamps measure:

| Metric | Measurement |
| --- | --- |
| Tokenizer/backend/package verification/device load/setup | 78.692499 s |
| TTFT from ready context/prefill start through first output flush | 0.456716 s |
| TTFT including fresh-process setup | 79.149216 s |
| Output inter-token rate, `(emitted-1)/(last_flush-first_flush)` | 8.810544 tokens/s |
| Decode call rate, 28 one-token calls including final EOG prediction | 8.913190 tokens/s |
| Accumulated one-token decode duration | 3.141412 s |

These are host end-to-end single-request measurements, not isolated kernel
profiling or a multi-run benchmark. Ready TTFT excludes setup and tokenization;
fresh-process TTFT includes setup but does not imply cold OS filesystem caches
or a power-cycled device. The first output can be whitespace/special-token text,
not necessarily the first semantic word. Decode includes full logits transfer
and bridge overhead; the output rate also includes greedy sampling and flushing.
Logs: sibling downloads/qwen36-perf-15p32.stdout.log and stderr.log. The timer
output is `NPU_PERF`; CLI regressions re-passed5/5 after instrumentation.

### Post-cleanup regression verification

Removed the superseded GET_ROWS/GGUF K-quant/matmul integration experiments,
their generated files and ignored Python caches. Their six shared DSL source
changes were restored exactly to the DSL repository's HEAD. The complete-model
session, original generated Qwen kernels, bridge, tests and documentation remain.

- The twelve selected Release build targets succeeded (sibling downloads log:
  npu-cleanup-build.log).
- The `npu|qwen36_session` CTest selection passed 20/20, including the real-model
  `test-npu-model-full` smoke test (npu-cleanup-ctest.log). This is a post-cleanup
  smoke/regression run, not a new independent full-logit numerical acceptance.
- Four DSL Python unittest modules ran 146 tests: 130 passed, seven skipped,
  nine failed (npu-cleanup-python-tests.log). An isolated temporary extraction
  of the unmodified DSL HEAD reproduced exactly the same nine failing tests.
  These baseline instruction-set freeze failures include the unrecorded
  `nl.tile` construct and an allowed-additions expectation involving
  `DEQUANT_I4_GM`; the frozen snapshot and tests were not weakened or changed.
- Both repositories passed `git diff --check`; the runtime working tree
  remained clean. Weights, external builds and evidence logs were preserved.

### Limits not covered by this acceptance

Only the measured short fixtures are numerically certified here: longest
prefill 15 tokens and eight continuation inputs. Long context, 512/2048-token
chunk boundaries, the advertised maximum capacity, arbitrary batched sampling,
performance, multi-sequence, MTP and production readiness are not established.
The runtime remains unmodified with its fixed ten-second submission wait;
long-prefill timeout adequacy remains a separate measured acceptance gate.

Historical external-oracle tests that require a clean pinned llama checkout
must use a separate clean pinned source tree now; do not weaken their dirty-tree
checks or reset this fork to rerun them.