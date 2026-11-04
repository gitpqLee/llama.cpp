# Explicit complete-model NPU execution

Current Chinese architecture overview and repository boundaries:
[NPU-architecture.md](NPU-architecture.md).
Detailed current contracts: [NPU-model-contract.md](NPU-model-contract.md).

## Generic provider boundary

The bridge now depends only on the DSL `npudsl_model_api` target and version-3
provider/session interface. Qwen-specific headers, architecture checks, dimensions,
layer patterns, filenames, tensor tables and identity header live in the DSL
provider, not the llama bridge. Context capacity and executed layer count come
from provider model information. Static providers can be added through the DSL
`NPUDSL_MODEL_PROVIDER_SOURCES` / `NPUDSL_MODEL_PROVIDER_FACTORIES` extension
points without changing bridge code; explicit startup registration is also supported.

Bound models resolve by `npu.dsl.abi`; the binder can resolve unbound scalar model
metadata only when the architecture identifies one provider unambiguously. Unknown
or duplicate ABIs, interface-version and capability mismatches fail closed. Model
validation callbacks do not include llama headers. Ordered package schemas remain
provider-owned while generic integrity and tokenizer fingerprinting remain in llama.

**Only Qwen currently has real numerical support.** The second provider is a
test-only seven-token/nine-context/two-layer MOCK, never a production registration.
The Qwen asset contract documented below is unchanged byte-for-byte; existing
bound assets require no rebind. This refactor does not add arbitrary GGUF loading,
another real model, long-context validation, or a dynamic plugin ABI.

Current evidence and blockers: [NPU-validation.md](NPU-validation.md).

Cross-machine prerequisites, delivery options and commands:
[NPU-deployment.md](NPU-deployment.md).

`llama-npu-model` is an **in-process complete-model bridge**, not ggml graph
offload and not a launcher for another program. It uses the standard public
`llama_model_load_from_file` API with `vocab_only=true`, passes the package
directory through `llama_context_params.npu_model_path`, tokenizes with llama,
calls `llama_decode`, samples with llama's CPU greedy sampler, and writes generated
token pieces to stdout. Diagnostics go to stderr. The prompt is not echoed.
There is no automatic chat template, subprocess, synthetic weight generation,
on-load quantization/requantization, or CPU **model** execution fallback.

## Build and invocation

The build adds [examples/npu-model/CMakeLists.txt](../../examples/npu-model/CMakeLists.txt)
under `GGML_NPU`, even when `LLAMA_BUILD_EXAMPLES=OFF`. It defines
`llama-npu-model` linked only to `llama`. The bridge itself requires
`GGML_NPU=ON`, a suitable `NPU_DSL_ROOT`, and the `npudsl_model_api` target/runtime.
Enabling a graph backend alone is not evidence that the complete-model bridge
is compiled or that a compatible package exists.

The command-line interface is `llama-npu-model --model <tokenizer GGUF>
--weights <packed package directory> --prompt <text> [--predict <count>]
[--ctx-size <tokens>]`. Quote paths and prompts containing spaces.
`--help` is standalone and does not initialize a device or read model assets.

Defaults are `--predict 32` and `--ctx-size 4096`. Numbers must be unsigned
decimal integers without signs, whitespace or suffixes. Syntax accepts context
1..2147483647 and prediction 0..2147483647, not exceeding context. Actual capacity
is checked against the selected provider (Qwen: 65792). Prompt text is limited to
1 MiB. Duplicate/unknown options and missing values are errors. An explicitly
empty prompt is allowed only if the tokenizer produces at least one token.
Tokenization uses `add_special=true, parse_special=true`; special-token strings
in the prompt are interpreted as such. The application conservatively requires
`prompt_token_count + predict <= ctx_size`, does not truncate, and never shifts
the context. It evaluates prompt chunks of up to 512 tokens, with consecutive
positions, sequence zero, and logits only for the last token of each call.
Generation stops at EOG or the prediction count. `predict=0` still verifies the
package, creates a device session and evaluates the prompt; it is not a cheap
metadata-only check.

The sampler is not attached to the NPU context. Prompt tokens are accepted with
`llama_sampler_accept`; `llama_sampler_sample` already accepts each generated
token internally, so the application does not double-accept. Model, context,
sampler and backend teardown use RAII. Exit status is 0 for successful execution
or help, 2 for invalid arguments, and 1 for assets, context/package rejection,
runtime or output errors. All nonzero decode returns are fatal, without retries.

## Required assets: not a generic GGUF loader

The current contract is [src/llama-npu.cpp](../../src/llama-npu.cpp). A normal
Qwen GGUF, a tokenizer from a similarly named checkpoint, or an unbound older
DSL export is **not** sufficient. Two matching assets must be supplied:

1. A valid tokenizer GGUF preserving the exact original vocabulary, tokenizer
   scalar metadata, merges, normalization data and special-token behavior, plus
   the model dimensions and the three `npu.dsl.*` string metadata fields below.
   Model tensor data is not loaded by this executable.
2. The real prepacked whole-model session package for exactly that checkpoint,
   layout and ABI, with the strict identity file below. Its manifest, dimensions,
   binary layouts and sizes must also pass `qwen36_session` validation; the
   identity file does not convert an arbitrary binary into that layout.

**Compatible assets and bounded real end-to-end numerical/state validation have
now been obtained locally; assets are not bundled in this source tree.** See
[the validation evidence](NPU-validation.md) and
[cross-machine delivery guide](NPU-deployment.md). The metadata-only `npu-package-bind` utility is described
in [examples/npu-model/README.bind.md](../../examples/npu-model/README.bind.md).
It uses the exported `llama_npu_tokenizer_sha256` helper and does not convert
weights or establish checkpoint equivalence. The caller must independently verify
the source checkpoint match and explicitly acknowledge external weights. Merely
adding arbitrary hashes, renaming a model, manufacturing dummy binaries, or
suppressing validation is not a supported path. Keep an immutable package
through verification and session use: these hashes are integrity/binding checks,
not digital signatures or protection against concurrent malicious modification.

### Qwen provider architecture and scalar metadata

The model must be loaded vocab-only, architecture `qwen35moe`, with exactly
248320 loaded tokens. These original scalar GGUF metadata values are required;
the bridge checks the loader's rendered scalar strings exactly:

| Key | Required value |
| --- | --- |
| `general.architecture` | `qwen35moe` |
| `qwen35moe.embedding_length` | `2048` |
| `qwen35moe.block_count` | `40` without MTP, or `41` with one MTP layer (see below) |
| `qwen35moe.nextn_predict_layers` | absent or `0` with block count `40`; exactly `1` with block count `41` |
| `qwen35moe.attention.head_count` | `16` |
| `qwen35moe.attention.head_count_kv` | `2` |
| `qwen35moe.attention.key_length` | `256` |
| `qwen35moe.attention.value_length` | `256` |
| `qwen35moe.expert_count` | `256` |
| `qwen35moe.expert_used_count` | `8` |
| `qwen35moe.expert_feed_forward_length` | `512` |
| `qwen35moe.expert_shared_feed_forward_length` | `512` |
| `qwen35moe.ssm.conv_kernel` | `4` |
| `qwen35moe.ssm.inner_size` | `4096` |
| `qwen35moe.ssm.state_size` | `128` |
| `qwen35moe.ssm.time_step_rank` | `32` |
| `qwen35moe.ssm.group_count` | `16` |
| `qwen35moe.full_attention_interval` | `4` |
| `qwen35moe.rope.dimension_count` | `64` |
| `npu.dsl.abi` | `qwen36-35b-a3b-v1` |

The complete DSL executes exactly **40 trunk layers** for ordinary generation.
An original GGUF with `block_count=41` and `nextn_predict_layers=1` includes
one additional MTP layer, which the DSL intentionally excludes. Both the binder
and context verifier accept this layout without rewriting the source GGUF's
block metadata. A block count of 40 with one MTP layer, 41 with absent/zero MTP,
or more than one MTP layer is rejected. Only an absent MTP key defaults to zero;
empty, malformed and oversized values are rejected, not treated as absent.

Additionally `qwen35moe.rope.freq_base` must parse completely to a finite number
whose float32 value equals 10000000, and
`qwen35moe.attention.layer_norm_rms_epsilon` likewise equals float32(1e-6).
`qwen35moe.rope.scaling.type` must be absent or exactly `none`. Required metadata
retrieved by the bridge is limited to 256 bytes per rendered value.

`npu.dsl.tokenizer_sha256` must be the 64-character lowercase hexadecimal
SHA-256 of the tokenizer stream described below. `npu.dsl.package_id` must be
the 64-character lowercase hexadecimal SHA-256 of **all raw bytes** of the
package identity file, including its final LF. These fields are GGUF strings,
not filenames or hashes of the GGUF file itself.

### Exact tokenizer fingerprint (`llama-npu-tokenizer-v2`)

This protocol fingerprints the **loaded llama vocabulary**, not a JSON tokenizer
file or the raw GGUF byte representation. Integers below are always unsigned
64-bit **little-endian** (eight bytes), including counts, score bits, token
attributes and token IDs. `field(s)` is the byte length of `s` as that u64,
followed by exactly its bytes, without a NUL or delimiter. Strings use their
loaded bytes; do not normalize, re-encode or sort token/merge arrays.
Hash the following concatenation using SHA-256:

1. `field("llama-npu-tokenizer-v2")`.
2. Count of scalar entries in the loaded model's `gguf_kv` whose key begins
   exactly with `tokenizer.`. Sort these entries by `std::map<std::string,...>`
   key order, then append `field(key), field(value)` for each. The model collects
   non-array values via `gguf_kv_to_str`; use precisely those rendered strings,
   including bool/numeric formatting, not guessed JSON representations.
3. Effective vocabulary settings, **after loader defaults and `kv_overrides`**,
    in this exact order (no extra counts or field names):
    - `field(get_tokenizer_model())`, `field(get_tokenizer_pre())`.
    - u64 of `get_type()`, `get_pre_type()`, `n_token_types()`, `max_token_len()`.
    - Special IDs from `token_bos()`, `token_eos()`, `token_eot()`, `token_eom()`,
       `token_unk()`, `token_sep()`, `token_nl()`, `token_pad()`, `token_mask()`,
       `token_fim_pre()`, `token_fim_suf()`, `token_fim_mid()`, `token_fim_pad()`,
       `token_fim_rep()`, `token_fim_sep()`. Cast each ID to uint32, then serialize
       as u64 (including the null ID). The prefix/middle/suffix getters are aliases
       of the corresponding FIM getters, so are not serialized again.
    - Boolean getters `get_add_space_prefix()`, `get_add_bos()`, `get_add_eos()`,
       `get_add_sep()`, `get_dna_auto_tags()`, `get_ignore_merges()`,
       `get_clean_spaces()`, `get_remove_extra_whitespaces()`,
       `get_escape_whitespaces()`, `get_treat_whitespace_as_suffix()`;
       each is a u64, 0 or 1.
    - The `get_normalizer_opts()` fields `nfc`, `lowercase`, `strip_accents`,
       likewise each a u64, 0 or 1.
4. Loaded token count. For each token in ascending ID order: `field(text)`,
   u64 of its float32 score's raw uint32 bit pattern (zero-extended, **not** a
    numeric float-to-integer conversion), u64 of its loaded attribute bitmask,
    and u64 of `is_eog(id)` (0 or 1).
   These are the loader's resulting attributes, not just raw GGUF token types.
5. Count returned by `get_bpe_merges()`, then `field(merge)` for each in rank
   order. Each reconstructed merge is `left + " " + right`; gaps in the loaded
   rank vector remain empty strings. This is not a sorted set of merges.
6. Byte count of `get_precompiled_charsmap()`, then those raw bytes (no second
   length prefix).
7. Count of `get_suppress_tokens()`, then each ID cast to uint32 and serialized
   as u64, in the loader's order.

Original scalar strings alone do not describe the effective tokenizer: an
override can change, for example, BOS insertion without changing `gguf_kv`.
Both the binder and verifier use the single `llama_npu_tokenizer_sha256`
implementation; do not implement a separate producer-side approximation.
Version 1 tokenizer digests are obsolete and are not accepted as version 2.
Rebind existing assets to regenerate the tokenizer digest, identity file and
GGUF package ID. The package identity header `llama-npu-qwen36-v1` and model ABI
`qwen36-35b-a3b-v1` are unchanged: only the tokenizer serialization is version 2.

Encode the resulting 32-byte digest as lowercase hex. The binding metadata uses
the `npu.dsl.` prefix, so it is not included in the `tokenizer.` scalar set;
there is no self-reference. A reliable producer first loads the finalized
tokenizer/dimension metadata, computes the fingerprint, writes the identity,
computes its digest, adds the three binding fields without changing tokenizer
content, and reloads/rechecks the result against the bridge. Do not copy a digest
from another checkpoint or assume token count alone establishes compatibility.

### Exact package identity file

The file [identity.txt](identity.txt) is inside the package directory (this link
names the required asset, not a supplied fixture). It is ASCII, **LF only**, no
BOM, nonempty, at most 8192 bytes, with a final LF. Exactly 45 lines, in order:

1. Literal `llama-npu-qwen36-v1`.
2. Literal `tokenizer ` followed by the tokenizer's lowercase 64-digit hash.
3. Hash entry for [manifest.txt](manifest.txt).
4. Hash entry for [layer_types.txt](layer_types.txt).
5. Hash entry for [globals.bin](globals.bin).
6. Forty hash entries for layer binary files numbered 00 through 39, using
   the exact filename pattern `layer_%02d.bin`, in ascending order.

Every hash entry consists of exactly 64 lowercase hex characters, **one ASCII
space**, the exact fixed filename, and LF. There are no sizes, comments, blank
lines, extra spaces, CR bytes or trailing lines. Each digest is SHA-256 of the
referenced file's complete raw bytes. The two text files must each be nonempty
and at most 16 MiB; the binaries must each be nonempty and at most 64 GiB.
The bridge canonicalizes each fixed path and requires a regular file whose
canonical parent is the canonical package directory (no symlink escape).

The layer-type file has exactly 40 lines: `linear_attention` for layers 0,1,2,
`full_attention` for layer 3, repeating this four-layer pattern through layer 39.
Its content and the manifest/binaries are hashed byte-for-byte, regardless of
the session parser's text conventions. The binder and bridge accept LF or CRLF
layer lines, removing exactly one CR immediately before LF for validation only;
spaces, embedded CRs, doubled CRs and extra lines are not trimmed or accepted.
A final layer line without a newline is also accepted, but a bare trailing CR
is not. No line-ending normalization is applied before hashing: LF and CRLF
files have different digests. The identity file itself remains strictly LF-only.
Generate the identity only after final
packing; changing any packed file requires regenerating the identity and the
GGUF's package ID. Changing tokenizer content also requires a fresh tokenizer
digest and identity. Ordinary file SHA-256 tools suffice for package files,
but **not** for the loaded-tokenizer digest.

## Supported state and failures

Only a single causal sequence zero with consecutive positions, token inputs and
last-only logits is supported. `ctx_type=LLAMA_CONTEXT_TYPE_MTP` is unsupported,
including when the source GGUF contains an MTP layer. Embeddings, multiple sequences, speculative
rollback, sliding/shifted positions, arbitrary RoPE overrides, context samplers,
callbacks, state save/load and general KV manipulation are not supported.
Full memory clearing resets a healthy session; unsupported void memory
operations latch rejection. Device failures may poison the session; do not
attempt to recover by retrying or silently constructing a normal CPU context.
The executable exposes none of those unsupported state operations.

## Tests and verification limits

[tests/test-npu-context.cpp](../../tests/test-npu-context.cpp) exercises real
llama context construction and decode with a link-time mock session. It checks
40-block absent/zero-MTP and 41-block one-MTP acceptance, full-vocabulary mock
logits, mismatched trunk/MTP counts, malformed/oversized optional metadata, and
MTP context rejection before mock session construction. These are host contract
tests, not device execution or numerical model validation.

[examples/npu-model/package-bind-test.cpp](../../examples/npu-model/package-bind-test.cpp)
tests the shared layer parser with LF/CRLF binary streams, malformed CR/spacing
and raw-byte hash distinctions. Its tiny metadata-only GGUF fixture exercises
actual vocab-only loads with tokenizer `kv_overrides`: add flags, special IDs,
normalizer options and pretype/ignore-merges. It checks unchanged original scalar
metadata, changed effective getters/fingerprints, no-op override stability and
an actual BOS-induced tokenization change. These tests require no external model
or device and do not claim checkpoint equivalence or numerical inference.

[tests/test-npu-model-contract.cpp](../../tests/test-npu-model-contract.cpp)
checks exported public default parameters (NPU opt-in is null and ordinary
model loading is not vocab-only), the actual executable's argument parser,
numeric/size bounds, duplicate/unknown options, and context-budget overflow.
These are real runtime assertions, not source-text searches. They do **not**
prove that a normal model context is unchanged, exercise the disabled bridge's
model rejection, or test numerical inference. No internal-class mock is linked.
The CLI CTests require exact exit codes and diagnostics, so a missing executable,
timeout, crash or missing DLL cannot count as an expected argument rejection.

`test-npu-model-full` is explicitly **skipped** (return 77) unless both
`LLAMA_NPU_MODEL_TOKENIZER` (file path) and `LLAMA_NPU_MODEL_WEIGHTS` (directory)
are configured. Supplying only one or lacking the executable is a configure
error. With both, it runs `Hello`, predicts at most two tokens, and uses context
128. It is an in-process **smoke test**, not a numerical equivalence test or a
golden-text assertion; it may stop early at EOG. A real verification campaign
must additionally check known prompt/logit/token references, mismatched and
tampered assets, context/state behavior and device execution/profiling.

For a sequential parent build using the existing build directory, reconfigure
after the parent subdirectory hook is in place, build targets `llama-npu-model`
and `test-npu-model-contract` in Release, then run CTest in Release with regex
`^test-npu-model-` and `--output-on-failure`. Do not launch another configure/build
in the same directory while a parent build is active. No model assets are
downloaded or fabricated by these tests.