# B-route complete-model capability and contract dispatch

## Delivered scope

A host C++17 **interface version 3** in the sibling DSL repository provides
provider-owned model/format/request constraints, structured diagnostics,
validated package-backed capability probing, and integration with the existing
metadata-only binder and complete-model llama bridge. Users supply a model and
external execution package, **not a manual provider choice**.

This is B-route: tokenizer -> llama context/decode -> provider session -> complete
DSL prefill/decode -> last-token FP32 logits. It is not an arbitrary-GGUF GGML
operator backend. Qwen remains the only provider with real numerical evidence.
MOCK providers are host-contract tests, not supported numerical models.

The original Qwen package ABI, identity header/order, asset bytes, effective
tokenizer fingerprint v2, kernel exports, packer and runtime are unchanged.
Interface version 3 is independent of package ABI `qwen36-35b-a3b-v1`: a host
descriptor migration does not rewrite an asset ABI.

### Bounded v3 delivery

The host core (`npudsl_model_core`) has no model/runtime/generated/Python
dependency. A separate registry aggregator (`npudsl_model_api`) explicitly links
factory archives. Qwen is a separate `npudsl_model_qwen36` target selected by
`NPUDSL_MODEL_QWEN36`, default ON for existing CLI/build compatibility. OFF
does not enter the Qwen build at all. Additional sources/factories and independent
archives are supplied through `NPUDSL_MODEL_PROVIDER_SOURCES`,
`NPUDSL_MODEL_PROVIDER_FACTORIES`, `NPUDSL_MODEL_PROVIDER_LIBRARIES`. Factory
archives depend on the host core, not on the aggregator: explicit factory calls
retain static extension linkage without whole-archive or initialization ordering.

The generic asset boundary is `asset_source` (size, bounded range read, location)
plus an ordered flat-file `asset_contract` integrity envelope. A provider-owned
`prepare` adapter validates its own layout and returns representation evidence
and an immutable shared payload. A privately constructed `prepared_assets`
handle retains source/payload ownership, immutable execution implementation, ABI and
package/tokenizer identity. The bridge/binder do not interpret model tensors or
require a manifest/blob serialization. Qwen explicitly uses the retained bounded
legacy schema helper; its runtime receives the same canonical package path and
allocates exactly as before. Test-only opaque record assets prove non-manifest
binding and public bridge execution, not a new real architecture or GGUF adapter.

Preparation and creation callbacks are installed together with
`provider::set_execution` in a private shared immutable implementation. An
unchanged descriptor copy shares the implementation identity. Replacing either
callback creates a new identity and rejects old handles before factory invocation,
including between a registry snapshot and its subsequently modified source
descriptor. Callback getters return copies. Handles also retain callback captures
for their own lifetime; identity matching is in-process type safety, not a plugin
authentication boundary.

The llama host can also configure/build with `NPUDSL_MODEL_QWEN36=OFF` and no
`GENERIC_COMPUTE_ROOT`. Only runtime-dependent providers introduce runtime
dependencies. Qwen-specific tests are conditional; generic record binding and
public context tests still run. The empty production registry cannot execute a
model until an application links an executor factory.

`create_session(provider, prepared_assets, session_options)` is the checked host
entry point. Wrong/null/unverified handles, interface/ABI/factory-ownership and
request errors reject before factory construction. The trusted provider callback
receives normalized explicit context/logical batch/physical batch/sequences and
IO/cache options. The returned session exposes stable `effective_config`: logical
options, actual allocated context, physical capacity, fixed-allocation status and
opaque state capabilities. Inconsistent effective reports reject before any
evaluation; the bridge also enforces them per decode. Qwen's allocated capacity
remains **65792**, even for context128: no smaller cache allocation is claimed.
Append and reset are the only current state capabilities. There is no generic KV
layout, copy/truncate/serialization callback or pretend implementation; unsupported
public operations remain fail closed. Existing semantic callbacks are retained.

The host owns canonical containment, declared filename/size limits, byte hashing,
tokenizer fingerprinting and immutable-use discipline. The provider owns layout,
tensor bounds and representation/semantic validation. `directory_assets` restricts
reads to declared names, nonzero bounded files, valid ranges and at most 16 MiB
per read. Alternate asset sources are trusted host code, not an untrusted plugin
boundary. `asset_identity.integrity_verified` is a trusted host attestation after
actual hash verification, not a signature/authentication primitive; binder probes
remain unverified and cannot execute through the checked creator. Directory-backed
handles own their adapter, not locked snapshots: files must remain immutable
through hashing, preparation, construction and use. Fixed-file integrity envelopes
are delivered; arbitrary network/archive security and exhaustive payload typing
are not. All original Qwen ABI/header/hash layouts and bound assets remain intact.

## Three independent identities

| Concern | Representation | Meaning / non-meaning |
| --- | --- | --- |
| Provider interface | interface version, capabilities, ABI, factory | Callable host/session compatibility, not quantization evidence |
| Execution contract | semantic fields, execution format, provider preparation, request/effective limits | Representation and semantics the exported computation can execute |
| Asset binding | ordered per-file SHA256, package ID, effective tokenizer hash | Integrity of this package/tokenizer pair, not a checkpoint whitelist or signature |

A new checkpoint with the same validated architecture, semantic dimensions and
execution format uses the same provider. Different weight bytes produce new file
hashes and a new package ID. Fresh binding must authorize those bytes; an old
bound GGUF must reject them. Checkpoint/repository names and recorded `source_id`
do not participate in provider eligibility. `source_id` remains unverified caller
provenance. Hashes do not establish tokenizer/checkpoint equivalence, authenticate
a publisher, or establish numerical accuracy.

The external execution package, **not unused source GGUF tensor types**, determines
the representation consumed by B-route. `general.file_type`, architecture and
provider selection never invent that representation. Selecting Qwen cannot make
Q4_K become signed group128 INT4. The binder performs no conversion/requantization;
the caller still acknowledges independently established checkpoint/tokenizer
equivalence.

## Dispatch algorithm

1. With `npu.dsl.abi`, route directly to that registered ABI. Unknown ABI,
   incompatible host version, or missing required capabilities fail; no fallback
   provider is tried. **Routing never skips contract validation.**
2. Without ABI, actual `general.architecture` only filters candidates. Probe each
   candidate's full model/format/package contract. Count only successful probes.
   Zero matches fails with candidate diagnostics; multiple matches fails as
   ambiguous, never registration-order selection. No CLI `--provider` is added.
3. Complete six-field execution-format metadata can support metadata-only
   capability probing. Otherwise the binder obtains evidence from a validated
   package parser for each candidate. Probes never create sessions or detect
   hardware. `package_probe` is a trusted host validation adapter, not an
   authentication boundary; production adapters invoke provider preparation.
4. Binding verifies structure, computes actual hashes, and writes a NEW
   metadata-only GGUF/identity through existing no-replace publication. New GGUF
   format fields come from package validation, not provider selection or original
   source GGUF quantization labels.
5. Runtime still requires an integrity-bound GGUF/package. The bridge routes its
   bound ABI, preflights model/request constraints, verifies identity/tokenizer
   and every file hash, validates package structure/format, then checks the full
   model contract again with package evidence. Only then may it create a session.
   The package identity header must match the routed provider asset envelope. An unbound
   vocabulary-only GGUF can be probed/bound by the binder, but is not silently
   executed without integrity binding.

`resolve()` retrieves a descriptor; `resolve_model()` validates eligibility.
Descriptor retrieval alone must not authorize session construction. All model
specific constraints live in providers, not generic llama bridge code.

## Structured contracts

### Model semantics

`model_info` describes vocabulary, capacity and executed trunk layers.
`model_fields` contains provider-owned exact/numeric metadata constraints;
numeric constraints require finite values with the same FP32 value rather than
textual float equality. `validate_model_contract()` accumulates field/expected/
actual diagnostics for architecture, actual loaded vocabulary, semantic fields
and format. Provider `validate_model` callbacks retain cross-field semantics and
legacy compatibility; callback failures become a `model.semantics` diagnostic.

Every metadata value read during model validation, including semantic-callback
reads, is cached and bounded to 256 bytes. Oversized values and throwing metadata
readers produce field-specific structured diagnostics with bounded size/error
summaries; they do not escape as bare metadata exceptions. The standalone bounded
`optional()`/`required()` helpers retain their throwing behavior.

Qwen constraints reside in its DSL provider and shared contract: embedding/head/
KV/expert/SSM/rotary shapes, expert selection, full-attention interval, RMS epsilon
and rotary base. Existing 40-trunk/optional-one-MTP block-count semantics remain
enforced. MTP/vision are not executed; scaled RoPE remains rejected. Equal byte
counts with different attention-head semantics do not authorize a model.

### Execution representation

Six fields are independent of ABI: `weights`, `scales`, `dense`, `head`, `layout`,
`group_size`. GGUF declarations use `npu.dsl.format.<field>`; group size is an
unsigned scalar. If any field is supplied, **all six are required and checked**,
even with validated package evidence. Missing/inconsistent claims fail closed.

Runtime preflight defers missing-format errors only if **all six keys are absent**.
A partial declaration is rejected immediately, before package path resolution,
hashing or provider creation. All-absent legacy declarations still require actual
validated package evidence and the full post-hash model-contract check.

The delivered Qwen representation is symmetric signed INT4 (low nibble first,
row-major), group128/group-major FP16 scales, FP16 dense/embedding tensors,
FP32 recurrent-state auxiliaries and a symmetric signed INT8 output head with
group128 scales. Tensor roles/sizes come from linked generated decode/prefill ABI.
Descriptor strings identify this composite representation; they are not a
universal quantization taxonomy. Four-bit GPTQ/AWQ/GGUF labels are not sufficient.

New provider schemas may designate an identity-covered metadata `format_file`
with exactly six unique `key value` lines. Missing/unknown/duplicate/malformed/
overflowing fields fail. The independently parsed declaration must match provider
constraints. It supplements structural validation, not numerical provenance.
No format file was added to existing Qwen assets.

### Explicit validated legacy parser

Existing Qwen packages lack a self-describing format file. Their provider explicitly
opts into `validated_legacy_format`; this is not the default. Format evidence
requires every required file/size, exact generated tensor names/byte counts,
alignment, nonoverlap, bounds, final blob extent and exact 40-layer pattern to
validate. The known legacy **serialization contract** then supplies representation.
Architecture or ABI alone supplies no evidence.

**Trust limit:** raw legacy bytes cannot prove signedness, scale ordering or
checkpoint provenance. A wrongly packed/malicious blob with identical schema can
pass structural validation and newly computed hashes. Producer/caller must honor
this documented legacy serialization contract; independent numerical acceptance
remains necessary. Arbitrary semantic reinterpretation of untyped bytes is not
detected. Explicit incompatible format declarations fail; absence uses only this
validated legacy path. The existing asset compatibility does not become a
checkpoint whitelist.

### Requests, IO and cache

`request_contract` / `session_options` describes context, logical/physical batch,
sequence count, input/output and K/V
cache representation; provider `request_limits` owns constraints.
`validate_request_contract()` accumulates all mismatches. Zero context selects
provider capacity, not unlimited capacity. Qwen supports ABI capacity/batch up to
65792, one sequence, sequential token IDs, last-token FP32 logits and FP16 K/V.
These are exported ABI limits, **not long-context numerical/performance certification**.

Public `type_k`/`type_v` enums are range-checked before GGML type helpers and
provider creation, for normal as well as opted-in contexts. Invalid enum values
return a null public context rather than aborting.

B-route enforces the physical cap, not just the logical cap. The reported logical
cap is `min(n_batch, n_ctx)`; the reported physical cap is
`min(logical_cap, n_ubatch ? n_ubatch : logical_cap)`. Unequal caps are valid, but
each legacy or extended decode call must fit both caps. Calls exceeding the
physical cap fail before provider evaluation, with no implicit microbatch split,
position/counter mutation or session poisoning; old logits are invalidated.
Zero `n_ubatch` selects the logical cap. Logical requests exceeding the provider
limit still fail before normalization/hashing. CLI and audit callers already set
equal caps; their existing explicit caller-side prompt chunking is unchanged.

Generic llama restrictions remain stricter: no embeddings/all-row outputs,
noncausal context, state serialization, sequence editing, custom RoPE options,
callbacks, sampler chains or unsupported context modes. These remain fail-closed
adapter errors. Session evaluation continues to check tokens/positions/remaining
capacity and preserve permanent device poison. No unexpressible bridge operation
is advertised as supported.

### Package validation and diagnostics

`validate_manifest_contract()` and `validate_package_contract()` are optional
legacy provider helpers. Qwen's `use_legacy_package` invokes them for metadata,
file sizes, tensor layout and representation evidence. Generic bridge/binder code
invokes `prepare_assets`, not these parsers. The parser helpers do no IO, hashing,
session creation or hardware detection. Host adapters own canonical containment,
bounded reads, actual SHA256 and no-replace publication.

`validation_report.errors` and `validation_error.errors` retain structured `field`,
`expected`, `actual`. Throwing adapters print those same fields. Callback failures
and malformed manifest rows preserve provider/tensor details in their diagnostic.
Hash/path failures remain existing explicit adapter errors. These structures are
not a stable cross-language error serialization schema.

## Interface migration and extension

All built-in/mock providers migrate to interface version 3, mandatory format,
request limits, preparation and effective session configuration. Semantic callbacks
remain; the factory signature and session configuration interface are breaking.
`resolve_model()` now requires the actual loaded vocabulary; optional package
probes return validated representation evidence. Version 1 and incomplete new
descriptors fail rather than silently acquiring default Qwen constraints. Version2
also fails; host interface changes do not require repacking/rebinding existing assets.

Static extensions supply immutable descriptors and compile-time factories in the
DSL API build. No generic llama bridge edit is required. New exported shapes or
representations may require new provider/package ABIs and real kernel work;
descriptor registration alone does not deliver new numerical support.

**Static/shared registry caveat remains:** executable-side registration is tested
only with static llama. Shared hosts must register build-time factories inside
the bridge library, not assume their separately linked executable registry is the
same object. This is not a stable C ABI, dynamic-plugin loader or thread-safe
runtime registry-mutation service.

## Delivered tests and evidence

Host tests cover interface migration, duplicate/unknown/incompatible ABI,
insufficient capabilities, actual format mismatch with/without explicit ABI,
missing format, compatible-provider ambiguity, package-backed unbound dispatch,
explicit/legacy declarations, malformed formats, wrong tensor semantics, IO/cache/
context/batch/sequence diagnostics, equal-byte Qwen semantic mismatches, different
weight contents rebound to the same provider and rejection by the old binding.
Test blobs/session tags are explicitly MOCK, never numerical evidence. V3 adds
standalone no-Qwen core builds, independent factory-archive linkage, immutable
handle/source lifetime checks, wrong provider/ABI/version/unverified/null handles,
physical/IO/cache/configuration errors and inconsistent effective reports.
The public bridge/binder fixture uses two opaque records and no manifest/schema.

Existing public context fixtures now register an explicitly test-only tiny schema;
tiny blobs cannot pretend to satisfy the Qwen numerical tensor ABI. Original
output/reset/poison regressions remain. All thirteen existing NPU/session Release
targets, the full requested `npu|qwen36_session` CTest selection and fresh real
hardware audit evidence are recorded in [NPU-validation.md](NPU-validation.md).

## Future / not delivered

Arbitrary GGUF quantization, new architecture mathematics, automatic conversion/
packing, signed provenance, exhaustive typed payload verification, dynamic
plugins, shared-host execution validation, new native topology detection,
multisequence scheduling, embeddings/all-row logits, variable Qwen cache formats,
MTP/vision execution, device recovery/cancellation and long-context/performance
certification are not delivered. No invented device detection or CPU model
fallback is introduced. The runtime's ten-second submission wait is unchanged.
No kernel, packer, generated kernel asset, checkpoint or bound GGUF edit is needed.