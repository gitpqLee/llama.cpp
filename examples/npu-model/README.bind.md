# Metadata-only package binding

Build target `npu-package-bind` in a complete-model NPU build (`GGML_NPU=ON`,
`NPU_DSL_ROOT`, and the generic `npudsl_model_api` target). The utility never creates a
device/session, runs model inference, or falls back to CPU model execution.

Invocation: `npu-package-bind --tokenizer <source-matching GGUF> --weights <packed dir> --output <NEW metadata-only GGUF> --source-id <explicit checkpoint revision ID> --acknowledge-external-weights`.

**Binding does not transform GGUF weights or establish checkpoint equivalence.**
Use real, already packed session weights and independently establish that they
and the tokenizer belong to the same checkpoint revision before acknowledging.
A matching architecture, token count, hash, model name, or caller-supplied source
ID is not proof of this equivalence. No compatible numerical fixture or verified
end-to-end generation is supplied or claimed by this utility.

`--source-id` is required, limited to 1..256 ASCII characters from letters,
digits, `.`, `_`, `-`, `/`, `:`, `@`, and `+`. Supply an immutable checkpoint
revision, for example `organization/checkpoint@revision`. It is stored as the
string `npu.dsl.source_id` in the output, as **unverified caller provenance**.
It is not incorporated into the unchanged v1 identity contract, and the runtime
does not authenticate or compare this field. No separate provenance file is made.

The binder routes by `npu.dsl.abi` but always checks the actual contract. An
unbound model's architecture only filters candidates; a unique full
model/format/package match is required. Missing format may use the explicitly
validated legacy package parser, never architecture alone. Ambiguity and unknown
ABIs are errors. Provider constraints, callbacks and ordered schemas supply all
model-specific validation; adding a provider does not require binder edits.
Only Qwen is a real supported model; the second provider is a test-only MOCK.

For the unchanged Qwen provider, preflight checks all 43 fixed required package files,
canonical containment and regular-file status, file-size bounds, the exact
40-layer pattern, and fixed GGUF scalar dimensions/vocabulary. It checks every
manifest weight name and size against the same generated ABI tables selected for
the linked session, with duplicate, missing, unexpected, unaligned, overlapping,
out-of-bounds, overflowing, and trailing-blob data rejected. This is structural
validation, not numerical or checkpoint-equivalence validation. All required files
are then streamed through SHA-256, including the raw text files and blob padding.

The original tokenizer is loaded vocab-only. The public
`llama_npu_tokenizer_sha256(model, output, size)` calls the runtime's existing
loaded-tokenizer fingerprint implementation. It needs a 65-byte buffer, writes
64 lowercase hexadecimal characters plus NUL, and returns false on invalid
arguments, hashing failure, or a build without the complete-model bridge. It
does not check model dimensions, verify packages, or initialize a device.

New bindings record six `npu.dsl.format.*` fields from validated external package
evidence. Partial/incompatible declarations fail. Old Qwen bindings without these
fields remain accepted only through the exact validated legacy tensor/layout
parser. Hashes bind corresponding packages, not a checkpoint whitelist. See
[the model contract design](../../docs/backend/NPU-model-contract.md) for the
legacy untyped-byte trust limit and interface-version migration.

GGUF APIs clone all KV metadata into an empty context, with **zero tensor
descriptors and zero tensor payload**. The binder removes only the original
shard bookkeeping (`split.no`, `split.count`, `split.tensors.count`) and replaces
the binding/provenance strings and validated execution-format fields. It never rewrites the input GGUF
or packed weights. The output is reloaded and its tokenizer fingerprint checked
before publication. The output ABI is `qwen36-35b-a3b-v1`.

The new package identity is exactly the existing contract documented in
[docs/backend/NPU.md](../../docs/backend/NPU.md): header, tokenizer hash, then
the ordered hashes of the manifest, layer-type file, globals, and layers 00..39.
The output package ID is SHA-256 of those exact identity bytes, including final
LF. `source-id` is deliberately not inserted into this file.

Existing output paths and existing package identities are refused, including
dangling symlinks. Output parent directories must already exist. Temporary files
live in exclusively reserved directories on their destination filesystems.
Publication uses atomic no-replace hard links; the filesystem must support hard
links. RAII removes temporary files and rolls back an identity published before
an output-publication failure. Two different destinations cannot be committed
as a crash-atomic transaction: power loss/process termination between publication
steps can leave an orphan identity, which must be reviewed manually, not silently
overwritten. No successful output is partial. Keep the whole package and input
tokenizer immutable during binding and runtime use; hashes are integrity checks,
not signatures or protection from malicious concurrent mutation.

Exit statuses are 0 for success/help, 2 for CLI errors, and 1 for asset, hashing,
or output failures. Errors and the limits of binding are printed explicitly.
`--help` is standalone and does not read assets.

With `LLAMA_BUILD_TESTS=ON`, target/CTest `test-npu-package-bind` covers synthetic
invalid assets, text-only manifest ABI/bounds, known SHA-256 bytes, small
synthetic tokenizers and fingerprint/KV-clone behavior, output refusal,
no-replace publication, and rollback. It creates no numerical model package and
performs no device execution. A successful real-package bind still requires real
checkpoint-matched assets, followed by independent numerical validation.