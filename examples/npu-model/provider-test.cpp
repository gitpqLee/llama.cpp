// Actual generic binder + public llama context + test-only small provider.
// No hardware execution or second-model numerical support is claimed.
#define NPU_PACKAGE_BIND_TEST
#include "package-bind.cpp"
#include "tests/mock_provider.h"
#ifdef NPUDSL_TEST_QWEN36
#include "qwen36_contract.h"
#endif

namespace {
unsigned checks = 0;
void check(bool value, const char * message) {
    ++checks; if (!value) { throw std::runtime_error(message); }
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception &) { rejected = true; }
    check(rejected, "expected rejection");
}
void write(const std::filesystem::path & path, const std::string & data) {
    std::ofstream out(path, std::ios::binary); out.write(data.data(), data.size()); out.close();
    check(bool(out), "fixture write");
}
void registry_tests() {
    using namespace npudsl_model;
    registry r;
    auto p = npudsl_model_test::mock_provider();
    r.add(p);
    check(r.resolve(p.abi).info.context == 9, "registry dispatch");
    rejects([&] { r.add(p); });
    rejects([&] { r.resolve("unknown"); });
    rejects([&] { r.resolve(p.abi, interface_version + 1); });
    rejects([&] { r.resolve(p.abi, interface_version, 1ull << 63); });
    auto invalid = p; invalid.abi = "bad-version"; invalid.version = interface_version + 1;
    rejects([&] { r.add(invalid); });
    invalid = p; invalid.abi = "bad-capability"; invalid.capabilities |= 1ull << 63;
    rejects([&] { r.add(invalid); });
    invalid = p; invalid.abi = "limited"; invalid.capabilities &= ~reset_state;
    r.add(invalid);
    rejects([&] { r.resolve("limited"); });
    invalid = p; invalid.abi = "bad-schema"; invalid.assets.files[0].name = "../escape";
    rejects([&] { r.add(invalid); });
    invalid = p; invalid.abi = "zero-vocab"; invalid.info.vocabulary = 0;
    rejects([&] { r.add(invalid); });
    std::map<std::string, std::string> values = {{"general.architecture", "llama"}, {"mock.width", "3"}, {"npu.dsl.abi", p.abi}};
    metadata get = [&](const char * key) -> std::optional<std::string> {
        auto it = values.find(key); return it == values.end() ? std::nullopt : std::optional<std::string>(it->second);
    };
    auto declare = [&](const execution_format & f) {
        values["npu.dsl.format.weights"] = f.weights; values["npu.dsl.format.scales"] = f.scales;
        values["npu.dsl.format.dense"] = f.dense; values["npu.dsl.format.head"] = f.head;
        values["npu.dsl.format.layout"] = f.layout; values["npu.dsl.format.group_size"] = std::to_string(f.group_size);
    };
    declare(p.format);
    check(r.resolve_model(get, 7).abi == p.abi, "explicit ABI validated despite architecture overlap");
    values.erase("npu.dsl.abi");
    check(r.resolve_model(get, 7).abi == p.abi, "incompatible capabilities do not cause ambiguity");
    auto twin = p; twin.abi = "MOCK-twin-v1"; r.add(twin);
    rejects([&] { r.resolve_model(get, 7); });
    registry only; only.add(p);
    check(only.resolve_model(get, 7).abi == p.abi, "unbound complete format capability dispatch");
    values["npu.dsl.format.weights"] = "Q4_K";
    const auto mismatch = validate_model_contract(p, get, 7);
    check(!mismatch.accepted() && mismatch.errors[0].field == "npu.dsl.format.weights", "field-level quant mismatch");
    rejects([&] { only.resolve_model(get, 7); });
    values["npu.dsl.abi"] = p.abi;
    rejects([&] { only.resolve_model(get, 7); }); // ABI never bypasses format validation
    declare(p.format);
    values.erase("npu.dsl.format.scales"); rejects([&] { only.resolve_model(get, 7); });
    declare(p.format);
    values["mock.width"] = "4"; rejects([&] { only.resolve_model(get, 7); });
    values["mock.width"] = "3";
    // Bounded reads must return structured diagnostics, even in callbacks.
    for (const char * field : {"general.architecture", "mock.width", "npu.dsl.format.weights"}) {
        const auto saved = values[field];
        values[field] = std::string(257, 'x');
        const auto report = validate_model_contract(p, get, 7, &p.format);
        const auto found = std::find_if(report.errors.begin(), report.errors.end(), [&](const diagnostic & e) {
            return e.field == field && e.expected == "0..256 bytes" && e.actual == "<oversized: 257 bytes>";
        });
        check(found != report.errors.end(), "oversized metadata retains bounded field diagnostic");
        try { report.require(); check(false, "structured report must throw"); }
        catch (const validation_error & e) { check(e.errors.size() == report.errors.size(), "throwing adapter preserves fields"); }
        values[field] = saved;
    }
    auto callback = p;
    callback.validate_model = [](const metadata & read, uint32_t) { (void) required(read, "callback.only"); };
    values["callback.only"] = std::string(4096, 'x');
    const auto callback_report = validate_model_contract(callback, get, 7);
    check(callback_report.errors[0].field == "callback.only" && callback_report.errors[0].actual.size() < 80,
          "callback-only oversized metadata is bounded and field-specific");
    metadata throwing = [&](const char * key) -> std::optional<std::string> {
        if (std::string(key) == "mock.width") { throw std::runtime_error(std::string(1024, 'e')); }
        return get(key);
    };
    const auto read_failure = validate_model_contract(p, throwing, 7);
    check(read_failure.errors[0].field == "mock.width" && read_failure.errors[0].actual.size() < 300,
          "metadata callback exceptions become bounded field diagnostics");
    request_contract request; request.context = 9; request.batch = 9;
    check(validate_request_contract(p, request).accepted(), "boundary IO/cache request");
    request.context = 10; request.batch = 10; request.sequences = 2;
    request.input = "embeddings-f32"; request.output = "all-logits-f16";
    request.cache_k = "q8_0"; request.cache_v = "f32";
    const auto rejected = validate_request_contract(p, request);
    check(rejected.errors.size() == 7, "all request limit and IO/cache mismatches reported");
    p.validate_model(get, 7);
    rejects([&] { p.validate_model(get, 8); });
    values["mock.width"] = "4"; rejects([&] { p.validate_model(get, 7); });
    values["mock.width"] = std::string(257, '3'); rejects([&] { p.validate_model(get, 7); });
    values.erase("mock.width"); rejects([&] { p.validate_model(get, 7); });
    values["npu.dsl.abi"] = "unknown"; rejects([&] { only.resolve_model(get, 7); });
    auto record = npudsl_model_test::record_provider();
    auto assets = prepare_assets(record, std::make_shared<npudsl_model_test::record_source>(), {std::string(64, 'a'), std::string(64, 'b'), true});
    session_options options; options.batch = 9;
    auto s = create_session(record, assets, options);
    uint32_t t[] = {1, 2}; float output[7];
    rejects([&] { s->evaluate(t, 2, 1, output, 7); });
    check(s->position() == 0 && s->submissions().kernels == 0, "argument errors do not mutate");
    s->evaluate(t, 2, 0, output, 7);
    check(s->position() == 2 && s->submissions().prefill == 1 && output[6] == 38, "mock evaluation");
    s->reset(); check(s->position() == 0 && s->submissions().kernels == 1, "reset preserves counters");
    npudsl_model_test::fail_next = true; rejects([&] { s->evaluate(t, 1, 0, output, 7); });
    s->reset(); check(s->poisoned(), "reset must not clear poison");
    rejects([&] { s->evaluate(t, 1, 0, output, 7); });
}
void package_contract_tests() {
    using namespace npudsl_model;
    auto p = npudsl_model_test::mock_provider();
    package_text text{{"layout.mock", "weights.mock mock.tensor 0 4\n"}, {"state.mock", "MOCK two layers\n"}};
    std::map<std::string, uint64_t> sizes{{"layout.mock", text.at("layout.mock").size()},
        {"state.mock", text.at("state.mock").size()}, {"weights.mock", 4}};
    check(validate_package_contract(p, text, sizes).weights == "MOCK-tags", "validated legacy format evidence");
    auto missing = p; missing.package.validated_legacy_format = false;
    rejects([&] { validate_package_contract(missing, text, sizes); });
    p.package.validated_legacy_format = false;
    p.package.format_file = "format.mock";
    p.package.files.push_back({"format.mock", 4096}); p.package.metadata_files.push_back("format.mock");
    use_legacy_package(p);
    registry r; r.add(p);
    const auto declaration = "weights MOCK-tags\nscales MOCK-none\ndense MOCK-tags\nhead MOCK-tags\nlayout MOCK-layout-v1\ngroup_size 1\n";
    text["format.mock"] = declaration; sizes["format.mock"] = text.at("format.mock").size();
    check(validate_package_contract(p, text, sizes).group_size == 1, "independent explicit package format declaration");
    text["format.mock"].replace(8, 9, "Q4_K");
    rejects([&] { validate_package_contract(p, text, sizes); });
    text["format.mock"] = std::string(declaration) + "group_size 1\n";
    rejects([&] { validate_package_contract(p, text, sizes); });
    text["format.mock"] = "weights MOCK-tags\n";
    rejects([&] { validate_package_contract(p, text, sizes); });
    text["format.mock"] = declaration;
    text["layout.mock"] = "weights.mock wrong.semantic.name 0 4\n";
    rejects([&] { validate_package_contract(p, text, sizes); });
    text["layout.mock"] = "weights.mock mock.tensor 0 4\n";
    sizes["weights.mock"] = 5; rejects([&] { validate_package_contract(p, text, sizes); });
    sizes["weights.mock"] = 4;
    std::map<std::string, std::string> values{{"general.architecture", "llama"}, {"mock.width", "3"}};
    metadata get = [&](const char * key) -> std::optional<std::string> {
        const auto it = values.find(key); return it == values.end() ? std::nullopt : std::optional<std::string>(it->second);
    };
    rejects([&] { r.resolve_model(get, 7); }); // architecture alone supplies no format
    check(r.resolve_model(get, 7, [&](const provider & candidate) {
        return validate_package_contract(candidate, text, sizes);
    }).abi == p.abi, "unbound dispatch obtains actual package contract evidence");
#ifdef NPUDSL_TEST_QWEN36
    const auto & qwen = providers().resolve(qwen36_contract::abi);
    values.clear();
    for (const auto & dimension : qwen36_contract::dimensions) { values[dimension.first] = dimension.second; }
    values["qwen35moe.block_count"] = "40"; values["qwen35moe.rope.freq_base"] = "10000000";
    values["qwen35moe.attention.layer_norm_rms_epsilon"] = "0.000001";
    check(validate_model_contract(qwen, get, qwen.info.vocabulary, &qwen.format).accepted(), "actual Qwen semantic field contract");
    // Preserve total Q projection width while changing head semantics. Byte-size
    // equality alone cannot authorize this model.
    values["qwen35moe.attention.head_count"] = "32";
    values["qwen35moe.attention.key_length"] = "128";
    const auto semantic = validate_model_contract(qwen, get, qwen.info.vocabulary, &qwen.format);
    check(!semantic.accepted() && semantic.errors[0].field == "qwen35moe.attention.head_count" &&
          semantic.errors[1].field == "qwen35moe.attention.key_length", "equal-byte shape semantic mismatch diagnosed by field");
    #endif
}
void bridge_tests() {
    using namespace npu_bind;
    using namespace npudsl_model_test;
    npudsl_model::providers().add(record_provider());
    const auto & p = npudsl_model::providers().resolve("MOCK-records-v3");
    staged_file scratch(fs::temp_directory_path() / ("npu-provider-test-" + std::to_string(std::random_device{}())));
    const auto root = fs::canonical(scratch.directory);
    check(p.package.tensors.empty() && p.package.manifest_file.empty(), "bridge fixture has no tensor/blob manifest schema");
    write(root / "weights.mock", "MOCK"); write(root / "state.mock", "MOCK two layers\n");
    validate_package(p, root);
    rejects([&] { validate_manifest(p, "weights.mock mock.tensor 1 4\n", {{"weights.mock", 5}}); });
    gguf_ptr kv(gguf_init_empty(), gguf_free);
    gguf_set_val_str(kv.get(), "general.architecture", "llama");
    gguf_set_val_u32(kv.get(), "mock.width", 3);
    gguf_set_val_str(kv.get(), "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(kv.get(), "tokenizer.ggml.pre", "gpt-2");
    const char * tokens[] = {"a", "b", "c", "d", "e", "f", "<|endoftext|>"};
    const int32_t types[] = {1, 1, 1, 1, 1, 1, 3};
    gguf_set_arr_str(kv.get(), "tokenizer.ggml.tokens", tokens, 7);
    gguf_set_arr_str(kv.get(), "tokenizer.ggml.merges", nullptr, 0);
    gguf_set_arr_data(kv.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types, 7);
    gguf_set_val_u32(kv.get(), "tokenizer.ggml.bos_token_id", 6);
    gguf_set_val_u32(kv.get(), "tokenizer.ggml.eos_token_id", 6);
    const auto source = root / "source.gguf", bound = root / "bound.gguf";
    check(gguf_write_to_file(kv.get(), source.string().c_str(), true), "write mock vocabulary");
    bind({source.string(), root.string(), bound.string(), "MOCK-no-numerical-model"});
    const auto identity = read_text(root / "identity.txt");
    auto model = load_tokenizer(bound);
    check(validate_model(model.get()).abi == p.abi, "binder selected mock ABI");
    check(identity == identity_text(p, root, fingerprint(model.get())), "provider-owned identity bytes/order");
    const auto dir = root.string();
    auto cp = llama_context_default_params(); cp.npu_model_path = dir.c_str(); cp.n_ctx = 0; cp.n_batch = 9;
    using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
    auto construct = [&](llama_context_params params) { return context_ptr(llama_init_from_model(model.get(), params), llama_free); };
    auto reject_context = [&](llama_context_params params) {
        const auto before = constructions; auto ctx = construct(params);
        check(!ctx && before == constructions, "reject before provider session construction");
    };
    auto oversized = cp; oversized.n_ctx = 10; reject_context(oversized);
        for (const int value : {-1, int(GGML_TYPE_COUNT), int(GGML_TYPE_COUNT) + 1, 2147483647}) {
          auto invalid = cp; invalid.type_k = static_cast<ggml_type>(value); reject_context(invalid);
          invalid = cp; invalid.type_v = static_cast<ggml_type>(value); reject_context(invalid);
          // The non-NPU public entry must also reject without calling GGML helpers.
          invalid.npu_model_path = nullptr; reject_context(invalid);
          invalid = cp; invalid.npu_model_path = nullptr; invalid.type_k = static_cast<ggml_type>(value);
          reject_context(invalid);
        }
        // Partial declarations fail in preflight, before even resolving the package
        // path (and therefore before hashing). Capture the actual public diagnostic.
        gguf_ptr bound_kv(gguf_init_from_file(bound.string().c_str(), {true, nullptr}), gguf_free);
        check(bool(bound_kv), "read bound metadata fixture");
        const char * fields[] = {"weights", "scales", "dense", "head", "layout", "group_size"};
        const auto missing_dir = (root / "nonexistent-package").string();
        for (size_t missing = 0; missing < 6; ++missing) {
          gguf_ptr partial(gguf_init_empty(), gguf_free); gguf_set_kv(partial.get(), bound_kv.get());
          const auto key = std::string("npu.dsl.format.") + fields[missing];
          gguf_remove_key(partial.get(), key.c_str());
          const auto path = root / ("partial-" + std::to_string(missing) + ".gguf");
          check(gguf_write_to_file(partial.get(), path.string().c_str(), true), "write partial format fixture");
          auto partial_model = load_tokenizer(path);
          std::string diagnostic;
          llama_log_set([](ggml_log_level, const char * text, void * data) {
            *static_cast<std::string *>(data) += text;
          }, &diagnostic);
          auto invalid = cp; invalid.npu_model_path = missing_dir.c_str();
          const auto before = constructions;
          auto rejected = context_ptr(llama_init_from_model(partial_model.get(), invalid), llama_free);
          llama_log_set(nullptr, nullptr);
          check(!rejected && before == constructions && diagnostic.find(key + ": expected") != std::string::npos,
              "partial format rejected by field before package IO/hash/session");
        }
        gguf_ptr legacy(gguf_init_empty(), gguf_free); gguf_set_kv(legacy.get(), bound_kv.get());
        for (const char * field : fields) { gguf_remove_key(legacy.get(), (std::string("npu.dsl.format.") + field).c_str()); }
        const auto legacy_path = root / "legacy.gguf";
        check(gguf_write_to_file(legacy.get(), legacy_path.string().c_str(), true), "write all-absent legacy fixture");
        auto legacy_model = load_tokenizer(legacy_path);
        auto legacy_context = context_ptr(llama_init_from_model(legacy_model.get(), cp), llama_free);
        check(bool(legacy_context), "all six absent may defer to validated package evidence");
        legacy_context.reset();
        auto physical = cp; physical.n_ubatch = 2;
        auto physical_context = construct(physical);
        check(bool(physical_context) && llama_n_batch(physical_context.get()) == 9 && llama_n_ubatch(physical_context.get()) == 2,
            "unequal logical/physical caps reported honestly");
        llama_token physical_ids[] = {1, 2, 3};
        check(llama_decode(physical_context.get(), llama_batch_get_one(physical_ids, 2)) == 0, "physical boundary accepted");
        check(llama_decode(physical_context.get(), llama_batch_get_one(physical_ids, 3)) != 0 &&
            !llama_get_logits(physical_context.get()) &&
            llama_memory_seq_pos_max(llama_get_memory(physical_context.get()), 0) == 1,
            "over physical cap rejects without splitting or state mutation");
        check(llama_decode(physical_context.get(), llama_batch_get_one(physical_ids, 1)) == 0 &&
            llama_get_logits(physical_context.get())[6] == 54, "physical rejection preserves healthy retry");
        auto ext = std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)>(llama_batch_ext_init(physical_context.get()), llama_batch_ext_free);
        for (int i = 0; i < 3; ++i) {
          const auto index = llama_batch_ext_add_token(ext.get(), 0, physical_ids[i]);
          const llama_pos position = 3 + i;
          check(index >= 0 && llama_batch_ext_set_pos(ext.get(), index, &position), "extended batch fixture");
        }
        check(llama_process(physical_context.get(), LLAMA_PROCESS_TYPE_DECODE, ext.get()) != 0 &&
            !llama_get_logits(physical_context.get()) &&
            llama_memory_seq_pos_max(llama_get_memory(physical_context.get()), 0) == 2,
            "extended decode also enforces physical cap without mutation");
        physical_context.reset();
        auto automatic = cp; automatic.n_ubatch = 0;
        auto automatic_context = construct(automatic);
        check(bool(automatic_context) && llama_n_ubatch(automatic_context.get()) == 9, "zero physical cap selects logical cap");
        automatic_context.reset();
    auto ctx = construct(cp);
    check(bool(ctx) && llama_n_ctx(ctx.get()) == 9, "provider default capacity, not Qwen context");
    wrong_effective_config = true;
    auto wrong_config = construct(cp);
    wrong_effective_config = false;
    check(!wrong_config, "public bridge rejects inconsistent effective provider config before execution");
    auto decode = [&](std::initializer_list<llama_token> ids) {
        std::vector<llama_token> input(ids); return llama_decode(ctx.get(), llama_batch_get_one(input.data(), int32_t(input.size())));
    };
    check(decode({1, 2}) == 0, "mock provider prefill through unchanged generic bridge");
    auto * logits = llama_get_logits_ith(ctx.get(), -1);
    check(logits && logits[0] == 32 && logits[6] == 38, "seven-element logits");
    check(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == 1, "provider position");
    check(decode({7}) != 0 && !llama_get_logits(ctx.get()), "mock vocabulary bound and output invalidation");
    check(decode({3}) == 0, "retry argument failure");
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    check(decode({1, 2}) == 0 && llama_get_logits(ctx.get())[6] == 38, "reset replay");
    fail_next = true;
    check(decode({3}) != 0 && !llama_get_logits(ctx.get()), "provider failure never publishes partial output");
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    check(decode({1}) != 0, "poison persists across reset");
    write(root / "weights.mock", "FAIL"); reject_context(cp); write(root / "weights.mock", "MOCK");
    write(root / "identity.txt", identity + "extra\n"); reject_context(cp); write(root / "identity.txt", identity);
    write(root / "state.mock", "MOCK bad layers\n");
    rejects([&] { validate_package(p, root); }); write(root / "state.mock", "MOCK two layers\n");
    auto restored = construct(cp); check(bool(restored), "restored package still accepted");
    restored.reset(); ctx.reset();
    // A different checkpoint's content with the same structure is not whitelisted
    // by its previous package hash. Fresh binding must verify its own identity.
    fs::remove(root / "identity.txt");
    write(root / "weights.mock", "NEW!");
    const auto rebound = root / "rebound.gguf";
    bind({source.string(), root.string(), rebound.string(), "MOCK-different-checkpoint"});
    auto new_model = load_tokenizer(rebound);
    auto new_context = context_ptr(llama_init_from_model(new_model.get(), cp), llama_free);
    check(bool(new_context), "same-format different weight contents accepted after rebind");
    check(read_text(root / "identity.txt") != identity, "new content gets distinct integrity binding");
    reject_context(cp); // old bound GGUF cannot authorize the new package
    auto bad_batch = cp; bad_batch.n_batch = 10;
    const auto before = constructions;
    auto bad_context = context_ptr(llama_init_from_model(new_model.get(), bad_batch), llama_free);
    check(!bad_context && constructions == before, "provider batch limit enforced before construction");
    std::cout << "PASS MOCK second provider: generic binder/context/registry; NOT model numerical support\n";
}
}
int main() {
    try {
        registry_tests(); package_contract_tests(); bridge_tests();
        std::cout << "PASS provider contracts: " << checks << " checks\n"; return 0;
    } catch (const std::exception & e) { std::cerr << "FAIL provider contracts: " << e.what() << '\n'; return 1; }
}