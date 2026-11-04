#define NPU_AUDIT_TEST
#include "npu-audit.cpp"

#include <random>

namespace {
void check(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception &) { rejected = true; }
    check(rejected, "expected rejection");
}

struct temp_directory {
    fs::path path;
    temp_directory() {
        for (int i = 0; i < 100; ++i) {
            const auto candidate = fs::temp_directory_path() / ("llama-npu-audit-test-" + std::to_string(std::random_device{}()));
            if (fs::create_directory(candidate)) { path = candidate; return; }
        }
        throw std::runtime_error("cannot create exclusive temporary fixture");
    }
    ~temp_directory() { std::error_code ec; fs::remove_all(path, ec); }
};

options args(std::initializer_list<const char *> values) {
    std::vector<std::string> strings;
    for (auto value : values) { strings.emplace_back(value); }
    std::vector<char *> pointers;
    for (auto & value : strings) { pointers.push_back(value.data()); }
    return parse(static_cast<int>(pointers.size()), pointers.data());
}

void test_arguments() {
    check(args({"audit", "--help"}).help, "help");
    const auto o = args({"audit", "--model", "m", "--weights", "w", "--tokens", "t", "--prompt-count", "3", "--output", "o", "--reference", "r"});
    check(o.prompt == 3 && o.reference == "r" && !o.help, "valid arguments");
    rejects([] { args({"audit"}); });
    rejects([] { args({"audit", "--help", "--model", "m"}); });
    rejects([] { args({"audit", "--bogus", "x"}); });
    for (const char * knob : {"--tau", "--mean-tol", "--max-tol", "--tolerance"}) {
        rejects([&] { args({"audit", "--model", "m", "--weights", "w", "--tokens", "t",
                           "--prompt-count", "3", "--output", "o", knob, "1"}); });
    }
    rejects([] { args({"audit", "--model"}); });
    rejects([] { args({"audit", "--model", "m", "--model", "n"}); });
    for (const char * number : {"0", "-1", "+1", "1x", "4097", "42949672960"}) {
        rejects([&] { args({"audit", "--model", "m", "--weights", "w", "--tokens", "t", "--prompt-count", number, "--output", "o"}); });
    }
    const std::string long_path(241, 'x');
    rejects([&] { args({"audit", "--model", long_path.c_str()}); });
}

void test_tokens_and_files(const fs::path & root) {
    options o;
    o.tokens = (root / "tokens.txt").string();
    o.prompt = 3;
    auto fixture = [&](const std::string & text) {
        { std::ofstream out(o.tokens, std::ios::binary); out << text; check(bool(out), "fixture write"); }
        return read_tokens(o);
    };
    check(fixture("0\t1\r\n42 99\v100\f").size() == 5, "whitespace token file");
    for (const std::string & text : std::vector<std::string>{"", "1 2", "1 2 -3", "1 2 +3", "1 2 3x", "1 2 4294967295", "1 2 4294967296", "1 2 00000000000", std::string("1 2 3\0x", 7)}) {
        rejects([&] { fixture(text); });
    }
    std::string limit;
    for (uint32_t i = 0; i < max_tokens; ++i) { limit += "1 "; }
    check(fixture(limit).size() == max_tokens, "maximum sequence length");
    rejects([&] { fixture(limit + "1"); });
    rejects([&] { fixture(std::string(65537, ' ')); });

    const auto path = (root / "exclusive.bin").string();
    auto file = exclusive(path);
    write(file.get(), "keep", 4);
    close_file(file);
    rejects([&] { exclusive(path); });
    check(read_text(path, 4) == "keep", "exclusive creation preserved existing bytes");
}

void test_comparisons() {
    const float expected[] = {1, -2, 3, 0};
    differences same;
    same.compare(expected, expected, 4);
    check(!same.mismatches && !same.argmax_mismatches && same.max_abs == 0, "equal full logits");
    float changed[] = {1, -2, 3, 0};
    changed[1] = std::nextafter(changed[1], 0.0f);
    differences one_bit;
    one_bit.compare(changed, expected, 4);
    check(one_bit.mismatches == 1 && one_bit.max_abs > 0 && !one_bit.argmax_mismatches, "detect non-argmax single-ULP difference");
    changed[0] = 4;
    differences best;
    best.compare(changed, expected, 4);
    check(best.argmax_mismatches == 1, "detect changed argmax");
    for (float value : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        changed[3] = value;
        rejects([&] { differences d; d.compare(changed, expected, 4); });
        rejects([&] { differences d; d.compare(expected, changed, 4); });
    }
}

bool accepts(const differences & d, const char * label) {
    file_ptr file(std::tmpfile(), std::fclose);
    check(bool(file), "temporary report file");
    return d.report(file.get(), label);
}

void test_policy() {
    const float ref[] = {1, -2, 3, 0};
    float rounded[] = {1, std::nextafter(-2.0f, 0.0f), 3, 0};
    differences ulp;
    ulp.compare(rounded, ref, 4);
    // Fail-first regression: the old zero-tolerance policy rejects this split.
    check(accepts(ulp, "split_continuation"), "split must accept rounding under golden policy");
    check(accepts(ulp, "independent_reference"), "independent reference accepts rounding");
    check(!accepts(ulp, "reset_replay"), "reset rejects even a non-argmax single ULP");
    differences same;
    same.compare(ref, ref, 4);
    check(accepts(same, "reset_replay"), "reset identical bits pass");
    check(!accepts(differences{}, "reset_replay"), "empty reset cannot pass");

    const float zero[] = {0, 0};
    const float signed_zero[] = {-0.0f, 0};
    differences sign;
    sign.compare(signed_zero, zero, 2);
    check(!accepts(sign, "reset_replay"), "reset rejects signed-zero bit difference");
    check(accepts(sign, "split_continuation"), "golden zero norm with zero delta passes");
    const float nonzero[] = {0.001f, 0};
    differences zero_norm;
    zero_norm.compare(nonzero, zero, 2);
    check(!accepts(zero_norm, "split_continuation"), "zero reference norm with nonzero delta fails");

    const float unit[] = {1, 0};
    const float mean_bad[] = {1, 0.05f};
    const float max_bad[] = {1, 0.11f};
    const float good[] = {1, 0.02f};
    for (const char * label : {"split_continuation", "sequential_prompt", "independent_reference"}) {
        differences mean;
        mean.compare(mean_bad, unit, 2); // mean .05 fails, max .05 passes
        check(!accepts(mean, label), "relative L2 mean alone must fail");
        differences worst;
        worst.compare(max_bad, unit, 2);
        for (int i = 0; i < 9; ++i) { worst.compare(unit, unit, 2); }
        check(!accepts(worst, label), "relative L2 max alone must fail despite mean .011");
        differences ok;
        ok.compare(good, unit, 2);
        check(accepts(ok, label), "both L2 criteria pass");
        const float mean_boundary_ref[] = {25, 0};
        const float mean_boundary_actual[] = {25, 1};
        differences mean_boundary;
        mean_boundary.compare(mean_boundary_actual, mean_boundary_ref, 2);
        check(accepts(mean_boundary, label), "mean threshold is inclusive");
        const float max_boundary_ref[] = {10, 0};
        const float max_boundary_actual[] = {10, 1};
        differences max_boundary;
        max_boundary.compare(max_boundary_actual, max_boundary_ref, 2);
        for (int i = 0; i < 3; ++i) { max_boundary.compare(unit, unit, 2); }
        check(accepts(max_boundary, label), "max threshold is inclusive");
        differences empty;
        check(!accepts(empty, label), "empty comparison cannot pass");

        // A large shared component isolates argmax policy from the L2 gates.
        const float far_ref[] = {2, 1, -1000};
        const float far_actual[] = {1, 2, -1000};
        differences far;
        far.compare(far_actual, far_ref, 3);
        check(far.argmax_mismatches == 1 && !accepts(far, label), "far-margin argmax must fail");
        for (float runner : {1.5f, 1.75f, 2.0f}) {
            const float near_ref[] = {2, runner, -1000};
            const float near_actual[] = {runner, 2.01f, -1000};
            differences near;
            near.compare(near_actual, near_ref, 3);
            check(near.argmax_mismatches == 1 && accepts(near, label), "near tie including tau boundary passes");
            check(!accepts(near, "reset_replay"), "near tie never relaxes reset");
        }
        const float just_far[] = {2, std::nextafter(1.5f, 0.0f), -1000};
        differences outside;
        outside.compare(far_actual, just_far, 3);
        check(!accepts(outside, label), "margin immediately above tau fails");
        const float top2_ref[] = {2, 1.75f, 1, -1000};
        const float third_choice[] = {2, 1.75f, 2.01f, -1000};
        differences top2;
        top2.compare(third_choice, top2_ref, 4);
        check(!accepts(top2, label), "reference near top2 cannot excuse a distant chosen token");
    }
    const float tied[] = {2, 2, 1};
    check(argmax(tied, 3) == 0, "first index wins exact ties");
    const float huge[] = {std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
    const float tiny[] = {std::numeric_limits<float>::denorm_min(), 0};
    const float tiny_actual[] = {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::denorm_min()};
    differences large, small;
    large.compare(huge, huge, 2);
    small.compare(tiny_actual, tiny, 2);
    check(accepts(large, "independent_reference"), "double norms avoid float overflow");
    check(!accepts(small, "independent_reference"), "double norms avoid float underflow");
        differences singleton;
        const auto single = singleton.compare(ref, ref, 1);
        check(std::isinf(single.reference_top2_margin) && single.argmax_equal &&
            accepts(singleton, "independent_reference"), "singleton vocabulary has exact argmax without runner-up");
    rejects([&] { differences d; d.compare(ref, ref, 0); });
        rejects([&] { accepts(ulp, "unknown_policy"); });
        check(ulp.rows == 1 && ulp.mismatches == 1 && ulp.mismatched_bits > 0 && ulp.max_abs > 0,
            "retain bit and absolute diagnostics on golden pass");
        check(sign.mismatched_bits == 1 && sign.max_abs == 0, "signed-zero diagnostics");
    for (float value : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        float invalid[] = {1, value};
        rejects([&] { differences d; d.compare(invalid, zero, 2); });
        rejects([&] { differences d; d.compare(zero, invalid, 2); });
    }
}

void test_reports() {
    const float ref[] = {2, 1.75f, -1000};
    const float actual[] = {1.75f, 2, -1000};
    differences d;
    const auto row = d.compare(actual, ref, 3);
    for (const char * label : {"reset_replay", "split_continuation", "independent_reference"}) {
        file_ptr file(std::tmpfile(), std::fclose);
        check(bool(file), "temporary diagnostic report");
        const bool passed = d.report(file.get(), label);
        report_row(file.get(), label, 0, row);
        check(std::fseek(file.get(), 0, SEEK_SET) == 0, "rewind diagnostic report");
        char buffer[4096] = {};
        const size_t count = std::fread(buffer, 1, sizeof(buffer) - 1, file.get());
        check(!std::ferror(file.get()) && count > 0 && std::feof(file.get()), "read complete report");
        const std::string text(buffer, count);
        const bool reset = std::strcmp(label, "reset_replay") == 0;
        check(passed != reset, "report result matches fixed criterion");
        for (const char * key : {"mismatched_elements=2", "mismatched_bits=", "argmax_mismatches=1",
                                "max_abs=0.25", "max_rel=", "rel_l2_mean=", "rel_l2_max=",
                                "near_ties=1", "far_argmax_mismatches=0", "comparison_row=0",
                                "argmax_equal=0", "reference_top2_margin=0.25"}) {
            check(text.find(key) != std::string::npos, "report retains complete diagnostics");
        }
        check(text.find(reset ? "criterion=bitexact" :
                        "criterion=golden_relL2_mean<=0.04_max<=0.10_and_(argmax_exact_OR_reference_chosen_gap<=0.5)")
              != std::string::npos, "report explicitly names criterion");
        check(text.find(reset ? "result=FAIL" : "result=PASS") != std::string::npos, "report names outcome");
    }
}
} // namespace

int main() {
    try {
        temp_directory temp;
        test_arguments();
        test_tokens_and_files(temp.path);
        test_comparisons();
        test_policy();
        test_reports();
        std::puts("PASS NPU audit host-only parsing, golden L2 mean/max, far/near-tie argmax, bitexact reset, finite/double norms, and no-clobber tests; NO model/device execution");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL NPU audit host-only tests: %s\n", e.what());
        return 1;
    }
}