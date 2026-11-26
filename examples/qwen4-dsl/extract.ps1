param([Parameter(Mandatory=$true)][string]$HostSource,
    [Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
$source = [IO.File]::ReadAllText($hostSource)
function Slice($begin, $end) {
    $a = $source.IndexOf($begin, [StringComparison]::Ordinal)
    if ($a -lt 0) { throw "Direct host marker missing: $begin" }
    $b = $source.IndexOf($end, $a, [StringComparison]::Ordinal)
    if ($b -lt $a) { throw "Direct host marker missing: $end" }
    $source.Substring($a, $b - $a)
}
$shared = Slice '#include <npu_runtime.hpp>' 'static int run(int argc, char** argv);'
# Use exceptions in the DLL, rather than terminating llama from inside the direct host.
$shared = $shared.Replace('std::cerr << "ERROR: " << m << std::endl;', '').Replace('std::exit(1);', 'throw std::runtime_error(m);')
$setup = Slice '    auto t0 = std::chrono::steady_clock::now();' '    std::ofstream profOut, traceOut;'
if ($setup.Contains('Model m(device, weights);')) {
    $setup = $setup.Replace('Model m(device, weights);', 'if (!indexerOverlay.empty()) die("This host requires a complete pack, not an overlay"); Model m(device, weights);')
}
$prefix = @'
template<class Serve>
static void with_direct_host(const std::string & weights, const std::string & indexerOverlay, Serve serve) {
    constexpr bool usePrefill = true;
    constexpr auto profBytesPrefill = kProfBytes;
'@
$suffix = @'
    // The device-visible mapping is expensive to read with scalar FP16 loads.
    // Copy the entire row in bulk, then run the same conversion and argmax.
    const bool stageLogits = std::getenv("QWEN4_STAGE_LOGITS") != nullptr;
    std::vector<uint16_t> stagedLogits(stageLogits ? kVocab : 0);
    serve([&](auto & r) {
        using Clock = std::chrono::steady_clock;
        ensureKv(r.pos + r.n);
        if (r.n == 1) {
            cfgDec->pos = r.pos;
            cfgDec->token = static_cast<uint32_t>(r.tokens[0]);
            r.ms = runLaunch(decode);
        } else {
            for (uint32_t i = 0; i < r.n; ++i) idsBuf[i] = static_cast<uint32_t>(r.tokens[i]);
            cfgPre->pos0 = r.pos;
            cfgPre->n_prompt = r.n;
            r.ms = runLaunch(prefill);
        }
        const auto readStart = r.profile ? Clock::now() : Clock::time_point{};
        const auto * src = logits;
        if (stageLogits) {
            std::memcpy(stagedLogits.data(), logits, kVocab * sizeof(*logits));
            src = stagedLogits.data();
        }
        const auto convertStart = r.profile ? Clock::now() : Clock::time_point{};
        ggml_fp16_to_fp32_row(src, r.logits, kVocab);
        const auto argmaxStart = r.profile ? Clock::now() : Clock::time_point{};
        const auto best = std::max_element(r.logits, r.logits + kVocab) - r.logits;
        const auto deviceToken = *tokOut;
        if (best != deviceToken) die("FP16 logits argmax disagrees with device token");
        if (r.profile) {
            using Ms = std::chrono::duration<double, std::milli>;
            r.copy_ms = Ms(convertStart - readStart).count();
            r.convert_ms = Ms(argmaxStart - convertStart).count();
            r.check_ms = Ms(Clock::now() - argmaxStart).count();
        }
    });
}
'@
[IO.File]::WriteAllText($OutputPath, "$shared`n$prefix`n$setup`n$suffix`n")