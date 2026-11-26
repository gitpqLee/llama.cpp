param([Parameter(Mandatory=$true)][string]$DslRoot,
      [Parameter(Mandatory=$true)][string]$DirectBuild,
      [Parameter(Mandatory=$true)][string]$LlamaBuild,
      [Parameter(Mandatory=$true)][string]$Out)
$ErrorActionPreference='Stop'
$null=Get-Command cl.exe
$dsl=(Resolve-Path $DslRoot).Path; $direct=(Resolve-Path $DirectBuild).Path
$lb=(Resolve-Path $LlamaBuild).Path; $llama=(Resolve-Path "$PSScriptRoot\..\..").Path
$outDir=[IO.Directory]::CreateDirectory($Out).FullName
$gc="$dsl\thirdparty\generic_compute"
& "$PSScriptRoot\extract.ps1" "$dsl\kernels\qwen4\qwen4_run.cpp" "$outDir\direct_host_extract.inc"
$flags=@('/nologo','/EHsc','/O2','/MD','/std:c++17','/DNOMINMAX','/DGGML_SHARED',"/I$outDir","/I$llama\ggml\include","/I$llama\ggml\src","/I$gc","/I$gc\src\host\npu_runtime\include","/I$dsl\runtime","/I$dsl\kernels\qwen4\generated")
$libs=@("$lb\ggml\src\ggml-base.lib","$direct\npu_framework\src\host\npu_runtime\Release\npu_runtime.lib","$direct\npu_framework\src\host\ze_wrapper\Release\ze_wrapper.lib","$gc\src\host\ze_wrapper\thirdparty\level-zero-win-sdk-1.22.4\lib\ze_loader.lib","$direct\Release\generated_kernels_registry.lib")
Push-Location $outDir
try {
    & cl.exe @flags /LD "$PSScriptRoot\same_kernel_backend.cpp" @libs /link /out:same_kernel_npu.dll
    if ($LASTEXITCODE) { throw 'Plugin build failed' }
    & cl.exe @flags "/I$llama\include" "$PSScriptRoot\same_kernel_bench.cpp" "$lb\src\llama.lib" "$lb\ggml\src\ggml.lib" "$lb\ggml\src\ggml-base.lib" /link /out:same_kernel_bench.exe
    if ($LASTEXITCODE) { throw 'Harness build failed' }
} finally { Pop-Location }