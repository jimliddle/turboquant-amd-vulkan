param(
    [string]$BenchExe = "C:\Users\jimli\projects\turboquants\build-vulkan-tests\bin\Release\llama-bench.exe",
    [string]$Model = "C:\Users\jimli\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf",
    [string]$OutputDir = "C:\Users\jimli\projects\turboquants\bench-results",
    [string]$Device = "Vulkan0",
    [int]$Repetitions = 3,
    [string]$PromptGenList = "512,128;2048,128;4096,256;4096,1024;8192,256"
)

$ErrorActionPreference = "Stop"
$PromptGen = $PromptGenList.Split(";", [System.StringSplitOptions]::RemoveEmptyEntries)

function Invoke-BenchRun {
    param(
        [string]$Label,
        [string[]]$ExtraArgs,
        [string]$JsonlPath,
        [string]$LogPath
    )

    $commonArgs = @(
        "-m", $Model,
        "-o", "jsonl",
        "-r", $Repetitions.ToString(),
        "-ngl", "999",
        "-sm", "none",
        "-fa", "1",
        "-ctk", "f16",
        "-ctv", "f16",
        "-dev", $Device
    )

    foreach ($pg in $PromptGen) {
        $commonArgs += @("-pg", $pg)
    }

    Write-Host "Running $Label..."
    $allArgs = @($commonArgs + $ExtraArgs)
    $proc = Start-Process -FilePath $BenchExe `
        -ArgumentList $allArgs `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $JsonlPath `
        -RedirectStandardError $LogPath
    if ($proc.ExitCode -ne 0) {
        throw "Benchmark run '$Label' failed with exit code $($proc.ExitCode). See $LogPath"
    }
}

function Read-BenchJsonl {
    param([string]$Path)

    $rows = @()
    foreach ($line in Get-Content -Path $Path) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        $rows += ($line | ConvertFrom-Json)
    }
    return $rows
}

function Get-BenchKey {
    param($Row)
    return "{0}|{1}" -f $Row.n_prompt, $Row.n_gen
}

function Get-BenchLabel {
    param($Row)
    if ($Row.n_prompt -gt 0 -and $Row.n_gen -eq 0) {
        return "pp$($Row.n_prompt)"
    }
    if ($Row.n_prompt -eq 0 -and $Row.n_gen -gt 0) {
        return "tg$($Row.n_gen)"
    }
    return "pp$($Row.n_prompt)+tg$($Row.n_gen)"
}

function Format-Delta {
    param([double]$Baseline, [double]$Candidate)
    if ($Baseline -eq 0) {
        return ""
    }
    return "{0:P1}" -f (($Candidate - $Baseline) / $Baseline)
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDir = Join-Path $OutputDir "turboquant-vulkan-$stamp"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$baselineJsonl = Join-Path $runDir "baseline.jsonl"
$baselineLog = Join-Path $runDir "baseline.stderr.log"
$turboJsonl = Join-Path $runDir "turboquant.jsonl"
$turboLog = Join-Path $runDir "turboquant.stderr.log"
$summaryPath = Join-Path $runDir "summary.md"

Invoke-BenchRun -Label "baseline" -ExtraArgs @("-nkvo", "0") -JsonlPath $baselineJsonl -LogPath $baselineLog
Invoke-BenchRun -Label "turboquant" -ExtraArgs @("--kv-codec", "turboquant", "--kv-tq-runtime", "vulkan") -JsonlPath $turboJsonl -LogPath $turboLog

$baselineRows = Read-BenchJsonl -Path $baselineJsonl
$turboRows = Read-BenchJsonl -Path $turboJsonl

$baselineByKey = @{}
foreach ($row in $baselineRows) {
    $baselineByKey[(Get-BenchKey -Row $row)] = $row
}

$summary = @()
$summary += "| n_prompt | n_gen | test | baseline t/s | turboquant t/s | delta |"
$summary += "| ---: | ---: | :--- | ---: | ---: | ---: |"

foreach ($row in $turboRows | Sort-Object n_prompt, n_gen, test) {
    $key = Get-BenchKey -Row $row
    if (-not $baselineByKey.ContainsKey($key)) {
        continue
    }

    $base = $baselineByKey[$key]
    $summary += "| {0} | {1} | {2} | {3:N2} | {4:N2} | {5} |" -f `
        $row.n_prompt, $row.n_gen, (Get-BenchLabel -Row $row), [double]$base.avg_ts, [double]$row.avg_ts, (Format-Delta -Baseline ([double]$base.avg_ts) -Candidate ([double]$row.avg_ts))
}

$summary += ""
$summary += "Artifacts:"
$summary += "- baseline jsonl: $baselineJsonl"
$summary += "- baseline stderr: $baselineLog"
$summary += "- turboquant jsonl: $turboJsonl"
$summary += "- turboquant stderr: $turboLog"

Set-Content -Path $summaryPath -Value $summary

Write-Host ""
Get-Content -Path $summaryPath
Write-Host ""
Write-Host "Summary written to $summaryPath"
