# TurboQuant Model Usage

This document explains how to use the TurboQuant fork with GGUF models stored in the local LM Studio model cache:

- `C:\Users\username\.lmstudio\models`

The focus here is practical usage, not build steps. It covers:

- `llama-bench`
- `llama-cli`
- `llama-server`

## Model Location

LM Studio stores downloaded models under a directory structure like:

```text
C:\Users\username\.lmstudio\models\<publisher>\<model-folder>\<file>.gguf
```

Example used in this repository:

```text
C:\Users\username\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf
```

## Finding GGUF Files

PowerShell:

```powershell
Get-ChildItem "C:\Users\username\.lmstudio\models" -Recurse -Filter *.gguf |
    Select-Object FullName
```

If you want one model family:

```powershell
Get-ChildItem "C:\Users\username\.lmstudio\models\unsloth\gpt-oss-20b-GGUF" -Filter *.gguf |
    Select-Object FullName
```

## Choosing A Model

For first validation runs, prefer:

- a known-working GGUF
- a quant that already fits the target hardware comfortably
- one model that you will reuse across baseline and TurboQuant runs

That keeps the comparison clean. In this repository, the practical proof model has been:

```text
C:\Users\username\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf
```

## Running A Standard Baseline

Use the normal KV path:

```powershell
.\build-vulkan-tests\bin\Release\llama-bench.exe `
  -m "C:\Users\username\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf" `
  -o jsonl `
  -r 3 `
  -ngl 999 `
  -sm none `
  -fa 1 `
  -ctk f16 `
  -ctv f16 `
  -dev Vulkan0 `
  -nkvo 0 `
  -pg 0,128 `
  -pg 512,128 `
  -pg 4096,256
```

This is the right reference point for:

- standard `llama.cpp` behavior inside this fork
- upstream comparison runs

## Running TurboQuant

Use the same model path, but switch the KV codec:

```powershell
.\build-vulkan-tests\bin\Release\llama-bench.exe `
  -m "C:\Users\username\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf" `
  -o jsonl `
  -r 3 `
  -ngl 999 `
  -sm none `
  -fa 1 `
  -ctk f16 `
  -ctv f16 `
  -dev Vulkan0 `
  --kv-codec turboquant `
  --kv-tq-runtime vulkan `
  -pg 0,128 `
  -pg 512,128 `
  -pg 4096,256
```

Important point:

- the GGUF file does not need to be converted into a special TurboQuant model format
- TurboQuant in this fork is a KV-cache/runtime path, not a different GGUF container requirement

## Running llama-cli

Interactive CLI usage works the same way: point the binary at the GGUF in the LM Studio cache and choose either the standard KV path or the TurboQuant path.

Baseline example:

```powershell
.\build-vulkan-tests\bin\Release\llama-cli.exe `
  -m "C:\Users\username\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf" `
  -ngl 999 `
  -sm none `
  -fa on `
  -ctk f16 `
  -ctv f16 `
  -dev Vulkan0 `
  -p "Write a short summary of TurboQuant."
```

TurboQuant example:

```powershell
.\build-vulkan-tests\bin\Release\llama-cli.exe `
  -m "C:\Users\username\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf" `
  -ngl 999 `
  -sm none `
  -fa on `
  -ctk f16 `
  -ctv f16 `
  -dev Vulkan0 `
  --kv-codec turboquant `
  --kv-tq-runtime vulkan `
  -p "Write a short summary of TurboQuant."
```

Practical notes for `llama-cli`:

- use the exact same GGUF path for baseline and TurboQuant if you want a fair comparison
- keep `-fa`, `-ctk`, `-ctv`, and device settings fixed between runs
- TurboQuant affects runtime KV behavior, so prompt length and generation length matter more than prompt-only smoke tests

## Running llama-server

`llama-server` can use the same GGUF files directly from the LM Studio cache.

Baseline example:

```powershell
.\build-vulkan-tests\bin\Release\llama-server.exe `
  -m "C:\Users\username\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf" `
  -ngl 999 `
  -sm none `
  -fa on `
  -ctk f16 `
  -ctv f16 `
  -dev Vulkan0 `
  --host 127.0.0.1 `
  --port 8080
```

TurboQuant example:

```powershell
.\build-vulkan-tests\bin\Release\llama-server.exe `
  -m "C:\Users\username\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf" `
  -ngl 999 `
  -sm none `
  -fa on `
  -ctk f16 `
  -ctv f16 `
  -dev Vulkan0 `
  --kv-codec turboquant `
  --kv-tq-runtime vulkan `
  --host 127.0.0.1 `
  --port 8080
```

Once the server is running, clients can use it the same way they would use a normal `llama-server` instance. The difference is only in how the backend KV cache is handled internally.

Practical notes for `llama-server`:

- use separate ports if you want to run baseline and TurboQuant side by side
- restart the server between mode changes unless you intentionally want separate processes
- long-context or generation-heavy requests are the best workloads for seeing TurboQuant behavior

## Checking Binary Availability

If you built only benchmark targets, `llama-cli.exe` or `llama-server.exe` may not exist yet. Build them explicitly:

```powershell
& "C:\Program Files\CMake\bin\cmake.exe" `
  --build .\build-vulkan-tests `
  --config Release `
  --target llama-cli llama-server
```

Expected outputs:

- `build-vulkan-tests/bin/Release/llama-cli.exe`
- `build-vulkan-tests/bin/Release/llama-server.exe`

## Running The Benchmark Harness

The simplest repeatable way to use models from the LM Studio cache is the harness:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bench-turboquant-vulkan.ps1 `
  -Model "C:\Users\username\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf" `
  -Device "Vulkan0" `
  -Repetitions 3 `
  -PromptGenList "0,128;512,0;512,128;2048,128;4096,256;4096,1024;8192,256"
```

This will:

- run the same GGUF in baseline mode
- run the same GGUF in TurboQuant mode
- write JSONL, stderr logs, and a Markdown summary under `bench-results/`

## Comparing Multiple GGUF Files

If you want to compare several GGUFs from the LM Studio cache, keep the command constant and only change `-Model`.

Good pattern:

1. pick one shape set
2. pick one repetition count
3. run baseline and TurboQuant for model A
4. run baseline and TurboQuant for model B
5. compare only after all runs use the same settings

Do not mix:

- different devices
- different `-fa` settings
- different cache types
- different repetition counts

if the goal is to judge TurboQuant rather than general model/runtime variance.

## Using A Variable For Model Paths

PowerShell example:

```powershell
$model = "C:\Users\username\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf"

.\build-vulkan-tests\bin\Release\llama-bench.exe `
  -m $model `
  -o jsonl `
  -r 3 `
  -ngl 999 `
  -sm none `
  -fa 1 `
  -ctk f16 `
  -ctv f16 `
  -dev Vulkan0 `
  --kv-codec turboquant `
  --kv-tq-runtime vulkan `
  -pg 4096,256
```

That makes it easier to switch between LM Studio models without rewriting the command.

## Upstream Comparison With The Same Model Cache

To compare this fork with clean upstream, keep the `-m` path identical and only change the binary:

Fork baseline:

```powershell
.\build-vulkan-tests\bin\Release\llama-bench.exe -m "<same gguf path>" ...
```

Upstream:

```powershell
.\upstream-clean\build-vulkan\bin\Release\llama-bench.exe -m "<same gguf path>" ...
```

That is the correct way to use the LM Studio model directory for apples-to-apples comparisons.

## Common Usage Notes

- GGUF models under the LM Studio cache can be used directly by path.
- No LM Studio runtime is required once the file is downloaded.
- TurboQuant changes KV-cache handling at runtime; it does not require a different GGUF file layout.
- For comparison work, keep one exact GGUF path fixed across all runs.
- The same cached GGUF can be used with `llama-bench`, `llama-cli`, and `llama-server`.
- For interactive or server testing, expect the biggest observable differences on longer contexts or sustained generation rather than trivial short prompts.

## Suggested Next Step

If you want to turn this into a model-by-model evaluation workflow, the next practical addition would be a small script that enumerates GGUFs under `C:\Users\jimli\.lmstudio\models` and runs the benchmark harness against each selected model automatically.
