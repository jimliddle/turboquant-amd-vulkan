# TurboQuant Fork Installation

This document covers practical installation and build steps for this TurboQuant-enabled `llama.cpp` fork.

## Scope

This guide is for building the fork itself, not installing LM Studio.

Validated local proof path:

- Windows
- Visual Studio 2022 Build Tools
- Vulkan SDK
- AMD Radeon 8060S via Vulkan

HIP/ROCm support is wired into the codebase, but the benchmark proof captured in this repository is currently the Vulkan path.

## Prerequisites

Install:

- Git
- Visual Studio 2022 Build Tools with C/C++ toolchain
- CMake
- Vulkan SDK

Optional but useful:

- CUDA toolkit if you want to compile-check the HIP/CUDA-side backend file layout in the Windows environment used during development

## Repository Setup

Clone the repository and enter it:

```powershell
git clone <your-fork-url> turboquants
cd turboquants
```

If the Vulkan SDK is not already on `PATH`, set `Vulkan_ROOT` explicitly for configure/build commands:

```powershell
$env:Vulkan_ROOT = "C:\VulkanSDK\1.4.341.1"
```

## Vulkan Build

Configure:

```powershell
& "C:\Program Files\CMake\bin\cmake.exe" `
  -S . `
  -B .\build-vulkan-tests `
  -DGGML_VULKAN=ON `
  -DVulkan_ROOT="C:\VulkanSDK\1.4.341.1"
```

Build:

```powershell
& "C:\Program Files\CMake\bin\cmake.exe" `
  --build .\build-vulkan-tests `
  --config Release `
  --target llama-bench test-turboquant-backend
```

Main outputs:

- `build-vulkan-tests/bin/Release/llama-bench.exe`
- `build-vulkan-tests/bin/Release/test-turboquant-backend.exe`

## Sanity Check

Run the backend test:

```powershell
.\build-vulkan-tests\bin\Release\test-turboquant-backend.exe
```

Expected outcome:

- test passes
- Vulkan backend lists the AMD GPU

## Running TurboQuant

Example `llama-bench` invocation:

```powershell
.\build-vulkan-tests\bin\Release\llama-bench.exe `
  -m "C:\Users\jimli\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf" `
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

Baseline comparison run:

```powershell
.\build-vulkan-tests\bin\Release\llama-bench.exe `
  -m "C:\Users\jimli\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf" `
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

## Harnessed Benchmark Run

Use the provided harness:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bench-turboquant-vulkan.ps1 `
  -Model "C:\Users\jimli\.lmstudio\models\unsloth\gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_S.gguf" `
  -Device "Vulkan0" `
  -Repetitions 3 `
  -PromptGenList "0,128;512,0;512,128;2048,128;4096,256;4096,1024;8192,256"
```

Results are written under `bench-results/`.

## Clean Upstream Comparison Build

Create a clean worktree at the current commit:

```powershell
git worktree add .\upstream-clean HEAD
```

Configure and build upstream Vulkan:

```powershell
$env:Vulkan_ROOT = "C:\VulkanSDK\1.4.341.1"

& "C:\Program Files\CMake\bin\cmake.exe" `
  -S .\upstream-clean `
  -B .\upstream-clean\build-vulkan `
  -DGGML_VULKAN=ON `
  -DVulkan_ROOT="C:\VulkanSDK\1.4.341.1"

& "C:\Program Files\CMake\bin\cmake.exe" `
  --build .\upstream-clean\build-vulkan `
  --config Release `
  --target llama-bench
```

This gives a direct standard `llama.cpp` comparison binary at:

- `upstream-clean/build-vulkan/bin/Release/llama-bench.exe`

## Troubleshooting

If configure fails with missing Vulkan headers or `glslc`:

- verify the Vulkan SDK is installed
- set `Vulkan_ROOT`
- confirm `C:\VulkanSDK\<version>\Bin\glslc.exe` exists

If CMake is installed but not on `PATH`, call it directly:

```powershell
& "C:\Program Files\CMake\bin\cmake.exe" --version
```

If the benchmark does not use the intended GPU:

- run `llama-bench --list-devices`
- pass `-dev Vulkan0`

If you want proof that the TurboQuant Vulkan path is active during debugging:

- set `GGML_VK_TURBOQUANT_PROFILE=1`
- run `llama-bench`
- inspect stderr for TurboQuant path counters
