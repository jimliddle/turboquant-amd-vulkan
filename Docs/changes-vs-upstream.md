# Changes Vs Upstream

This is a short reviewer-oriented map of what changed relative to upstream `llama.cpp` at commit `389c7d495`.

The goal of this file is not to justify the project. It is to make the implementation surface easy to inspect.

## Scope

Main implementation footprint:

- around 25 tracked file modifications in the upstream tree
- new TurboQuant-specific source files under `src/` and `ggml/src/`
- new tests
- new benchmark/docs/scripts outside the core runtime

## Where To Look First

If you only want the critical code paths, start here:

1. `src/llama-kv-cache.cpp`
2. `src/llama-kv-cache.h`
3. `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
4. `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp`
5. `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp`
6. `src/llama-turboquant-runtime.cpp`
7. `ggml/src/ggml-turboquant-codec.cpp`

## Change Areas

### 1. Public/API and CLI surface

Files:

- `include/llama.h`
- `common/arg.cpp`
- `common/common.cpp`
- `common/common.h`
- `src/llama.cpp`

Purpose:

- adds `--kv-codec turboquant`
- adds TurboQuant runtime/config flags
- threads TurboQuant options through public/runtime-facing parameter structures

### 2. Core KV-cache integration

Files:

- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`
- `src/llama-context.cpp`
- `src/llama-model.cpp`
- `src/llama-memory.h`
- `src/llama-memory-hybrid.cpp`
- `src/llama-memory-hybrid.h`
- `src/llama-memory-hybrid-iswa.cpp`
- `src/llama-memory-hybrid-iswa.h`
- `src/llama-kv-cache-iswa.cpp`
- `src/llama-kv-cache-iswa.h`

Purpose:

- adds TurboQuant as a KV codec mode
- maintains compressed KV shadow state
- routes live KV reads/writes through codec-aware boundaries
- seeds and syncs backend-visible compressed state
- integrates TurboQuant behavior into the existing memory/context flow

### 3. TurboQuant-specific runtime and codec layer

New files:

- `src/llama-turboquant.h`
- `src/llama-turboquant.cpp`
- `src/llama-turboquant-codec.h`
- `src/llama-turboquant-codec.cpp`
- `src/llama-turboquant-runtime.h`
- `src/llama-turboquant-runtime.cpp`
- `src/llama-turboquant-runtime-impl.h`
- `src/llama-turboquant-runtime-vulkan.cpp`
- `src/llama-turboquant-runtime-hip.cpp`

Purpose:

- defines the codec/runtime boundary
- implements the CPU/reference row codec surface
- dispatches live materialization/sync to backend-specific paths
- isolates HIP/Vulkan backend ABI from the higher-level KV cache logic

### 4. Shared `ggml` TurboQuant layer

New files:

- `ggml/src/ggml-turboquant-codec.h`
- `ggml/src/ggml-turboquant-codec.cpp`

Files changed:

- `ggml/src/CMakeLists.txt`

Purpose:

- moves the row codec into shared `ggml` code
- lets backend code use the same packed-row contract as the `llama.cpp` side

### 5. Vulkan backend integration

Files:

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp`
- `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp`
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/turboquant_materialize.comp`

Purpose:

- adds backend-owned resident compressed KV handling
- adds Vulkan materialization shader support
- adds direct compressed attention support on the validated Vulkan path
- integrates packed-K / packed-V handling into flash attention selection and execution

### 6. HIP / ROCm backend hook surface

Files:

- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `src/llama-turboquant-runtime-hip.cpp`

Purpose:

- adds the backend hook/export surface for HIP-side TurboQuant integration
- compile-validates the HIP path in the current codebase structure

Note:

- the strongest runtime validation and benchmarking in this repo is currently the Vulkan path on AMD hardware

### 7. Benchmarks and tests

Files:

- `tools/llama-bench/llama-bench.cpp`
- `tests/CMakeLists.txt`
- `tests/test-turboquant-codec.cpp`
- `tests/test-turboquant-backend.cpp`
- `scripts/bench-turboquant-vulkan.ps1`

Purpose:

- extends `llama-bench` so TurboQuant mode can be benchmarked cleanly
- adds codec and Vulkan backend regression coverage
- adds a repeatable paired benchmark harness

## What This Is

- an experimental TurboQuant integration on top of `llama.cpp`
- a benchmarked AMD Vulkan proof path
- a codebase that can be inspected subsystem by subsystem

## What This Is Not

- not a polished upstream patch series
- not a claim that every implementation detail is final
- not a minimal diff
- not a claim that all dense KV-compatible compatibility paths have been eliminated

## Suggested Review Order

For someone reviewing the implementation technically, this is the most useful order:

1. `include/llama.h`
2. `common/arg.cpp`
3. `src/llama-kv-cache.h`
4. `src/llama-kv-cache.cpp`
5. `src/llama-turboquant-runtime.cpp`
6. `ggml/src/ggml-turboquant-codec.cpp`
7. `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
8. `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp`
9. `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp`
10. `tools/llama-bench/llama-bench.cpp`
