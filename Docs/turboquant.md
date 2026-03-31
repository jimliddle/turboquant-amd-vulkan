# TurboQuant

This repository contains a TurboQuant-enabled `llama.cpp` fork with explicit KV-cache codec support and AMD-oriented Vulkan/HIP runtime work.

Use the documents below as the main entry points:

- `docs/turboquant-benchmarks.md`
  Current benchmark methodology and results, including clean upstream comparisons.
- `docs/turboquant-design.md`
  How the fork was built, architecture choices made, and why the implementation takes this shape.
- `docs/turboquant-installation.md`
  Practical installation and build instructions for this fork.
- `docs/turboquant-model-usage.md`
  How to use GGUF models from the LM Studio cache with baseline, TurboQuant, and upstream comparison runs.
- `docs/turboquant-explained.md`
  Plain-English explanation of what was built, what the benchmark results mean, and where the implementation could still improve.

## Current Status

- `--kv-codec turboquant` is implemented as a first-class KV-cache mode.
- TurboQuant settings are wired through API, CLI, context creation, and KV-cache construction.
- Runtime selection supports `auto`, `hip`, and `vulkan`.
- A shared row codec exists in `ggml` and is used by both the `llama.cpp` side and backend integrations.
- Vulkan currently provides the strongest validated proof path on AMD hardware.
- A clean upstream `llama.cpp` Vulkan build has been benchmarked at the same commit for direct comparison.

## Main Runtime Surface

- `--kv-codec {none|turboquant}`
- `--kv-tq-runtime {auto|hip|vulkan}`
- `--kv-tq-group-size N`
- `--kv-tq-residual-bits N`
- `--kv-tq-qjl` / `--no-kv-tq-qjl`
- `--kv-tq-fallback` / `--no-kv-tq-fallback`

## Practical Summary

The current state is beyond a prototype flag. On the validated AMD Vulkan setup used in this repository, the TurboQuant path is materially faster than both:

- clean upstream `llama.cpp`
- this fork's standard KV mode

The benchmark details and raw artifact locations are documented in `docs/turboquant-benchmarks.md`.
