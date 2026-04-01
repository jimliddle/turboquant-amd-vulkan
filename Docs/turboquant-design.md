# TurboQuant Build And Design Notes

This document explains how this fork was built, what architectural choices were made, and why the implementation looks the way it does.

## Objective

The goal was not to bolt on a synthetic codec flag. The goal was to create a modified `llama.cpp` that can:

- keep KV cache state in a TurboQuant-style compressed representation
- run on AMD hardware
- support Vulkan now and ROCm/HIP as the parallel backend path
- prove the implementation with real `llama-bench` comparisons against standard `llama.cpp`

## Base Choice

The project uses upstream `llama.cpp` as the base runtime.

Reasons:

- `llama.cpp` is the actual inference runtime being extended
- license is clear and permissive
- Vulkan and HIP/`ggml` backend structure already exists
- benchmark tooling already exists in-tree via `llama-bench`

This was intentionally not built as a standalone side repository or a patch against LM Studio itself.

## High-Level Architecture

TurboQuant was implemented as a KV codec boundary, not as a new plain `ggml_type`.

That decision matters because TurboQuant needs:

- side metadata
- packed-row layout rules
- runtime/backend-specific decode paths
- attention-path integration

Trying to force all of that into a simple tensor element type would have made the backend contract brittle and hard to extend.

The major layers are:

1. public/API and CLI controls
2. KV cache codec configuration and shadow state
3. shared row codec
4. backend runtime dispatch
5. backend-native Vulkan/HIP integration
6. benchmark and validation tooling

## Implementation Choices

### 1. Explicit KV codec surface

The fork adds explicit KV codec control rather than hiding behavior behind cache types.

Key options:

- `--kv-codec turboquant`
- `--kv-tq-runtime auto|hip|vulkan`
- `--kv-tq-group-size`
- `--kv-tq-residual-bits`
- `--kv-tq-qjl`
- `--kv-tq-fallback`

Why:

- keeps the standard path intact
- makes A/B benchmarking clean
- keeps TurboQuant behavior orthogonal to `type_k` and `type_v`

### 2. Shared row codec

A shared TurboQuant row codec was added under `ggml/src/ggml-turboquant-codec.*`, with `llama.cpp` wrappers layered on top.

Why:

- backend code needs the same pack/unpack contract as the `llama.cpp` side
- avoids duplicating packing logic across runtime layers
- makes serialization, shadowing, and backend decode agree on one physical row format

### 3. Compressed host shadow first

Before direct compressed attention worked, the implementation kept a host-side compressed KV shadow and used reference materialization paths.

Why:

- gave a correctness path early
- made state save/load and live-path validation possible before GPU-native attention work was complete
- exposed integration boundaries cleanly

This was not the final performance architecture. It was a deliberate stepping stone.

### 4. Backend-owned resident compressed state

The Vulkan backend now keeps compressed KV state resident in backend-owned buffers.

Why:

- repeated restaging of packed rows was a real overhead
- compressed attention cannot be credible if every access reconstructs dense KV through generic host paths
- resident backend state is the right substrate for direct attention integration

### 5. Direct compressed flash attention

The important turning point was moving beyond materialization and teaching Vulkan flash attention to use compressed TurboQuant K/V directly on the coopmat path.

Why:

- earlier code existed but was not actually exercised in the real benchmark path
- once the resident shadow was seeded correctly and keyed to the same tensor/view identity as flash attention, the benchmark behavior changed materially
- that is what shifted the benchmark story from "slightly slower or flat" to "clearly faster"

### 6. Upstream comparison discipline

The project does not treat "TurboQuant beats this fork's baseline mode" as enough.

A clean upstream worktree was built at the same commit and benchmarked on the same machine.

Why:

- isolates TurboQuant gains from unrelated fork drift
- makes the benchmark result defensible
- shows the fork baseline and upstream are in the same band, which supports the validity of the comparison

## Important Debugging Lesson

The main performance breakthrough did not come from adding more shader variants. It came from discovering that the direct compressed attention path was not being exercised at all in real `llama-bench` runs.

The critical issues were:

- resident cache identity mismatch
- view-offset accounting problems
- backend shadow not being seeded for the attention path's tensor identity

Until those were fixed, benchmarks were measuring transitional plumbing, not real TurboQuant attention.

## What Was Deliberately Deferred

Some work was intentionally not treated as done:

- ROCm/HIP hardware benchmark validation
- memory-footprint reporting as part of the benchmark package
- an upstream-quality patch split
- paper-exact reproduction claims for every TurboQuant detail

The target here was proof on AMD hardware first, then cleanup and packaging.

## Status

The repository now contains:

- a working Vulkan TurboQuant path for AMD GPU benchmarking
- a benchmark harness
- a clean upstream comparison build path
- enough documentation and artifacts to explain both the implementation and the observed performance

That is the right place to be before deciding whether to harden this into a more formal release or upstreaming effort.
