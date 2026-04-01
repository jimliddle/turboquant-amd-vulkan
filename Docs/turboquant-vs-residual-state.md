# TurboQuant Vs Residual-State Inference

This document explains how the work in this repository relates to the idea that the residual stream, not the KV cache, may be the true information-bearing state in transformer inference.

## Short Version

The two ideas are not mutually exclusive.

- TurboQuant improves the current mainstream transformer inference regime, where retained K/V history is part of the implementation.
- Residual-state inference questions whether retained K/V history should be the main state representation at all.

So the relationship is:

- TurboQuant is a practical optimization of the current paradigm.
- Residual-state inference may point toward a future paradigm.

## What Was Built Here

This repository improves inference inside the standard KV-based transformer execution model.

That means:

- previous keys and values are retained
- later tokens attend over that retained state
- KV storage and access become a major memory and throughput issue

TurboQuant helps by making that retained KV state smaller and cheaper to use.

On the validated AMD Vulkan setup in this repository, that produced real wins over:

- the fork's baseline mode
- clean upstream `llama.cpp`

So the work here is real and useful now.

## What The Residual-State Argument Changes

The residual-state view says something deeper:

- the residual stream at a layer may already be the sufficient state
- keys and values are deterministic projections from that state
- therefore the KV cache may be a computational shortcut rather than the true information store

If that holds broadly and can be turned into an efficient runtime, then the implications are significant:

- the KV cache is not the final state abstraction
- memory scaling could potentially change much more fundamentally
- the field may be optimizing a workaround rather than the final design

## Why This Does Not Invalidate TurboQuant

Even if residual-state inference is correct in principle, that still leaves an engineering question:

- can it beat highly optimized KV-based systems on real hardware and real workloads?

Those are different levels of truth:

- representational truth
  what information is actually sufficient
- computational truth
  what operations are cheapest to perform
- systems truth
  what runs fastest and most reliably on actual hardware and software stacks

TurboQuant wins in the current systems truth category.

That matters because deployed software runs in the present architecture, not the hypothetical best architecture.

## A Useful Analogy

Think of it this way:

- TurboQuant is like making the current engine much more efficient.
- Residual-state inference is like asking whether we should be using a different engine layout entirely.

Both are valuable, but they operate at different levels.

## Strategic Interpretation

The practical interpretation is:

### Near term

TurboQuant-style work is worth doing because:

- it applies to current model/runtime assumptions
- it can produce measurable speed and memory improvements now
- it is benchmarkable today

### Long term

Residual-state work is worth exploring because:

- it may reduce dependence on ever-growing KV state
- it may offer a more fundamental path out of long-context scaling pain
- it could eventually obsolete parts of the KV optimization stack

## The Most Important Open Question

The real open question is not:

- "Is the KV cache redundant in principle?"

The more important question is:

- "Can residual-state-based inference outperform compressed-KV inference on real hardware, real models, and real workloads?"

Until that answer is clearly yes, TurboQuant remains highly relevant.

## How This Repository Should Be Viewed

This repository should be viewed as:

- a strong implementation of "best-in-class KV-based inference optimization" on the tested AMD Vulkan path
- not necessarily the final word on transformer state management

That is still a meaningful achievement.

## Recommended Next Step

If this repository evolves further, a strong next research direction would be:

- keep TurboQuant as the practical KV path
- begin a parallel prototype track for residual-state-centric inference
- compare the two directly on:
  - memory footprint
  - prompt-ingest scaling
  - generation throughput
  - mixed prompt+generation workloads

That would answer the real architectural question instead of arguing it abstractly.
