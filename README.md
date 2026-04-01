This project is a modified version of `llama.cpp` that adds a new way to store and use the model's KV cache during inference.

In simple terms:

- normal `llama.cpp` stores some working memory for the model in a standard format
- this fork can store that same working memory in a much more compressed format called TurboQuant
- on AMD hardware, the fork can use that compressed format efficiently enough to speed up real workloads

The important point is that this is not just a theory or a paper exercise. It has been built, tested, and benchmarked on an AMD machine using Vulkan.

## What Problem It Solves

When a large language model answers a prompt, it keeps an internal history of what it has already processed. That history is stored in what is usually called the KV cache.

Why that matters:

- longer prompts create more KV cache
- longer conversations create more KV cache
- generating more tokens means the model keeps reusing that KV cache

For large models, KV cache handling becomes a real performance cost.

TurboQuant aims to reduce that cost by storing KV data in a much smaller form while still letting the model use it effectively.

## What TurboQuant Does Not Fix

TurboQuant is not a complete fix for every long-context slowdown.

The most important limitation is prefill.

Prefill means:

- loading a long prompt
- computing all the model work needed to turn that prompt into KV cache state

TurboQuant helps with KV-cache efficiency, but it does not remove the fundamental compute cost of processing a very large prompt in the first place.

That means two things can both be true:

- TurboQuant can make long-context usage fit in memory more easily
- very large prompt ingestion can still be too slow to feel practical on local hardware

This is especially relevant for people who imagine "6x less KV memory" automatically means "6x bigger usable context." In practice, usable context is limited by both:

- memory
- speed

TurboQuant helps strongly with the first and can help meaningfully with decode-side speed, but it does not by itself eliminate prefill bottlenecks.

## Why This Matters To A User

If it works well, you can get:

- faster text generation
- better throughput on longer prompts
- better scaling as context length grows
- potentially lower memory pressure from KV cache handling

In plain terms: the model can feel snappier, especially when it is generating over a longer context instead of just answering one very short prompt.

## What The Benchmarks Mean

The benchmark names use a shorthand:

- `tg128`
  Means generation-only, 128 generated tokens.
- `pp512`
  Means prompt processing only, 512 prompt tokens.
- `pp4096+tg256`
  Means process a 4096-token prompt and then generate 256 tokens.

Why the mixed tests matter:

- prompt-only tests show prompt ingestion speed
- generation-only tests show sustained decode speed
- mixed prompt+generation tests are closer to normal real usage

## Current Result In Plain English

On the AMD Vulkan machine used for validation, TurboQuant is now faster than both:

- clean upstream `llama.cpp`
- this fork running the normal non-TurboQuant KV path

That is the important headline.

Here is the latest larger benchmark comparison:

<img width="600" height="308" alt="image" src="https://github.com/user-attachments/assets/672c8f89-072d-4cb7-8b5d-766c36afa4d5" />


And here is the TurboQuant uplift over clean upstream:

<img width="563" height="283" alt="image" src="https://github.com/user-attachments/assets/c1bba0f9-6b7d-4d78-8ea4-8a0c8f8f70a9" />


## What The Graphs Say

The main pattern is straightforward:

- small pure prompt gains exist, but they are modest
- the strongest gains appear in generation-heavy and mixed prompt+generation workloads
- the longer-context mixed tests show some of the best results

That is exactly where TurboQuant is expected to matter most, because that is where KV-cache handling becomes more important.

## About The Result

This was not judged only against the fork's own baseline mode.

A clean upstream `llama.cpp` build was also created and benchmarked on:

- the same machine
- the same GPU backend
- the same GGUF model
- the same benchmark shapes

That matters because it shows the gains are not just caused by unrelated drift in the fork.

## What Was Hard About It

One of the biggest problems during development was that the "fast path" had been coded, but the real benchmark was not actually using it.

In plain terms:

- the optimized path existed
- the benchmark still kept falling back to a more ordinary path
- that made early benchmark results look weak or mixed

The real breakthrough was fixing the backend integration so the compressed attention path was actually being exercised during real Vulkan flash attention.

Once that happened, the performance picture changed in a meaningful way.

## How You Can Use It

You can use this fork with GGUF models already downloaded by LM Studio from:

- `C:\Users\username\.lmstudio\models`

You do not need a special TurboQuant model file format.

You use the same GGUF, but run the fork with:

- normal mode for baseline
- `--kv-codec turboquant --kv-tq-runtime vulkan` for TurboQuant mode

That is covered in:

- `docs/turboquant-model-usage.md`
- `docs/turboquant-installation.md`

## How It Could Be Improved Further

The current result is good, but it is not the end of the road.

Likely improvement areas:

- broader backend coverage
  Vulkan is the validated proof path today. HIP/ROCm still needs the same level of real hardware benchmark proof.
- more models
  Right now the result is proven on a strong large-model case, but broader model-family coverage would make the case stronger.
- memory reporting
  Throughput is now good. The next useful proof would be showing memory behavior and not just speed.
- more direct compressed attention coverage
  There is still room to push more of the compressed path deeper into the backend instead of relying on transitional compatibility paths.
- cleanup for upstream quality
  The code works and benchmarks well, but there is still a difference between "working research-grade fork" and "polished upstream-ready patch set."

## Summary

For a non-specialist, the takeaway is:

- this fork is a real modified `llama.cpp`
- it runs ordinary GGUF models
- it has a TurboQuant mode for KV cache handling
- on the tested AMD Vulkan setup, that mode is meaningfully faster on important workloads
