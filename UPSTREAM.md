# Upstream validation note

The SGLang Qwen3.8-27B cookbook currently describes a DGX Spark configuration but labels the SM121 recipe as not yet validated on that platform.

This repository supplies one-host validation for the official FP8 checkpoint with the dedicated SGLang image.

## Evidence suitable for an upstream issue or documentation PR

- Hardware: NVIDIA GB10, SM121, 128 GB unified memory
- Checkpoint: `Qwen/Qwen3.8-27B-FP8`
- Pinned revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Pinned image: `lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1`
- SGLang build: `0.0.0.dev0+qwen38.27b.g561c8f3`
- Configured full-attention backend: FlashInfer
- Hybrid GDN decode, prefill, and verify kernels selected by the runtime: Triton
- Working prefill chunk: 8,192
- Working static-memory fraction on this host: 0.70
- Native in-checkpoint MTP: EAGLE, 3 steps, top-k 1, 4 draft tokens
- Successful matrix: 72/72 measured requests
- Maximum tested prompt: 32,768 tokens
- Maximum tested concurrency: 8

## Memory qualification

A pre-readiness attempt at `--mem-fraction-static 0.75` crossed the experiment's sealed 512 MiB swap-growth guard by 26,406,912 bytes during CUDA-graph capture. It produced no primary trace and was stopped by an ownership-scoped watchdog. Both measured arms used 0.70 and completed without a safety incident.

This does not establish that 0.75 or 0.95 will fail on every DGX Spark. It establishes that 0.70 was the validated safe operating point on this host under the published guard.

## Suggested cookbook wording

> DGX Spark / GB10 was validated with the official Qwen3.8-27B-FP8 checkpoint using the dedicated `qwen38-27b` image, FlashInfer attention, 8,192-token prefill chunks, and `--mem-fraction-static 0.70`. Because GB10 memory is shared with the host, treat the memory fraction as an operating-point parameter and preserve host-memory and swap headroom during CUDA-graph capture. Native in-checkpoint MTP with EAGLE also completed successfully.
