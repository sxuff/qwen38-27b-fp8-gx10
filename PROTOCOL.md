# Qwen3.8-27B-FP8 throughput benchmark protocol

## Question

Measure text-serving throughput, latency, context scaling, concurrency scaling, and native-MTP behavior for the exact official Qwen3.8-27B-FP8 checkpoint on one NVIDIA GB10.

## Frozen target

- Repository: `Qwen/Qwen3.8-27B-FP8`
- Revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Snapshot verification: 81/81 files, 30,890,049,597 bytes
- Runtime: dedicated launch-day `lmsysorg/sglang:qwen38-27b` image, immutable digest recorded before primary collection
- Hardware: one NVIDIA GB10 / SM121, 128 GB unified memory

## Scope

Included:

- Official FP8 checkpoint
- Text-only engine throughput
- Real no-thinking chat smoke
- Serial context sweep
- Continuous-batching concurrency sweep
- Native in-checkpoint MTP compatibility and matched A/B when supported
- Host memory, swap, GPU temperature, power, and utilization telemetry

Excluded:

- Model-quality benchmarks
- BF16 or community quant comparisons
- Video throughput
- Claims about other GPUs or runtimes
- Publication

## Common server treatment

Use the SGLang Qwen3.8 DGX Spark recipe with:

- `--trust-remote-code`
- `--attention-backend flashinfer`
- `--chunked-prefill-size 8192`
- `--reasoning-parser qwen3`
- `--tool-call-parser qwen3_coder`
- one server and one checkpoint copy

The official recipe uses `--mem-fraction-static 0.95`. Compatibility attempt `no-spec-a1` froze `0.75`, but crossed the sealed 512 MiB swap-growth guard during CUDA-graph capture before readiness and before any primary trace. Amendment 1 lowers the common value to `0.70` while preserving the 15 GiB host `MemAvailable` floor. This changes cache capacity, not the checkpoint. The deviation is disclosed and common across comparable arms.

No-spec uses no speculative flags. MTP, if compatible in the exact image, adds only:

```text
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

If FlashInfer MTP is incompatible on SM121, preserve the failure. A Triton retry may be run only as a separately labeled compatibility amendment and requires a matched Triton no-spec control before any speed comparison.

## Request contract

### Readiness

One OpenAI chat-completions request absent from the benchmark corpus:

- prompt asks for `12 × 17`
- temperature 0
- top-p 1
- seed 42
- thinking disabled
- preserved thinking disabled
- response must contain `204`

### Engine measurements

Use OpenAI streaming completions with:

- temperature 0
- top-p 1
- seed 42
- `ignore_eos: true`
- exactly 256 requested completion tokens
- unique first-prefix benchmark keys to prevent cross-request prefix-cache reuse

Retain raw SSE, request and response hashes, usage, finish reason, actual prompt/completion tokens, TTFT, end-to-end latency, reconstructed decode interval, and decode tokens/s.

## Matrix

### Serial context sweep

Target input lengths:

- 512
- 4,096
- 16,384
- 32,768

For each target:

- one unmeasured warm-up
- three measured repetitions
- concurrency 1
- 256 requested output tokens

Report actual tokenizer and server usage counts. Targets are labels, not substitutes for measured token counts.

### Concurrency sweep

- target input: 1,024 tokens
- output: 256 tokens
- concurrency: 1, 4, 8
- eight requests per cell

Report successful/planned requests, aggregate completion tok/s, per-request TTFT, per-request end-to-end latency, and per-request decode tok/s.

## Safety and eligibility

Before each server load, record memory and swap baselines. Sample once per second.

Frozen guards:

- `MemAvailable >= 15 GiB`
- swap growth `<= 512 MiB` from the arm baseline

A breach stops only the experiment-owned `qwen38-bench-server` container and makes the arm ineligible.

A row is successful only when:

- no transport/runtime error occurred
- the server returned usage
- completion count is exactly 256
- TTFT and end-to-end timing are finite
- finish behavior is retained

Failed and missing rows remain in denominators.

## Comparison gates

- Do not claim MTP speedup unless no-spec and MTP use the same image, checkpoint, backend, server flags, prompts, and request bytes except the speculative configuration.
- Report output-hash parity separately from speed.
- Positive MTP identification requires startup evidence plus proposal/acceptance telemetry when exposed.
- A faster arm with changed deterministic output is not lossless.
- Launch-day source claims and server startup alone are not measured results.

## Run order and limitation

Run no-spec first, then MTP, one fresh model load each. The order is not counterbalanced, so small arm differences remain sensitive to thermal and run-order effects. Large claims require a later ABBA confirmation.
