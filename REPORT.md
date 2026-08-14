# Qwen3.8-27B-FP8 on one GX10

## Scope

Official FP8 checkpoint, pinned SGLang image, text-only serving. Three measured serial repetitions per context target and eight requests per concurrency cell.

## no-spec

### Serial context sweep

| Target input | Complete | Actual prompt tokens | Median TTFT | Median decode | Median E2E |
|---:|---:|---:|---:|---:|---:|
| 512 | 3/3 | 512 | 0.38 s | 7.86 tok/s | 32.82 s |
| 4,096 | 3/3 | 4096 | 2.85 s | 7.80 tok/s | 35.56 s |
| 16,384 | 3/3 | 16384 | 19.56 s | 7.50 tok/s | 53.54 s |
| 32,768 | 3/3 | 32768 | 41.59 s | 7.18 tok/s | 77.12 s |

### Concurrency sweep

| Concurrency | Complete | Aggregate output | Median TTFT | p95 TTFT | Median per-request decode |
|---:|---:|---:|---:|---:|---:|
| 1 | 8/8 | 7.72 tok/s | 0.64 s | 0.65 s | 7.85 tok/s |
| 4 | 8/8 | 28.94 tok/s | 2.58 s | 2.97 s | 7.77 tok/s |
| 8 | 8/8 | 47.05 tok/s | 8.79 s | 8.79 s | 7.34 tok/s |

### Telemetry

- Samples: 1579
- Minimum MemAvailable: 22.82 GiB
- Maximum swap growth: 77.97 MiB
- Maximum temperature: 90.00 °C
- Mean / max sampled GPU power: 34.64 / 83.50 W
- Safety incident: False

## mtp

### Serial context sweep

| Target input | Complete | Actual prompt tokens | Median TTFT | Median decode | Median E2E |
|---:|---:|---:|---:|---:|---:|
| 512 | 3/3 | 512 | 0.44 s | 20.11 tok/s | 13.12 s |
| 4,096 | 3/3 | 4096 | 2.88 s | 21.78 tok/s | 14.59 s |
| 16,384 | 3/3 | 16384 | 19.09 s | 21.93 tok/s | 30.71 s |
| 32,768 | 3/3 | 32768 | 41.20 s | 20.87 tok/s | 53.58 s |

### Concurrency sweep

| Concurrency | Complete | Aggregate output | Median TTFT | p95 TTFT | Median per-request decode |
|---:|---:|---:|---:|---:|---:|
| 1 | 8/8 | 19.97 tok/s | 0.71 s | 0.72 s | 21.20 tok/s |
| 4 | 8/8 | 58.22 tok/s | 1.82 s | 2.31 s | 18.27 tok/s |
| 8 | 8/8 | 83.89 tok/s | 8.79 s | 8.79 s | 16.65 tok/s |

### Telemetry

- Samples: 1113
- Minimum MemAvailable: 26.27 GiB
- Maximum swap growth: 487.98 MiB
- Maximum temperature: 90.00 °C
- Mean / max sampled GPU power: 35.32 / 88.34 W
- Safety incident: False

## MTP matched comparison

- Paired eligible serial rows: 12
- Byte-identical outputs: 6/12
- Median decode speed ratio, MTP/no-spec: 2.85x
- Median end-to-end speed ratio, MTP/no-spec: 1.92x

## Evidence boundary

These measurements describe the exact pinned checkpoint, runtime image, launch flags, and GX10 host. They are not model-quality results and do not generalize to BF16, GGUF, other GPUs, or other serving engines.
