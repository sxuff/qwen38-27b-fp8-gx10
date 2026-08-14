# Amendments

## Amendment 1: lower common SGLang static-memory fraction

- Time: 2026-08-14, before readiness and before any primary trace
- Superseded compatibility attempt: `results/pilot/no-spec-a1`
- Old common value: `--mem-fraction-static 0.75`
- New common value: `--mem-fraction-static 0.70`
- Unchanged guards: `MemAvailable >= 15 GiB`; swap growth `<= 512 MiB`
- Unchanged checkpoint, image, backend, prompts, request contract, matrix, and analysis

### Evidence

Attempt `no-spec-a1` loaded all 66 FP8 weight shards successfully. During CUDA-graph capture it reached:

- `MemAvailable`: 20,734,709,760 bytes
- swap growth: 563,277,824 bytes
- swap-growth limit: 536,870,912 bytes
- overage: 26,406,912 bytes

The watchdog stopped only the experiment-owned container. The server never reached readiness and primary trace count remained zero.

### Rationale

The observed failure is a launch-time unified-memory allocation breach, not a model or request failure. Lowering the common static-memory fraction preserves the preregistered safety guard and reduces cache capacity without changing model weights. Both no-spec and MTP use the amended value.
