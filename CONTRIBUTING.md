# Contributing

Keep performance claims tied to reproducible artifacts.

For a new hardware or runtime result, include:

1. An immutable model revision and runtime image or commit.
2. Exact launch flags.
3. Exact prompt and completion token counts.
4. Warm-up policy and complete denominators.
5. Raw request rows and machine-readable aggregates.
6. Memory and thermal telemetry where available.
7. Output parity when comparing speculative decoding.
8. A clear separation between preregistered metrics and post-hoc diagnostics.

Do not replace failed rows with successful reruns without retaining and labeling the original failure.
