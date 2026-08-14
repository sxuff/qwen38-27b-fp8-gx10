# Public results

This directory exposes the compact, machine-readable receipts needed to reproduce every table while keeping host-level raw streams out of the public repository.

## Published

- `summary.json`: regenerated aggregate metrics
- `posthoc-diagnostics.json`: explicitly labeled post-hoc calculations
- `telemetry-summary.json`: aggregate memory, swap, temperature, power, and utilization by arm
- `no-spec/` and `mtp/`:
  - every measured serial and concurrency row
  - request, prompt, response, and raw-SSE SHA-256 commitments
  - exact token counts and timings
  - concurrency-cell wall times
  - compact run summaries and source identities
- `private-archive-manifest.json`: SHA-256 and byte count for every omitted source-archive file

## Kept private

The private scientific archive retains the raw SSE bodies, one-second telemetry, full server logs, smoke-test HTTP envelopes, and run-control markers. Its 96 files and 4,306,709 bytes are committed by `private-archive-manifest.json`. The original archive verifier is preserved at `environment/source-archive-final-verification.json`.

The published JSONL rows replace private raw paths with `raw_sse_sha256` and `raw_sse_bytes`. Other request, response, token, and timing fields remain available for independent aggregation.

Verify the curated publication from the repository root:

```bash
python3 scripts/verify_publication.py
```

The verifier enforces the public allowlist, regenerates `summary.json` and `REPORT.md` from row-level receipts, checks matched requests across arms, validates the private-archive commitment, and scans for private paths and credentials.
