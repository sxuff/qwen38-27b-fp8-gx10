#!/usr/bin/env python3
"""Bounded fail-closed test for watchdog tool failures and owned cleanup."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        fake = temp / "docker"
        fake.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = inspect ]; then\n"
            "  printf '%s\\n' "
            "'[{\"Config\":{\"Labels\":{\"io.sxuff.qwen38.recipe\":\"true\","
            "\"io.sxuff.qwen38.run\":\"run-1\"}},\"State\":{\"Running\":true}}]'\n"
            "  exit 0\n"
            "fi\n"
            "if [ \"$1\" = rm ]; then exit 0; fi\n"
            "exit 1\n"
        )
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        output = temp / "telemetry"
        env = dict(os.environ)
        env["PATH"] = str(temp)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "telemetry.py"),
                "--output",
                str(output),
                "--container-id",
                "container-1",
                "--run-id",
                "run-1",
                "--interval",
                "0.01",
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 2
        incident = json.loads((output / "safety-incident.json").read_text())
        assert incident["violation"] == "watchdog_exception"
        assert "nvidia-smi" in incident["detail"]
        assert incident["action"] == "removed owned container"
    print("SAFETY_HELPERS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
