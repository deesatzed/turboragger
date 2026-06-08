#!/usr/bin/env python3
from pathlib import Path

from turboragger.artifacts import write_json_artifact
from turboragger.probe import probe_environment


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = write_json_artifact(
        root / "artifacts" / "environment_probe.json",
        probe_environment(root),
        command="PYTHONPATH=src python3 scripts/probe_environment.py",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
