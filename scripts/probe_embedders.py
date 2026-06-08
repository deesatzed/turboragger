#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from turboragger.artifacts import write_json_artifact
from turboragger.embedder_probe import probe_stronger_embedders


COMMAND = "PYTHONPATH=src python3 scripts/probe_embedders.py"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    payload = probe_stronger_embedders()
    path = write_json_artifact(root / "artifacts" / "embedder_availability.json", payload, command=COMMAND)
    print(json.dumps({"artifact": str(path.relative_to(root)), "status": payload["status"]}, indent=2))
    return 0 if payload["status"] == "available" else 1


if __name__ == "__main__":
    raise SystemExit(main())
