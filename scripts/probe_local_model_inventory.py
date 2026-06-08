#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from turboragger.artifacts import write_json_artifact
from turboragger.local_model_inventory import build_inventory


COMMAND = "PYTHONPATH=src python3 scripts/probe_local_model_inventory.py"
DEFAULT_ROOTS = [
    Path("/Volumes/WS4TB/WS4TBr"),
    Path("/Volumes/WS4TB/repo421sn"),
    Path.home() / ".cache" / "huggingface" / "hub",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory local model weight files relevant to SOTA retrieval work.")
    parser.add_argument("--root", action="append", type=Path, help="Root directory to scan. Can be repeated.")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    scan_roots = args.root or DEFAULT_ROOTS
    payload = build_inventory(
        scan_roots,
        timeout_seconds=args.timeout_seconds,
        benchmarked_model_dirs=benchmarked_model_dirs(root),
    )
    path = write_json_artifact(
        root / "artifacts" / "local_model_inventory.json",
        payload,
        command=COMMAND,
    )
    print(json.dumps({"artifact": str(path.relative_to(root)), "summary": payload["summary"]}, indent=2))
    return 0


def benchmarked_model_dirs(root: Path) -> list[Path]:
    dirs: set[Path] = set()
    for artifact in sorted((root / "artifacts" / "runs").glob("*.json")):
        try:
            payload = json.loads(artifact.read_text())
        except json.JSONDecodeError:
            continue
        config = payload.get("retriever_config", {})
        model_path = config.get("model_path")
        if model_path:
            dirs.add(Path(model_path))
        selected_model_path = config.get("selected_model_path")
        if selected_model_path:
            dirs.add(Path(selected_model_path))
        for path in config.get("model_paths", {}).values():
            dirs.add(Path(path))
    return sorted(dirs, key=str)


if __name__ == "__main__":
    raise SystemExit(main())
