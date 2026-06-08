from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_json_artifact(path: Path, payload: dict[str, Any], command: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        **payload,
        "command": command,
        "timestamp_utc": utc_timestamp(),
    }
    path.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n")
    return path


def write_markdown_artifact(path: Path, title: str, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [f"# {title}", "", f"Generated: {utc_timestamp()}", "", *lines, ""]
    path.write_text("\n".join(content))
    return path
