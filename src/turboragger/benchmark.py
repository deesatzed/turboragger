from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


def build_candidate_payload(
    *,
    run_id: str,
    candidate_name: str,
    retrieval_mode: str,
    dataset: Mapping[str, Any],
    retriever_config: Mapping[str, Any],
    dependency: Mapping[str, Any],
    report: Mapping[str, Any],
    scores: Mapping[str, Any],
    command: str,
    hyperparameters: Mapping[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    failures = dict(report.get("failures", {}))
    score_failure_count = int(scores.get("failure_count", 0))
    failure_count = score_failure_count + len(failures)
    metrics = dict(scores.get("metrics", {}))

    return {
        "schema_version": 1,
        "run_id": run_id,
        "candidate": candidate_name,
        "retrieval_mode": retrieval_mode,
        "command": command,
        "timestamp_utc": utc_timestamp(),
        "dataset": dict(dataset),
        "dependency": dict(dependency),
        "retriever_config": dict(retriever_config),
        "hyperparameters": dict(hyperparameters),
        "runtime_seconds": runtime_seconds,
        "query_count": int(scores.get("queries_tested", 0)),
        "failure_count": failure_count,
        "metrics": metrics,
        "per_query_scores": dict(scores.get("per_query_scores", {})),
        "runs": dict(report.get("runs", {})),
        "branch_outputs": dict(report.get("branch_outputs", {})),
        "failures": failures,
    }


def write_candidate_run(root: Path, payload: Mapping[str, Any]) -> Path:
    run_id = str(payload["run_id"])
    path = root / "artifacts" / "runs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {**dict(payload), "artifact_path": str(path.relative_to(root))}
    path.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n")
    return path


def write_leaderboard(root: Path) -> Path:
    runs_dir = root / "artifacts" / "runs"
    entries = []
    for path in sorted(runs_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        entries.append(
            {
                "run_id": payload["run_id"],
                "candidate": payload["candidate"],
                "retrieval_mode": payload["retrieval_mode"],
                "metrics": payload.get("metrics", {}),
                "query_count": payload.get("query_count", 0),
                "failure_count": payload.get("failure_count", 0),
                "artifact_path": payload.get("artifact_path", str(path.relative_to(root))),
                "timestamp_utc": payload.get("timestamp_utc"),
            }
        )

    entries.sort(
        key=lambda entry: (
            -float(entry.get("metrics", {}).get("nDCG@10", 0.0)),
            int(entry.get("failure_count", 0)),
            str(entry.get("run_id", "")),
        )
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": utc_timestamp(),
        "sort_key": "nDCG@10 desc",
        "best_run": entries[0] if entries else None,
        "runs": entries,
    }

    path = root / "artifacts" / "leaderboard.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
