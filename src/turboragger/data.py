from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def candidate_nfcorpus_paths(root: Path) -> list[Path]:
    candidates = []
    env_path = os.environ.get("TURBORAGGER_NFCORPUS_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            root / "datasets" / "nfcorpus",
            root / "data" / "nfcorpus",
            root / "newragcity" / "datasets" / "nfcorpus",
            root / "newragcity" / "data" / "nfcorpus",
            root / "newragcity" / "ersatz_rag" / "regulus" / "backend" / "benchmarks" / "datasets" / "nfcorpus",
            Path("/Volumes/WS4TB/newragcity/UltraRAG-main/datasets/nfcorpus"),
            Path("/Volumes/WS4TB/WS4TBr/newragcity/UltraRAG-main/datasets/nfcorpus"),
        ]
    )
    return _dedupe(candidates)


def find_nfcorpus(root: Path) -> dict[str, Any]:
    checked = []
    for path in candidate_nfcorpus_paths(root):
        checked.append(str(path))
        if _is_nfcorpus(path):
            return {
                "status": "found",
                "path": str(path),
                "checked": checked,
                "fingerprint": fingerprint_dataset(path),
            }
    return {"status": "missing", "path": None, "checked": checked, "fingerprint": None}


def load_nfcorpus(
    dataset_path: Path,
    qrels_split: str = "test",
) -> tuple[dict[str, dict], dict[str, str], dict[str, dict[str, int]]]:
    if not _is_nfcorpus(dataset_path):
        raise FileNotFoundError(f"nfcorpus files not found under {dataset_path}")
    corpus = _read_jsonl(dataset_path / "corpus.jsonl")
    queries_raw = _read_jsonl(dataset_path / "queries.jsonl")
    queries = {query["_id"]: query["text"] for query in queries_raw.values()}
    qrels = load_nfcorpus_qrels(dataset_path, split=qrels_split)
    return corpus, queries, qrels


def load_nfcorpus_qrels(dataset_path: Path, split: str = "test") -> dict[str, dict[str, int]]:
    qrels_path = dataset_path / "qrels" / f"{split}.tsv"
    if not qrels_path.is_file():
        raise FileNotFoundError(f"nfcorpus qrels split not found: {qrels_path}")
    return _read_qrels(qrels_path)


def fingerprint_dataset(dataset_path: Path) -> dict[str, Any]:
    files = [dataset_path / "corpus.jsonl", dataset_path / "queries.jsonl", dataset_path / "qrels" / "test.tsv"]
    digest = hashlib.sha256()
    file_info = []
    for file_path in files:
        data = file_path.read_bytes()
        digest.update(file_path.name.encode("utf-8"))
        digest.update(data)
        file_info.append({"path": str(file_path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return {"dataset_sha256": digest.hexdigest(), "files": file_info}


def _is_nfcorpus(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "corpus.jsonl").is_file()
        and (path / "queries.jsonl").is_file()
        and (path / "qrels" / "test.tsv").is_file()
    )


def _read_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["_id"])] = record
    return records


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open() as handle:
        header = next(handle, "")
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            query_id, corpus_id, score = parts[:3]
            qrels.setdefault(query_id, {})[corpus_id] = int(score)
    return qrels


def _dedupe(paths: list[Path]) -> list[Path]:
    seen = set()
    result = []
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result
