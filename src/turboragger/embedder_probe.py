from __future__ import annotations

from pathlib import Path
from typing import Iterable


STRONG_EMBEDDER_CANDIDATES = [
    {
        "model_id": "BAAI/bge-m3",
        "priority": 1,
        "family": "BGE-M3",
        "reason": "Goal-prioritized dense/sparse/ColBERT-capable embedder.",
        "explicit_paths": [
            "/Volumes/WS4TB/WS4TBr/Partial_Apps_WS/dec24_apps/MedAiTools/bge-m3",
        ],
    },
    {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "priority": 2,
        "family": "Qwen3-Embedding",
        "reason": "Goal-prioritized newer general embedder candidate.",
        "explicit_paths": [],
    },
    {
        "model_id": "nomic-ai/nomic-embed-text-v1.5",
        "priority": 3,
        "family": "Nomic",
        "reason": "Earlier project handoff claimed a Nomic embedder may exist on disk.",
        "explicit_paths": [],
    },
    {
        "model_id": "intfloat/e5-large-v2",
        "priority": 4,
        "family": "E5",
        "reason": "Strong general dense retrieval baseline candidate.",
        "explicit_paths": [],
    },
    {
        "model_id": "Alibaba-NLP/gte-large-en-v1.5",
        "priority": 5,
        "family": "GTE",
        "reason": "Strong general dense retrieval baseline candidate.",
        "explicit_paths": [],
    },
    {
        "model_id": "BAAI/bge-large-zh-v1.5",
        "priority": 6,
        "family": "BGE",
        "reason": "Complete local BGE snapshot discovered under sibling ragflow cache; not the preferred English/M3 model, but a runnable stronger BGE-family dense candidate.",
        "explicit_paths": [
            "/Volumes/WS4TB/WS4TBr/aP2A/ragflow/huggingface.co/BAAI/bge-large-zh-v1.5",
        ],
    },
    {
        "model_id": "Xenova/bge-small-en-v1.5",
        "priority": 7,
        "family": "BGE",
        "reason": "Complete local English BGE ONNX fallback discovered under finESS cache.",
        "explicit_paths": [
            "/Volumes/WS4TB/WS4TBr/finESS/node_modules/@xenova/transformers/.cache/Xenova/bge-small-en-v1.5",
            "/Volumes/WS4TB/WS4TBr/finESS/data/.cache/transformers/Xenova/bge-small-en-v1.5",
        ],
    },
]


def default_cache_roots() -> list[Path]:
    return [Path.home() / ".cache" / "huggingface" / "hub"]


def probe_stronger_embedders(cache_roots: Iterable[Path] | None = None) -> dict:
    roots = [Path(root).expanduser() for root in (cache_roots or default_cache_roots())]
    candidates = []
    runnable = []

    for candidate in STRONG_EMBEDDER_CANDIDATES:
        model_id = candidate["model_id"]
        snapshot = find_model_snapshot(model_id, roots)
        checked_paths = [
            {
                "path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "complete": False,
                "status": "missing",
                "missing_files": ["snapshot directory"],
                "files_present": [],
            }
            for path in candidate_cache_paths(model_id, roots)
            if not path.is_dir()
        ]
        if snapshot is not None:
            path_probe = probe_candidate_path(snapshot)
            checked_paths.append(path_probe)
        for explicit_path in candidate.get("explicit_paths", []):
            path_probe = probe_candidate_path(Path(explicit_path).expanduser())
            checked_paths.append(path_probe)

        complete_paths = [probe["path"] for probe in checked_paths if probe["complete"]]
        status = "available" if complete_paths else "missing_or_incomplete"
        result = {
            **candidate,
            "status": status,
            "complete_paths": complete_paths,
            "checked_paths": checked_paths,
        }
        candidates.append(result)
        if complete_paths:
            runnable.append(result)

    return {
        "schema_version": 1,
        "status": "available" if runnable else "no_stronger_embedder_available_locally",
        "cache_roots": [str(root) for root in roots],
        "candidates": candidates,
        "runnable_candidates": runnable,
    }


def find_model_snapshot(model_id: str, cache_roots: Iterable[Path] | None = None) -> Path | None:
    for snapshots in candidate_cache_paths(model_id, cache_roots or default_cache_roots()):
        if not snapshots.is_dir():
            continue
        for snapshot in sorted(snapshots.iterdir()):
            if snapshot.is_dir() and probe_candidate_path(snapshot)["complete"]:
                return snapshot
    return None


def candidate_cache_paths(model_id: str, cache_roots: Iterable[Path] | None = None) -> list[Path]:
    repo_dir_name = "models--" + model_id.replace("/", "--")
    return [Path(root).expanduser() / repo_dir_name / "snapshots" for root in cache_roots or default_cache_roots()]


def probe_candidate_path(path: Path) -> dict:
    required_any = [
        ("config.json",),
        ("tokenizer.json", "vocab.txt", "sentencepiece.bpe.model", "spiece.model"),
        ("model.safetensors", "pytorch_model.bin"),
    ]
    missing = []
    for alternatives in required_any:
        if not any((path / filename).is_file() for filename in alternatives):
            missing.append(" or ".join(alternatives))
    if (path / "onnx" / "model_quantized.onnx").is_file() and (path / "tokenizer.json").is_file() and (path / "config.json").is_file():
        missing = []
    exists = path.exists()
    complete = exists and path.is_dir() and not missing
    return {
        "path": str(path),
        "exists": exists,
        "is_dir": path.is_dir(),
        "complete": complete,
        "status": "complete" if complete else "incomplete" if exists else "missing",
        "missing_files": missing,
        "files_present": sorted(child.name for child in path.iterdir()) if path.is_dir() else [],
    }
