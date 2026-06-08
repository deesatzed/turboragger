from __future__ import annotations

from pathlib import Path
from typing import Iterable

from turboragger.dense import optional_kernels_disabled


RERANKER_CANDIDATES = [
    {
        "model_id": "openbmb/MiniCPM-Reranker-Light",
        "priority": 1,
        "family": "MiniCPM",
        "reason": "Configured in sibling newragcity reranker service.",
        "explicit_paths": [],
    },
    {
        "model_id": "BAAI/bge-reranker-base",
        "priority": 2,
        "family": "BGE reranker",
        "reason": "Common local/open reranker candidate.",
        "explicit_paths": [],
    },
    {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "priority": 3,
        "family": "BGE reranker",
        "reason": "Goal-compatible stronger reranker candidate.",
        "explicit_paths": [
            "/Volumes/WS4TB/WS4TBr/CPfrac/cam-rag-platform/benchmarks/mteb/results/bge-reranker-v2-m3",
        ],
    },
    {
        "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "priority": 4,
        "family": "CrossEncoder",
        "reason": "Small sentence-transformers reranker fallback candidate.",
        "explicit_paths": [],
    },
]


def default_cache_roots() -> list[Path]:
    return [Path.home() / ".cache" / "huggingface" / "hub"]


def probe_rerankers(cache_roots: Iterable[Path] | None = None) -> dict:
    roots = [Path(root).expanduser() for root in (cache_roots or default_cache_roots())]
    import_status = probe_reranker_imports()
    candidates = []
    runnable = []

    for candidate in RERANKER_CANDIDATES:
        model_id = candidate["model_id"]
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
            for path in reranker_cache_paths(model_id, roots)
            if not path.is_dir()
        ]
        snapshot = find_reranker_snapshot(model_id, roots)
        if snapshot is not None:
            checked_paths.append(probe_reranker_path(snapshot))
        for explicit_path in candidate.get("explicit_paths", []):
            checked_paths.append(probe_reranker_path(Path(explicit_path).expanduser()))

        complete_paths = [probe["path"] for probe in checked_paths if probe["complete"]]
        result = {
            **candidate,
            "status": "available" if complete_paths else "missing_or_incomplete",
            "complete_paths": complete_paths,
            "checked_paths": checked_paths,
        }
        candidates.append(result)
        if complete_paths:
            runnable.append(result)

    return {
        "schema_version": 1,
        "status": "available" if runnable and import_status["guarded_imports_usable"] else "no_reranker_available",
        "cache_roots": [str(root) for root in roots],
        "import_status": import_status,
        "candidates": candidates,
        "runnable_candidates": runnable,
    }


def probe_reranker_imports() -> dict:
    plain = {}
    guarded = {}
    for module in ["sentence_transformers", "FlagEmbedding"]:
        plain[module] = _try_import(module, guarded=False)
    for module in ["sentence_transformers", "FlagEmbedding"]:
        guarded[module] = _try_import(module, guarded=True)
    return {
        "plain": plain,
        "guarded": guarded,
        "guarded_imports_usable": all(item["usable"] for item in guarded.values()),
    }


def find_reranker_snapshot(model_id: str, cache_roots: Iterable[Path] | None = None) -> Path | None:
    for snapshots in reranker_cache_paths(model_id, cache_roots or default_cache_roots()):
        if not snapshots.is_dir():
            continue
        for snapshot in sorted(snapshots.iterdir()):
            if snapshot.is_dir() and probe_reranker_path(snapshot)["complete"]:
                return snapshot
    return None


def reranker_cache_paths(model_id: str, cache_roots: Iterable[Path] | None = None) -> list[Path]:
    repo_dir_name = "models--" + model_id.replace("/", "--")
    return [Path(root).expanduser() / repo_dir_name / "snapshots" for root in cache_roots or default_cache_roots()]


def probe_reranker_path(path: Path) -> dict:
    required_any = [
        ("config.json",),
        ("tokenizer.json", "vocab.txt", "sentencepiece.bpe.model", "spiece.model"),
        ("model.safetensors", "pytorch_model.bin"),
    ]
    missing = []
    for alternatives in required_any:
        if not any((path / filename).is_file() for filename in alternatives):
            missing.append(" or ".join(alternatives))
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


def _try_import(module: str, guarded: bool) -> dict:
    try:
        if guarded:
            with optional_kernels_disabled():
                imported = __import__(module)
        else:
            imported = __import__(module)
        return {"usable": True, "version": str(getattr(imported, "__version__", ""))}
    except Exception as exc:
        return {"usable": False, "error_type": type(exc).__name__, "error": str(exc)}
