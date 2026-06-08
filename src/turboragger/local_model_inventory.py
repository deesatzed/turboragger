from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable


MODEL_FILENAMES = {
    "model.onnx",
    "model_quantized.onnx",
    "pytorch_model.bin",
    "model.safetensors",
}

GOAL_PRIORITY_EMBEDDING_SIGNALS = {
    "bge-m3",
    "qwen3-embedding",
    "nomic-embed",
    "e5-large",
    "gte-large",
    "nv-embed",
    "arctic-embed",
    "bce-embedding",
    "jina-embeddings",
    "contriever",
    "colbert",
}

RERANKER_SIGNALS = {
    "reranker",
    "rerank",
    "cross-encoder",
    "ms-marco-minilm",
}

KNOWN_MEASURED_SIGNALS = {
    "bge-small-en-v1.5",
    "all-minilm-l6-v2",
    "bge-large-zh-v1.5",
}

GENERAL_LANGUAGE_MODEL_SIGNALS = {
    "qwen2.",
    "qwen2-",
    "qwen2_",
    "qwen2.5",
    "gemma",
    "llama",
    "mistral",
}


def discover_model_files(roots: Iterable[Path], timeout_seconds: int = 300) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        command = [
            "find",
            str(root),
            "-type",
            "f",
            "(",
            "-name",
            "model.onnx",
            "-o",
            "-name",
            "model_quantized.onnx",
            "-o",
            "-name",
            "pytorch_model.bin",
            "-o",
            "-name",
            "model.safetensors",
            ")",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        for line in completed.stdout.splitlines():
            path = Path(line.strip())
            if path.name in MODEL_FILENAMES:
                found.add(path)
    return sorted(found, key=str)


def classify_model_file(path: Path) -> dict:
    path = Path(path)
    normalized = str(path).lower()
    signals = sorted(
        signal
        for signal in [
            *GOAL_PRIORITY_EMBEDDING_SIGNALS,
            *RERANKER_SIGNALS,
            *KNOWN_MEASURED_SIGNALS,
            *GENERAL_LANGUAGE_MODEL_SIGNALS,
        ]
        if signal in normalized
    )

    if any(signal in normalized for signal in KNOWN_MEASURED_SIGNALS):
        category = "known_measured_local_model"
        sota_relevant = False
    elif any(signal in normalized for signal in RERANKER_SIGNALS):
        category = "reranker_candidate"
        sota_relevant = True
    elif any(signal in normalized for signal in GOAL_PRIORITY_EMBEDDING_SIGNALS):
        category = "goal_priority_embedding"
        sota_relevant = True
    elif any(signal in normalized for signal in GENERAL_LANGUAGE_MODEL_SIGNALS):
        category = "general_language_model"
        sota_relevant = False
    else:
        category = "uncategorized_model_file"
        sota_relevant = False

    return {
        "path": str(path),
        "filename": path.name,
        "model_dir": str(_model_dir_for_file(path)),
        "category": category,
        "signals": signals,
        "sota_relevant": sota_relevant,
        "unmeasured_sota_candidate": sota_relevant and category != "known_measured_local_model",
    }


def summarize_inventory(entries: Iterable[dict]) -> dict:
    entries = list(entries)
    categories = Counter(entry["category"] for entry in entries)
    return {
        "file_count": len(entries),
        "categories": dict(sorted(categories.items())),
        "sota_relevant_count": sum(1 for entry in entries if entry["sota_relevant"]),
        "unmeasured_sota_candidate_count": sum(
            1 for entry in entries if entry["unmeasured_sota_candidate"]
        ),
        "unmeasured_sota_candidate_paths": [
            entry["path"] for entry in entries if entry["unmeasured_sota_candidate"]
        ],
    }


def mark_benchmarked_entries(entries: Iterable[dict], benchmarked_model_dirs: Iterable[Path]) -> list[dict]:
    benchmarked = {str(Path(path).expanduser()) for path in benchmarked_model_dirs}
    marked = []
    for entry in entries:
        updated = dict(entry)
        is_benchmarked = updated["model_dir"] in benchmarked
        updated["benchmarked"] = is_benchmarked
        if is_benchmarked:
            updated["unmeasured_sota_candidate"] = False
        marked.append(updated)
    return marked


def build_inventory(
    roots: Iterable[Path],
    timeout_seconds: int = 300,
    benchmarked_model_dirs: Iterable[Path] | None = None,
) -> dict:
    files = discover_model_files(roots, timeout_seconds=timeout_seconds)
    entries = [classify_model_file(path) for path in files]
    entries = mark_benchmarked_entries(entries, benchmarked_model_dirs or [])
    return {
        "schema_version": 1,
        "roots": [str(Path(root).expanduser()) for root in roots],
        "benchmarked_model_dirs": [str(Path(path).expanduser()) for path in benchmarked_model_dirs or []],
        "summary": summarize_inventory(entries),
        "model_files": entries,
    }


def _model_dir_for_file(path: Path) -> Path:
    if path.parent.name == "onnx":
        return path.parent.parent
    return path.parent
