from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path
from typing import Any

from turboragger.data import find_nfcorpus
from turboragger.dense import find_cached_minilm_snapshot


MODULES = ["pytrec_eval", "rank_bm25", "beir", "turbovec", "sentence_transformers", "mlx", "mlx_lm"]


def probe_environment(root: Path) -> dict[str, Any]:
    modules = {module_name: _probe_module(module_name) for module_name in MODULES}
    modules["turbovec"]["local_source"] = _probe_turbovec_source(root)
    accelerator = _accelerator_status(modules)
    return {
        "status": "ok",
        "python": sys.version,
        "platform": platform.platform(),
        "modules": modules,
        "models": {"all_minilm_l6_v2": _probe_minilm_cache()},
        "nfcorpus": find_nfcorpus(root),
        "accelerator": accelerator,
    }


def _probe_module(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"usable": False, "error_type": type(exc).__name__, "error": str(exc)}
    return {"usable": True, "version": str(getattr(module, "__version__", ""))}


def _probe_turbovec_source(root: Path) -> dict[str, Any]:
    source_path = root / "turbovec" / "turbovec-python" / "python"
    init_path = source_path / "turbovec" / "__init__.py"
    extension_candidates = list((source_path / "turbovec").glob("_turbovec*.so"))
    return {
        "source_path": str(source_path),
        "source_init_exists": init_path.exists(),
        "compiled_extension_exists": bool(extension_candidates),
        "compiled_extensions": [str(path) for path in extension_candidates],
    }


def _probe_minilm_cache() -> dict[str, Any]:
    snapshot = find_cached_minilm_snapshot()
    if snapshot is None:
        return {"usable": False, "snapshot": None}
    return {
        "usable": True,
        "snapshot": str(snapshot),
        "loader": "direct_transformers_with_optional_kernels_disabled",
    }


def _accelerator_status(modules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if modules.get("mlx_lm", {}).get("usable"):
        return {"mode": "mlx", "usable": True}
    if modules.get("mlx", {}).get("usable"):
        return {"mode": "mlx_core_only", "usable": False, "reason": "mlx imports but mlx_lm is unavailable"}
    return {"mode": "cpu_or_unknown", "usable": True}
