#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from turboragger.historical_audit import audit_newragcity_historical_claim


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result_path = root / "newragcity/ersatz_rag/regulus/backend/benchmarks/results/beir_unified_results.json"
    source_path = root / "newragcity/ersatz_rag/regulus/backend/benchmarks/beir_unified_benchmark.py"
    audit = audit_newragcity_historical_claim(result_path, source_path)
    output_path = root / "artifacts/historical_newragcity_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"written": str(output_path.relative_to(root)), "verdict": audit["verdict"]}, indent=2))
    return 0 if audit["verdict"] in {"invalid", "needs_rescore"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
