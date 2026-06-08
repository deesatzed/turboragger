# REPO_MAP.md

## Project Type

`turboragger` is currently a workspace container for a new local RAG methodology, not a git repository and not yet an implemented application. The root contains design/handoff artifacts plus child git repos used as source material or reusable components.

## Tech Stack

- Target: Python 3.11+, local/air-gapped RAG on Apple Silicon.
- Retrieval components under consideration: BEIR/nfcorpus scoring, MiniLM baseline, BGE-M3/medical embedders, sparse retrieval, RRF, HyDE, ColBERT-style late interaction, graph-hop retrieval, coverage loop.
- Local acceleration/storage components: MLX/MLX-LM where available, `turbovec`/TurboQuant for compressed vector storage and filtered search.
- Provenance component: `Agent_Pidgeon` deterministic semantic contracts, receipts, and hash-chained traces.

## Package Manager

No root package manager exists yet. Child repos have their own package surfaces:

- `turbovec/`: Rust crate plus Python bindings via maturin.
- `Agent_Pidgeon/`: Python project with unittest suite.
- `newragcity/`: prior RAG work and BEIR benchmark scripts.

## Commands

| Purpose | Command | Verified |
|---|---|---|
| Confirm root is not git | `git rev-parse --show-toplevel` | Yes: failed, root is not a git repo |
| Inspect child repo SHAs | `git -C <child> branch --show-current && git -C <child> rev-parse --short HEAD` | Yes |
| Probe local imports | `python3 - <<'PY' ... import pytrec_eval, rank_bm25, sentence_transformers, beir, mlx, mlx_lm, turbovec ... PY` | Yes |
| Run Agent_Pidgeon tests | `cd Agent_Pidgeon && python3 -m unittest discover -s tests -v` | Not run in this audit |
| Step 0 baseline reproduction | Build/run nfcorpus Recall@100 scoreboard against current MiniLM dense baseline | Not ready: dataset path not found and dependency issues remain |

## Entry Points

- No root `turboragger` entry point exists.
- Design entry point: `BRAINSTORM_turboragger.md`.
- Resume entry point: `HANDOFF_LATEST.md`.
- Benchmark source: `newragcity/ersatz_rag/regulus/backend/benchmarks/beir_unified_benchmark.py`.
- Current dense retriever source: `newragcity/ersatz_rag/regulus/backend/app/leann_vector.py`.
- Provenance source: `Agent_Pidgeon/src/agent_pidgin/resolver.py`, `Agent_Pidgeon/src/agent_pidgin/flight_recorder.py`.

## Major Folders

- `turbovec/`: compressed vector index and Python/Rust bindings.
- `Agent_Pidgeon/`: deterministic semantic contract and provenance layer.
- `newragcity/`: prior RAG stack, benchmark artifacts, and known baseline failure evidence.
- `codex-chatgpt-control/`: child git repo present but not part of the current handoff plan.

## Existing Patterns To Preserve

- Treat `turboragger` root as a design workspace until a root package/app is created.
- Do not build multiple retrieval branches simultaneously. Build a stable fan-out/RRF harness, then enable one branch at a time against a fixed baseline.
- Do not trust placeholder aggregate benchmark claims. Trust only reproducible benchmark artifacts and rerun evidence.
- Keep Agent_Pidgeon resolution/trust deterministic; embeddings may suggest pointers but must not replace exact pointer resolution.

## Tests and Verification

Current live probe results:

- `pytrec_eval`: import OK.
- `rank_bm25`: import OK.
- `beir`: import OK.
- `turbovec`: current probe finds local Python source under `turbovec/turbovec-python/python`, but the compiled `_turbovec` extension is missing.
- `sentence_transformers`: import failed with `ValueError: Either a revision or a version must be specified.`
- direct cached MiniLM via `transformers`: usable with in-process optional `kernels` suppression.
- `mlx_lm`: import failed because no Metal device is available in this session.
- nfcorpus dataset found at `/Volumes/WS4TB/WS4TBr/newragcity/UltraRAG-main/datasets/nfcorpus`.

## Likely Files For Current Task

- `HANDOFF_LATEST.md`
- `HANDOFF_2026-06-07.md`
- `BRAINSTORM_turboragger.md`
- `newragcity/ersatz_rag/regulus/backend/benchmarks/results/beir_unified_results.json`
- `newragcity/ersatz_rag/regulus/backend/benchmarks/beir_unified_benchmark.py`
- `newragcity/ersatz_rag/regulus/backend/app/leann_vector.py`
- `newragcity/ersatz_rag/cognitron/cognitron/core/confidence.py`
- `turbovec/README.md`
- `Agent_Pidgeon/README.md`

## Unknowns

- Whether the original nfcorpus dataset/config can be recovered locally.
- Whether the old `Recall@100 ~= 0.18` result can be reproduced without repairing paths/dependencies.
- Which embedder wins on nfcorpus: BGE-M3, Qwen3-Embedding, medical-domain dense model, or a hybrid.
- Whether BGE-M3 ColBERT vectors quantized through turbovec fit the target corpus on 64 GB unified memory.
- Whether MLX access is available from a non-headless runtime even though it is unavailable in this session.
