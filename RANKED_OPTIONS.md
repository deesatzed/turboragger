# RANKED_OPTIONS.md

## Decision Needed

Which next development direction gives `turboragger` the highest probability of reaching the active NFCorpus SOTA target without weakening the benchmark or tuning on test qrels?

## Goal Being Optimized

Maximize the probability of a saved local benchmark artifact reaching BEIR NFCorpus `test` `nDCG@10 >= 0.4699` while preserving the reproducible artifact schema, train/dev/test separation, local-first preference, and secondary `Recall@100` reporting.

## Current State

- Current best local run: `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion`.
- Current best artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion_20260608T192209Z.json`.
- Current best metrics: `nDCG@10 = 0.3675830427079456`, `Recall@100 = 0.3328785827037792`.
- Remaining gap to selected SOTA target: `0.10231695729205437` absolute `nDCG@10`.
- Same-source fusion, calibration, rank-score fusion, pooling, branch-depth, train/dev GBDT classification, train/dev GBDT regression, learned cascades, lexical fields, PRF, LEANN no-recompute, and locally discovered BCE embedding have been measured and did not close the gap.

## Options

| Rank | Option | Estimated Chance of Achieving Goal | Why | Main Risk | When To Choose |
|---:|---|---:|---|---|---|
| 1 | Stronger English retrieval model acquisition and benchmark | 45% | The current gap is too large for additional same-source fusion; a stronger semantic retriever is the most plausible missing signal. | Network, storage, license, or runtime limits; large models may be slow on CPU. | Choose first if local model acquisition is approved. |
| 2 | Local reranker acquisition and top-k reranking | 35% | `nDCG@10` is mostly top-10 ordering; a real cross-encoder/reranker could improve ranking without changing the corpus or qrels. | No complete reranker is currently local; runtime may be high; dependency imports need guarded loading. | Choose when a complete BGE/Qwen3/MiniCPM/cross-encoder reranker is available. |
| 3 | Domain-supervised fine-tuning or adapter training | 25% | NFCorpus `train` and `dev` can provide a domain signal beyond current frozen embeddings and GBDT fusion. | Overfit/leakage risk, slow iteration, and uncertain gains from small data. | Choose if stronger pretrained models are unavailable but local training resources are acceptable. |
| 4 | Distinct retrieval family: SPLADE, ColBERT, graph/path retrieval, or coverage search | 20% | A different retrieval family may recover documents the current dense/sparse/fusion pool misses. | More engineering and storage work before knowing whether it moves `nDCG@10`. | Choose after confirming no stronger plain retriever/reranker can be added. |
| 5 | External API target-sanity benchmark | 15% for local methodology progress | A hosted model can validate the target and expose what kind of signal is missing, but it does not satisfy the local/offline methodology goal by itself. | Requires explicit user approval and may shift the project from local method to paid API comparison. | Choose only as a bounded diagnostic, not as silent completion evidence. |

## Recommended Option

Start with Rank 1, then Rank 2. The scoreboard already exists; the bottleneck is now signal quality, not harness architecture. The highest-confidence path is:

1. Add one complete stronger English retriever locally.
2. Benchmark it directly with `scripts/run_nfcorpus_candidate.py`.
3. If it improves candidate quality, fuse it with the current graded-regression anchor using `dev` calibration only.
4. Add a local reranker and rerank the strongest candidate pool if a complete reranker becomes available.

## Why Not The Others

- More same-source fusion has already produced only sub-`0.002` movements around the current anchor.
- HyDE/PRF-style query expansion has already been measured in a simple form and did not improve the current best.
- Graph-hop and coverage loops are behaviorally interesting, but they are downstream of having a better candidate pool.
- Fine-tuning should stay behind stronger pretrained retrieval/reranking because it has higher leakage and overfit risk.

## Confidence Level

Medium. The benchmark harness and current results are artifact-backed, but SOTA success depends on adding a stronger signal that is not currently available in the local model inventory.

## Evidence Used

- `GOAL.md`: active SOTA target and completion contract.
- `artifacts/leaderboard.json`: current measured leaderboard.
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion_20260608T192209Z.json`: current best artifact.
- `artifacts/local_model_inventory.json`: no unmeasured SOTA-relevant local retriever/reranker candidates remain.
- `SOTA_ATTEMPT_REPORT.md`: measured rejected branches and root-cause analysis.

## Assumptions

- The user wants methodology development, not a product UI pass.
- The successful milestone must be benchmark-backed.
- Local/air-gapped remains a core preference unless the user explicitly approves external access.
- BEIR NFCorpus remains the first benchmark because it is the current artifact-backed target and medical-domain stress case.

## Decision To Record

Do not keep spending long runs on same-source calibration. Acquire or discover a stronger English retriever or reranker, run it through the existing harness, and only then revisit fusion.
