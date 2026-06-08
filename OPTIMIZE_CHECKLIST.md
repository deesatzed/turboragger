# OPTIMIZE_CHECKLIST.md

## Current Frontier

- Goal: achieve SOTA or better on BEIR NFCorpus `test` under `GOAL.md`.
- Primary metric: `nDCG@10`.
- SOTA target: `0.4699`.
- Current best local artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion_20260608T192209Z.json`.
- Current best local score: `nDCG@10 = 0.3675830427079456`, `Recall@100 = 0.3328785827037792`.
- Remaining primary gap: `0.10231695729205437`.

## Optimize State

- Frontier mode: evidence-gated benchmark optimization.
- Current optimize submode: exploit/fusion audit over existing local source set.
- Candidate brief count: 11 active/recent.
- Promoted line count: 1 current local anchor.
- Current smoke queue: none.
- Current full-eval queue: stronger English retriever or local reranker once available.
- Stagnation check: same-source fusion, learned calibration, deeper same-source candidate pools, graded-regression deeper candidate pools, current-source token MaxSim late interaction, second-stage late-interaction calibration, tiny SciFact-style domain fine-tuning, and dev-selected tiny SciFact checkpoints have not closed the SOTA gap.
- Next concrete action: obtain or discover a materially stronger English retrieval/reranking signal; do not spend more long runs on same-source calibration unless the candidate changes the signal family.

## Verification Surface

- Focused candidate tests: `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_candidate_runner.py' -v`.
- Full suite: `PYTHONPATH=src python3 -m unittest discover -s tests -v`.
- Compile: `PYTHONPATH=src python3 -m compileall -q src scripts`.
- Artifact validation: `python3 -m json.tool <artifact> >/tmp/<name>.pretty`.
- Leaderboard: `artifacts/leaderboard.json`.

## Stop Conditions

- Do not mark the goal complete until a saved run reaches `nDCG@10 >= 0.4699` under the same split and metric.
- Do not move the target after seeing local results.
- Do not tune on NFCorpus test qrels.
- If no stronger model or distinct signal is available, preserve the current attempt report instead of weakening the goal.
