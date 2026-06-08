# SOTA Progress

## 2026-06-08 - Target Gate

Status: SOTA target established before new candidate implementation.

Artifacts created:

- `artifacts/sota_target.json`
- `SOTA_TARGET.md`

Primary target:

- Source: MTEB English Retrieval leaderboard data table
- System: `voyage-3-m-exp`
- Reported NFCorpus `nDCG@10`: `46.99`
- Normalized target: `0.4699`

Comparator targets:

- Official BEIR EvalAI NFCorpus comparator: `0.385` from `ZA+NM+Unicamp (InParsv2)`
- Best observed current open-weight MTEB comparator: `0.4517` from `nvidia/NV-Embed-v2`

Current local baseline:

- `nDCG@10`: `0.3160012178022206`
- `Recall@100`: `0.31150992401169303`
- Queries: `323`
- Failure count: `0`

Current primary gap:

- Absolute `nDCG@10` gap to selected target: `0.1538987821977794`

Next stage:

1. Rerun the environment probe.
2. Rerun the direct MiniLM baseline.
3. Implement a generalized candidate runner that writes `artifacts/runs/<run_id>.json` and `artifacts/leaderboard.json`.

## 2026-06-08 - Baseline Gate Rerun

Commands run:

```bash
PYTHONPATH=src python3 scripts/probe_environment.py
PYTHONPATH=src python3 scripts/run_nfcorpus_baseline.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src scripts
python3 -m json.tool artifacts/sota_target.json >/tmp/sota_target.pretty
python3 -m json.tool artifacts/environment_probe.json >/tmp/environment_probe.pretty
python3 -m json.tool artifacts/baseline_minilm_nfcorpus.json >/tmp/baseline_minilm.pretty
```

Observed outcomes:

- Environment probe exited 0 and rewrote `artifacts/environment_probe.json`.
- Probe still emitted the known no-Metal MLX warning at interpreter shutdown.
- Direct MiniLM baseline exited 0 and reproduced:
  - `nDCG@10`: `0.3160012178022206`
  - `Recall@100`: `0.31150992401169303`
  - Queries: `323`
  - Failure count: `0`
- Unit tests passed: 15 tests, 0 failures.
- Compileall exited 0.
- JSON validation exited 0 for target, probe, and baseline artifacts.

Next stage:

1. Implement a generalized candidate runner.
2. Populate `artifacts/runs/`.
3. Populate `artifacts/leaderboard.json`.
4. Re-run direct MiniLM, BM25, and MiniLM+BM25 RRF through the unified candidate artifact schema.

## 2026-06-08 - Candidate Runner And First Leaderboard

Code added:

- `src/turboragger/benchmark.py`
- `scripts/run_nfcorpus_candidate.py`
- `tests/test_benchmark.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate all-baselines
```

Generated artifacts:

- `artifacts/runs/bm25_20260608T132446Z.json`
- `artifacts/runs/minilm_dense_20260608T132449Z.json`
- `artifacts/runs/minilm_bm25_rrf_20260608T132506Z.json`
- `artifacts/leaderboard.json`

Leaderboard:

| Candidate | nDCG@10 | Recall@100 | Queries | Failures |
|---|---:|---:|---:|---:|
| `minilm_bm25_rrf` | `0.3344802256991052` | `0.3121492868681654` | 323 | 0 |
| `minilm_dense` | `0.3160012178022206` | `0.31150992401169303` | 323 | 0 |
| `bm25` | `0.3064137649277972` | `0.23307829323957305` | 323 | 0 |

Current best local result:

- `minilm_bm25_rrf`
- Gap to selected SOTA target `0.4699`: `0.1354197743008948`

Verification:

- New benchmark test failed before implementation with `ModuleNotFoundError: No module named 'turboragger.benchmark'`.
- New benchmark tests passed after implementation: 2 tests, 0 failures.
- Full unit suite passed after implementation: 17 tests, 0 failures.
- Compileall exited 0.
- Generated RRF run artifact and leaderboard both parse as JSON.

Next stage:

1. Probe stronger embedder availability.
2. Add the strongest locally feasible dense candidate to `scripts/run_nfcorpus_candidate.py`.
3. Run and compare against `artifacts/leaderboard.json`.

## 2026-06-08 - Stronger Embedder Availability Probe And Runtime Gate

Code added:

- `src/turboragger/embedder_probe.py`
- `scripts/probe_embedders.py`
- `tests/test_embedder_probe.py`

Command:

```bash
PYTHONPATH=src python3 scripts/probe_embedders.py
```

Generated artifact:

- `artifacts/embedder_availability.json`

Observed result:

- Status: `available`
- Probe exit code after adding local fallback: 0

Checked candidates:

| Candidate | Result |
|---|---|
| `BAAI/bge-m3` | Missing from Hugging Face cache; known workspace path is an incomplete stub without weights. |
| `Qwen/Qwen3-Embedding-0.6B` | Missing from Hugging Face cache. |
| `nomic-ai/nomic-embed-text-v1.5` | Missing from Hugging Face cache. |
| `intfloat/e5-large-v2` | Missing from Hugging Face cache. |
| `Alibaba-NLP/gte-large-en-v1.5` | Missing from Hugging Face cache. |
| `BAAI/bge-large-zh-v1.5` | Complete local fallback snapshot available under sibling ragflow cache. |
| `Xenova/bge-small-en-v1.5` | Complete local English ONNX fallback snapshot available under sibling finESS cache. |

BGE benchmark:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_large_zh
```

Runtime-gate artifact:

- `artifacts/bge_large_zh_runtime_gate.json`

Result:

- Model loaded successfully through direct `transformers`.
- The latest `max_length=256` run completed after `411.749291` seconds.
- Latest run artifact: `artifacts/runs/bge_large_zh_20260608T134449Z.json`
- `nDCG@10`: `0.14143051730446513`
- `Recall@100`: `0.16162218356917438`
- Queries: 323
- Failures: 0
- `artifacts/leaderboard.json` includes the BGE fallback below the existing baseline branches.

Verification:

- Probe tests passed: 3 tests, 0 failures.
- Full unit suite passed after probe implementation: 20 tests, 0 failures.
- Compileall exited 0.
- `artifacts/embedder_availability.json` parses as JSON.
- `artifacts/bge_large_zh_runtime_gate.json` parses as JSON.

Next stage:

1. Make a stronger English retrieval embedder available locally.
2. Benchmark the stronger model through `scripts/run_nfcorpus_candidate.py`.
3. Add reranking/fusion only after the stronger dense branch improves the candidate pool.

## 2026-06-08 - Dev-Calibrated Rank-Score Fusion

Code updated:

- `src/turboragger/calibration.py`
- `scripts/run_nfcorpus_candidate.py`
- `tests/test_calibration.py`
- `tests/test_candidate_runner.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_rank_score_fusion
```

Generated artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_rank_score_fusion_20260608T174836Z.json`

Result:

- Dev split calibration selected `rank_weight = 4.0` from `[0.0, 0.25, 0.5, 1.0, 2.0, 4.0]`.
- Dev calibration metrics: `nDCG@10 = 0.3430752044743363`, `Recall@100 = 0.3188384175956566`, 324 queries.
- Test metrics: `nDCG@10 = 0.36566973749080495`, `Recall@100 = 0.32302981358524474`, 323 queries, 0 failures.
- Runtime: `312.021744` seconds.

Decision:

- Not promoted. It loses `0.00047867597184941824` absolute `nDCG@10` versus the current best `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- Current gap to selected SOTA target `0.4699`: `0.10423026250919504`.

Next stage:

1. Stop spending benchmark time on fusion-only changes over the same four weak source branches unless a stronger source appears.
2. Make an approved stronger English retriever or local reranker available.
3. Benchmark that stronger model through `scripts/run_nfcorpus_candidate.py` before any further fusion calibration.

## 2026-06-08 - Train-Split Learned Feature Fusion

Code updated:

- `src/turboragger/learned_fusion.py`
- `scripts/run_nfcorpus_candidate.py`
- `tests/test_learned_fusion.py`
- `tests/test_candidate_runner.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_train_dev_learned_fusion
```

Generated artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_learned_fusion_20260608T180407Z.json`

Result:

- Fitted local `sklearn.linear_model.LogisticRegression` on NFCorpus `train` branch outputs only.
- Training rows: `629617`.
- Positive training rows: `30964`.
- Train query count: `2590`.
- Test metrics: `nDCG@10 = 0.35887473712153073`, `Recall@100 = 0.31843407013584935`, 323 queries, 0 failures.
- Runtime: `349.383313` seconds.

Decision:

- Not promoted. It loses `0.007273676341123636` absolute `nDCG@10` versus the current best `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- Current gap from this branch to selected SOTA target `0.4699`: `0.11102526287846926`.

Next stage:

1. Treat supervised score/rank feature fusion over the same source set as covered and rejected.
2. Make a stronger English retriever or local reranker available before further expensive calibration.
3. Reuse the learned-fusion module only after a stronger candidate pool exists.

## 2026-06-08 - Train-Split GBDT Feature Fusion

Code updated:

- `src/turboragger/learned_fusion.py`
- `scripts/run_nfcorpus_candidate.py`
- `tests/test_learned_fusion.py`
- `tests/test_candidate_runner.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion
```

Generated artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion_20260608T181638Z.json`

Result:

- Fitted local `sklearn.ensemble.HistGradientBoostingClassifier` on NFCorpus `train` branch outputs only.
- Model params: `max_iter = 80`, `max_leaf_nodes = 15`, `learning_rate = 0.05`, `class_weight = balanced`.
- Training rows: `629617`.
- Positive training rows: `30964`.
- Train query count: `2590`.
- Test metrics: `nDCG@10 = 0.36273469704993305`, `Recall@100 = 0.32849490503295947`, 323 queries, 0 failures.
- Runtime: `358.284685` seconds.

Decision:

- Not promoted. It loses `0.0034137164127213127` absolute `nDCG@10` versus the current best `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- It improves `Recall@100` by `0.007308533158182075`, but `nDCG@10` remains the primary SOTA metric.
- Current gap from this branch to selected SOTA target `0.4699`: `0.10716530295006693`.

Next stage:

1. Treat nonlinear supervised feature fusion over the current branch set as measured and rejected for the primary metric.
2. Stop adding learned scorers over the same source set unless a stronger retriever/reranker changes the candidate pool.
3. Make a stronger English retriever or local reranker available and benchmark it through the same artifact schema.

## 2026-06-08 - Train/Dev GBDT Cascade

Code updated:

- `src/turboragger/learned_fusion.py`
- `scripts/run_nfcorpus_candidate.py`
- `tests/test_learned_fusion.py`
- `tests/test_candidate_runner.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_cascade
```

Generated artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_cascade_20260608T183008Z.json`

Result:

- Fitted local GBDT on NFCorpus `train` branch outputs only.
- Selected cascade `anchor_k = 3` on NFCorpus `dev` from `[0, 3, 5, 10, 20]`.
- Dev calibration metrics: `nDCG@10 = 0.34344212545018865`, `Recall@100 = 0.32309497909000856`, 324 queries.
- Test metrics: `nDCG@10 = 0.36218572616170835`, `Recall@100 = 0.32849490503295947`, 323 queries, 0 failures.
- Runtime: `365.989368` seconds.

Decision:

- Not promoted. It loses `0.003962687300946011` absolute `nDCG@10` versus the current best `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- It preserves the recall-positive GBDT surface, but top-10 ordering remains worse.
- Current gap from this branch to selected SOTA target `0.4699`: `0.10771427383829163`.

Next stage:

1. Treat learned cascades over the current branch set as measured and rejected for the primary metric.
2. Make a stronger English retriever or local reranker available before further expensive learned reranking.
3. Keep the cascade helper as reusable infrastructure for a stronger future candidate pool.

## 2026-06-08 - Train/Dev GBDT Score Fusion

Code updated:

- `src/turboragger/learned_fusion.py`
- `scripts/run_nfcorpus_candidate.py`
- `tests/test_candidate_runner.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion
```

Generated artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion_20260608T184157Z.json`

Result:

- Fitted local GBDT on NFCorpus `train` branch outputs only.
- Selected final two-ranker score-fusion weights on NFCorpus `dev` only.
- Selected weights: `score_fusion_primary = 1.5`, `gbdt_secondary = 2.0`.
- Dev calibration metrics: `nDCG@10 = 0.3463843538452143`, `Recall@100 = 0.3246700972325609`, 324 queries.
- Training rows: `629617`.
- Positive training rows: `30964`.
- Train query count: `2590`.
- Test metrics: `nDCG@10 = 0.3670950977369987`, `Recall@100 = 0.3331795970806332`, 323 queries, 0 failures.
- Runtime: `366.071979` seconds.

Decision:

- Promoted as the new current best local result.
- It improves the previous best by `0.000946684274344356` absolute `nDCG@10`.
- It improves `Recall@100` by `0.0119932252058558`.
- It remains `0.10280490226300126` absolute `nDCG@10` below the selected SOTA target `0.4699`.
- It remains `0.017904902263001303` absolute `nDCG@10` below the official BEIR comparator `0.385`.

Next stage:

1. Use this branch as the current local anchor.
2. Do not continue same-source fusion-only tuning unless a stronger retriever/reranker changes the candidate pool.
3. Make an approved stronger English retriever or local reranker available, then benchmark it through the same artifact schema.

## 2026-06-08 - Train/Dev GBDT Five-Source Calibrated Score Fusion

Code updated:

- `scripts/run_nfcorpus_candidate.py`
- `tests/test_candidate_runner.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_calibrated_score_fusion
```

Generated artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_calibrated_score_fusion_20260608T185555Z.json`

Result:

- Fitted local GBDT on NFCorpus `train` branch outputs only.
- Selected five-source score-fusion weights on NFCorpus `dev` only.
- Calibrated branches: `bge_small_cls_onnx`, `bge_small_mean_onnx`, `xenova_minilm_onnx`, `bm25`, `gbdt_feature_fusion`.
- Selected weights: `bge_small_cls_onnx = 0.5`, `bge_small_mean_onnx = 2.0`, `xenova_minilm_onnx = 0.5`, `bm25 = 0.5`, `gbdt_feature_fusion = 1.5`.
- Dev calibration metrics: `nDCG@10 = 0.3491764253742419`, `Recall@100 = 0.3236656324926965`, 324 queries.
- Calibration grid size: `3124`.
- Training rows: `629617`.
- Positive training rows: `30964`.
- Train query count: `2590`.
- Test metrics: `nDCG@10 = 0.3660561464830782`, `Recall@100 = 0.32611330897080015`, 323 queries, 0 failures.
- Runtime: `522.18218` seconds.

Decision:

- Not promoted.
- It loses `0.001038951253920506` absolute `nDCG@10` versus the current best.
- It loses `0.007066288109833063` `Recall@100` versus the current best.
- It remains `0.10384385351692177` absolute `nDCG@10` below the selected SOTA target `0.4699`.
- It remains `0.018943853516921794` absolute `nDCG@10` below the official BEIR comparator `0.385`.

Next stage:

1. Treat five-source GBDT calibration over the current branch set as measured and rejected.
2. Keep the two-ranker GBDT score-fusion branch as the current local anchor.
3. Shift effort toward a stronger English retriever, local reranker, or a materially different training signal.

## 2026-06-08 - Deep-Pool Train/Dev GBDT Score Fusion

Code updated:

- `scripts/run_nfcorpus_candidate.py`
- `tests/test_candidate_runner.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_dev_score_fusion
```

Generated artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_dev_score_fusion_20260608T190950Z.json`

Result:

- Fitted local GBDT on NFCorpus `train` branch outputs only.
- Used `branch_k = 300` for the score-fusion primary, GBDT secondary, and final two-ranker score fusion.
- Selected final two-ranker score-fusion weights on NFCorpus `dev` only.
- Selected weights: `score_fusion_primary = 2.0`, `gbdt_secondary = 0.5`.
- Dev calibration metrics: `nDCG@10 = 0.3433560955879096`, `Recall@100 = 0.32859137470597494`, 324 queries.
- Training rows: `629617`.
- Positive training rows: `30964`.
- Train query count: `2590`.
- Test metrics: `nDCG@10 = 0.36562875673411727`, `Recall@100 = 0.33854940883436946`, 323 queries, 0 failures.
- Runtime: `366.618768` seconds.

Decision:

- Not promoted.
- It loses `0.001466341002881455` absolute `nDCG@10` versus the current best.
- It improves `Recall@100` by `0.005369811753736242`, but `nDCG@10` remains the primary SOTA metric.
- It remains `0.10427124326588272` absolute `nDCG@10` below the selected SOTA target `0.4699`.
- It remains `0.019371243265882743` absolute `nDCG@10` below the official BEIR comparator `0.385`.

Next stage:

1. Treat deeper same-source candidate pools as measured and rejected for the primary metric.
2. Keep the non-deep two-ranker GBDT score-fusion branch as the current local anchor.
3. Shift effort toward a stronger English retriever, local reranker, or a materially different training signal.

## 2026-06-08 - Train/Dev GBDT Regression Score Fusion

Code updated:

- `src/turboragger/learned_fusion.py`
- `scripts/run_nfcorpus_candidate.py`
- `tests/test_learned_fusion.py`
- `tests/test_candidate_runner.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion
```

Generated artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion_20260608T192209Z.json`

Result:

- Fitted local `HistGradientBoostingRegressor` on NFCorpus `train` branch outputs only.
- Used graded train relevance targets instead of binary positive/negative labels.
- Selected final two-ranker score-fusion weights on NFCorpus `dev` only.
- Selected weights: `score_fusion_primary = 1.0`, `gbdt_regression_secondary = 1.5`.
- Dev calibration metrics: `nDCG@10 = 0.34652923156548504`, `Recall@100 = 0.32365113103814974`, 324 queries.
- Training rows: `629617`.
- Positive training rows: `30964`.
- Train query count: `2590`.
- Positive sample weight: `19.333839297248417`.
- Test metrics: `nDCG@10 = 0.3675830427079456`, `Recall@100 = 0.3328785827037792`, 323 queries, 0 failures.
- Runtime: `347.032307` seconds.

Decision:

- Promoted as the new current best local result.
- It improves the previous best by `0.00048794497094689637` absolute `nDCG@10`.
- It reduces `Recall@100` by `0.0003010143768540363`.
- It remains `0.10231695729205437` absolute `nDCG@10` below the selected SOTA target `0.4699`.
- It remains `0.01741695729205439` absolute `nDCG@10` below the official BEIR comparator `0.385`.

Next stage:

1. Use this branch as the current local anchor.
2. Continue looking for a stronger English retriever, local reranker, or materially different learning objective.
3. Do not weaken the target or tune on NFCorpus test qrels.

## 2026-06-08 - Direct Train/Dev GBDT Regression Feature Fusion

Code updated:

- `scripts/run_nfcorpus_candidate.py`
- `tests/test_candidate_runner.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_fusion
```

Generated artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_fusion_20260608T193443Z.json`

Result:

- Fitted local `HistGradientBoostingRegressor` on NFCorpus `train` branch outputs only.
- Used graded train relevance targets instead of binary positive/negative labels.
- Test metrics: `nDCG@10 = 0.36434599974495163`, `Recall@100 = 0.325698652336419`, 323 queries, 0 failures.
- Runtime: `333.79718` seconds.

Decision:

- Not promoted.
- It loses `0.0032370429629939856` absolute `nDCG@10` versus the current best.
- It loses `0.007179930367360199` `Recall@100` versus the current best.
- It remains `0.10555400025504835` absolute `nDCG@10` below the selected SOTA target `0.4699`.
- It remains `0.020654000255048377` absolute `nDCG@10` below the official BEIR comparator `0.385`.

Next stage:

1. Treat the direct graded-regression ranker as measured and rejected.
2. Keep the dev-calibrated graded-regression score-fusion branch as the current local anchor.
3. Shift effort toward a stronger English retriever, local reranker, or another materially different learning objective.

## 2026-06-08 - English ONNX BGE And Hybrid Fusion

Code updated:

- `src/turboragger/dense.py`
- `scripts/run_nfcorpus_candidate.py`
- `src/turboragger/embedder_probe.py`
- `tests/test_dense.py`

Artifacts:

- `artifacts/runs/bge_small_en_onnx_20260608T135538Z.json`
- `artifacts/runs/bge_small_en_bm25_rrf_20260608T135851Z.json`
- `artifacts/runs/bge_small_minilm_bm25_rrf_20260608T140218Z.json`
- `artifacts/leaderboard.json`

New best local run:

- Candidate: `bge_small_minilm_bm25_rrf`
- `nDCG@10`: `0.3563276091075661`
- `Recall@100`: `0.32206494945280123`
- Queries: 323
- Failures: 0
- Runtime: `159.059456` seconds

Gap:

- To selected SOTA target `0.4699`: `0.11357239089243386`
- To official BEIR comparator `0.385`: `0.0286723908924339`

Reranker status:

- Sibling reranker services point to `openbmb/MiniCPM-Reranker-Light` via `sentence_transformers` or external backends.
- Plain `sentence_transformers` and `FlagEmbedding` imports still fail in this environment.
- Guarded imports using the optional-kernels suppression work for both packages.
- `artifacts/reranker_availability.json` records no complete local reranker model for MiniCPM, BGE reranker, or a small cross-encoder candidate.
- No reranker branch has valid local benchmark evidence yet.

Next stage:

1. Repair/project-isolate reranker and stronger-embedder dependencies.
2. Make one stronger English model available locally.
3. Benchmark it through the same candidate runner.

## 2026-06-08 - Query Expansion PRF Ablation

Code updated:

- `src/turboragger/lexical.py`
- `tests/test_lexical.py`
- `scripts/run_nfcorpus_candidate.py`

Artifacts:

- `artifacts/runs/bm25_prf_20260608T141609Z.json`
- `artifacts/runs/bge_small_minilm_bm25_prf_rrf_20260608T141628Z.json`

Results:

| Candidate | nDCG@10 | Recall@100 | Interpretation |
|---|---:|---:|---|
| `bm25_prf` | `0.2643284676291605` | `0.22671140398062603` | Worse than plain BM25. |
| `bge_small_minilm_bm25_prf_rrf` | `0.34788409229565553` | `0.3208886711792187` | Worse than current best hybrid. |

Conclusion:

- The fixed pseudo-relevance query expansion branch does not improve the local methodology.
- Best at this stage remained `bge_small_minilm_bm25_rrf` at `0.3563276091075661` `nDCG@10`.

## 2026-06-08 - Score-Level Fusion Ablation

Code updated:

- `src/turboragger/score_fusion.py`
- `tests/test_score_fusion.py`
- `tests/test_candidate_runner.py`
- `scripts/run_nfcorpus_candidate.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_minilm_bm25_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_minilm_bm25_score_fusion_20260608T142822Z.json`

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_minilm_bm25_score_fusion` | `0.3622186429303806` | `0.3194178792738884` | `156.206243` | 323 | 0 |

Conclusion:

- Score-level min-max fusion is now the best measured local branch.
- It improves over the previous best RRF branch by `0.005891033822814485` absolute `nDCG@10`.
- It lowers `Recall@100` versus the previous best RRF branch by `0.002647070178912817`.
- It is still not SOTA: gap to selected target `0.4699` is `0.10768135706961942`.
- It also remains below the official BEIR comparator `0.385` by `0.02278135706961939`.

Verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_candidate_runner.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_score_fusion.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src scripts
python3 -m json.tool artifacts/runs/bge_small_minilm_bm25_score_fusion_20260608T142822Z.json >/tmp/score_fusion.pretty
python3 -m json.tool artifacts/leaderboard.json >/tmp/leaderboard.pretty
```

Observed outcomes:

- Candidate runner test failed before implementation with `ValueError: Unsupported candidate: bge_small_minilm_bm25_score_fusion`.
- Candidate runner test passed after implementation.
- Score fusion tests passed.
- Full unit suite passed: 29 tests, 0 failures.
- Compileall exited 0.
- New run artifact and leaderboard both parse as JSON.

Next stage:

1. Do not tune score-fusion weights on NFCorpus test qrels.
2. Make a stronger English retriever or reranker available locally.
3. Benchmark the stronger branch through the same artifact schema.
4. If no stronger local model can be made available, produce `SOTA_ATTEMPT_REPORT.md` rather than weakening the target.

## 2026-06-08 - Historical Newragcity Claim Audit

Claim audited:

- `newragcity/ersatz_rag/regulus/backend/benchmarks/results/beir_unified_results.json`
- Claimed `nDCG@10 = 0.5085946124009167`, `Recall@100 = 0.18392811903482234`

Audit command:

```bash
PYTHONPATH=src python3 scripts/audit_historical_newragcity.py
```

Audit artifact:

- `artifacts/historical_newragcity_audit.json`

Verdict:

- `invalid`

Reasons:

- Saved result has no retrieved doc IDs, runs, or branch outputs, so it cannot be rescored.
- Saved per-query recall contains 12 values above `1.0`; maximum observed value is `2.0`.
- Source evaluator computes IDCG from retrieved top-10 relevances rather than all qrel relevances.
- Source evaluator has double-count risk by counting top-10 relevant hits and then counting top-100 relevant hits again.
- Import smoke test for `ThreeApproachRAG` from the root environment is blocked by an installed `app` package shadowing the local `app/` directory.

Conclusion:

- The historical `0.5085946124009167` value must not be used as SOTA evidence.
- A corrected rerun through the root harness would be needed before any `newragcity` route can be reconsidered.

## 2026-06-08 - Xenova MiniLM ONNX Replacement Ablation

New local model path discovered:

- `/Volumes/WS4TB/WS4TBr/whsjan14/node_modules/@xenova/transformers/.cache/Xenova/all-MiniLM-L6-v2`

Code updated:

- `scripts/run_nfcorpus_candidate.py`
- `tests/test_candidate_runner.py`

Commands:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate xenova_minilm_onnx
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_xenova_minilm_bm25_score_fusion
```

Artifacts:

- `artifacts/runs/xenova_minilm_onnx_20260608T144741Z.json`
- `artifacts/runs/bge_small_xenova_minilm_bm25_score_fusion_20260608T144823Z.json`

Results:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `xenova_minilm_onnx` | `0.31689265638802383` | `0.30671003953431614` | `18.827889` | 323 | 0 |
| `bge_small_xenova_minilm_bm25_score_fusion` | `0.3660878569380838` | `0.3162754206179384` | `158.708362` | 323 | 0 |

Conclusion:

- ONNX MiniLM dense-only improves `nDCG@10` over direct MiniLM by `0.0008914385858032059`, but has lower `Recall@100`.
- Replacing direct MiniLM with ONNX MiniLM inside the BGE-small/BM25 score-fusion branch produces a new best local `nDCG@10`.
- New best improves over the previous score-fusion best by `0.0038692140077032366` absolute `nDCG@10`.
- New best lowers `Recall@100` versus the previous score-fusion best by `0.003142458655950031`.
- SOTA remains unachieved: gap to selected target `0.4699` is `0.10381214306191616`.
- Gap to official BEIR comparator `0.385` is `0.018912143061916187`.

## 2026-06-08 - Corrected LEANN No-Recompute Ablation

Rationale:

- The historical `newragcity` route depends on LEANN.
- Its old result was invalid, but a corrected root-side LEANN candidate could still test whether LEANN/HNSW adds useful retrieval signal.
- LEANN search failed in default recompute mode because this environment cannot start the embedding server, so the candidate builds a no-recompute, non-compact HNSW index with the exact cached MiniLM snapshot.

Code updated:

- `src/turboragger/leann_bridge.py`
- `tests/test_leann_bridge.py`
- `scripts/run_nfcorpus_candidate.py`
- `tests/test_candidate_runner.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate leann_minilm_no_recompute
```

Artifacts:

- `artifacts/runs/leann_minilm_no_recompute_20260608T145957Z.json`
- `artifacts/leann_indexes/leann_minilm_no_recompute`

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `leann_minilm_no_recompute` | `0.3138633685494098` | `0.3111533551882048` | `316.96866` | 323 | 0 |

Conclusion:

- The corrected LEANN branch is valid evidence but not a SOTA path.
- It underperforms direct MiniLM by `0.0021378492528107973` absolute `nDCG@10`.
- It underperforms the current best local branch by `0.052224488388673995` absolute `nDCG@10`.
- It remains `0.15603663145059016` absolute `nDCG@10` below the selected SOTA target.

## 2026-06-08 - Dual-MiniLM Score-Fusion Ablation

Rationale:

- Direct MiniLM and ONNX MiniLM have slightly different score surfaces.
- The previous best used ONNX MiniLM but not direct MiniLM.
- A fair next ablation is equal-weight score fusion over BGE-small ONNX, direct MiniLM, ONNX MiniLM, and BM25 without tuning weights on test qrels.

Code updated:

- `scripts/run_nfcorpus_candidate.py`
- `tests/test_candidate_runner.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_minilm_bm25_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_dual_minilm_bm25_score_fusion_20260608T150916Z.json`

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_dual_minilm_bm25_score_fusion` | `0.3559227197172415` | `0.322866194510992` | `179.24664` | 323 | 0 |

Conclusion:

- Dual-MiniLM fusion is not promoted because it reduces `nDCG@10`.
- It improves `Recall@100` over the current best by `0.006590773893053625`.
- It loses `0.010165137220842335` absolute `nDCG@10` versus the current best branch.
- It remains `0.1139772802827585` absolute `nDCG@10` below the selected SOTA target.

## 2026-06-08 - CombMNZ Score-Fusion Ablation

Rationale:

- The current best branch uses equal-weight min-max score summation over BGE-small ONNX, ONNX MiniLM, and BM25.
- A no-qrels next ablation is to boost documents returned by multiple independent branches using CombMNZ-style agreement weighting.
- This keeps the same local models and avoids test-qrels weight tuning.

Code updated:

- `src/turboragger/score_fusion.py`
- `tests/test_score_fusion.py`
- `tests/test_candidate_runner.py`
- `scripts/run_nfcorpus_candidate.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_xenova_minilm_bm25_mnz_fusion
```

Artifact:

- `artifacts/runs/bge_small_xenova_minilm_bm25_mnz_fusion_20260608T151744Z.json`

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_xenova_minilm_bm25_mnz_fusion` | `0.3594854251733051` | `0.31779820735251124` | `233.373193` | 323 | 0 |

Conclusion:

- CombMNZ agreement boosting is not promoted because it reduces the primary metric.
- It loses `0.0066024317647787045` absolute `nDCG@10` versus the current best branch.
- It improves `Recall@100` by `0.0015227867345728452` versus the current best branch.
- It remains `0.11041457482669487` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.02551457482669489` absolute `nDCG@10` below the official BEIR comparator.

## 2026-06-08 - Deep Candidate-Pool Score-Fusion Ablation

Rationale:

- The current best branch fuses only the top 100 returned by each component retriever.
- A non-leaking next ablation is to retrieve a wider branch pool, then still evaluate only the final fused top 100.
- This tests whether the current method is candidate-pool limited without tuning fusion weights on NFCorpus test qrels.

Code updated:

- `src/turboragger/score_fusion.py`
- `tests/test_score_fusion.py`
- `tests/test_candidate_runner.py`
- `scripts/run_nfcorpus_candidate.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_xenova_minilm_bm25_deep_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_xenova_minilm_bm25_deep_score_fusion_20260608T152801Z.json`

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Branch k | Queries | Failures |
|---|---:|---:|---:|---:|---:|---:|
| `bge_small_xenova_minilm_bm25_deep_score_fusion` | `0.3621651141202009` | `0.3273489539542475` | `195.823889` | 300 | 323 | 0 |

Conclusion:

- Deep score fusion is not promoted because it reduces the primary metric.
- It loses `0.003922742817882907` absolute `nDCG@10` versus the current best branch.
- It improves `Recall@100` by `0.011073533336309116` versus the current best branch.
- It remains `0.10773488587979907` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.022834885879799094` absolute `nDCG@10` below the official BEIR comparator.
- The recall gain makes a reranker over a wider pool more attractive, but this branch alone does not achieve SOTA.

## 2026-06-08 - BGE Mean-Pooling Score-Fusion Ablation

Rationale:

- The current best branch uses `Xenova/bge-small-en-v1.5` ONNX with CLS pooling.
- The same local ONNX model can also be pooled by attention-mask mean pooling, which is a no-qrels embedding variant.
- This tests whether the BGE pooling choice is suppressing the English dense signal without changing qrels, corpus, query text, or fusion weights.

Code updated:

- `tests/test_candidate_runner.py`
- `scripts/run_nfcorpus_candidate.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_mean_xenova_minilm_bm25_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_mean_xenova_minilm_bm25_score_fusion_20260608T153812Z.json`

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | BGE pooling | Queries | Failures |
|---|---:|---:|---:|---|---:|---:|
| `bge_small_mean_xenova_minilm_bm25_score_fusion` | `0.3634426387237848` | `0.31832067766164274` | `191.308131` | `mean` | 323 | 0 |

Conclusion:

- BGE mean pooling is not promoted because it reduces the primary metric.
- It loses `0.0026452182142990277` absolute `nDCG@10` versus the current best branch.
- It improves `Recall@100` by `0.0020452570437043405` versus the current best branch.
- It remains `0.10645736127621519` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.021557361276215214` absolute `nDCG@10` below the official BEIR comparator.

## 2026-06-08 - Combined BGE CLS+Mean Pooling Score-Fusion

Rationale:

- CLS pooling remained the best BGE-small ONNX branch, but mean pooling ranked second and improved recall.
- A fair no-qrels ablation is to include both BGE pooling views as independent equal-weight branches.
- This tests whether the BGE pooling views are complementary without changing qrels, corpus, query text, or fusion weights.

Code updated:

- `tests/test_candidate_runner.py`
- `scripts/run_nfcorpus_candidate.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_score_fusion_20260608T154445Z.json`

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | BGE poolings | Queries | Failures |
|---|---:|---:|---:|---|---:|---:|
| `bge_small_dual_pool_xenova_minilm_bm25_score_fusion` | `0.36614841346265437` | `0.3211863718747774` | `301.019192` | `cls, mean` | 323 | 0 |

Conclusion:

- This is the new best measured local branch.
- It improves over the previous best by `0.00006055652457054306` absolute `nDCG@10`.
- It improves `Recall@100` by `0.004910951256838991` versus the previous best.
- It remains `0.10375158653734562` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.018851586537345644` absolute `nDCG@10` below the official BEIR comparator.
- The improvement is real but tiny; SOTA still requires a stronger English retriever or real reranker.

## 2026-06-08 - Dev-Calibrated Fusion Weight Ablation

Rationale:

- NFCorpus includes `qrels/dev.tsv`, so fusion weights can be selected without using test qrels.
- The calibration branch uses the current best four source signals: BGE-small ONNX CLS pooling, BGE-small ONNX mean pooling, ONNX MiniLM, and BM25.
- It searches a fixed grid on dev and evaluates the selected weights once on test.

Code updated:

- `src/turboragger/calibration.py`
- `src/turboragger/data.py`
- `tests/test_calibration.py`
- `tests/test_candidate_runner.py`
- `scripts/run_nfcorpus_candidate.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_score_fusion_20260608T155633Z.json`

Calibration:

- Split: `dev`
- Queries: 324
- Grid size: 624
- Selected weights:
  - `bge_small_cls_onnx = 0.5`
  - `bge_small_mean_onnx = 2.0`
  - `xenova_minilm_onnx = 1.5`
  - `bm25 = 1.0`
- Dev metrics: `nDCG@10 = 0.34526833215967884`, `Recall@100 = 0.3208808336328131`

Test result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_score_fusion` | `0.3648389532581644` | `0.32369539341221953` | `341.703909` | 323 | 0 |

Conclusion:

- Dev-calibrated fusion is not promoted because it reduces the primary test metric.
- It loses `0.001309460204489965` absolute `nDCG@10` versus the current best branch.
- It improves `Recall@100` by `0.0025090215374421465` versus the current best branch.
- It remains `0.10506104674183558` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.02016104674183561` absolute `nDCG@10` below the official BEIR comparator.
- This validates the no-test-leak calibration path, but the available source signals are still too weak for SOTA.

## 2026-06-08 - Title-Only BM25 Fusion Ablation

Rationale:

- NFCorpus corpus records include title and text fields.
- Title-only BM25 is a new non-leaking local lexical signal that might complement dense and full-text BM25 retrieval.
- This branch adds title-only BM25 to the current best equal-weight fusion recipe.

Code updated:

- `src/turboragger/lexical.py`
- `tests/test_lexical.py`
- `tests/test_candidate_runner.py`
- `scripts/run_nfcorpus_candidate.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_title_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_title_score_fusion_20260608T160744Z.json`

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_dual_pool_xenova_minilm_bm25_title_score_fusion` | `0.3568054177005105` | `0.312460387549029` | `350.379514` | 323 | 0 |

Conclusion:

- Title-only BM25 fusion is not promoted because it reduces both primary and secondary metrics.
- It loses `0.009342995762143869` absolute `nDCG@10` versus the current best branch.
- It loses `0.008725984325748393` `Recall@100` versus the current best branch.
- It remains `0.11309458229948949` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.028194582299489512` absolute `nDCG@10` below the official BEIR comparator.

## 2026-06-08 - Text-Only BM25 Fusion Ablation

Rationale:

- Text-only BM25 is the counterpart to the title-only branch and isolates the body-text lexical signal from the full title+text BM25 branch.
- This branch adds text-only BM25 to the current best equal-weight fusion recipe without using test qrels for tuning.

Code updated:

- `tests/test_candidate_runner.py`
- `scripts/run_nfcorpus_candidate.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_text_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_text_score_fusion_20260608T162004Z.json`

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_dual_pool_xenova_minilm_bm25_text_score_fusion` | `0.36448715421857897` | `0.3152911085718485` | `399.920804` | 323 | 0 |

Conclusion:

- Text-only BM25 fusion is not promoted because it reduces both primary and secondary metrics.
- It loses `0.0016612592440753962` absolute `nDCG@10` versus the current best branch.
- It loses `0.005895263302928886` `Recall@100` versus the current best branch.
- It remains `0.10541284578142102` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.020512845781421052` absolute `nDCG@10` below the official BEIR comparator.

## 2026-06-08 - Field-Aware Dev-Calibrated BM25 Fusion

Rationale:

- Equal-weight title-only and text-only lexical branches underperformed, but dev-calibrated weights might safely preserve useful field-specific lexical signal without using test qrels.
- This branch calibrates over BGE-small CLS, BGE-small mean, ONNX MiniLM, full BM25, title-only BM25, and text-only BM25 using NFCorpus `dev` qrels only.

Code updated:

- `tests/test_candidate_runner.py`
- `scripts/run_nfcorpus_candidate.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_fields_dev_calibrated_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_fields_dev_calibrated_score_fusion_20260608T163215Z.json`

Calibration:

- Split: `dev`
- Queries: 324
- Grid size: 15624
- Selected weights:
  - `bge_small_cls_onnx = 0.5`
  - `bge_small_mean_onnx = 2.0`
  - `xenova_minilm_onnx = 1.5`
  - `bm25 = 0.5`
  - `bm25_title = 0.0`
  - `bm25_text = 0.5`
- Dev metrics: `nDCG@10 = 0.3455687342182085`, `Recall@100 = 0.3208251731052093`

Test result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_dual_pool_xenova_minilm_bm25_fields_dev_calibrated_score_fusion` | `0.36473218877158997` | `0.32396224357142317` | `1308.000998` | 323 | 0 |

Conclusion:

- Field-aware dev calibration is not promoted because it reduces the primary test metric.
- It loses `0.0014162246910643995` absolute `nDCG@10` versus the current best branch.
- It improves `Recall@100` by `0.00277587169664577` versus the current best branch.
- It remains `0.10516781122841002` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.020267811228410053` absolute `nDCG@10` below the official BEIR comparator.
- The exhaustive calibration runtime was high for a rejected branch, so future calibration should be faster or tied to a stronger signal source.

## 2026-06-08 - Broad Local Model Inventory And BCE Embedding Benchmark

Rationale:

- After local fusion and calibration branches failed to reach SOTA, the next required gate was stronger model availability.
- A broader bounded local inventory found one unmeasured SOTA-relevant embedding candidate: `maidalun1020/bce-embedding-base_v1`.
- The paired `bce-reranker-base_v1` was not found under obvious local paths.

Code updated:

- `src/turboragger/local_model_inventory.py`
- `scripts/probe_local_model_inventory.py`
- `src/turboragger/dense.py`
- `scripts/run_nfcorpus_candidate.py`
- `tests/test_local_model_inventory.py`
- `tests/test_dense.py`
- `tests/test_candidate_runner.py`

Inventory command:

```bash
PYTHONPATH=src python3 scripts/probe_local_model_inventory.py --timeout-seconds 300
```

Inventory artifact:

- `artifacts/local_model_inventory.json`

Final inventory summary:

- Model files found: 59
- SOTA-relevant local candidates found: 1
- Unmeasured SOTA candidates after BCE benchmark: 0
- Newly discovered model path: `/Volumes/WS4TB/WS4TBr/aP2A/ragflow/huggingface.co/maidalun1020/bce-embedding-base_v1`

BCE benchmark command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bce_embedding_base_v1
```

BCE artifact:

- `artifacts/runs/bce_embedding_base_v1_20260608T171437Z.json`

BCE result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bce_embedding_base_v1` | `0.2621479854747279` | `0.27654921213670386` | `226.425135` | 323 | 0 |

Conclusion:

- BCE is not promoted because it reduces both primary and secondary metrics.
- It loses `0.10400042798792647` absolute `nDCG@10` versus the current best branch.
- It loses `0.04463715973807353` `Recall@100` versus the current best branch.
- It remains `0.2077520145252721` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.12285201452527211` absolute `nDCG@10` below the official BEIR comparator.
- The bounded local inventory now has no unmeasured SOTA-relevant retriever/reranker candidates left; further SOTA progress needs a new approved model source.

## 2026-06-08 - Rank-Score Hybrid Fusion Ablation

Rationale:

- Current score fusion and RRF capture different evidence: normalized branch scores and rank agreement.
- This branch adds a fixed RRF-style rank bonus to the current best min-max score-fusion source set without using qrels to tune the rank weight.

Code updated:

- `src/turboragger/score_fusion.py`
- `scripts/run_nfcorpus_candidate.py`
- `tests/test_score_fusion.py`
- `tests/test_candidate_runner.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion_20260608T173359Z.json`

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion` | `0.3659607648002612` | `0.3220841618662724` | `383.576669` | 323 | 0 |

Conclusion:

- Rank-score fusion is not promoted because it reduces the primary test metric.
- It loses `0.00018764866239318057` absolute `nDCG@10` versus the current best branch.
- It improves `Recall@100` by `0.0008977899914950349` versus the current best branch.
- It remains `0.1039392351997388` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.019039235199738824` absolute `nDCG@10` below the official BEIR comparator.

## 2026-06-08 - Late-Interaction Rerank Ablation

Rationale:

- `GOAL.md` identifies late interaction/reranking as a distinct next retrieval-family direction.
- No complete local cross-encoder/reranker is available, so this branch tests whether BGE-small ONNX token-level MaxSim can improve top-10 ordering over the current local score-fusion candidate pool.

Code path:

- `src/turboragger/late_interaction.py`
- `scripts/run_nfcorpus_candidate.py`

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_late_interaction_score_fusion_rerank
```

Artifact:

- `artifacts/runs/bge_small_late_interaction_score_fusion_rerank_20260608T195336Z.json`

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_late_interaction_score_fusion_rerank` | `0.36006186813204677` | `0.3211863718747774` | `342.137017` | 323 | 0 |

Conclusion:

- The branch is not promoted because it reduces both primary and secondary metrics.
- It loses `0.007521174575898848` absolute `nDCG@10` versus the current best branch.
- It loses `0.011692210829001792` `Recall@100` versus the current best branch.
- It remains `0.10983813186795321` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.02493813186795324` absolute `nDCG@10` below the official BEIR comparator.
- Late interaction remains a plausible methodology family, but this measured BGE-small token-MaxSim version is not the missing SOTA signal.

## 2026-06-08 - Current-Best Plus Late-Interaction Dev Score Fusion

Rationale:

- The standalone late-interaction branch failed, but it could still contain complementary ranking signal.
- This branch dev-calibrates the current best graded-regression score-fusion anchor against the BGE-small token-MaxSim late-interaction branch, using NFCorpus `dev` only for the second-stage fusion weights.

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_late_interaction_gbdt_regression_dev_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_late_interaction_gbdt_regression_dev_score_fusion_20260608T200608Z.json`

Dev calibration:

- Selected weights: `current_best_primary = 0.5`, `late_interaction_secondary = 2.0`.
- Dev metrics: `nDCG@10 = 0.3506070294397712`, `Recall@100 = 0.319279558808475`.

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_late_interaction_gbdt_regression_dev_score_fusion` | `0.36542794274338575` | `0.32149384348966625` | `717.679263` | 323 | 0 |

Conclusion:

- The branch is not promoted because it reduces both primary and secondary metrics.
- It loses `0.00215509996455987` absolute `nDCG@10` versus the current best branch.
- It loses `0.011384739214112927` `Recall@100` versus the current best branch.
- It remains `0.10447205725661424` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.01957205725661426` absolute `nDCG@10` below the official BEIR comparator.
- Current-source token MaxSim appears exhausted as a SOTA-moving path unless paired with a stronger reranker model or different token-level training signal.

## 2026-06-08 - Deep-Pool Graded-Regression Dev Score Fusion

Rationale:

- The current best uses graded GBDT regression with `branch_k = 100`.
- A previous deep-pool classifier branch improved recall but hurt `nDCG@10`; this branch tests whether deeper candidate exposure helps the graded-regression variant.

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_regression_dev_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_regression_dev_score_fusion_20260608T202346Z.json`

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_regression_dev_score_fusion` | `0.363902495492805` | `0.33490230529221543` | `363.911605` | 323 | 0 |

Conclusion:

- The branch is not promoted because it reduces the primary metric.
- It loses `0.003680547215140606` absolute `nDCG@10` versus the current best branch.
- It improves `Recall@100` by `0.0020237225884362497` versus the current best branch.
- It remains `0.10599750450719497` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.021097504507194997` absolute `nDCG@10` below the official BEIR comparator.
- Deeper same-source candidate pools are useful for recall evidence but still degrade top-10 ordering under the current local source set.

## 2026-06-08 - SciFact-Finetuned MiniLM Dense Branch

Rationale:

- A bounded local inventory listed uncategorized SentenceTransformer-style checkpoints under `CPfrac/cam-rag-platform/output/scifact-finetuned`.
- The root checkpoint is complete and loads through the existing guarded direct-`transformers` dense retriever.
- This branch tests whether a local domain-supervised MiniLM checkpoint fine-tuned on SciFact-style biomedical pairs transfers to NFCorpus without network access, credentials, or test-qrels tuning.

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate scifact_finetuned_minilm
```

Artifact:

- `artifacts/runs/scifact_finetuned_minilm_20260608T203922Z.json`

Result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `scifact_finetuned_minilm` | `0.3113948750306552` | `0.30306730089962974` | `20.385284` | 323 | 0 |

Conclusion:

- The branch is not promoted because it reduces both primary and secondary metrics.
- It loses `0.0561881676772904` absolute `nDCG@10` versus the current best branch.
- It loses `0.029811281804149437` `Recall@100` versus the current best branch.
- It is also `0.004606342771565408` absolute `nDCG@10` below the direct MiniLM baseline.
- It remains `0.15850512496934477` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.07360512496934479` absolute `nDCG@10` below the official BEIR comparator.
- Tiny local SciFact-style fine-tuning is not a SOTA path for NFCorpus under this artifact surface; future fine-tuning must use a stronger base model, larger domain data, or a reranker objective.

## 2026-06-08 - Dev-Selected SciFact-Finetuned MiniLM Checkpoint

Rationale:

- The root SciFact-finetuned MiniLM checkpoint failed, but five intermediate checkpoints are also complete local SentenceTransformer snapshots.
- Selecting among those checkpoints by NFCorpus `test` would be leakage, so this branch selects by NFCorpus `dev` `nDCG@10` only, then evaluates the selected checkpoint on `test`.

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate scifact_dev_selected_minilm
```

Artifact:

- `artifacts/runs/scifact_dev_selected_minilm_20260608T205002Z.json`

Dev selection:

| Checkpoint | Dev nDCG@10 | Dev Recall@100 |
|---|---:|---:|
| `checkpoint-9` | `0.30328459075499065` | `0.30134690653738905` |
| `checkpoint-18` | `0.3019942589158048` | `0.3009931117316315` |
| `checkpoint-36` | `0.30050877866450537` | `0.3011734211067443` |
| `checkpoint-45` | `0.30007330269955695` | `0.29974749301213877` |
| `final` | `0.30007330269955695` | `0.29974749301213877` |
| `checkpoint-27` | `0.2997546758359658` | `0.3019637733442272` |

Selected checkpoint:

- `checkpoint-9`
- `/Volumes/WS4TB/WS4TBr/CPfrac/cam-rag-platform/output/scifact-finetuned/checkpoint-9`

Test result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `scifact_dev_selected_minilm` | `0.316500938704734` | `0.31019343221102363` | `115.023363` | 323 | 0 |

Conclusion:

- The branch is not promoted because it is far below the current best.
- It loses `0.051082104003211615` absolute `nDCG@10` versus the current best branch.
- It loses `0.022685150492755546` `Recall@100` versus the current best branch.
- It improves direct MiniLM by only `0.0004997209025133786` absolute `nDCG@10`.
- It improves the final SciFact checkpoint by `0.005106063674078787` absolute `nDCG@10`, validating the dev-selection guard but not creating a SOTA path.
- It remains `0.15339906129526598` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.068499061295266` absolute `nDCG@10` below the official BEIR comparator.

## 2026-06-08 - Current-Best Plus Dev-Selected SciFact Score Fusion

Rationale:

- The standalone dev-selected SciFact checkpoint branch was weak, but it could still have complementary signal against the current best learned fusion anchor.
- This branch dev-calibrates the current graded-regression score-fusion anchor against the dev-selected SciFact checkpoint branch without using NFCorpus `test` qrels for training, checkpoint selection, or weight calibration.

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_gbdt_regression_scifact_dev_selected_dev_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_gbdt_regression_scifact_dev_selected_dev_score_fusion_20260608T210404Z.json`

Dev calibration:

| Branch | Selected weight |
|---|---:|
| `current_best_primary` | `0.5` |
| `scifact_secondary` | `0.0` |

Calibration metrics:

- Dev `nDCG@10 = 0.34652923156548504`
- Dev `Recall@100 = 0.32345304232384053`

SciFact selected checkpoint:

- `checkpoint-9`
- Dev `nDCG@10 = 0.30328459075499065`
- Dev `Recall@100 = 0.30134690653738905`

Test result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_gbdt_regression_scifact_dev_selected_dev_score_fusion` | `0.3675830427079456` | `0.33050103876645803` | `480.304969` | 323 | 0 |

Conclusion:

- The branch is not promoted because it ties the current best on primary `nDCG@10` and reduces `Recall@100`.
- It changes primary `nDCG@10` by `0.0` versus the current best.
- It loses `0.002377543937321147` `Recall@100` versus the current best.
- It remains `0.10231695729205437` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.01741695729205439` absolute `nDCG@10` below the official BEIR comparator.
- Dev calibration assigned the SciFact secondary branch weight `0.0`, which is direct evidence that this tiny fine-tuned checkpoint family is not contributing useful complementary signal to the current anchor.

## 2026-06-08 - BGE-Small Dense Pseudo-Relevance Feedback

Rationale:

- Broader read-only discovery did not find a complete stronger local English retriever or reranker beyond already measured candidates.
- Since model acquisition was not available in-session, this branch tests a distinct qrels-free query-expansion method: Rocchio-style dense pseudo-relevance feedback over BGE-small ONNX top documents.
- The method uses the initial dense top-10 document vectors to shift the query vector before final retrieval; NFCorpus `test` qrels are used only for final evaluation.

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_en_onnx_dense_prf
```

Artifact:

- `artifacts/runs/bge_small_en_onnx_dense_prf_20260608T212303Z.json`

Configuration:

| Parameter | Value |
|---|---:|
| `feedback_docs` | `10` |
| `query_weight` | `1.0` |
| `feedback_weight` | `0.5` |

Test result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_en_onnx_dense_prf` | `0.3425690018844359` | `0.3105355971312376` | `144.642333` | 323 | 0 |

Conclusion:

- The branch is not promoted because it remains well below the current best.
- It improves direct `bge_small_en_onnx` by `0.0003240370649028601` absolute `nDCG@10`, but loses `0.0015122627919349485` `Recall@100` versus direct BGE-small.
- It loses `0.02501404082350972` absolute `nDCG@10` and `0.022342985572541607` `Recall@100` versus the current best branch.
- It remains `0.1273309981155641` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.04243099811556411` absolute `nDCG@10` below the official BEIR comparator.
- Dense PRF gives a tiny direct-BGE ranking gain but not a SOTA-moving signal; further query expansion should involve a stronger generator/retriever or train/dev-selected parameters rather than more fixed PRF over the same weak model.

## 2026-06-08 - Dev-Selected BGE-Small Dense Pseudo-Relevance Feedback

Rationale:

- Fixed dense PRF improved direct BGE-small by only `0.0003240370649028601` `nDCG@10`.
- This branch keeps test qrels untouched and selects dense PRF feedback parameters on NFCorpus `dev`, then evaluates the selected configuration on `test`.

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_en_onnx_dense_prf_dev_selected
```

Artifact:

- `artifacts/runs/bge_small_en_onnx_dense_prf_dev_selected_20260608T213136Z.json`

Dev selection:

| Rank | feedback_docs | feedback_weight | Dev nDCG@10 | Dev Recall@100 |
|---:|---:|---:|---:|---:|
| 1 | `3` | `0.5` | `0.3307341905658255` | `0.3191282770446407` |
| 2 | `3` | `0.25` | `0.32840166288952555` | `0.3184996922046842` |
| 3 | `5` | `0.25` | `0.3281405592255335` | `0.3188856761217982` |
| 4 | `3` | `1.0` | `0.32741001743642584` | `0.3233710542646078` |
| 5 | `5` | `0.5` | `0.32518496541432734` | `0.32128918709614923` |

Selected parameters:

- `feedback_docs = 3`
- `query_weight = 1.0`
- `feedback_weight = 0.5`
- Candidate grid size: 20

Test result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_en_onnx_dense_prf_dev_selected` | `0.34933673642384316` | `0.3180000296586443` | `150.554259` | 323 | 0 |

Conclusion:

- The branch is not promoted because it remains below the current best.
- It improves fixed dense PRF by `0.006767734539407266` `nDCG@10` and `0.00746443252740675` `Recall@100`.
- It improves direct BGE-small ONNX by `0.007091771604310126` `nDCG@10` and `0.005952169735471802` `Recall@100`.
- It loses `0.018246306284102454` absolute `nDCG@10` and `0.014878553045134857` `Recall@100` versus the current best branch.
- It remains `0.12056326357615682` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.035663263576156845` absolute `nDCG@10` below the official BEIR comparator.
- Dev-selected dense PRF is a legitimate query-expansion improvement over direct BGE-small, but it still does not replace the need for a stronger retriever/reranker.

## 2026-06-08 - Current-Best Plus Dev-Selected Dense PRF Score Fusion

Rationale:

- Dev-selected dense PRF improved direct BGE-small ONNX by `0.007091771604310126` `nDCG@10`, but the standalone branch remained below the graded-regression anchor.
- This branch tests whether the dev-selected PRF signal is complementary when fused against the current best graded-regression score-fusion anchor.
- The branch preserves split hygiene: train qrels fit the graded-regression anchor, dev qrels select dense PRF parameters and second-stage fusion weights, and test qrels are used only for final evaluation.

Command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_small_gbdt_regression_dense_prf_dev_selected_dev_score_fusion
```

Artifact:

- `artifacts/runs/bge_small_gbdt_regression_dense_prf_dev_selected_dev_score_fusion_20260608T214206Z.json`

Dev selection and calibration:

| Item | Value |
|---|---:|
| Dense PRF `feedback_docs` | `3` |
| Dense PRF `query_weight` | `1.0` |
| Dense PRF `feedback_weight` | `0.5` |
| Dense PRF dev `nDCG@10` | `0.3307341905658255` |
| Dense PRF dev `Recall@100` | `0.3191282770446407` |
| Second-stage `current_best_primary` | `2.0` |
| Second-stage `dense_prf_secondary` | `0.5` |
| Second-stage dev `nDCG@10` | `0.34775638278253135` |
| Second-stage dev `Recall@100` | `0.33530330911288864` |

Test result:

| Candidate | nDCG@10 | Recall@100 | Runtime seconds | Queries | Failures |
|---|---:|---:|---:|---:|---:|
| `bge_small_gbdt_regression_dense_prf_dev_selected_dev_score_fusion` | `0.369215383024794` | `0.33559899868294146` | `507.268217` | 323 | 0 |

Conclusion:

- The branch is promoted as the new current best.
- It improves the previous best by `0.0016323403168483908` absolute `nDCG@10`.
- It improves the previous best by `0.002720415979162283` `Recall@100`.
- It remains `0.10068461697520598` absolute `nDCG@10` below the selected SOTA target.
- It remains `0.015784616975206` absolute `nDCG@10` below the official BEIR comparator.
- Dense PRF is useful as a complement to the graded-regression anchor, but the remaining gap is still too large for more same-source fusion to be the primary SOTA path.
