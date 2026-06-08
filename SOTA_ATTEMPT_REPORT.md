# SOTA Attempt Report

Generated: 2026-06-08

## Verdict

SOTA is not achieved under the current local/offline model set.

Primary target:

- Benchmark: BEIR NFCorpus `test`
- Metric: `nDCG@10`
- Target: `0.4699`
- Target artifact: `artifacts/sota_target.json`

Best current local result:

- Candidate: `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion`
- Artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion_20260608T192209Z.json`
- `nDCG@10`: `0.3675830427079456`
- `Recall@100`: `0.3328785827037792`
- Queries: 323
- Failures: 0
- Gap to SOTA target: `0.10231695729205437` absolute `nDCG@10`
- Gap to official BEIR comparator `0.385`: `0.01741695729205439` absolute `nDCG@10`

## What Worked

- The benchmark harness is now reproducible and artifact-backed.
- The current best local method improves the direct MiniLM baseline from `0.3160012178022206` to `0.3675830427079456` `nDCG@10`.
- The train/dev GBDT score-fusion branch improved the previous local best by `0.000946684274344356` absolute `nDCG@10` and improved `Recall@100` by `0.0119932252058558` without training or calibrating on test qrels.
- The historical `newragcity` `0.5085946124009167` claim was audited and rejected as invalid evidence.
- Local ONNX BGE-small plus ONNX MiniLM plus BM25 is the strongest available local source set found in this workspace.

## What Did Not Close The Gap

Measured but rejected branches include:

- Direct MiniLM baseline: `0.3160012178022206`
- BM25 baseline: `0.3064137649277972`
- MiniLM + BM25 RRF: `0.3344802256991052`
- BGE-large-zh fallback: `0.14143051730446513`
- BGE-small ONNX dense: `0.34224496481953304`
- BGE-small ONNX dense pseudo-relevance feedback: `0.3425690018844359`
- BGE-small + MiniLM + BM25 RRF: `0.3563276091075661`
- BM25 pseudo-relevance feedback: `0.2643284676291605`
- BGE-small + MiniLM + BM25 PRF RRF: `0.34788409229565553`
- BGE-small + direct MiniLM + BM25 score fusion: `0.3622186429303806`
- Xenova MiniLM ONNX dense: `0.31689265638802383`
- BGE-small + Xenova MiniLM + BM25 score fusion: `0.3660878569380838`
- BGE-small dual-pool + Xenova MiniLM + BM25 rank-score fusion: `0.3659607648002612`
- BGE-small dual-pool + Xenova MiniLM + BM25 dev-calibrated rank-score fusion: `0.36566973749080495`
- BGE-small dual-pool + Xenova MiniLM + BM25 train-split learned feature fusion: `0.35887473712153073`
- BGE-small dual-pool + Xenova MiniLM + BM25 train-split GBDT feature fusion: `0.36273469704993305`
- BGE-small dual-pool + Xenova MiniLM + BM25 train/dev GBDT cascade: `0.36218572616170835`
- BGE-small dual-pool + Xenova MiniLM + BM25 train/dev GBDT score fusion: `0.3670950977369987` then-current best, now superseded by graded-regression score fusion
- BGE-small dual-pool + Xenova MiniLM + BM25 train/dev five-source GBDT calibrated score fusion: `0.3660561464830782`
- BGE-small dual-pool + Xenova MiniLM + BM25 deep-pool train/dev GBDT score fusion: `0.36562875673411727`
- BGE-small dual-pool + Xenova MiniLM + BM25 train/dev GBDT regression score fusion: `0.3675830427079456` current best, still below SOTA
- BGE-small dual-pool + Xenova MiniLM + BM25 deep-pool train/dev GBDT regression score fusion: `0.363902495492805`
- BGE-small dual-pool + Xenova MiniLM + BM25 direct train/dev GBDT regression fusion: `0.36434599974495163`
- BGE-small late-interaction MaxSim rerank over score-fusion pool: `0.36006186813204677`
- BGE-small late-interaction plus current-best GBDT regression dev score fusion: `0.36542794274338575`
- Dev-selected SciFact-finetuned MiniLM checkpoint branch: `0.316500938704734`
- Current-best plus dev-selected SciFact checkpoint dev score fusion: `0.3675830427079456` tied current best but lost recall and selected `scifact_secondary = 0.0`
- SciFact-finetuned MiniLM dense branch: `0.3113948750306552`
- Corrected LEANN MiniLM no-recompute: `0.3138633685494098`
- Dual-MiniLM score fusion: `0.3559227197172415`
- CombMNZ agreement fusion: `0.3594854251733051`
- Deep branch-pool score fusion: `0.3621651141202009`
- BGE mean-pooling fusion: `0.3634426387237848`
- Dev-calibrated current-source fusion: `0.3648389532581644`
- Title-only BM25 field fusion: `0.3568054177005105`
- Text-only BM25 field fusion: `0.36448715421857897`
- Field-aware dev-calibrated BM25 fusion: `0.36473218877158997`
- BCE embedding dense branch: `0.2621479854747279`

## Root Causes

- No complete stronger English embedding model is locally available for the preferred SOTA-moving candidates such as BGE-M3, Qwen3-Embedding, Nomic, E5, or GTE.
- The broader local inventory found `maidalun1020/bce-embedding-base_v1`, but the benchmark scored only `0.2621479854747279` `nDCG@10`.
- A complete local SciFact-style fine-tuned MiniLM checkpoint was discovered and benchmarked, but it scored only `0.3113948750306552` `nDCG@10`, below the direct MiniLM baseline.
- Dev-selecting among the local SciFact-style MiniLM checkpoint family improved that branch to `0.316500938704734` `nDCG@10`, but it remains MiniLM-level and far below the current best.
- Dev-calibrating the current best against the dev-selected SciFact checkpoint branch assigned the SciFact secondary branch weight `0.0`, tied current-best `nDCG@10`, and reduced `Recall@100`, so the checkpoint family also lacks useful complementary fusion signal.
- No complete local reranker model is available for the checked MiniCPM, BGE reranker, or cross-encoder candidates.
- No obvious local `bce-reranker-base_v1` path was found.
- Fusion, rank-score fusion, dev-calibrated rank-score fusion, supervised feature fusion, nonlinear supervised feature fusion, learned cascades, train/dev two-ranker score fusion, five-source GBDT calibration, deep-pool GBDT fusion, deep-pool GBDT regression, current-source token MaxSim late interaction, second-stage late-interaction calibration, tiny SciFact-style fine-tuning, dev-selected tiny SciFact checkpoints, pooling, lexical-field, depth, and no-test-leak calibration changes extract small gains from the current weak source set but do not create the missing semantic ranking signal.
- Dense pseudo-relevance feedback improves direct BGE-small `nDCG@10` by only `0.0003240370649028601` and reduces recall, so fixed PRF over the same dense model is not a SOTA-moving query-expansion path.
- The best learned branch proves train/dev supervision can recover a small local gain, but the remaining `0.10231695729205437` `nDCG@10` gap is too large for same-source fusion alone to be a credible SOTA path.
- The only large local fallback found, `BAAI/bge-large-zh-v1.5`, is a poor English NFCorpus fit.
- Exhaustive five-source dev calibration is expensive and still misses the primary metric.

## Recommended Scope Change

To continue toward SOTA, make one stronger English retrieval or reranking model available locally, then benchmark it through the existing artifact schema before adding more fusion changes.

Priority order:

1. Approved local model acquisition: `BAAI/bge-m3`, Qwen3-Embedding, Nomic, E5/GTE, or another strong English retrieval model.
2. Approved local reranker acquisition: BGE reranker, Qwen3 reranker, MiniCPM reranker, or a small cross-encoder reranker.
3. If network/API use is approved, run a bounded external-model comparison only as a target sanity check, not as a silent replacement for the local methodology.
4. After a stronger model improves the candidate pool, revisit dev-calibrated fusion and reranking.

## Verification

Latest verification commands:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src scripts
python3 -m json.tool artifacts/runs/bce_embedding_base_v1_20260608T171437Z.json >/tmp/bce_embedding.pretty
python3 -m json.tool artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion_20260608T173359Z.json >/tmp/rank_score_fusion.pretty
python3 -m json.tool artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_rank_score_fusion_20260608T174836Z.json >/tmp/dev_calibrated_rank_score_fusion.pretty
python3 -m json.tool artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_learned_fusion_20260608T180407Z.json >/tmp/train_dev_learned_fusion.pretty
python3 -m json.tool artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion_20260608T181638Z.json >/tmp/train_dev_gbdt_fusion.pretty
python3 -m json.tool artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_cascade_20260608T183008Z.json >/tmp/train_dev_gbdt_dev_cascade.pretty
python3 -m json.tool artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion_20260608T184157Z.json >/tmp/train_dev_gbdt_dev_score_fusion.pretty
python3 -m json.tool artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_calibrated_score_fusion_20260608T185555Z.json >/tmp/train_dev_gbdt_dev_calibrated_score_fusion.pretty
python3 -m json.tool artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_dev_score_fusion_20260608T190950Z.json >/tmp/deep_train_dev_gbdt_dev_score_fusion.pretty
python3 -m json.tool artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion_20260608T192209Z.json >/tmp/train_dev_gbdt_regression_dev_score_fusion.pretty
python3 -m json.tool artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_fusion_20260608T193443Z.json >/tmp/train_dev_gbdt_regression_fusion.pretty
python3 -m json.tool artifacts/runs/bge_small_late_interaction_score_fusion_rerank_20260608T195336Z.json >/tmp/late_interaction.pretty
python3 -m json.tool artifacts/runs/bge_small_late_interaction_gbdt_regression_dev_score_fusion_20260608T200608Z.json >/tmp/late_gbdt.pretty
python3 -m json.tool artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_regression_dev_score_fusion_20260608T202346Z.json >/tmp/deep_train_dev_gbdt_regression_dev_score_fusion.pretty
python3 -m json.tool artifacts/runs/scifact_finetuned_minilm_20260608T203922Z.json >/tmp/scifact_finetuned_minilm.pretty
python3 -m json.tool artifacts/runs/scifact_dev_selected_minilm_20260608T205002Z.json >/tmp/scifact_dev_selected_minilm.pretty
python3 -m json.tool artifacts/runs/bge_small_gbdt_regression_scifact_dev_selected_dev_score_fusion_20260608T210404Z.json >/tmp/current_best_scifact_dev_score_fusion.pretty
python3 -m json.tool artifacts/runs/bge_small_en_onnx_dense_prf_20260608T212303Z.json >/tmp/bge_dense_prf.pretty
python3 -m json.tool artifacts/local_model_inventory.json >/tmp/local_model_inventory.pretty
python3 -m json.tool artifacts/leaderboard.json >/tmp/leaderboard.pretty
python3 -m json.tool artifacts/sota_target.json >/tmp/sota_target.pretty
```

Observed outcomes:

- Unit tests passed: 86 tests, 0 failures.
- Compileall exited 0.
- JSON validation exited 0 for the latest BCE run, rank-score fusion run, dev-calibrated rank-score fusion run, train-split learned feature fusion run, train-split GBDT feature fusion run, train/dev GBDT cascade run, train/dev GBDT score-fusion run, train/dev GBDT five-source calibrated score-fusion run, deep-pool train/dev GBDT score-fusion run, train/dev GBDT regression score-fusion run, direct train/dev GBDT regression run, late-interaction rerank run, late-interaction plus current-best GBDT regression dev score-fusion run, deep-pool train/dev GBDT regression score-fusion run, SciFact-finetuned MiniLM run, dev-selected SciFact-finetuned MiniLM run, current-best plus dev-selected SciFact score-fusion run, dense PRF run, local model inventory, leaderboard, and SOTA target.
- Root `turboragger` now has sandbox-limited git metadata; root `git diff --check` exits 0, but root files remain untracked locally because `.git` metadata writes are restricted in this environment. The GitHub mirror was pushed through the documented `/private/tmp` staging repo workaround.
- Child repo diff checks exited 0; `turbovec` still has only the known untracked `.DS_Store`.
