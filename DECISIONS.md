# DECISIONS.md

## 2026-06-08 - Add Late-Interaction Rerank Candidate Without Promoting It

Decision: add `bge_small_late_interaction_score_fusion_rerank` as a runnable candidate branch, but do not promote it until it has a saved NFCorpus test artifact.

Reasons:

- The current `GOAL.md` next directions prioritize a stronger ranking signal over more same-source calibration.
- No complete stronger local English retriever or reranker is currently available.
- Token-level MaxSim over the existing BGE-small ONNX model is a distinct local reranking/late-interaction ablation that can be benchmarked without credentials or test-qrels tuning.
- The branch reranks the existing BGE-small dual-pool + Xenova MiniLM + BM25 score-fusion pool, so it tests top-10 ordering rather than adding another plain weighted fusion.

Consequences:

- New module: `src/turboragger/late_interaction.py`.
- New candidate: `bge_small_late_interaction_score_fusion_rerank`.
- Verification so far is test and compile coverage only; no `nDCG@10` movement can be claimed until the benchmark command writes a run artifact.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion`.

## 2026-06-08 - Reject Direct Train/Dev GBDT Regression Feature Fusion

Decision: keep `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_fusion` as a valid graded-regression ablation, but do not promote it.

Reasons:

- The branch trained a local `HistGradientBoostingRegressor` on NFCorpus `train` only, using graded relevance targets.
- The branch evaluated the regressor directly as the final ranker instead of dev-calibrating it against the score-fusion primary ranker.
- Test scoring produced `nDCG@10 = 0.36434599974495163`, below the current best `0.3675830427079456`.
- It also lowered `Recall@100` from `0.3328785827037792` to `0.325698652336419`.

Consequences:

- Run artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_fusion_20260608T193443Z.json`.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion`.
- The graded regressor is useful as a secondary signal inside dev-calibrated score fusion, but not as the standalone final ranker.

## 2026-06-08 - Promote Train/Dev GBDT Regression Score Fusion As Current Best

Decision: promote `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion` as the current best local method, while explicitly not claiming SOTA.

Reasons:

- The branch trained a local `HistGradientBoostingRegressor` on NFCorpus `train` only, using graded relevance targets rather than binary labels.
- The branch selected two-ranker score-fusion weights on NFCorpus `dev` only: `score_fusion_primary = 1.0`, `gbdt_regression_secondary = 1.5`.
- The branch used no NFCorpus `test` qrels for training or calibration.
- Test scoring produced `nDCG@10 = 0.3675830427079456`, above the previous best `0.3670950977369987`.
- It slightly reduced `Recall@100` from `0.3331795970806332` to `0.3328785827037792`, but `GOAL.md` defines `nDCG@10` as the primary SOTA metric.

Consequences:

- New best artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion_20260608T192209Z.json`.
- The absolute primary-metric improvement is `0.00048794497094689637` `nDCG@10`.
- SOTA remains unachieved; the current gap to the selected target `0.4699` is `0.10231695729205437`.
- The strongest local direction is now graded-relevance learned fusion plus any future stronger retriever/reranker source.

## 2026-06-08 - Reject Deep-Pool Train/Dev GBDT Score Fusion

Decision: keep `bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_dev_score_fusion` as a valid no-test-leak depth ablation, but do not promote it.

Reasons:

- The branch trained a local GBDT feature scorer on NFCorpus `train` only.
- The branch selected two-ranker score-fusion weights on NFCorpus `dev` only.
- The branch used `branch_k = 300` to expose deeper same-source candidates before final top-100 fusion.
- Test scoring produced `nDCG@10 = 0.36562875673411727`, below the current best `0.3670950977369987`.
- It improved `Recall@100` from `0.3331795970806332` to `0.33854940883436946`, but `GOAL.md` defines `nDCG@10` as the primary SOTA metric.

Consequences:

- Run artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_dev_score_fusion_20260608T190950Z.json`.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion`.
- Deeper same-source candidate pools are useful for recall but do not improve top-10 ordering under the current branch set.

## 2026-06-08 - Reject Train/Dev GBDT Five-Source Calibration

Decision: keep `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_calibrated_score_fusion` as a valid no-test-leak learned calibration ablation, but do not promote it.

Reasons:

- The branch trained a local GBDT feature scorer on NFCorpus `train` only.
- The branch selected five-source score-fusion weights on NFCorpus `dev` only across BGE CLS, BGE mean, Xenova MiniLM, BM25, and the train-fitted GBDT branch.
- The branch used no NFCorpus `test` qrels for training or calibration.
- Dev calibration selected `bge_small_cls_onnx = 0.5`, `bge_small_mean_onnx = 2.0`, `xenova_minilm_onnx = 0.5`, `bm25 = 0.5`, and `gbdt_feature_fusion = 1.5`.
- Test scoring produced `nDCG@10 = 0.3660561464830782`, below the current best `0.3670950977369987`.
- It also lowered `Recall@100` from `0.3331795970806332` to `0.32611330897080015`.

Consequences:

- Run artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_calibrated_score_fusion_20260608T185555Z.json`.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion`.
- Same-source learned calibration is now further covered; the next SOTA-moving branch should use a stronger retriever, a real local reranker, or a materially different training signal.

## 2026-06-08 - Promote Train/Dev GBDT Score Fusion As Current Best

Decision: promote `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion` as the current best local method, while explicitly not claiming SOTA.

Reasons:

- The branch trained a local GBDT feature scorer on NFCorpus `train` only.
- The branch selected two-ranker score-fusion weights on NFCorpus `dev` only: `score_fusion_primary = 1.5`, `gbdt_secondary = 2.0`.
- The branch used no NFCorpus `test` qrels for training or calibration.
- Test scoring produced `nDCG@10 = 0.3670950977369987`, above the previous best `0.36614841346265437`.
- It also improved `Recall@100` from `0.3211863718747774` to `0.3331795970806332`.

Consequences:

- New best artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion_20260608T184157Z.json`.
- The absolute primary-metric improvement is `0.000946684274344356` `nDCG@10`.
- SOTA remains unachieved; the current gap to the selected target `0.4699` is `0.10280490226300126`.
- The strongest local direction now is not more same-source fusion alone; the two-ranker learned fusion is a better base for future work if a stronger retriever or reranker becomes available.

## 2026-06-08 - Reject Train/Dev GBDT Cascade For Primary Metric

Decision: keep `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_cascade` as a valid no-test-leak cascade ablation, but do not promote it.

Reasons:

- The branch trained a local GBDT feature scorer on NFCorpus `train` only.
- The branch selected cascade `anchor_k = 3` on NFCorpus `dev` qrels only from `[0, 3, 5, 10, 20]`.
- The cascade used current best score fusion as the primary ranker and the train-fitted GBDT branch as the secondary ranker.
- Test scoring produced `nDCG@10 = 0.36218572616170835`, below the current best `0.36614841346265437`.
- It preserved the GBDT branch `Recall@100 = 0.32849490503295947`, but `GOAL.md` defines `nDCG@10` as the primary SOTA metric.

Consequences:

- Run artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_cascade_20260608T183008Z.json`.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- The recall-positive learned branch is useful for downstream reranking evidence, but learned cascades over the same local branch set still do not improve top-10 ordering.

## 2026-06-08 - Reject Train-Split GBDT Feature Fusion For Primary Metric

Decision: keep `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion` as a valid no-test-leak nonlinear supervised ablation, but do not promote it.

Reasons:

- The branch trained a local `sklearn.ensemble.HistGradientBoostingClassifier` on NFCorpus `train` branch outputs only.
- The branch used no NFCorpus `test` qrels for parameter fitting.
- Features were built from the current best source set: per-branch normalized score, reciprocal rank, and branch presence count for BGE-small ONNX CLS pooling, BGE-small ONNX mean pooling, Xenova MiniLM ONNX, and BM25.
- Training generated `629617` candidate rows from `2590` train queries, including `30964` positive rows.
- Test scoring produced `nDCG@10 = 0.36273469704993305`, below the current best `0.36614841346265437`.
- It improved `Recall@100` to `0.32849490503295947`, but `GOAL.md` defines `nDCG@10` as the primary SOTA metric.

Consequences:

- Run artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion_20260608T181638Z.json`.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- The GBDT branch is useful recall evidence, but a nonlinear learned scorer over the same weak branch set still does not supply enough top-10 ordering quality for SOTA.

## 2026-06-08 - Reject Train-Split Learned Feature Fusion For Primary Metric

Decision: keep `bge_small_dual_pool_xenova_minilm_bm25_train_dev_learned_fusion` as a valid no-test-leak supervised ablation, but do not promote it.

Reasons:

- The branch trained a local `sklearn.linear_model.LogisticRegression` feature fusion on NFCorpus `train` branch outputs only.
- The branch used no NFCorpus `test` qrels for parameter fitting.
- Features were built from the current best source set: per-branch normalized score, reciprocal rank, and branch presence count for BGE-small ONNX CLS pooling, BGE-small ONNX mean pooling, Xenova MiniLM ONNX, and BM25.
- Training generated `629617` candidate rows from `2590` train queries, including `30964` positive rows.
- Test scoring produced `nDCG@10 = 0.35887473712153073`, below the current best `0.36614841346265437`.
- It also reduced `Recall@100` to `0.31843407013584935`, below the current best `0.3211863718747774`.

Consequences:

- Run artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_learned_fusion_20260608T180407Z.json`.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- `sklearn` is an observed local dependency for this optional branch, not a new required global dependency for the base harness.
- Supervised score/rank feature fusion over the same source set does not create the missing SOTA-moving ranking signal.

## 2026-06-08 - Reject Dev-Calibrated Rank-Score Fusion For Primary Metric

Decision: keep `bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_rank_score_fusion` as a valid no-test-leak calibration ablation, but do not promote it.

Reasons:

- The branch used NFCorpus `dev` qrels only to select the rank-score fusion parameter, then evaluated once on NFCorpus `test`.
- The branch used the current best source set: BGE-small ONNX CLS pooling, BGE-small ONNX mean pooling, Xenova MiniLM ONNX, and BM25.
- Dev calibration searched `rank_weight_grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]` and selected `rank_weight = 4.0`.
- Test scoring produced `nDCG@10 = 0.36566973749080495`, below the current best `0.36614841346265437`.
- It improved `Recall@100` to `0.32302981358524474`, but `GOAL.md` defines `nDCG@10` as the primary SOTA metric.

Consequences:

- Run artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_rank_score_fusion_20260608T174836Z.json`.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- Rank calibration does not close the SOTA gap; further progress needs a stronger English retriever or reranker rather than more fusion-only tuning over the same weak source set.

## 2026-06-08 - Reject Rank-Score Hybrid Fusion For Primary Metric

Decision: keep `bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion` as a measured no-qrels fusion ablation, but do not promote it.

Reasons:

- The branch used the current best source set and added a fixed RRF-style rank bonus to min-max score fusion.
- The branch used no test-qrels tuning: `rank_weight = 1.0`, `rrf_k = 60`, and equal source weights.
- It scored `nDCG@10 = 0.3659607648002612`, below the current best `0.36614841346265437`.
- It improved `Recall@100` to `0.3220841618662724`, but `GOAL.md` defines `nDCG@10` as the primary SOTA metric.

Consequences:

- Run artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion_20260608T173359Z.json`.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- Rank evidence is useful for recall but does not supply the missing top-10 ranking signal.

## 2026-06-08 - Reject BCE Embedding As A SOTA Path

Decision: benchmark `maidalun1020/bce-embedding-base_v1` as a newly discovered local embedding candidate, but do not promote it.

Reasons:

- Broad local model inventory found a complete BCE embedding snapshot under the sibling ragflow cache.
- The model loads through direct `transformers` as `XLMRobertaModel` and emits 768-dimensional embeddings.
- The local README recommends CLS pooling, so the benchmark used direct-transformers CLS pooling with `max_length = 512`.
- The full NFCorpus test run scored `nDCG@10 = 0.2621479854747279`, below MiniLM, BM25, BGE-small, and the current best fusion.
- It also scored `Recall@100 = 0.27654921213670386`, below the current best `0.3211863718747774`.

Consequences:

- Run artifact: `artifacts/runs/bce_embedding_base_v1_20260608T171437Z.json`.
- Local inventory artifact: `artifacts/local_model_inventory.json`.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- The refreshed inventory has `unmeasured_sota_candidate_count = 0`; no stronger unmeasured local retriever/reranker remains in the bounded scan roots.

## 2026-06-08 - Reject Field-Aware Dev Calibration For Primary Metric

Decision: keep `bge_small_dual_pool_xenova_minilm_bm25_fields_dev_calibrated_score_fusion` as a valid no-test-leak calibration ablation, but do not promote it.

Reasons:

- The branch used NFCorpus `dev` qrels only to calibrate weights over BGE-small CLS, BGE-small mean, ONNX MiniLM, full BM25, title-only BM25, and text-only BM25.
- Dev calibration searched `15624` weight sets and selected `bge_small_cls_onnx = 0.5`, `bge_small_mean_onnx = 2.0`, `xenova_minilm_onnx = 1.5`, `bm25 = 0.5`, `bm25_title = 0.0`, and `bm25_text = 0.5`.
- Test scoring produced `nDCG@10 = 0.36473218877158997`, below the current best `0.36614841346265437`.
- It improved `Recall@100` to `0.32396224357142317`, but `GOAL.md` defines `nDCG@10` as the primary SOTA metric.
- Runtime was `1308.000998` seconds, so exhaustive five-source calibration is costly for a non-promoted result.

Consequences:

- Run artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_fields_dev_calibrated_score_fusion_20260608T163215Z.json`.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- Field-specific lexical sources should not be pursued further without a stronger retriever/reranker or a faster calibration strategy.

## 2026-06-08 - Reject Text-Only BM25 Fusion

Decision: keep `bge_small_dual_pool_xenova_minilm_bm25_text_score_fusion` as a measured lexical-field ablation, but do not promote it.

Reasons:

- Text-only BM25 was the natural counterpart to the title-only branch because NFCorpus records include separate title and text fields.
- The branch scored `nDCG@10 = 0.36448715421857897`, below the current best `0.36614841346265437`.
- It also reduced `Recall@100` from `0.3211863718747774` to `0.3152911085718485`.
- Text-only BM25 is less damaging than title-only BM25, but it still adds no SOTA-moving signal under equal-weight score fusion.

Consequences:

- Run artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_text_score_fusion_20260608T162004Z.json`.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- Field-specific lexical ablations are now covered for title and text; further SOTA progress needs a stronger retriever/reranker rather than more equal-weight BM25 field duplication.

## 2026-06-08 - Reject Title-Only BM25 Fusion

Decision: keep `bge_small_dual_pool_xenova_minilm_bm25_title_score_fusion` as a measured lexical-field ablation, but do not promote it.

Reasons:

- NFCorpus documents have informative title fields, so a title-only BM25 branch was a plausible new local signal.
- The equal-weight fusion branch scored `nDCG@10 = 0.3568054177005105`, below the current best `0.36614841346265437`.
- It also reduced `Recall@100` from `0.3211863718747774` to `0.312460387549029`.
- The title-only branch appears to add lexical noise or overemphasize short-title matches under equal-weight score fusion.

Consequences:

- Run artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_title_score_fusion_20260608T160744Z.json`.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- Future title use should require dev calibration or a reranker, not equal-weight addition.

## 2026-06-08 - Reject Dev-Calibrated Fusion Weights For Primary Metric

Decision: keep `bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_score_fusion` as a valid no-test-leak calibration ablation, but do not promote it as the current best method.

Reasons:

- NFCorpus includes `dev.tsv`, so fixed fusion weights can be selected without using test qrels.
- The branch selected weights on dev only: `bge_small_cls_onnx = 0.5`, `bge_small_mean_onnx = 2.0`, `xenova_minilm_onnx = 1.5`, `bm25 = 1.0`.
- Dev calibration scored `nDCG@10 = 0.34526833215967884`, `Recall@100 = 0.3208808336328131` over 324 dev queries.
- Fixed-weight test evaluation scored `nDCG@10 = 0.3648389532581644`, below the current best `0.36614841346265437`.
- It improved `Recall@100` by `0.0025090215374421465`, but lost `0.001309460204489965` absolute `nDCG@10`.

Consequences:

- Run artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_score_fusion_20260608T155633Z.json`.
- Current best remains `bge_small_dual_pool_xenova_minilm_bm25_score_fusion`.
- Dev-calibrated weights are evidence that calibration is safe to run, but this grid over the current weak source set does not close the SOTA gap.

## 2026-06-08 - Promote Combined BGE CLS+Mean Pooling Fusion As Current Best

Decision: promote `bge_small_dual_pool_xenova_minilm_bm25_score_fusion` as the current best local method, while explicitly not claiming SOTA.

Reasons:

- The branch fuses `Xenova/bge-small-en-v1.5` ONNX twice, once with CLS pooling and once with mean pooling, plus `Xenova/all-MiniLM-L6-v2` ONNX and BM25.
- It scored `nDCG@10 = 0.36614841346265437`, narrowly above the previous best `0.3660878569380838`.
- It improved `Recall@100` from `0.3162754206179384` to `0.3211863718747774`.
- The branch uses equal weights and does not tune on NFCorpus test qrels.

Consequences:

- New best artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_score_fusion_20260608T154445Z.json`.
- The absolute improvement is small: `0.00006055652457054306` `nDCG@10`.
- SOTA remains unachieved; the current gap to the selected target `0.4699` is `0.10375158653734562`.
- Further progress still needs a stronger English retriever or a genuine reranker; pooling fusion alone is not a sufficient SOTA path.

## 2026-06-08 - Reject BGE Mean-Pooling Fusion For Primary Metric

Decision: keep `bge_small_mean_xenova_minilm_bm25_score_fusion` as a measured BGE pooling ablation, but do not promote it as the current best method.

Reasons:

- The branch used the current best local source set except that `Xenova/bge-small-en-v1.5` ONNX used mean pooling instead of CLS pooling.
- The run scored `nDCG@10 = 0.3634426387237848`, below the current best `0.3660878569380838`.
- It improved `Recall@100` by `0.0020452570437043405`, but lost `0.0026452182142990277` absolute `nDCG@10`.
- `GOAL.md` defines `nDCG@10` as the primary SOTA metric.

Consequences:

- Run artifact: `artifacts/runs/bge_small_mean_xenova_minilm_bm25_score_fusion_20260608T153812Z.json`.
- Current best remains `bge_small_xenova_minilm_bm25_score_fusion` with BGE CLS pooling.
- Mean pooling is a useful ablation but not enough to close the SOTA gap; a combined CLS+mean branch may be worth one measured no-qrels test because the score surfaces differ.

## 2026-06-08 - Reject Deep Candidate-Pool Score Fusion For Primary Metric

Decision: keep `bge_small_xenova_minilm_bm25_deep_score_fusion` as a measured recall-positive ablation, but do not promote it as the current best method.

Reasons:

- `GOAL.md` defines `nDCG@10` as the primary SOTA metric.
- The branch used the current best three sources with the same equal weights and min-max score summation, but requested `branch_k = 300` from each branch before returning the fused top 100.
- The run scored `nDCG@10 = 0.3621651141202009`, below the current best `0.3660878569380838`.
- It improved `Recall@100` by `0.011073533336309116`, which shows the wider branch pool finds more relevant documents, but the top-10 ordering degraded by `0.003922742817882907` absolute `nDCG@10`.

Consequences:

- Run artifact: `artifacts/runs/bge_small_xenova_minilm_bm25_deep_score_fusion_20260608T152801Z.json`.
- Current best remains `bge_small_xenova_minilm_bm25_score_fusion`.
- Deeper pooling is useful evidence for downstream reranking, but by itself it is not enough to reach the SOTA target.

## 2026-06-08 - Reject CombMNZ Agreement Fusion For Primary Metric

Decision: keep `bge_small_xenova_minilm_bm25_mnz_fusion` as a measured no-qrels fusion ablation, but do not promote it as the current best method.

Reasons:

- `GOAL.md` defines `nDCG@10` as the primary SOTA metric.
- The branch used the current best three sources, but changed score fusion from CombSUM-style min-max summation to CombMNZ-style branch-agreement boosting.
- The run scored `nDCG@10 = 0.3594854251733051`, below the current best `0.3660878569380838`.
- It improved `Recall@100` by `0.0015227867345728452`, but lost `0.0066024317647787045` absolute `nDCG@10`, so it does not move the SOTA target.

Consequences:

- Run artifact: `artifacts/runs/bge_small_xenova_minilm_bm25_mnz_fusion_20260608T151744Z.json`.
- Current best remains `bge_small_xenova_minilm_bm25_score_fusion`.
- Agreement boosting alone is not the next best SOTA path; the next credible branch still needs a stronger English retriever, a working local reranker, or a non-test-leaking calibration source.

## 2026-06-08 - Reject Dual-MiniLM Score Fusion For Primary Metric

Decision: do not promote `bge_small_dual_minilm_bm25_score_fusion` despite its higher `Recall@100`.

Reasons:

- `GOAL.md` defines `nDCG@10` as the primary SOTA metric.
- The four-branch equal-weight fusion scored `nDCG@10 = 0.3559227197172415`, below the current best `0.3660878569380838`.
- It improves `Recall@100` to `0.322866194510992`, but the gain does not offset the primary-metric regression.
- The branch used no qrels-tuned weights, so it is a valid ablation but not a methodology improvement for the SOTA target.

Consequences:

- Run artifact: `artifacts/runs/bge_small_dual_minilm_bm25_score_fusion_20260608T150916Z.json`.
- Current best remains `bge_small_xenova_minilm_bm25_score_fusion`.
- Future fusion attempts should either bring in a genuinely new signal source or use non-test/dev-safe calibration, not more equal-weight MiniLM variants.

## 2026-06-08 - Reject Corrected LEANN MiniLM As A SOTA Path

Decision: keep the root-side `leann_minilm_no_recompute` candidate as measured evidence, but do not promote it as a methodology improvement.

Reasons:

- The historical `newragcity` route depends on LEANN, so a corrected LEANN rerun was worth testing after rejecting the old invalid metric artifact.
- Default LEANN recompute search cannot run in this environment because the embedding server cannot start.
- A no-recompute, non-compact HNSW index using the exact cached MiniLM snapshot can run without the server and under the corrected root metric harness.
- The measured run scored `nDCG@10 = 0.3138633685494098`, below direct MiniLM `0.3160012178022206`.

Consequences:

- Run artifact: `artifacts/runs/leann_minilm_no_recompute_20260608T145957Z.json`.
- LEANN is now covered as a corrected ablation rather than a stale historical claim.
- Future LEANN work should focus on storage/latency or a stronger embedding model, not MiniLM quality on NFCorpus.
- The current best branch remains `bge_small_xenova_minilm_bm25_score_fusion`.

## 2026-06-08 - Reject Historical Newragcity 0.5086 Claim As SOTA Evidence

Decision: do not promote the historical `newragcity` `nDCG@10 = 0.5085946124009167` artifact as a SOTA result.

Reasons:

- `artifacts/historical_newragcity_audit.json` marks the claim `invalid`.
- The saved result lacks retrieved document IDs, runs, or branch outputs, so it cannot be rescored under the corrected root metric implementation.
- Saved per-query `Recall@100` contains 12 values above `1.0`, with a maximum of `2.0`.
- The source evaluator computes IDCG from retrieved top-10 relevances rather than all qrel relevances.
- The source evaluator has a recall double-count risk by counting top-10 relevant hits and then top-100 relevant hits again.
- A root import smoke test shows the local `app/` package is shadowed by an installed `app` package, so the historical route is not cleanly runnable as-is.

Consequences:

- The old `0.5085946124009167` number is treated as a false-positive historical claim, not a target-beating artifact.
- Future `newragcity` work must rerun retrieval through the root harness or another corrected metric implementation with saved runs.
- SOTA remains unachieved until a corrected artifact reaches `nDCG@10 >= 0.4699`.

## 2026-06-08 - Use Xenova MiniLM ONNX As Current Best Fusion Component

Decision: add `Xenova/all-MiniLM-L6-v2` ONNX from a sibling cache as a measured MiniLM replacement branch and promote the BGE-small + Xenova MiniLM + BM25 score-fusion result as the current best local branch.

Reasons:

- A complete local ONNX MiniLM cache exists at `/Volumes/WS4TB/WS4TBr/whsjan14/node_modules/@xenova/transformers/.cache/Xenova/all-MiniLM-L6-v2`.
- It runs through the existing `OnnxDenseRetriever` without network access, credentials, paid APIs, or child repo edits.
- Dense-only ONNX MiniLM slightly improves `nDCG@10` over direct MiniLM, and the fused branch improves the current leaderboard best.

Consequences:

- New best artifact: `artifacts/runs/bge_small_xenova_minilm_bm25_score_fusion_20260608T144823Z.json`.
- Best local `nDCG@10` improved from `0.3622186429303806` to `0.3660878569380838`.
- `Recall@100` decreased from `0.3194178792738884` to `0.3162754206179384`.
- The result remains below the official BEIR comparator `0.385` and the selected SOTA target `0.4699`.
- Next credible progress still requires a stronger English retrieval model, a valid reranker, or a corrected `newragcity` rerun through the root metric surface.

## 2026-06-08 - Promote Score-Level Fusion As Current Best Local Branch

Decision: add `bge_small_minilm_bm25_score_fusion` as a measured candidate using per-branch min-max score normalization over local BGE-small ONNX, direct MiniLM, and BM25 with equal weights.

Reasons:

- The previous best RRF branch showed that the three sources are complementary, but rank-only fusion discards branch score shape.
- Score fusion is deterministic, uses no qrels-tuned weights, and fits the existing candidate artifact schema.
- The branch is runnable with existing local models and dependencies; no network, credentials, paid APIs, or child repo edits are required.

Consequences:

- New best artifact: `artifacts/runs/bge_small_minilm_bm25_score_fusion_20260608T142822Z.json`.
- Best local `nDCG@10` improved from `0.3563276091075661` to `0.3622186429303806`.
- `Recall@100` decreased from `0.32206494945280123` to `0.3194178792738884`, so the branch improves top-10 ranking quality while slightly narrowing the candidate pool.
- The result remains below the official BEIR comparator `0.385` and the selected SOTA target `0.4699`.
- Do not tune fusion weights on NFCorpus test qrels; the next credible SOTA path is a stronger English retriever or reranker.

## 2026-06-08 - Treat Stronger Embedder Availability And Runtime As Evidence Gates

Decision: do not fake or approximate the required stronger embedder branch. Record both local model availability and full-benchmark runtime as separate evidence gates.

Reasons:

- `GOAL.md` requires evidence-backed candidate benchmark artifacts, not placeholder claims.
- The only default Hugging Face cache model with complete embedding weights is still `sentence-transformers/all-MiniLM-L6-v2`.
- The previously referenced local BGE-M3 path exists only as a stub without config, tokenizer, or model weights.
- A full local `BAAI/bge-large-zh-v1.5` snapshot was found under a sibling ragflow cache and loads with direct `transformers`.
- The BGE fallback eventually completed after long CPU runtime but scored far below existing local baselines.

Consequences:

- `artifacts/embedder_availability.json` records exact checked paths and missing files.
- `artifacts/bge_large_zh_runtime_gate.json` records the BGE fallback run, runtime, and poor score.
- `artifacts/leaderboard.json` now includes the BGE fallback below the previous baseline branches.
- The next SOTA-moving action is a stronger English retrieval embedder, not reranking on the same weak candidate pool and not the Chinese-focused BGE fallback.
- Completion remains impossible until at least one stronger English/general retrieval model produces a competitive full NFCorpus score or an approved download/install path is available.

## 2026-06-08 - Use Local English ONNX BGE As A Measured Fallback

Decision: add `Xenova/bge-small-en-v1.5` from the sibling finESS cache as an ONNX Runtime dense candidate and fuse it with MiniLM and BM25 through the existing RRF harness.

Reasons:

- It is a complete local English BGE-family model with `config.json`, `tokenizer.json`, and `onnx/model_quantized.onnx`.
- `onnxruntime` and `tokenizers` are installed and usable in the current environment.
- It can be benchmarked without network access, credentials, paid APIs, or child repo edits.

Consequences:

- The best local `nDCG@10` improved from `0.3344802256991052` to `0.3563276091075661`.
- The best local run is now `bge_small_minilm_bm25_rrf`.
- At that stage the result was still not SOTA; it remained `0.11357239089243386` absolute `nDCG@10` below the selected `0.4699` target.
- The next highest-leverage path is a stronger English retrieval model or a repaired local reranker, not more weak-model fusion.

## 2026-06-08 - Separate Reranker Import Repair From Model Availability

Decision: treat reranker feasibility as two gates: import/runtime repair and model availability.

Reasons:

- Plain `sentence_transformers` and `FlagEmbedding` imports fail through the optional `kernels.LayerRepository` path.
- The existing `optional_kernels_disabled()` guard repairs both imports in-process without global site-package edits.
- No complete local reranker model snapshot is available for the checked MiniCPM, BGE reranker, or cross-encoder candidates.

Consequences:

- `artifacts/reranker_availability.json` records the import repair and model absence separately.
- No reranker candidate is added to the leaderboard yet.
- The next reranker work should focus on making a complete approved local reranker model available, then using the guarded import path for execution.

## 2026-06-08 - Reject Current BM25 PRF Query Expansion

Decision: do not promote the current BM25 pseudo-relevance feedback branch.

Reasons:

- `bm25_prf` scored `nDCG@10 = 0.2643284676291605`, worse than plain BM25.
- `bge_small_minilm_bm25_prf_rrf` scored `nDCG@10 = 0.34788409229565553`, worse than the current best `bge_small_minilm_bm25_rrf`.
- The fixed PRF configuration uses only query text and top BM25 documents, so it is a fair no-qrels ablation, but it appears to introduce lexical drift on NFCorpus.

Consequences:

- Query expansion is covered as a measured branch, but it is not part of the current best methodology.
- Future query expansion should use a better expansion model or stricter expansion constraints rather than this simple BM25 PRF branch.

## 2026-06-08 - Create Unified Candidate Runner Before Stronger Model Work

Decision: add a generalized NFCorpus candidate runner before probing or adding stronger embedders.

Reasons:

- `GOAL.md` requires candidate runs under `artifacts/runs/<run_id>.json` and a sorted `artifacts/leaderboard.json`.
- The existing baseline script is a one-off baseline artifact, not a reusable branch comparison surface.
- MiniLM, BM25, and MiniLM+BM25 RRF can be measured with existing code and dependencies, making them the correct first runner validation set.

Consequences:

- `scripts/run_nfcorpus_candidate.py --candidate all-baselines` now writes three comparable run artifacts and a leaderboard.
- The first RRF branch improves `nDCG@10` from `0.3160012178022206` to `0.3344802256991052`.
- This is progress, not SOTA; the best local run remains `0.1354197743008948` absolute `nDCG@10` below the selected `0.4699` target.
- Stronger dense embedder work should now plug into the runner rather than creating separate benchmark scripts.

## 2026-06-08 - Use Strict Current MTEB NFCorpus SOTA Target

Decision: set the active SOTA target to the newer MTEB English Retrieval NFCorpus value from `voyage-3-m-exp`: reported `46.99`, normalized to `0.4699` `nDCG@10`.

Reasons:

- `GOAL.md` requires live SOTA target discovery before new candidate implementation.
- The official BEIR EvalAI table remains the canonical BEIR challenge comparator, but its best observed NFCorpus value is lower at `0.385` and the table is older.
- The current MTEB retrieval table is the stricter public target and therefore better matches the user's request to achieve SOTA rather than only clear the older official comparator.

Consequences:

- The goal is deliberately hard: the current direct MiniLM baseline is `0.1538987821977794` absolute `nDCG@10` below the selected primary target.
- `0.385` remains an official BEIR comparator gate, and `0.4517` from `nvidia/NV-Embed-v2` remains the best observed current open-weight comparator gate.
- The target must not be moved after seeing local candidate results unless the source is later proven invalid.
- If local/open-weight branches cannot reach `0.4699`, the correct outcome is `SOTA_ATTEMPT_REPORT.md`, not a weaker completion claim.

## 2026-06-08 - Root Harness Before Retrieval Expansion

Decision: create a minimal root `turboragger` Python harness instead of editing child repos or starting BGE-M3/HyDE/ColBERT work.

Reasons:

- `GOAL.md` requires baseline reproduction before methodology expansion.
- The root workspace had no package/test command, so there was no durable verification surface.
- Child repos are source material and should not be rewritten for the first milestone.

Consequences:

- The current implementation covers metrics, RRF, result contracts, artifact writing, and neutral harness wiring.
- The MiniLM/nfcorpus baseline is still blocked until nfcorpus and `sentence_transformers` are available.
- No methodology-improvement claim is made.

## 2026-06-08 - Record Baseline State As Evidence

Decision: the baseline command writes `artifacts/baseline_minilm_nfcorpus.json` and `artifacts/baseline_status.md` even when it cannot run MiniLM. If nfcorpus is present but MiniLM is blocked, it writes a BM25 lexical replacement baseline with the same metric surface.

Reasons:

- Missing data/dependency state is part of the evidence surface.
- Future agents need exact blockers, checked paths, command, and timestamp.
- A replacement baseline is better than fake continuity with the historical 0.18 result.

Consequences:

- `scripts/run_nfcorpus_baseline.py` currently exits 0 with BM25 replacement metrics in this environment.
- Completion remains open until the MiniLM dense baseline is repaired or the replacement baseline is explicitly accepted as the first benchmark anchor and compared through the neutral harness.

## 2026-06-08 - Do Not Patch Global Site-Packages

Decision: do not edit the global `transformers` or `kernels` site-packages to force `sentence_transformers` to import.

Reasons:

- The workspace has no permission to safely mutate the shared global Python environment.
- The observed root cause is a dependency compatibility problem in `transformers -> kernels.LayerRepository`.
- A project-local environment or pinned dependency set is safer than modifying shared packages.

Consequences:

- MiniLM dense execution remains blocked in this environment.
- BM25 replacement baseline is used only as a temporary benchmark anchor.

## 2026-06-08 - Use Direct Transformers MiniLM Instead Of Broken sentence_transformers Wrapper

Decision: load cached `sentence-transformers/all-MiniLM-L6-v2` directly with `transformers.AutoTokenizer` and `AutoModel`, using mean pooling and in-process optional-kernels suppression, rather than requiring the broken `sentence_transformers` wrapper.

Reasons:

- The full MiniLM snapshot exists locally under `~/.cache/huggingface/hub`.
- `sentence_transformers` fails because of a `transformers -> kernels.LayerRepository` compatibility bug.
- Direct `transformers` loading works when the optional `kernels` package is made unavailable only inside the process import path.
- This avoids modifying global site-packages.

Consequences:

- The baseline now produces dense MiniLM metrics through the neutral harness.
- The measured direct baseline is `Recall@100 = 0.31150992401169303`, `nDCG@10 = 0.3160012178022206`.
- This does not exactly reproduce the old newragcity artifact and should be treated as a new direct dense baseline unless exact historical reproduction is later required.
