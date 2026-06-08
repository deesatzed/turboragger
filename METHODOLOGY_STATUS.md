# METHODOLOGY_STATUS.md

## SOTA Target Gate

Status: SOTA target established; generalized candidate leaderboard implemented and populated with local dense, sparse, ONNX, and hybrid branches.

Target artifacts:

- `artifacts/sota_target.json`
- `SOTA_TARGET.md`
- `SOTA_PROGRESS.md`

Selected primary target:

- Benchmark: BEIR NFCorpus `test`
- Primary metric: `nDCG@10`
- Source: MTEB English Retrieval leaderboard data table, accessed 2026-06-08
- System: `voyage-3-m-exp`
- Reported value: `46.99`
- Normalized target: `0.4699`

Comparator gates:

- Official BEIR EvalAI NFCorpus comparator: `0.385` from `ZA+NM+Unicamp (InParsv2)`.
- Best observed current open-weight MTEB comparator: `0.4517` from `nvidia/NV-Embed-v2`.

Direct MiniLM starting gap:

- Direct MiniLM baseline `nDCG@10`: `0.3160012178022206`
- Gap to official BEIR comparator: `0.0689987821977794`
- Gap to open-weight MTEB comparator: `0.13569878219777942`
- Gap to selected primary SOTA target: `0.1538987821977794`

Updated local gap after best measured branch:

- Current best local leaderboard `nDCG@10`: `0.369215383024794` from `bge_small_gbdt_regression_dense_prf_dev_selected_dev_score_fusion`
- Gap to official BEIR comparator: `0.015784616975206`
- Gap to open-weight MTEB comparator: `0.082484616975206`
- Gap to selected primary SOTA target: `0.10068461697520598`
- Latest no-qrels agreement-fusion ablation: `bge_small_xenova_minilm_bm25_mnz_fusion` scored `nDCG@10 = 0.3594854251733051`, `Recall@100 = 0.31779820735251124`, and is not promoted.
- Latest candidate-pool depth ablation: `bge_small_xenova_minilm_bm25_deep_score_fusion` scored `nDCG@10 = 0.3621651141202009`, `Recall@100 = 0.3273489539542475`, and is not promoted.
- Latest BGE pooling ablation: `bge_small_mean_xenova_minilm_bm25_score_fusion` scored `nDCG@10 = 0.3634426387237848`, `Recall@100 = 0.31832067766164274`, and is not promoted.
- Latest promoted branch: `bge_small_dual_pool_xenova_minilm_bm25_score_fusion` scored `nDCG@10 = 0.36614841346265437`, `Recall@100 = 0.3211863718747774`.
- Latest no-test-leak calibration ablation: `bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_score_fusion` used dev-selected weights and scored `nDCG@10 = 0.3648389532581644`, `Recall@100 = 0.32369539341221953` on test, and is not promoted.
- Latest title-lexical ablation: `bge_small_dual_pool_xenova_minilm_bm25_title_score_fusion` added title-only BM25 and scored `nDCG@10 = 0.3568054177005105`, `Recall@100 = 0.312460387549029`, and is not promoted.
- Latest text-lexical ablation: `bge_small_dual_pool_xenova_minilm_bm25_text_score_fusion` added text-only BM25 and scored `nDCG@10 = 0.36448715421857897`, `Recall@100 = 0.3152911085718485`, and is not promoted.
- Latest field-aware dev-calibration ablation: `bge_small_dual_pool_xenova_minilm_bm25_fields_dev_calibrated_score_fusion` scored `nDCG@10 = 0.36473218877158997`, `Recall@100 = 0.32396224357142317`, selected `bm25_title = 0.0`, and is not promoted.
- Latest broad local model inventory: `artifacts/local_model_inventory.json` found one additional SOTA-relevant embedding candidate, `maidalun1020/bce-embedding-base_v1`, and no unmeasured candidates remain after benchmarking it.
- Latest newly discovered embedding benchmark: `bce_embedding_base_v1` scored `nDCG@10 = 0.2621479854747279`, `Recall@100 = 0.27654921213670386`, and is not promoted.
- Latest rank-score fusion ablation: `bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion` scored `nDCG@10 = 0.3659607648002612`, `Recall@100 = 0.3220841618662724`, and is not promoted.
- Latest dev-calibrated rank-score fusion ablation: `bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_rank_score_fusion` selected `rank_weight = 4.0` on dev, scored `nDCG@10 = 0.36566973749080495`, `Recall@100 = 0.32302981358524474` on test, and is not promoted.
- Latest supervised feature-fusion ablation: `bge_small_dual_pool_xenova_minilm_bm25_train_dev_learned_fusion` trained a local logistic feature fusion on train branch outputs, scored `nDCG@10 = 0.35887473712153073`, `Recall@100 = 0.31843407013584935` on test, and is not promoted.
- Latest nonlinear supervised feature-fusion ablation: `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion` trained a local GBDT feature scorer on train branch outputs, scored `nDCG@10 = 0.36273469704993305`, `Recall@100 = 0.32849490503295947` on test, and is not promoted.
- Latest train/dev learned cascade ablation: `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_cascade` selected `anchor_k = 3` on dev after GBDT train fitting, scored `nDCG@10 = 0.36218572616170835`, `Recall@100 = 0.32849490503295947` on test, and is not promoted.
- Previous promoted train/dev two-ranker learned fusion: `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion` selected `score_fusion_primary = 1.5` and `gbdt_secondary = 2.0` on dev after GBDT train fitting, scored `nDCG@10 = 0.3670950977369987`, `Recall@100 = 0.3331795970806332` on test, and is now superseded by the graded-regression branch on primary `nDCG@10`.
- Latest five-source train/dev learned calibration ablation: `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_calibrated_score_fusion` selected weights for BGE CLS, BGE mean, Xenova MiniLM, BM25, and the train-fitted GBDT branch on dev, scored `nDCG@10 = 0.3660561464830782`, `Recall@100 = 0.32611330897080015` on test, and is not promoted.
- Latest deep-pool train/dev learned fusion ablation: `bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_dev_score_fusion` used `branch_k = 300`, selected `score_fusion_primary = 2.0` and `gbdt_secondary = 0.5` on dev after GBDT train fitting, scored `nDCG@10 = 0.36562875673411727`, `Recall@100 = 0.33854940883436946` on test, and is not promoted.
- Previous promoted train/dev graded-regression learned fusion: `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion` trained a GBDT regressor on graded train relevance targets, selected `score_fusion_primary = 1.0` and `gbdt_regression_secondary = 1.5` on dev, scored `nDCG@10 = 0.3675830427079456`, `Recall@100 = 0.3328785827037792` on test, and is now superseded by current-best plus dense PRF complement fusion.
- Latest direct graded-regression learned fusion ablation: `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_fusion` trained the same GBDT regressor on graded train relevance targets, scored `nDCG@10 = 0.36434599974495163`, `Recall@100 = 0.325698652336419` on test, and is not promoted.
- Latest late-interaction rerank ablation: `bge_small_late_interaction_score_fusion_rerank` reranked the BGE-small dual-pool + Xenova MiniLM + BM25 score-fusion pool with BGE-small ONNX token MaxSim, scored `nDCG@10 = 0.36006186813204677`, `Recall@100 = 0.3211863718747774` on test, and is not promoted.
- Latest current-best plus late-interaction calibration ablation: `bge_small_late_interaction_gbdt_regression_dev_score_fusion` selected `current_best_primary = 0.5` and `late_interaction_secondary = 2.0` on dev, scored `nDCG@10 = 0.36542794274338575`, `Recall@100 = 0.32149384348966625` on test, and is not promoted.
- Latest deep-pool graded-regression ablation: `bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_regression_dev_score_fusion` used `branch_k = 300`, scored `nDCG@10 = 0.363902495492805`, `Recall@100 = 0.33490230529221543` on test, and is not promoted.
- Latest local domain-supervised checkpoint ablation: `scifact_finetuned_minilm` loaded a complete SciFact-style fine-tuned MiniLM checkpoint, scored `nDCG@10 = 0.3113948750306552`, `Recall@100 = 0.30306730089962974` on test, and is not promoted.
- Latest no-test-leak SciFact checkpoint-selection ablation: `scifact_dev_selected_minilm` selected `checkpoint-9` on NFCorpus dev, scored `nDCG@10 = 0.316500938704734`, `Recall@100 = 0.31019343221102363` on test, and is not promoted.
- Latest current-best plus no-test-leak SciFact checkpoint-selection fusion: `bge_small_gbdt_regression_scifact_dev_selected_dev_score_fusion` selected `current_best_primary = 0.5` and `scifact_secondary = 0.0` on dev, scored `nDCG@10 = 0.3675830427079456`, `Recall@100 = 0.33050103876645803` on test, and is not promoted because it tied the then-current best primary metric while lowering recall.
- Latest dense pseudo-relevance feedback ablation: `bge_small_en_onnx_dense_prf` shifted BGE-small ONNX query vectors toward top-10 pseudo-relevant document vectors, scored `nDCG@10 = 0.3425690018844359`, `Recall@100 = 0.3105355971312376` on test, and is not promoted.
- Latest no-test-leak dense PRF parameter-selection ablation: `bge_small_en_onnx_dense_prf_dev_selected` selected `feedback_docs = 3` and `feedback_weight = 0.5` on NFCorpus dev, scored `nDCG@10 = 0.34933673642384316`, `Recall@100 = 0.3180000296586443` on test, and is not promoted.
- Latest promoted current-best plus no-test-leak dense PRF complement fusion: `bge_small_gbdt_regression_dense_prf_dev_selected_dev_score_fusion` selected `current_best_primary = 2.0` and `dense_prf_secondary = 0.5` on dev after reusing the dev-selected dense PRF branch, scored `nDCG@10 = 0.369215383024794`, `Recall@100 = 0.33559899868294146` on test, and is the new current best while still below SOTA.

Interpretation: the current best local methodology is now a second-stage dev-calibrated fusion over the previous train/dev graded-regression anchor plus dev-selected BGE-small dense PRF. This is real local progress because both `nDCG@10` and `Recall@100` improved on `test`, but it is not SOTA. Agreement boosting improves recall slightly but hurts top-10 ranking quality. Rank-score fusion and dev-calibrated rank-score fusion improve recall slightly but miss the newer best `nDCG@10`. Train-split supervised feature fusion over the same branch outputs is valid but hurts both `nDCG@10` and `Recall@100`; the nonlinear GBDT classifier variant improves recall but still hurts `nDCG@10` by itself; the train/dev cascade keeps the recall gain but also misses the current best top-10 ranking. The dev-calibrated classifier two-ranker fusion was the first learned branch to improve both primary `nDCG@10` and `Recall@100`; the graded-regression variant improved `nDCG@10`, and dense PRF complement fusion improved both metrics again. Deeper graded-regression candidate pools improve recall but miss top-10 ordering. The direct graded-regression ranker is worse, confirming the regressor is useful as a secondary signal rather than as the standalone final ranker. Current-source token MaxSim late interaction also misses, and dev-calibrating it against the current best still misses, suggesting that late interaction needs a stronger model or better trained reranker signal rather than BGE-small token embeddings alone. The discovered SciFact-finetuned MiniLM checkpoint also misses and underperforms direct MiniLM; dev-selecting among the checkpoint family recovers a tiny MiniLM-level gain but still misses badly. Dev-calibrating that checkpoint family against the current best assigns the SciFact branch weight `0.0`, so it does not supply useful complementary signal. Dense PRF supplies enough complementary signal to improve the current anchor, but the remaining `0.10068461697520598` `nDCG@10` gap is still too large for same-source fusion alone. The next methodology stage still needs a stronger English retrieval embedder or working reranker without moving the target after seeing results.

## Candidate Leaderboard Status

Artifact: `artifacts/leaderboard.json`

Current runs:

| Candidate | Retrieval mode | nDCG@10 | Recall@100 | Queries | Failures |
|---|---|---:|---:|---:|---:|
| `bge_small_gbdt_regression_dense_prf_dev_selected_dev_score_fusion` | `multi_dense_sparse_dense_prf_dev_selected_gbdt_regression_dev_score_fusion` | `0.369215383024794` | `0.33559899868294146` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion` | `multi_dense_sparse_train_dev_gbdt_regression_dev_score_fusion` | `0.3675830427079456` | `0.3328785827037792` | 323 | 0 |
| `bge_small_gbdt_regression_scifact_dev_selected_dev_score_fusion` | `multi_dense_sparse_scifact_dev_selected_gbdt_regression_dev_score_fusion` | `0.3675830427079456` | `0.33050103876645803` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion` | `multi_dense_sparse_train_dev_gbdt_dev_score_fusion` | `0.3670950977369987` | `0.3331795970806332` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_score_fusion` | `multi_dense_sparse_score_fusion` | `0.36614841346265437` | `0.3211863718747774` | 323 | 0 |
| `bge_small_xenova_minilm_bm25_score_fusion` | `multi_dense_sparse_score_fusion` | `0.3660878569380838` | `0.3162754206179384` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_calibrated_score_fusion` | `multi_dense_sparse_train_dev_gbdt_dev_calibrated_score_fusion` | `0.3660561464830782` | `0.32611330897080015` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion` | `multi_dense_sparse_rank_score_fusion` | `0.3659607648002612` | `0.3220841618662724` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_rank_score_fusion` | `multi_dense_sparse_dev_calibrated_rank_score_fusion` | `0.36566973749080495` | `0.32302981358524474` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_dev_score_fusion` | `multi_dense_sparse_deep_train_dev_gbdt_dev_score_fusion` | `0.36562875673411727` | `0.33854940883436946` | 323 | 0 |
| `bge_small_late_interaction_gbdt_regression_dev_score_fusion` | `multi_dense_sparse_late_interaction_gbdt_regression_dev_score_fusion` | `0.36542794274338575` | `0.32149384348966625` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_score_fusion` | `multi_dense_sparse_dev_calibrated_score_fusion` | `0.3648389532581644` | `0.32369539341221953` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_fields_dev_calibrated_score_fusion` | `multi_dense_sparse_fields_dev_calibrated_score_fusion` | `0.36473218877158997` | `0.32396224357142317` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_text_score_fusion` | `multi_dense_sparse_text_score_fusion` | `0.36448715421857897` | `0.3152911085718485` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_fusion` | `multi_dense_sparse_train_dev_gbdt_regression_feature_fusion` | `0.36434599974495163` | `0.325698652336419` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_regression_dev_score_fusion` | `multi_dense_sparse_deep_train_dev_gbdt_regression_dev_score_fusion` | `0.363902495492805` | `0.33490230529221543` | 323 | 0 |
| `bge_small_mean_xenova_minilm_bm25_score_fusion` | `multi_dense_sparse_score_fusion` | `0.3634426387237848` | `0.31832067766164274` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion` | `multi_dense_sparse_train_dev_gbdt_feature_fusion` | `0.36273469704993305` | `0.32849490503295947` | 323 | 0 |
| `bge_small_minilm_bm25_score_fusion` | `multi_dense_sparse_score_fusion` | `0.3622186429303806` | `0.3194178792738884` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_cascade` | `multi_dense_sparse_train_dev_gbdt_dev_cascade` | `0.36218572616170835` | `0.32849490503295947` | 323 | 0 |
| `bge_small_xenova_minilm_bm25_deep_score_fusion` | `multi_dense_sparse_deep_score_fusion` | `0.3621651141202009` | `0.3273489539542475` | 323 | 0 |
| `bge_small_late_interaction_score_fusion_rerank` | `multi_dense_sparse_late_interaction_rerank` | `0.36006186813204677` | `0.3211863718747774` | 323 | 0 |
| `bge_small_xenova_minilm_bm25_mnz_fusion` | `multi_dense_sparse_mnz_fusion` | `0.3594854251733051` | `0.31779820735251124` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_train_dev_learned_fusion` | `multi_dense_sparse_train_dev_learned_linear_fusion` | `0.35887473712153073` | `0.31843407013584935` | 323 | 0 |
| `bge_small_dual_pool_xenova_minilm_bm25_title_score_fusion` | `multi_dense_sparse_title_score_fusion` | `0.3568054177005105` | `0.312460387549029` | 323 | 0 |
| `bge_small_minilm_bm25_rrf` | `multi_dense_sparse_rrf` | `0.3563276091075661` | `0.32206494945280123` | 323 | 0 |
| `bge_small_dual_minilm_bm25_score_fusion` | `multi_dense_sparse_score_fusion` | `0.3559227197172415` | `0.322866194510992` | 323 | 0 |
| `bge_small_minilm_bm25_prf_rrf` | `multi_dense_sparse_prf_rrf` | `0.34788409229565553` | `0.3208886711792187` | 323 | 0 |
| `bge_small_en_bm25_rrf` | `dense_onnx_sparse_rrf` | `0.3429287918550373` | `0.30287197379472947` | 323 | 0 |
| `bge_small_en_onnx` | `dense_onnx` | `0.34224496481953304` | `0.3120478599231725` | 323 | 0 |
| `minilm_bm25_rrf` | `dense_sparse_rrf` | `0.3344802256991052` | `0.3121492868681654` | 323 | 0 |
| `xenova_minilm_onnx` | `dense_onnx` | `0.31689265638802383` | `0.30671003953431614` | 323 | 0 |
| `scifact_dev_selected_minilm` | `dense_dev_selected` | `0.316500938704734` | `0.31019343221102363` | 323 | 0 |
| `minilm_dense` | `dense` | `0.3160012178022206` | `0.31150992401169303` | 323 | 0 |
| `leann_minilm_no_recompute` | `leann_dense_hnsw_no_recompute` | `0.3138633685494098` | `0.3111533551882048` | 323 | 0 |
| `scifact_finetuned_minilm` | `dense` | `0.3113948750306552` | `0.30306730089962974` | 323 | 0 |
| `bm25` | `lexical` | `0.3064137649277972` | `0.23307829323957305` | 323 | 0 |
| `bm25_prf` | `lexical_prf` | `0.2643284676291605` | `0.22671140398062603` | 323 | 0 |
| `bce_embedding_base_v1` | `dense` | `0.2621479854747279` | `0.27654921213670386` | 323 | 0 |
| `bge_large_zh` | `dense` | `0.14143051730446513` | `0.16162218356917438` | 323 | 0 |

Run artifacts:

- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion_20260608T192209Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion_20260608T184157Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_score_fusion_20260608T154445Z.json`
- `artifacts/runs/bge_small_xenova_minilm_bm25_score_fusion_20260608T144823Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_calibrated_score_fusion_20260608T185555Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion_20260608T173359Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_rank_score_fusion_20260608T174836Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_dev_score_fusion_20260608T190950Z.json`
- `artifacts/runs/bge_small_late_interaction_gbdt_regression_dev_score_fusion_20260608T200608Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_score_fusion_20260608T155633Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_fields_dev_calibrated_score_fusion_20260608T163215Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_text_score_fusion_20260608T162004Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_fusion_20260608T193443Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_regression_dev_score_fusion_20260608T202346Z.json`
- `artifacts/runs/bge_small_mean_xenova_minilm_bm25_score_fusion_20260608T153812Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion_20260608T181638Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_cascade_20260608T183008Z.json`
- `artifacts/runs/bge_small_xenova_minilm_bm25_deep_score_fusion_20260608T152801Z.json`
- `artifacts/runs/bge_small_late_interaction_score_fusion_rerank_20260608T195336Z.json`
- `artifacts/runs/bge_small_xenova_minilm_bm25_mnz_fusion_20260608T151744Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_learned_fusion_20260608T180407Z.json`
- `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_title_score_fusion_20260608T160744Z.json`
- `artifacts/runs/bge_small_dual_minilm_bm25_score_fusion_20260608T150916Z.json`
- `artifacts/runs/xenova_minilm_onnx_20260608T144741Z.json`
- `artifacts/runs/scifact_dev_selected_minilm_20260608T205002Z.json`
- `artifacts/runs/leann_minilm_no_recompute_20260608T145957Z.json`
- `artifacts/runs/scifact_finetuned_minilm_20260608T203922Z.json`
- `artifacts/runs/bge_small_minilm_bm25_score_fusion_20260608T142822Z.json`
- `artifacts/runs/bm25_20260608T132446Z.json`
- `artifacts/runs/minilm_dense_20260608T132449Z.json`
- `artifacts/runs/minilm_bm25_rrf_20260608T132506Z.json`
- `artifacts/runs/bge_small_en_onnx_20260608T135538Z.json`
- `artifacts/runs/bge_small_en_bm25_rrf_20260608T135851Z.json`
- `artifacts/runs/bge_small_minilm_bm25_rrf_20260608T140218Z.json`
- `artifacts/runs/bm25_prf_20260608T141609Z.json`
- `artifacts/runs/bge_small_minilm_bm25_prf_rrf_20260608T141628Z.json`
- `artifacts/runs/bge_large_zh_20260608T134449Z.json`
- `artifacts/runs/bce_embedding_base_v1_20260608T171437Z.json`

## Query Expansion Status

Status: deterministic pseudo-relevance feedback was implemented and benchmarked; it did not improve the best local run.

Artifacts:

- `artifacts/runs/bm25_prf_20260608T141609Z.json`
- `artifacts/runs/bge_small_minilm_bm25_prf_rrf_20260608T141628Z.json`

Results:

- `bm25_prf`: `nDCG@10 = 0.2643284676291605`, worse than plain BM25.
- `bge_small_minilm_bm25_prf_rrf`: `nDCG@10 = 0.34788409229565553`, worse than `bge_small_minilm_bm25_rrf`.

Interpretation: query expansion has been covered as a measured branch, but this fixed BM25 pseudo-relevance feedback implementation introduces enough lexical drift to hurt ranking quality on NFCorpus.

## Stronger Embedder Availability

Artifact: `artifacts/embedder_availability.json`

Status: one stronger local fallback embedder was benchmarked, but it performed poorly.

Checked candidates:

| Candidate | Status | Evidence |
|---|---|---|
| `BAAI/bge-m3` | missing or incomplete | Hugging Face cache path missing; `/Volumes/WS4TB/WS4TBr/Partial_Apps_WS/dec24_apps/MedAiTools/bge-m3` exists but has only `modules.json` and `sentence_bert_config.json`, with no `config.json`, tokenizer, or model weights. |
| `Qwen/Qwen3-Embedding-0.6B` | missing | Expected Hugging Face snapshot path missing. |
| `nomic-ai/nomic-embed-text-v1.5` | missing | Expected Hugging Face snapshot path missing despite older handoff claim that Nomic may be on disk. |
| `intfloat/e5-large-v2` | missing | Expected Hugging Face snapshot path missing. |
| `Alibaba-NLP/gte-large-en-v1.5` | missing | Expected Hugging Face snapshot path missing. |
| `BAAI/bge-large-zh-v1.5` | available fallback | Complete 1.2 GB local PyTorch snapshot found at `/Volumes/WS4TB/WS4TBr/aP2A/ragflow/huggingface.co/BAAI/bge-large-zh-v1.5`. |
| `Xenova/bge-small-en-v1.5` | available fallback | Complete local English ONNX snapshot found under `/Volumes/WS4TB/WS4TBr/finESS/node_modules/@xenova/transformers/.cache/Xenova/bge-small-en-v1.5`. |

Benchmark result:

- Command: `PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bge_large_zh`
- Artifact: `artifacts/bge_large_zh_runtime_gate.json`
- Latest run artifact: `artifacts/runs/bge_large_zh_20260608T134449Z.json`
- Runtime: `411.749291` seconds
- Result: `nDCG@10 = 0.14143051730446513`, `Recall@100 = 0.16162218356917438`, 323 queries, 0 failures.
- At that point, before the later ONNX hybrid and score-fusion runs, `minilm_bm25_rrf` remained best measured local run at `0.3344802256991052` `nDCG@10`.

Interpretation: the required stronger-embedder stage now has completed fallback scores. `Xenova/bge-small-en-v1.5` improves the best local result when fused with MiniLM and BM25; `BAAI/bge-large-zh-v1.5` is Chinese-focused and performs far worse than the MiniLM/BM25 branches on English NFCorpus. The next safe action is to obtain an approved stronger English retrieval embedder or a working local reranker and rerun through the same artifact schema.

## Reranker Availability

Status: no usable local reranker is available yet.

Artifact: `artifacts/reranker_availability.json`

Evidence:

- Sibling `newragcity` reranker service points to `openbmb/MiniCPM-Reranker-Light` through `sentence_transformers`.
- Plain `sentence_transformers` and `FlagEmbedding` imports both fail with `ValueError: Either a revision or a version must be specified.`
- Guarded imports using the existing optional-kernels suppression both work, so the dependency issue is repairable in-process without global package edits.
- No complete local reranker model artifact has been confirmed in the active benchmark environment.
- Checked reranker model candidates: `openbmb/MiniCPM-Reranker-Light`, `BAAI/bge-reranker-base`, `BAAI/bge-reranker-v2-m3`, and `cross-encoder/ms-marco-MiniLM-L-6-v2`.

Interpretation: reranking remains a required future branch. The import side is now understood and fixable through a local guard, but no reranker benchmark can be honestly run until a complete local reranker model is available.

## Current Baseline Status

Status: direct MiniLM dense baseline established; historical newragcity metric not exactly reproduced.

The historical newragcity artifact records `Recall@100 = 0.18392811903482234` and `nDCG@10 = 0.5085946124009167`. This run did not exactly reproduce that old integrated result. Instead, it established a direct MiniLM dense baseline through the new neutral harness.

Historical-claim audit update:

- Artifact: `artifacts/historical_newragcity_audit.json`
- Verdict: `invalid`
- The saved result has no retrieved doc IDs, contains 12 per-query `Recall@100` values above `1.0`, and its source evaluator computes IDCG from retrieved top-10 relevances rather than all qrel relevances.
- The historical `0.5085946124009167` number is therefore not accepted as SOTA evidence.

Corrected LEANN update:

- Artifact: `artifacts/runs/leann_minilm_no_recompute_20260608T145957Z.json`
- Result: `nDCG@10 = 0.3138633685494098`, `Recall@100 = 0.3111533551882048`
- Interpretation: a root-side no-recompute, non-compact LEANN/HNSW MiniLM index can run under the corrected metric harness, but it underperforms direct MiniLM and is not a SOTA path.

Current baseline command:

```bash
PYTHONPATH=src python3 scripts/run_nfcorpus_baseline.py
```

Observed result: exit code 0, with artifacts:

- `artifacts/baseline_minilm_nfcorpus.json`
- `artifacts/baseline_bm25_nfcorpus.json`
- `artifacts/baseline_status.md`

Current direct MiniLM baseline:

- Mode: direct `transformers` MiniLM mean pooling through the neutral harness.
- Dataset: `/Volumes/WS4TB/WS4TBr/newragcity/UltraRAG-main/datasets/nfcorpus`
- Dataset fingerprint: `79cb102227a4395b63e595fb53535b775d71b9e6e17581f342169eb7f97dc4d2`
- Queries tested: 323
- Failure count: 0
- `Recall@100`: `0.31150992401169303`
- `nDCG@10`: `0.3160012178022206`

Wrapper caveat:

- `sentence_transformers` fails to import because `transformers` imports `kernels.LayerRepository` without a required revision/version.
- The harness therefore loads MiniLM directly from the cached snapshot and disables optional `kernels` only during the model import.

## Measured Dense-Only Harness Result

MiniLM dense-only nfcorpus metrics exist through the neutral harness. The wrapper remains blocked, but direct cached `transformers` loading works.

Implemented harness pieces:

- `src/turboragger/contracts.py`
- `src/turboragger/metrics.py`
- `src/turboragger/rrf.py`
- `src/turboragger/harness.py`
- `src/turboragger/artifacts.py`
- `src/turboragger/data.py`
- `src/turboragger/probe.py`
- `src/turboragger/lexical.py`
- `src/turboragger/dense.py`

## Dependency and Data Status

Environment probe command:

```bash
PYTHONPATH=src python3 scripts/probe_environment.py
```

Artifact: `artifacts/environment_probe.json`

Current probe summary:

| Item | Status |
|---|---|
| `pytrec_eval` | usable |
| `rank_bm25` | usable |
| `beir` | usable |
| `mlx` | usable |
| `mlx_lm` | not usable: no Metal device available |
| `sentence_transformers` | not usable: `ValueError: Either a revision or a version must be specified.` |
| direct cached MiniLM via `transformers` | usable |
| `turbovec` | not usable as Python package; local source exists but compiled `_turbovec` extension is missing |
| nfcorpus | found at `/Volumes/WS4TB/WS4TBr/newragcity/UltraRAG-main/datasets/nfcorpus` |

## Branch-by-Branch Next-Step Recommendation

1. Baseline decision: decide whether direct MiniLM is the accepted anchor or whether exact old newragcity reproduction is required.
2. If direct MiniLM is accepted, begin embedder bakeoff using the same harness and metric surface.
3. If historical reproduction is required, inspect old newragcity integration differences before bakeoff.
4. Hybrid/RRF expansion: add sparse, HyDE, and BGE-M3 multi-mode branches one at a time with ablations.
5. Defer `turbovec`/ColBERT until the Python extension is built and storage sizing is measured.

## Explicit Next Gaps

- BGE-M3 integration: ready for a measured bakeoff only after accepting direct MiniLM as the baseline anchor or choosing to reproduce the old newragcity integrated path.
- Medical embedder integration: ready for a measured bakeoff only after the baseline-anchor decision.
- HyDE/query rewrites: defer until at least one stronger dense/embedder branch is measured.
- ColBERT/turbovec storage: deferred until `turbovec` imports in the harness environment and storage sizing is measured.
- Graph-hop retrieval: deferred until the candidate pool improves.
- Coverage loop: deferred until retrieval branches produce measurable gains and loop budgets can be defined.
- Fine-tuning: deferred until the off-the-shelf pipeline is stable.
- Agent_Pidgeon receipts: deferred until retrieval path artifacts are stable; the deterministic trust boundary must be preserved.
