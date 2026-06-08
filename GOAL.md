# GOAL.md

/goal

OUTCOME:
Build `turboragger` from the current direct MiniLM baseline into a reproducible local retrieval methodology that achieves SOTA or better on BEIR NFCorpus `test` under a clearly defined, currently verified SOTA target. Completion requires a saved benchmark artifact showing that the local pipeline beats the selected SOTA target on the primary metric `nDCG@10` and preserves a defensible secondary retrieval metric surface including `Recall@100`, with all code, configs, and evidence needed to rerun the result.

CURRENT EVIDENCE SNAPSHOT:
- Active primary SOTA target: MTEB English Retrieval NFCorpus `nDCG@10 = 0.4699` from `voyage-3-m-exp`, recorded in `artifacts/sota_target.json` and `SOTA_TARGET.md`.
- Official BEIR comparator gate: `nDCG@10 = 0.385`.
- Best current local artifact: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion_20260608T192209Z.json`.
- Best current local score: `nDCG@10 = 0.3675830427079456`, `Recall@100 = 0.3328785827037792`, 323 queries, 0 failures.
- Current gap to selected primary target: `0.10231695729205437` absolute `nDCG@10`.
- Current gap to official BEIR comparator: `0.01741695729205439` absolute `nDCG@10`.
- Current best local method: dev-calibrated min-max score fusion over the previous score-fusion ranker and a train-fitted graded-relevance GBDT regression feature ranker; the underlying source set remains `Xenova/bge-small-en-v1.5` ONNX with both CLS and mean pooling, `Xenova/all-MiniLM-L6-v2` ONNX, and BM25.
- Latest measured fusion ablation: `artifacts/runs/bge_small_xenova_minilm_bm25_mnz_fusion_20260608T151744Z.json` scored `nDCG@10 = 0.3594854251733051`, `Recall@100 = 0.31779820735251124`, and was not promoted because it lost `0.0066024317647787045` primary `nDCG@10` versus the then-current best.
- Latest measured depth ablation: `artifacts/runs/bge_small_xenova_minilm_bm25_deep_score_fusion_20260608T152801Z.json` scored `nDCG@10 = 0.3621651141202009`, `Recall@100 = 0.3273489539542475`, and was not promoted because it lost `0.003922742817882907` primary `nDCG@10` versus the then-current best despite improving `Recall@100` by `0.011073533336309116`.
- Latest measured BGE pooling ablation: `artifacts/runs/bge_small_mean_xenova_minilm_bm25_score_fusion_20260608T153812Z.json` scored `nDCG@10 = 0.3634426387237848`, `Recall@100 = 0.31832067766164274`, and was not promoted because it lost `0.0026452182142990277` primary `nDCG@10` versus the then-current best.
- Latest promoted BGE pooling branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_score_fusion_20260608T154445Z.json` improves the previous best by `0.00006055652457054306` absolute `nDCG@10` and improves `Recall@100` by `0.004910951256838991`, but remains far below SOTA.
- Latest measured no-test-leak calibration branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_score_fusion_20260608T155633Z.json` selected weights on NFCorpus `dev` qrels only, then scored `nDCG@10 = 0.3648389532581644`, `Recall@100 = 0.32369539341221953` on `test`, and was not promoted because it lost `0.001309460204489965` primary `nDCG@10` versus the then-current best.
- Latest measured title-lexical branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_title_score_fusion_20260608T160744Z.json` added title-only BM25 and scored `nDCG@10 = 0.3568054177005105`, `Recall@100 = 0.312460387549029`, so it is rejected.
- Latest measured text-lexical branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_text_score_fusion_20260608T162004Z.json` added text-only BM25 and scored `nDCG@10 = 0.36448715421857897`, `Recall@100 = 0.3152911085718485`, so it is rejected.
- Latest measured field-aware dev-calibration branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_fields_dev_calibrated_score_fusion_20260608T163215Z.json` selected weights on NFCorpus `dev` qrels only, then scored `nDCG@10 = 0.36473218877158997`, `Recall@100 = 0.32396224357142317` on `test`, so it is rejected despite the recall gain.
- Latest broader local model inventory: `artifacts/local_model_inventory.json` found one additional SOTA-relevant local embedding candidate, `maidalun1020/bce-embedding-base_v1`, and that candidate has now been benchmarked.
- Latest measured newly discovered embedding branch: `artifacts/runs/bce_embedding_base_v1_20260608T171437Z.json` scored `nDCG@10 = 0.2621479854747279`, `Recall@100 = 0.27654921213670386`, so it is rejected.
- Latest measured rank-score hybrid fusion branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion_20260608T173359Z.json` scored `nDCG@10 = 0.3659607648002612`, `Recall@100 = 0.3220841618662724`, so it is rejected despite the recall gain.
- Latest measured no-test-leak rank-score calibration branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_rank_score_fusion_20260608T174836Z.json` selected `rank_weight = 4.0` on NFCorpus `dev`, then scored `nDCG@10 = 0.36566973749080495`, `Recall@100 = 0.32302981358524474` on `test`, so it was rejected because it lost `0.00047867597184941824` primary `nDCG@10` versus the then-current best.
- Latest measured no-test-leak supervised feature branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_learned_fusion_20260608T180407Z.json` trained a local `sklearn` logistic feature fusion on NFCorpus `train` branch outputs only, then scored `nDCG@10 = 0.35887473712153073`, `Recall@100 = 0.31843407013584935` on `test`, so it was rejected because it lost `0.007273676341123636` primary `nDCG@10` versus the then-current best.
- Latest measured no-test-leak nonlinear supervised feature branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion_20260608T181638Z.json` trained a local `sklearn` `HistGradientBoostingClassifier` on NFCorpus `train` branch outputs only, then scored `nDCG@10 = 0.36273469704993305`, `Recall@100 = 0.32849490503295947` on `test`, so it was rejected because it lost `0.0034137164127213127` primary `nDCG@10` versus the then-current best despite a recall gain.
- Latest measured no-test-leak learned cascade branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_cascade_20260608T183008Z.json` trained the GBDT branch on `train`, selected cascade `anchor_k = 3` on `dev`, then scored `nDCG@10 = 0.36218572616170835`, `Recall@100 = 0.32849490503295947` on `test`, so it was rejected because it lost `0.003962687300946011` primary `nDCG@10` versus the then-current best despite the recall gain.
- Latest promoted no-test-leak two-ranker learned fusion branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion_20260608T184157Z.json` trained the GBDT branch on `train`, selected two-ranker fusion weights on `dev`, then scored `nDCG@10 = 0.3670950977369987`, `Recall@100 = 0.3331795970806332` on `test`; it improves the previous best by `0.000946684274344356` primary `nDCG@10` and `0.0119932252058558` `Recall@100`, but remains far below SOTA.
- Latest measured no-test-leak five-source learned calibration branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_calibrated_score_fusion_20260608T185555Z.json` trained the GBDT branch on `train`, calibrated BGE CLS, BGE mean, Xenova MiniLM, BM25, and GBDT branch weights on `dev`, then scored `nDCG@10 = 0.3660561464830782`, `Recall@100 = 0.32611330897080015` on `test`; it was rejected because it lost `0.001038951253920506` primary `nDCG@10` and `0.007066288109833063` `Recall@100` versus the then-current best.
- Latest measured deep-pool two-ranker learned fusion branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_dev_score_fusion_20260608T190950Z.json` trained the GBDT branch on `train`, used `branch_k = 300`, selected two-ranker fusion weights on `dev`, then scored `nDCG@10 = 0.36562875673411727`, `Recall@100 = 0.33854940883436946` on `test`; it was rejected because it lost `0.001466341002881455` primary `nDCG@10` versus the then-current best despite improving `Recall@100` by `0.005369811753736242`.
- Latest promoted no-test-leak graded-regression branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion_20260608T192209Z.json` trained a GBDT regressor on graded NFCorpus `train` relevance targets, selected two-ranker fusion weights on `dev`, then scored `nDCG@10 = 0.3675830427079456`, `Recall@100 = 0.3328785827037792` on `test`; it improves the previous best by `0.00048794497094689637` primary `nDCG@10` but loses `0.0003010143768540363` `Recall@100`, and remains far below SOTA.
- Latest measured direct graded-regression branch: `artifacts/runs/bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_fusion_20260608T193443Z.json` trained the same GBDT regressor on graded NFCorpus `train` relevance targets, then scored `nDCG@10 = 0.36434599974495163`, `Recall@100 = 0.325698652336419` on `test`; it is rejected because it loses `0.0032370429629939856` primary `nDCG@10` and `0.007179930367360199` `Recall@100` versus the current best.
- Latest measured late-interaction rerank branch: `artifacts/runs/bge_small_late_interaction_score_fusion_rerank_20260608T195336Z.json` reranked the BGE-small dual-pool + Xenova MiniLM + BM25 score-fusion pool with BGE-small ONNX token-level MaxSim, then scored `nDCG@10 = 0.36006186813204677`, `Recall@100 = 0.3211863718747774` on `test`; it is rejected because it loses `0.007521174575898848` primary `nDCG@10` and `0.011692210829001792` `Recall@100` versus the current best.
- Latest measured current-best plus late-interaction calibration branch: `artifacts/runs/bge_small_late_interaction_gbdt_regression_dev_score_fusion_20260608T200608Z.json` dev-calibrated the current graded-regression score-fusion anchor against the BGE-small token-MaxSim late-interaction branch, selected `current_best_primary = 0.5` and `late_interaction_secondary = 2.0` on `dev`, then scored `nDCG@10 = 0.36542794274338575`, `Recall@100 = 0.32149384348966625` on `test`; it is rejected because it loses `0.00215509996455987` primary `nDCG@10` and `0.011384739214112927` `Recall@100` versus the current best.
- Historical `newragcity` `0.5085946124009167` claim is rejected as non-comparable evidence by `artifacts/historical_newragcity_audit.json`.
- Next credible SOTA direction: make a new stronger English retriever or real local reranker available, then benchmark it through `scripts/run_nfcorpus_candidate.py`; agreement-only fusion, rank-score fusion, branch-depth, calibration, pooling, supervised feature fusion, nonlinear feature fusion, learned cascades, lexical-field changes, current-source token MaxSim late interaction, second-stage late-interaction calibration, and the newly discovered BCE embedding have not closed the SOTA gap, so do not tune fusion weights on NFCorpus test qrels and do not weaken the target.

NEXT BEST SOTA DIRECTIONS:
1. Stronger English retrieval model acquisition and benchmark, estimated success probability `45%`: add a complete local/open-weight retriever such as `BAAI/bge-m3`, Qwen3-Embedding, Nomic, E5/GTE, or `nvidia/NV-Embed-v2`-class model, then run it through the existing artifact schema. This is the highest-probability path because the remaining `0.10231695729205437` `nDCG@10` gap is too large for same-source fusion alone.
2. Local reranker acquisition and top-k reranking, estimated success probability `35%`: add a complete BGE/Qwen3/MiniCPM/cross-encoder reranker, rerank the current best candidate pool, and preserve train/dev/test separation. This directly targets top-10 ordering, where `nDCG@10` is decided.
3. Domain-supervised fine-tuning or adapter training, estimated success probability `25%`: fine-tune a local embedder or reranker using NFCorpus `train` and select hyperparameters on `dev`, with test used only once for final evaluation. This may move the semantic signal, but has higher leakage, overfit, and runtime risk.
4. Distinct retrieval signal expansion, estimated success probability `20%`: add a genuinely new retrieval family such as SPLADE-style sparse retrieval, ColBERT/late interaction, graph/path retrieval, or evidence-coverage search before revisiting fusion. This is more promising than more same-source calibration but requires new storage/runtime work.
5. External API target-sanity benchmark, estimated success probability `15%` for local methodology progress and high value for calibration: with explicit user approval only, run a bounded hosted model comparison to validate the target and identify the signal gap. This can guide local work but does not satisfy the local/offline SOTA goal unless the final winning pipeline is approved under the goal constraints.

PROOF OF DONE:
1. Establish the current SOTA target before implementation:
   - Create `artifacts/sota_target.json` and `SOTA_TARGET.md`.
   - Verify the current public NFCorpus `nDCG@10` target from the best available live sources, in this order: official BEIR/eval.ai or MTEB task leaderboard if accessible; PapersWithCode/leaderboard mirrors if official sources are unavailable; recent peer-reviewed/preprint tables only if leaderboards cannot provide per-task NFCorpus numbers.
   - Record source URLs, access date, metric definition, task split, whether the target is retrieval-only or reranking, and the exact target value.
   - Do not use a stale remembered target without live verification.
2. Preserve the current baseline evidence:
   - Run `PYTHONPATH=src python3 scripts/probe_environment.py`.
   - Run `PYTHONPATH=src python3 scripts/run_nfcorpus_baseline.py`.
   - Confirm `artifacts/baseline_minilm_nfcorpus.json` records 323 queries, 0 failures, `Recall@100`, `nDCG@10`, dataset fingerprint, model path/config, and command.
3. Implement a benchmark runner that can compare named retrieval/reranking pipelines using the same dataset, qrels, metrics, and artifact schema:
   - Required output: `artifacts/runs/<run_id>.json` for each candidate.
   - Required index: `artifacts/leaderboard.json` sorted by `nDCG@10`.
   - Required summary: `SOTA_PROGRESS.md`.
4. Implement and benchmark candidate branches in staged ablations. At minimum:
   - current direct MiniLM dense baseline;
   - BM25 lexical baseline;
   - MiniLM + BM25 RRF;
   - at least one stronger general embedder available locally or installable without secrets, prioritized as BGE-M3, Qwen3-Embedding, Nomic, E5/GTE, or equivalent;
   - if feasible locally, one reranker branch such as BGE reranker, Qwen3 reranker, cross-encoder MiniLM, or equivalent;
   - if feasible locally, one query-expansion/HyDE branch;
   - if feasible locally, one hybrid branch combining dense + sparse + rerank/fusion.
5. For each branch, save:
   - command;
   - timestamp;
   - dependency/model availability;
   - model names and exact local paths/revisions where available;
   - retrieval mode;
   - hyperparameters;
   - latency/runtime summary;
   - query count;
   - failure count;
   - `nDCG@10`;
   - `Recall@100`;
   - per-query scores or enough per-query data to audit aggregate metrics.
6. Achieve SOTA:
   - The best local run must have `nDCG@10 >= artifacts/sota_target.json.primary_target.nDCG@10`.
   - If the selected public target uses percentages, normalize to decimal before comparison.
   - Ties count only if exact metric conventions match and the local run is reproducible.
   - Prefer beating target by at least `0.001` absolute `nDCG@10` to avoid rounding ambiguity.
7. Run verification:
   - `PYTHONPATH=src python3 -m unittest discover -s tests -v` exits 0.
   - `PYTHONPATH=src python3 -m compileall -q src scripts` exits 0.
   - The SOTA benchmark command exits 0 and regenerates the winning artifact.
   - `git diff --check` passes for every touched child git repo; if root remains non-git, state that explicitly.
8. Produce final artifacts:
   - `SOTA_RESULT.md` with target, winning score, comparison delta, command, artifacts, limitations, and exact rerun instructions.
   - `METHODOLOGY_STATUS.md` updated with the accepted SOTA result and deferred risks.
   - `PROGRESS.md` updated with command outputs and remaining risks.
   - `DECISIONS.md` updated for meaningful model, fusion, reranking, dependency, and benchmark-scope decisions.
9. Provide a final changed-file summary with exact verification commands and observed outcomes.

SCOPE:
- Modify root-level `turboragger` workspace files needed for SOTA benchmark execution, retrieval/reranking branches, tests, scripts, configs, artifacts, and documentation.
- Allowed paths include `src/`, `tests/`, `scripts/`, `artifacts/`, `configs/`, `pyproject.toml`, `PROGRESS.md`, `DECISIONS.md`, `METHODOLOGY_STATUS.md`, `SOTA_TARGET.md`, `SOTA_PROGRESS.md`, and `SOTA_RESULT.md`.
- Read/reference `turbovec/`, `Agent_Pidgeon/`, and `newragcity/` as source material.
- Child repo edits are allowed only when narrowly required and documented before/with the edit. Do not perform broad child-repo rewrites.
- Do not build product UI, hosted APIs, production deployment, or clinical decision-support behavior in this goal.

CONSTRAINTS:
- Evidence first: do not claim SOTA, “near SOTA,” “beats,” or “improved” without saved benchmark artifacts generated in the current run.
- Do not move the target after seeing results unless the target source is invalid; if the target changes, document why in `DECISIONS.md` and preserve the old target.
- Do not train, tune, or select using test qrels except for final evaluation. If fine-tuning is attempted, use only train/dev/synthetic data and document leakage safeguards.
- Do not use placeholder benchmark claims, unverifiable tables, screenshots, or manually edited result artifacts as proof.
- Preserve the same NFCorpus test split, qrels, and metric definitions across candidate runs.
- Track both `nDCG@10` and `Recall@100`; do not optimize one while hiding regressions in the other.
- Add dependencies only when necessary for candidate branches or benchmark verification, and justify them in `DECISIONS.md`.
- Prefer local/open-weight/offline-compatible models. Cloud models or paid APIs require explicit user approval and must not be used silently.
- Keep build validation staged: one candidate branch or fusion change at a time, with ablations.
- Do not remove or weaken tests to make the goal pass.
- Do not alter qrels, corpus text, query text, or metric functions to inflate scores.

SAFETY / PROVENANCE:
- Treat NFCorpus as a retrieval benchmark only, not clinical decision support.
- Do not implement treatment advice, dosing advice, patient-specific medical recommendations, or regulated workflow behavior.
- Preserve auditability: every benchmark score must link to command, config, code path, dataset fingerprint, model identity, and generated artifact.
- Preserve the Agent_Pidgeon boundary: if receipts are added later, exact resolver, policy, trust, and receipt semantics remain deterministic.
- Prefer explicit uncertainty over fake completeness.

ITERATION:
Before editing, inspect current `GOAL.md`, `METHODOLOGY_STATUS.md`, `PROGRESS.md`, `DECISIONS.md`, `REPO_MAP.md`, `RISK_NOTES.md`, `RANKED_OPTIONS.md`, `artifacts/baseline_minilm_nfcorpus.json`, and `artifacts/environment_probe.json`.
Work in stages:
1. SOTA target discovery and artifact creation.
2. Benchmark runner/generalized candidate artifact schema.
3. Baseline re-run and leaderboard creation.
4. Candidate embedder availability probe.
5. One stronger dense branch.
6. Sparse+dense RRF branch.
7. Reranker branch.
8. HyDE/query-expansion branch if feasible.
9. Best-fusion branch and final SOTA comparison.
After each stage:
- run the nearest relevant tests;
- run the relevant benchmark command;
- update `SOTA_PROGRESS.md` and `PROGRESS.md`;
- keep old artifacts rather than overwriting evidence unless the file is explicitly a latest pointer/index.
If a branch fails, document the failure and proceed to the next plausible branch unless the failure corrupts the benchmark surface.

STOP:
Pause and summarize instead of continuing if:
- current SOTA target cannot be established from any credible source;
- the same dependency/model installation failure persists after 3 distinct repair attempts;
- required models need credentials, paid APIs, restricted licenses, or network access not available in this environment;
- benchmark runtime becomes impractical without a product decision on scope, model size, or hardware;
- a candidate would require modifying qrels/corpus/query text or otherwise invalidating the benchmark;
- MLX/Metal or GPU access is required but unavailable and no CPU fallback is practical;
- achieving the target appears impossible after completing all feasible local candidate branches.

COMPLETE:
Mark complete only when every PROOF OF DONE item has fresh evidence and the saved winning local artifact has `nDCG@10` greater than or equal to the verified current SOTA target under matching metric/split rules. If all feasible local branches fail to reach SOTA, do not mark complete; produce `SOTA_ATTEMPT_REPORT.md` with the best score, target gap, failed branches, root causes, and recommended scope changes.

ASSUMPTIONS:
- Primary SOTA target is BEIR NFCorpus `test` `nDCG@10`.
- Secondary metric is `Recall@100`.
- Direct MiniLM baseline from the completed first goal is the current local starting point unless exact old newragcity reproduction is explicitly required.
- Local/open-weight/offline-compatible approaches are preferred.
- The intended methodology path is retrieval quality first, then reranking/fusion, then latency/storage optimizations such as `turbovec`.
