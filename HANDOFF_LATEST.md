# turboragger — Handoff Packet
**Generated:** 2026-06-07
**Branch:** N/A — `turboragger/` is a working directory, not a git repo (sub-repos are git; see below)
**Last Commit:** N/A for the container; sub-repo SHAs listed under Recent Changes

> **READ FIRST:** This is a **brainstorm/research session handoff, not a code handoff.**
> Zero code was written this session. The deliverable is a *validated design + sequenced
> plan*. The authoritative artifact is **`BRAINSTORM_turboragger.md`** (12 sections) plus
> persistent memory in `.claude/.../memory/`. This packet summarizes and points there.

---

## Quick Resume Checklist
- [ ] Read `BRAINSTORM_turboragger.md` in full (it is the design record; §2 diagnosis is load-bearing)
- [ ] Read the 5 memory files in `/Users/o2satz/.claude/projects/-Volumes-WS4TB-turboragger/memory/`
- [ ] Confirm the decision still stands: **the problem is retrieval RECALL (0.18), not generation**
- [ ] Before ANY build: execute **Step 0 — reproduce the 0.18 baseline** (see "Next Steps")
- [ ] Review "Current Blockers" and "Open Questions" below

## AI Continuity Checklist
- [x] Latest design doc reviewed (`BRAINSTORM_turboragger.md`)
- [x] Open assumptions imported (see Reality Checks / memory `turboragger-reality-checks`)
- [x] Open debt items imported (see §10 Open Attacks)
- [x] Open error references imported (the 0.18 recall measurement, the deepConf/MLX logprob conflict)
- [ ] Verification suite executed — **NOT RUN** (no code exists yet; Step 0 is the first verification)
- [x] Next actions prioritized (P0/P1/P2 below)

---

## What This Project Does
**turboragger** is a planned **private, air-gapped RAG system** for Apple M4 Pro (64 GB) that
fixes a measured retrieval-recall collapse (Recall@100 = 0.18 on BEIR nfcorpus) in prior
work (newragcity). It fans out a query across multiple retrieval methods in parallel, fuses
them with Reciprocal Rank Fusion, loops until every sub-claim has evidence, and emits a
cryptographic provenance receipt. A second, larger product (agent capability discovery via
Agent_Pidgeon A2A) shares the same fine-tuned-embedder engine.

**Tech Stack (target):** Python 3.11+, MLX (Apple Silicon), MLX-LM, sentence-transformers /
FlagEmbedding (PyTorch+MPS), turbovec (Rust+Python, TurboQuant), rank_bm25, pytrec_eval.
**Architecture Pattern:** fan-out / fuse retrieval engine ("uniform retriever contract" + RRF),
fully local. Two products on one shared embedder+fine-tune+receipt stack.

---

## Project Structure
```
turboragger/
├── BRAINSTORM_turboragger.md   ← ★ THE DESIGN RECORD (12 sections) — read this first
├── HANDOFF_2026-06-07.md       ← this file
├── 2605.22817v1.pdf            ← input: Vector Policy Optimization (VPO) paper
├── RAG_Engine_Corpus.md        ← input: Google Agentic RAG (SCA loop, cross-corpus routing)
├── .claude/                    ← session config
├── turbovec/      (git @main)  ← Rust+Python vector index on TurboQuant (4-bit, SIMD, allowlist)
├── Agent_Pidgeon/ (git @main)  ← deterministic semantic-contract + provenance/receipts layer
└── newragcity/    (git @main)  ← PRIOR WORK: DKR+Ersatz(LEANN+PageIndex+deepConf)+RoT+UltraRAG
                                   the 0.18 recall measured here = the reason this project exists
```

**Entry Points:** None yet (no turboragger code written). Sub-repos have their own.

**Key Modules / Assets:**
| Module | Path | Purpose | Status |
|--------|------|---------|--------|
| Design record | `BRAINSTORM_turboragger.md` | All decisions + sequenced plan | ✅ complete |
| turbovec | `turbovec/` | 4-bit vector storage (incl. ColBERT vecs later) | ✅ works, ⚠️ not built into local Python (needs maturin) |
| Agent_Pidgeon resolver | `Agent_Pidgeon/src/agent_pidgin/resolver.py` | exact `catalog.get(pointer)` — the sacred deterministic path | ✅ works |
| Agent_Pidgeon receipts | `Agent_Pidgeon/src/agent_pidgin/flight_recorder.py` | hash-chained provenance | ✅ works |
| deepConf | `newragcity/ersatz_rag/cognitron/cognitron/core/confidence.py` | confidence scoring | ❌ depends on cloud logprobs; dies on MLX (returns 0.5) |
| LEANN retriever | `newragcity/ersatz_rag/regulus/backend/app/leann_vector.py` | current dense retriever | ❌ recall 0.18; embedder is MiniLM-256tok |
| RoT | `newragcity/servers/rot_reasoning/` | visual reasoning compression | ❌ untrained scaffolding, no VL model; SHELVED |
| BEIR harness | `newragcity/ersatz_rag/regulus/backend/benchmarks/` | recall/nDCG scoring | ⚠️ exists; nfcorpus data location unconfirmed |

---

## How to Run

### Local Development
```bash
# No turboragger code exists yet. The first "run" is reproducing the baseline (Step 0).
# Environment probe results (verified 2026-06-07, M4 Pro 64GB):
#   installed: mlx, mlx-lm, mlx-vlm, vllm(CPU), transformers 5.9, sentence-transformers(5.5 — but
#              NOT importable in the python used for probes), rank_bm25, pytrec_eval
#   NOT importable in probe python: turbovec, beir, sentence_transformers
#   cached embed models: all-MiniLM-L6-v2 (weak), nomic-ai (strong, on disk)
#   local LLMs: Qwen3.6-35B-A3B-MLX-oQ8, privacy-filter-nemotron-mlx-8bit
```

### Tests
```bash
# No turboragger tests. Sub-repo tests exist (e.g. Agent_Pidgeon):
cd Agent_Pidgeon && python3 -m unittest discover -s tests -v
```
**Current Status:** N/A for turboragger (no code).
**Known Failures:** newragcity unified benchmarks return HARDCODED placeholder data (ndcg=0.463) —
do NOT trust them. Only the real BEIR nfcorpus run (recall 0.18) and component unit tests are real.

### Verification Suite
```bash
# THE first verification to build (Step 0): reproduce Recall@100 = 0.18 on nfcorpus
# with the current MiniLM dense retriever, using pytrec_eval (installed).
```
**Pass Condition:** measured Recall@100 ≈ 0.18 ± noise on nfcorpus. If it can't be reproduced,
the system is not understood — STOP and resolve before building.

---

## Current State Assessment

### What's Working ✅
- **The diagnosis** — Recall@100=0.18 vs nDCG@10=0.51 on nfcorpus is a real measurement (the
  retriever is *blind*: misses 82% of relevant docs before ranking).
- **The architecture** — fan-out/fuse engine with uniform `retrieve(query,k)->[(id,score)]`
  contract + RRF fusion. Design complete, validated against probes.
- **turbovec, Agent_Pidgeon resolver + receipts** — real, tested code (in their own repos).

### What's Incomplete ⚠️
- **Everything in turboragger is design-only.** No harness, no retrievers, no fusion code yet.
- **Embedder decision unmade** — fork between BGE-M3 (PyTorch/MPS, 3-modes-in-one) vs
  Qwen3-Embedding-8B (native MLX, single-mode) vs medical (MedCPT/BMRetriever) dense + BGE hybrid.

### What's Broken ❌
- **deepConf on local/MLX** — reads OpenAI/Anthropic `top_logprobs`; MLX exposes none → returns
  constant 0.5. Privacy and logprob-confidence conflict. Mitigation: agreement-based confidence.
- **LEANN/MiniLM retriever** — 0.18 recall; MiniLM truncates at 256 tokens (suspect #1).
- **RoT** — untrained, no Qwen2.5-VL model present. Shelved for v1.

### Current Blockers 🚧
- **Baseline reproducibility** — nfcorpus data location and exact config of the original 0.18 run
  unconfirmed. Step 0 may require re-establishing the baseline cleanly first.
- **Embedder fork** — needs a measured decision (Recall@100 on nfcorpus), not an MTEB-average guess.

### Feature Completion Matrix
| Feature | Status | Evidence | Gap to Done | Priority |
|---------|--------|----------|-------------|----------|
| Recall diagnosis | ✅ | `newragcity/.../beir_unified_results.json` (0.18) | none | — |
| Fan-out/fuse architecture | ✅ design | `BRAINSTORM_turboragger.md §6,§7` | implementation | P0 |
| Step 0 baseline reproduction | ❌ | — | build + run scoreboard | P0 |
| Retriever contract + RRF harness | ❌ | `BRAINSTORM §7` | build (Wave 1 step 1) | P0 |
| BGE-M3 embedder swap | ❌ | `BRAINSTORM §11` | decide fork + integrate | P1 |
| Hybrid (dense+sparse+ColBERT) | ❌ | `BRAINSTORM §11` | flag on BGE-M3 | P1 |
| HyDE branch | ❌ | `BRAINSTORM §5 B1` | ~20 lines mlx-lm | P2 |
| turbovec storage for ColBERT vecs | ❌ | `BRAINSTORM §5 B2★` | maturin build + integrate | P2 |
| Coverage loop (B3) | ❌ | `BRAINSTORM §5 B3` | Wave 4 | P2 |
| Graph-hop (B4) | ❌ | `BRAINSTORM §5 B4` | Wave 3 | P2 |
| Embedder fine-tune (synthetic) | ❌ | `BRAINSTORM §12` | Wave 5 (LAST) | P2 |
| Agent_Pidgeon A2A discovery | ❌ | `BRAINSTORM §12` | separate product, after RAG | P2 |
| Provenance receipts on answers | ❌ | `BRAINSTORM §5 M5′` | wire Agent_Pidgeon | P2 |

---

## Recent Changes
This session produced documents only (no code). Sub-repo states for reference:

| Date | SHA | Repo | Note |
|------|-----|------|------|
| 2026-05-30 | efe29a1 | turbovec | Release 0.7.0 (Py) / 0.8.0 (Rust) |
| 2026-05-09 | 5d3e36e | Agent_Pidgeon | Variant B gap-mitigation merged |
| 2026-01-29 | 372b752 | newragcity | Qwen3 upgrade plan (note: stale; predates this diagnosis) |

**This session's artifacts (uncommitted, in `turboragger/` working dir):**
- `BRAINSTORM_turboragger.md` (created + extended to 12 sections)
- `HANDOFF_2026-06-07.md` (this file)
- 5 memory files under `.claude/projects/-Volumes-WS4TB-turboragger/memory/`

**Stashed Work:** none.

---

## Configuration & Secrets
### Environment Variables
| Variable | Purpose | Where to Get |
|----------|---------|--------------|
| (none required for local/air-gapped v1) | — | the privacy pitch = no cloud keys needed |
| `OPENROUTER_API_KEY` | only if using cloud LLMs (against the air-gapped goal) | user selects models via OpenRouter |

### External Dependencies
| Service | Purpose | Local Alternative |
|---------|---------|-------------------|
| (none — by design) | air-gapped | MLX local models already on disk |

---

## Known Issues & Tech Debt (from §10 Open Attacks — NOT yet addressed)
- [ ] RRF fusing 5 noisy retrievers can HURT precision even as recall rises (measure nDCG@10 alongside).
- [ ] Coverage loop (B3) can run forever — needs a budget/termination guarantee.
- [ ] Graph extraction at ingest (B4) is expensive and quality-sensitive.
- [ ] Agreement-confidence fails when methods agree on the same WRONG chunk (correlated errors).
- [ ] Multi-vector (ColBERT) storage even at 4-bit is N_tokens× larger — does it fit 64GB at target
      corpus size? Needs real measurement. This is where turbovec/TurboQuant must prove itself.
- [ ] Product vs technique: moat beyond "good recall" is the air-gapped + provenance + medical combo.

---

## Next Steps (Priority Order)
1. **P0 — Step 0: reproduce the 0.18 baseline.** Build a minimal scoreboard (pytrec_eval is
   installed) that runs the current MiniLM dense retriever on nfcorpus and measures Recall@100.
   "Done" = number ≈ 0.18 reproduced. *Nothing else proceeds until this is real.*
2. **P0 — Build the fan-out harness skeleton.** Uniform `retrieve(query,k)->[(id,score)]`
   contract + RRF fusion, with dense as the only branch. "Done" = harness reproduces 0.18
   (proves it's neutral). This is the PERMANENT skeleton every wave plugs into.
3. **P1 — Resolve the embedder fork by MEASUREMENT.** Swap MiniLM → BGE-M3 dense (and/or a
   medical model); measure Recall@100 on nfcorpus. "Done" = a chosen embedder with a number.
   Likely the single biggest jump (MiniLM's 256-token truncation is suspect #1).
4. **P1 — Add hybrid (BGE-M3 sparse + ColBERT) branches + RRF.** "Done" = recall rises again
   (the +2-5pt the BGE-M3 paper promises); this also delivers the "moat" nearly free.
5. **P2 — HyDE, then graph-hop, then coverage loop, then fine-tune (Wave 5), then receipts.**
   Fine-tune LAST (don't fine-tune a weak model; don't bake in assumptions before pipeline stable).

**Build discipline (from the project's own scar tissue):** architecture is fully parallel;
**validation is staged** — turn ONE branch on at a time against the fixed 0.18 baseline so each
proves its recall contribution. newragcity suffered "5 catastrophic drifts" from changing
components in isolation without this discipline. Each step validated before the next (per CLAUDE.md).

---

## Key Files Reference
| File | Purpose | When to Modify |
|------|---------|----------------|
| `BRAINSTORM_turboragger.md` | the design record | when a design decision changes |
| `.claude/.../memory/MEMORY.md` | memory index | when adding/removing a memory |
| `newragcity/.../leann_vector.py` | current dense retriever (the thing to replace) | Step 3 embedder swap |
| `Agent_Pidgeon/src/agent_pidgin/resolver.py` | the SACRED deterministic path | NEVER put embeddings here |
| `turbovec/` (README) | 4-bit vector index | when wiring ColBERT vec storage |

---

## Open Questions / Decisions Needed
- **Embedder fork** — BGE-M3 (PyTorch/MPS, 3-in-1) vs Qwen3-Embedding-8B (native MLX, single-mode)
  vs medical (MedCPT/BMRetriever) + BGE hybrid? **Decide by Recall@100 on nfcorpus, not MTEB.**
- **Confidence approach** — agreement-based (works on MLX, air-gapped) vs switch to vllm for
  logprobs (heavier, but enables deepConf)? Leaning agreement-based (preserves privacy pitch).
- **Two products, one team** — RAG (nearer revenue) vs A2A capability discovery (bigger vision).
  Working assumption: RAG first funds/validates the shared embedder+fine-tune+receipt stack;
  discovery reuses it. NOT parallel.
- **Baseline data** — is the original nfcorpus data + config recoverable, or must Step 0
  re-establish the baseline from scratch?

---

## Appendix: Machine-Readable Summary
```json
{
  "project": "turboragger",
  "generated": "2026-06-07",
  "session_type": "brainstorm_research_no_code",
  "repo": {
    "branch": "n/a (working dir, not git)",
    "commit": "n/a",
    "commit_date": "n/a",
    "uncommitted_changes": true,
    "stashed_work": 0,
    "subrepos": {
      "turbovec": {"sha": "efe29a1", "branch": "main"},
      "Agent_Pidgeon": {"sha": "5d3e36e", "branch": "main"},
      "newragcity": {"sha": "372b752", "branch": "main"}
    }
  },
  "stack": {
    "language": "python",
    "language_version": "3.11+",
    "framework": "MLX + FlagEmbedding/sentence-transformers + turbovec",
    "framework_version": "mlx 0.31, mlx-lm 0.31.3, transformers 5.9"
  },
  "health": {
    "tests_passing": null,
    "tests_failing": null,
    "tests_skipped": null,
    "lint_clean": null,
    "type_check_clean": null,
    "note": "no turboragger code exists yet; nothing to test"
  },
  "status": {
    "working": ["recall_diagnosis", "fan_out_fuse_architecture_design", "turbovec", "agent_pidgeon_resolver_receipts"],
    "incomplete": ["all_turboragger_implementation", "embedder_decision"],
    "broken": ["deepConf_on_mlx", "leann_minilm_retriever_0.18", "rot_untrained"],
    "blockers": ["baseline_reproducibility_unconfirmed", "embedder_fork_needs_measurement"]
  },
  "continuity": {
    "previous_handoff_loaded": false,
    "design_record": "BRAINSTORM_turboragger.md",
    "memory_files": 5,
    "assumptions_imported": 7,
    "debt_items_imported": 6,
    "error_refs_imported": 3
  },
  "key_finding": "Recall@100=0.18 vs nDCG@10=0.51 on BEIR nfcorpus — retriever is blind, misses 82% of relevant docs. Fix recall first; generation/confidence is downstream polish.",
  "feature_completion_matrix": [
    {"feature": "recall_diagnosis", "status": "✅", "evidence": "newragcity/.../beir_unified_results.json", "priority": "P0"},
    {"feature": "fan_out_fuse_architecture", "status": "✅design", "evidence": "BRAINSTORM §6,§7", "priority": "P0"},
    {"feature": "step0_baseline_reproduction", "status": "❌", "evidence": "-", "priority": "P0"},
    {"feature": "retriever_contract_rrf_harness", "status": "❌", "evidence": "BRAINSTORM §7", "priority": "P0"},
    {"feature": "bge_m3_embedder_swap", "status": "❌", "evidence": "BRAINSTORM §11", "priority": "P1"},
    {"feature": "hybrid_dense_sparse_colbert", "status": "❌", "evidence": "BRAINSTORM §11", "priority": "P1"},
    {"feature": "hyde_branch", "status": "❌", "evidence": "BRAINSTORM §5 B1", "priority": "P2"},
    {"feature": "turbovec_colbert_storage", "status": "❌", "evidence": "BRAINSTORM §5 B2", "priority": "P2"},
    {"feature": "coverage_loop_b3", "status": "❌", "evidence": "BRAINSTORM §5 B3", "priority": "P2"},
    {"feature": "graph_hop_b4", "status": "❌", "evidence": "BRAINSTORM §5 B4", "priority": "P2"},
    {"feature": "embedder_finetune_synthetic", "status": "❌", "evidence": "BRAINSTORM §12", "priority": "P2"},
    {"feature": "a2a_capability_discovery", "status": "❌", "evidence": "BRAINSTORM §12", "priority": "P2"},
    {"feature": "provenance_receipts_on_answers", "status": "❌", "evidence": "BRAINSTORM §5 M5prime", "priority": "P2"}
  ],
  "verification_suite": {
    "command": "build + run nfcorpus Recall@100 scoreboard (pytrec_eval) on current MiniLM dense",
    "pass_condition": "Recall@100 ~= 0.18 reproduced",
    "result": "not_run"
  },
  "next_steps": [
    {"task": "Step 0: reproduce 0.18 baseline with scoreboard", "priority": "P0", "scope": "small"},
    {"task": "Build fan-out harness skeleton (contract + RRF, dense-only)", "priority": "P0", "scope": "medium"},
    {"task": "Resolve embedder fork by measuring Recall@100 on nfcorpus (BGE-M3/medical)", "priority": "P1", "scope": "medium"},
    {"task": "Add hybrid dense+sparse+ColBERT branches + RRF", "priority": "P1", "scope": "medium"},
    {"task": "HyDE, graph-hop, coverage loop, fine-tune (Wave5), receipts", "priority": "P2", "scope": "large"}
  ]
}
```
