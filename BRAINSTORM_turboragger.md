# turboragger — Brainstorm & Architecture Notes

**Date:** 2026-06-07
**Status:** Brainstorm captured. No code built yet. This is the design record of the session.
**Machine target:** Apple M4 Pro, 64 GB unified memory. Fully local / air-gapped.

---

## 0. Purpose of this document

Preserve the diagnosis, the rejected ideas (and *why* they were rejected), and the
surviving breakthrough architecture produced during the brainstorm. Read this before
resuming so we do not re-derive or re-litigate decisions already made.

---

## 1. The inputs we assessed

| Input | What it is | What it gives turboragger |
|---|---|---|
| **turbovec** (repo) | Rust+Python vector index on Google's TurboQuant (data-oblivious 2/4-bit quant, SIMD scoring, **search-time allowlist filtering in-kernel**, online ingest, no train step) | The storage/speed primitive. 10M docs in 4 GB. Allowlist-in-kernel = near-free hybrid/ACL filtering. Drop-in adapters for LangChain/LlamaIndex/Haystack/Agno. **Not yet built into local Python — needs `maturin` build.** |
| **RAG_Engine_Corpus.md** | Google Agentic RAG (Vertex) docs | The orchestration pattern: planner routes across corpora by `description`; **Reasoning Agent with Sufficient Context Awareness (SCA)** loops retrieval until context is sufficient. |
| **2605.22817v1.pdf** | **Vector Policy Optimization (VPO)** paper | When an LLM is wrapped in test-time *search*, post-training should maximize **diversity of a competent solution set** (Pareto front of a *vector-valued* reward), not collapse to one best answer. Multi-answer-in-one-rollout + stochastic reward scalarization (w ~ Dir(α)). Beats GRPO on best@k; **gap widens as search budget grows.** Caveat: benefit **shrinks when reward components are collinear** (simplex collapses to a line). |
| **MLX stack** (installed) | mlx, mlx-lm, mlx-vlm, mlx-audio, mlx-openai-server, vllm (CPU), sentence-transformers, transformers 5.9, ollama | On-device inference AND fine-tune (`dflash-mlx`, `mlx-tune`). Already have quantized models (Qwen3.6-35B-A3B-MLX-oQ8, Nemotron 8-bit). |
| **newragcity** (prior work) | Working multi-method RAG: **DKR** (deterministic exact lookup), **Ersatz** (LEANN vector + PageIndex + **deepConf**), **RoT** (render-of-thought visual compression), **UltraRAG** MCP orchestration (YAML loops/branches), **The Vault** (parallel→aggregate→`min()` confidence→audit trail) | ~70% of the substrate already exists. BUT see §3 reality checks. |
| **Agent_Pidgeon** (prior work) | Deterministic semantic-contract + provenance layer: receipts, **hash-chained flight-recorder traces**, policy enforcement (block high-risk actions pre-execution), semantic diff, HMAC catalog trust | Real, tested provenance machinery. Makes any answer cryptographically auditable. |

---

## 2. THE DIAGNOSIS (the most important finding)

Real measured result from newragcity on **BEIR nfcorpus** (medical corpus, 323 queries):

```
system: ThreeApproachRAG (PageIndex + LEANN + deepConf)
nDCG@10   = 0.51   ← ranking is OK when the answer is present
Recall@100 = 0.18  ← 82% of relevant docs NEVER enter the top 100
```

**Interpretation:** LEANN is not slow-and-mediocre at ranking. It is *blind*. The
retriever misses 4 of 5 relevant docs **before** ranking happens. No reranker,
confidence gate, VPO, or clever generation can rescue an answer built on documents
that were never retrieved.

**=> The breakthrough is not a better generator or confidence layer. It is solving
the recall collapse. Everything downstream is polishing a stone missing 82% of its mass.**

Why single-dense-vector recall caps out: one vector per chunk only finds *globally*
similar docs. It misses (a) rare-term matches (BM25 catches these), (b) second-hop
relevance ("the drug that treats X"), (c) vocabulary mismatch (query and answer use
different words). nfcorpus is medical = worst case for vocab mismatch = exactly the
ABXorcist/AETOS domain.

---

## 3. Reality checks that killed earlier ideas (probe results — verified, not assumed)

| Assumption (earlier pitch) | Probe result | Consequence |
|---|---|---|
| deepConf = ready-made reward/confidence layer | deepConf reads **OpenAI/Anthropic API logprobs** (`top_logprobs` dicts; returns 0.5 if absent) | **MLX exposes NO token logprobs.** Going local kills deepConf → constant 0.5. Privacy vs logprob-confidence are in direct conflict via this code path. |
| turbovec is a drop-in swap | **Not importable in local Python.** Rust crate, needs `maturin` build. | Buildable, but "drop-in" was fiction. |
| RoT visual compression is usable | Real code (`cot_compressor_v2.py`, PIL, SAM/CLIP/Projector) but **untrained**; **no Qwen2.5-VL model present** in /models | RoT is untrained scaffolding. **Shelved for v1.** |
| Agent_Pidgeon semantic_diff = hallucination tripwire | Actual code `diff_step_lists` diffs **lists of catalog pointers** (removed guardrails between agent plans), **NOT free-text answer vs source chunks** | The "tripwire" must be *built* (local NLI), not wired up. The **receipt/hash-chain machinery is real** and survives. |
| newragcity benchmarks validate the system | `ACTUAL_BENCHMARK_RESULTS.md`: unified benchmarks return **hardcoded PLACEHOLDER data** (ndcg=0.463). Only component unit tests + the BEIR nfcorpus run (0.18 recall) are real. | Trust only the 0.18 number; it's the real one. |
| mlx-lm supports speculative decode | `stream_generate(... draft_model ...)` param exists | Speculative decode via draft model is real. True trained MTP heads (Medusa/EAGLE) are not one-liners. |

---

## 4. Ideas considered and their verdicts

**VPO inside The Vault (M1′)** — FATAL for v1. VPO is RL *training* (G rollouts × m answers
× K scalarizations + reward model). Not a laptop afternoon; weeks of uncertain work.
Cheap version (just prompt for 5 answers) collapses to paraphrases (paper's own Multi-RLVR
result). **Demoted to research bet.**

**Pareto-Retrieval / VPO axes (M2′)** — WOUNDED. DKR-exact and Ersatz-semantic are
*collinear* on most queries — precisely where the paper says VPO fails. **Demoted.**

**turbovec replaces LEANN (M3′)** — Reframed. Don't rip out (drift risk; repo had "5
catastrophic drifts" from isolated component changes). **Add as parallel branch, A/B on
golden set, keep LEANN until turbovec measurably wins on OUR data.** BUT see §2: LEANN's
problem is recall, not just speed — so a *better retrieval paradigm* (multi-vector) matters
more than a faster single-vector index.

**RoT × VPO (M4′)** — FATAL. Multiplies two unproven training pipelines. **Cut.**

**Provenance-Verified RAG (M5′)** — SURVIVES (partially). Keep the real part:
hash-chained receipt of the *retrieval path* {query, corpus IDs, chunk hashes, winning
method, confidence, policy decision}. The hallucination-detection diff is an aspirational
add-on requiring a local NLI model.

---

## 5. THE BREAKTHROUGH — five recall directions

All aimed at the 0.18 recall collapse. All local-feasible.

- **B1 — Hypothetical-document / query-imagination recall.** Don't search with the query;
  have the local LLM write a fake ideal answer (HyDE) + generate diverse query rewrites
  (VPO's diversity idea applied to *queries* — short, cheap, collapse-free). Union results.
  Attacks vocabulary mismatch directly. *"Imagines the document that would answer you, then finds that."*

- **B2 ★ — Quantized multi-vector late interaction (THE MOAT).** ColBERT-style: one vector
  *per token*, match at token granularity (MaxSim). Known SOTA recall jump (+15–25 pts on
  BEIR over single-vector). Catch = storage/speed — **exactly what TurboQuant/turbovec
  solves.** The real meld: multi-vector recall (quality) made affordable by 4-bit quant
  (storage). **Nobody has shipped quantized late-interaction, air-gapped, on a laptop.**

- **B3 — Recall-driven self-correcting loop.** Google's SCA re-queries on insufficient
  *context*; ours re-queries on suspect *recall*. LLM decomposes query into sub-claims;
  for each uncovered sub-claim, spawn a targeted search. Trigger = **coverage of decomposed
  sub-questions**, not vibes.

- **B4 — Graph-hop retrieval.** Second-hop misses need a graph. Extract entities/relations
  at ingest (local LLM, one-time) → at query, vector-retrieve seeds, then **walk graph 1–2
  hops** to pull connected chunks vector search can't reach. GraphRAG, but fully local +
  incremental (turbovec online ingest = graph grows without rebuilds).

- **B5 — Ensemble-recall union (cheapest, highest floor).** dense + BM25 + RRF. **DKR is
  already a sparse retriever** — you have both halves but route between them instead of
  fusing. Union DKR(sparse) + Ersatz(dense) + B1(HyDE), RRF. Likely 0.18 → 0.40+ with
  code you mostly have.

### The two starred items = patentable/publishable core
**Quantized multi-vector late-interaction retrieval running air-gapped on a laptop**,
directly fixing the measured 0.18 recall.

---

## 6. THE ARCHITECTURE — fan-out / fuse recall engine

```
Query
  │
  ▼  ① decompose into sub-claims (local LLM)         ← B3 trigger
  │
  ▼  ② per sub-claim: diverse queries + HyDE         ← B1 (cheap diversity)
  │      (N sub-queries)
  │
  ▼  ③ EACH sub-query fans out IN PARALLEL to:       ← runtime-parallel
  │      • sparse / exact (DKR)                       ← B5
  │      • dense (turbovec 4-bit)                     ← B5
  │      • HyDE                                       ← B1
  │      • multi-vector late interaction (ColBERT)    ← B2 ★ recall breakthrough
  │           on TurboQuant 4-bit (turbovec)          ← ★ storage breakthrough
  │      • graph 1–2 hop expansion                    ← B4
  │
  ▼  ④ RRF FUSION → high-recall candidate pool        ← rank-based, scale-free
  │
  ▼  ⑤ coverage check: every sub-claim has evidence?  ← B3 loop
  │      NO → spawn targeted search, goto ③
  │      YES ↓
  ▼  ⑥ answer + agreement-confidence + Agent_Pidgeon receipt
```

**Why RRF is the enabler:** Reciprocal Rank Fusion combines lists by *rank position*, not
score. BM25 (0–50), cosine (0–1), ColBERT MaxSim (unbounded) are incomparable as scores
but perfectly comparable as ranks. This is the technical reason five wildly different
retrievers fuse in one line — **the glue that makes "run them all in parallel" possible.**

**Uniform retriever interface:** every method is `query -> [(doc_id, score)]`. Adding a
method = one more fan-out branch. Removing = delete a branch. No method knows about another.

---

## 7. PARALLELIZATION — the key discipline

Two meanings of "parallel"; only one is safe:

- **Parallel-at-RUNTIME** ✅ = the product. All retrievers fire per query, fuse via RRF.
  Recall rises *because* nothing is sequential. This IS the breakthrough.
- **Parallel-at-BUILD** ⚠️ = the trap. Building all 5 untested, wiring, hoping. Recall
  stays 0.18 and you can't tell which of five broke it. This is the "5 catastrophic drifts."

**Resolution: architecture fully parallel; validation staged.** Build the fan-out harness
parallel from line one; turn branches on one at a time against the fixed 0.18 baseline so
each proves its recall contribution. Same end state, no mystery failures. **The architecture
never changes between waves — you flip branches on, you don't rebuild.**

| Wave | Branches lit | Proves | Gate (recall@100) |
|---|---|---|---|
| 0 | harness + fusion + dense only | reproduce baseline | = 0.18 |
| 1 | + sparse (DKR) + HyDE | cheap floor-raise (B5+B1) | ~0.40 |
| 2 | + multi-vector on turbovec (B2★) | the moat | ~0.55+ |
| 3 | + graph-hop (B4) | second-hop coverage | multi-hop recall ↑ |
| 4 | + coverage loop (B3) | self-digging | fewer zero-evidence sub-claims |

---

## 8. MITIGATIONS carried forward

| Problem | Mitigation (real) |
|---|---|
| MLX has no logprobs → deepConf dies local | Replace logprob-confidence with **agreement-based confidence** (cross-candidate self-consistency) + retrieval score + local **NLI entailment** (answer vs chunk). Gets *better* with more candidates — which the fan-out already produces. (Alt: use vllm which exposes logprobs, but heavier.) |
| VPO needs RL training | Don't train. Get diversity at **inference**: diverse decoding (temp + MMR/DPP) + diverse prompting + **diversity-by-method** (each candidate from a different retrieval core). |
| Multi-answer collapse | Force diversity *structurally* — route each candidate through a different core. Diversity from method, not from model trying to be diverse. |
| turbovec swap = drift risk | Add as parallel branch; A/B vs LEANN on golden set; keep LEANN until turbovec wins on OUR data. |
| RoT untrained, no VL model | Shelved for v1. Compression is an optimization, not a feature. |
| semantic_diff ≠ answer checker | Keep real receipts; build NLI tripwire separately and honestly. |

---

## 9. THE WOW STORY (one demo, four moments)

Ask a hard medical question → watch it **decompose and fan out** → **recall counter climbs
past where normal RAG dies (0.18 → 0.55)** → answer appears with a **signed receipt** →
**"and nothing left this laptop."**

1. *Finds what others miss* — the recall number climbing live (THE PROOF)
2. *Laptop-scale impossible* — quantized multi-vector, air-gapped, on one M4 (THE HOW)
3. *Self-digging agent* — decompose + re-search loop (THE BEHAVIOR)
4. *Private + provable* — the receipt (THE TRUST)

**One-liner:** *A private RAG that finds the 80% of relevant documents everyone else's
search silently misses — because it matches at the word level, not the paragraph level,
and keeps searching until every part of your question has evidence. All on one laptop,
with a receipt proving where every fact came from.*

---

## 11. EMBEDDER FINDINGS (2026-06-07 web search) — MAJOR PIVOT

**Root-cause suspect confirmed:** current dense embedder is `all-MiniLM-L6-v2` —
384-dim, **256-token context (truncates most of each chunk before embedding)**, 2021-era,
~50MB, positioned as "MVP starter, not production." This alone is a prime suspect for the
0.18 recall (§2). Found via `ersatz_rag/regulus/backend/app/leann_vector.py:10`.

### ★ BGE-M3 collapses 3 of the 5 fan-out branches into ONE model
`BAAI/bge-m3` emits **dense + sparse + ColBERT multi-vector in a single forward pass**:
```python
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
out = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=True)
# → dense_vecs (1024-dim), lexical_weights (learned sparse, better than BM25), colbert_vecs
```
- 1024-dim, **8192-token context** (32× MiniLM's 256).
- MTEB retrieval nDCG@10 ≈ 63.0; **hybrid (all 3 modes) beats any single mode by 2–5 nDCG pts (BGE-M3 paper).**
- **Consequence: the B2★ "moat" (multi-vector ColBERT) is NOT a from-scratch build — it's a
  flag on an off-the-shelf model.** turbovec's job shifts to *storing* bge colbert_vecs at
  4-bit (exactly what TurboQuant is for). Wave 1 and Wave 2 effectively MERGE.

### Embedder comparison
| Model | Dim | Context | MTEB nDCG@10 | Notes |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (current) | 384 | 256 | low ("starter") | the suspect |
| nomic-embed-v1.5 (cached on disk) | 768 | 8192 | 62.4 | single-mode upgrade |
| **BGE-M3** | 1024 | 8192 | 63.0 +hybrid 2–5pt | dense+sparse+ColBERT in one |
| Qwen3-Embedding-8B | — | 32K | **70.6 (MTEB #1)** | native MLX 4-bit, 44K tok/s, but single-mode |

### Medical-domain models (nfcorpus + ABXorcist/AETOS are medical → domain models likely win)
- **MedCPT** (220M) — contrastive-trained on 255M PubMed click logs, SOTA on biomedical IR.
- **BMRetriever** (410M / 2B) — LLM-backbone biomedical retriever; 410M beats baselines up
  to 11.7× larger; reported to retrieve symptom/medication-relevant passages where general
  models drift. More recent (2024-25) than MedCPT.

### THE REAL FORK (decide before committing)
1. **BGE-M3 on PyTorch/MPS** — one model, all 3 modes, hybrid built in. NOT native MLX
   (runs on M4 via MPS). Best architecture fit.
2. **Qwen3-Embedding on native MLX** — higher single-mode score, fast, but build sparse +
   multi-vector separately.
3. **Medical-domain (MedCPT/BMRetriever) dense + BGE-M3 sparse/colbert** — best recall on
   medical data, hybrid. Most promising for the actual use case; needs measurement.

Decision rule: **whichever wins Recall@100 on nfcorpus.** MTEB averages don't count here.

### Revised Wave 1 (cheaper + stronger than §7)
| Step | Action | Why |
|---|---|---|
| 0 | reproduce 0.18 with MiniLM | baseline must be real |
| 1 | harness + retriever contract + RRF, dense-only | neutral skeleton |
| 2 | swap MiniLM → BGE-M3 dense (or medical model) | bigger model + 32× context — likely biggest single jump |
| 3 | add BGE-M3 sparse + ColBERT (same model, already loaded) | hybrid +2–5pt AND the moat, nearly free |
| 4 | add HyDE | vocab-mismatch residual |

### Caveats
- FlagEmbedding = PyTorch/transformers (MPS on M4), not native MLX. `sentence_transformers`
  NOT currently installed in this Python; `rank_bm25` + `pytrec_eval` ARE.
- BGE-M3 colbert_vecs = per-token → storage grows → makes the "fits in 64 GB?" question
  (§10) MORE pressing. This is where turbovec/TurboQuant earns its place.
- nfcorpus measured number is the only score that counts.

Sources: BGE-M3 HF + paper (arxiv 2402.03216), MTEB Mar-2026 leaderboard, Qwen3-Embedding
MLX server (github jakedahn/qwen3-embeddings-mlx), BMRetriever (arxiv 2404.18443), MedCPT.

---

## 12. FINE-TUNING THE EMBEDDER — value + the Agent_Pidgeon A2A extension

### Fine-tuning value (web search 2026-06): HIGH, but do it LAST
Domain fine-tuning is the highest-leverage recall lever reported, strongest in exactly our
conditions (medical/specialized vocab + low recall). Reported gains: +33% recall;
+26% Recall@60 (0.751→0.951, Atlassian JIRA); +10% Recall@10/nDCG@10; medical precision
78%→87%. Trigger checklist (we tick every box): specialized vocabulary (medical/ICD), low
recall (our 0.18), domain-specific content.

**No labels needed — synthetic query generation:** feed each chunk to LLM "generate 3
realistic questions whose answer is in this passage" → (question, chunk) positive pairs.
~500 docs → 5000+ pairs. **On our stack the local MLX LLM generates these FREE + air-gapped**
(cloud recipes quote $10-30 API; ours is $0 and private). mlx-tune trains in <1 day on M4.

**Sequencing (critical): fine-tune LAST.** (1) Must know real baseline first — don't
fine-tune the weak MiniLM; fine-tune the strong off-the-shelf model only if it still falls
short. (2) It bakes in chunking/model assumptions — redo if pipeline changes. (3) Can mask
architecture bugs. Slots in as **Wave 5**, after off-the-shelf swap + hybrid + HyDE.

**Product moat:** a hospital CANNOT send records to OpenAI to fine-tune. We fine-tune
on-prem, air-gapped, on their actual corpus. "An embedder trained on your private data
without your data leaving the building" is structurally impossible for cloud players.

### THE A2A EXTENSION — same embedder, two products
The SAME fine-tuning recipe serves Agent_Pidgeon A2A: fine-tune on (synthetic intent
phrasing → catalog pointer) pairs. Generate the same way: feed each catalog entry
(pointer + description + type_signature — verified present in catalogs/core.json) to the
local LLM "write 5 ways an agent might describe wanting to do this." One MLX fine-tuning
capability, two corpora.

**THE SACRED BOUNDARY (non-negotiable):** Agent_Pidgeon's resolver is `catalog.get(pointer)`
— EXACT dict lookup, KeyError on miss, deterministic by design ("LLMs do not define pointer
truth"). Embeddings may cross the boundary in ONE direction only:
> Embeddings help an intent *find* a contract. They NEVER help a contract *prove* itself.
> Discovery is fuzzy; resolution, trust, receipts, policy stay EXACT.
Putting an embedding inside the resolution/trust path destroys the repo's whole value
(a receipt saying "85% similar to clinical.phi.scrub" is worthless for audit).

**Architecture:**
```
free-text intent / A2A    ┌─ FUZZY (fine-tuned embed) ─┐   ┌─ EXACT (unchanged) ──┐
task description ───────▶ │ semantic pointer SUGGESTION │─▶ │ resolver.get(pointer) │─▶ receipt
                          │ (top-k candidates)          │   │ deterministic + hash  │
                          └─────────────────────────────┘   └───────────────────────┘
                           embeds pointer+description          proposes → human/policy
                           +type_signature                     CONFIRMS → then resolve
```

**Five A2A uses, ranked by safety:**
1. ✅ Intent → pointer suggestion (embed suggests, resolver decides) — strongest
2. ✅ **Cross-agent capability discovery** (A2A peer: "got a contract for de-identifying
   records?" → semantic search over peer catalogs). = Google cross-corpus routing-by-
   description applied to AGENT CAPABILITIES. **May be a bigger product than RAG — see §13.**
3. ✅ Duplicate/overlap detection at catalog authoring time
4. ⚠️ Drift-detection SIGNAL only (deterministic semantic_diff still gives the verdict)
5. ⚠️ Anomaly screening on proposed actions (soft pre-filter; hard policy gate unchanged)

Sources: HF nvidia domain-embedding-finetune, LlamaIndex synthetic-data, philschmid,
AWS SageMaker RAG, iWeaver.

---

## 10. OPEN ATTACKS not yet addressed (next contrarian pass)

- Fusing 5 noisy retrievers can **hurt precision** even as recall rises (RRF dilutes good lists with bad).
- The coverage loop (B3) can **run forever** — needs a budget/termination guarantee.
- Graph extraction at ingest (B4) is **expensive** and quality-sensitive.
- Agreement-confidence fails when **3 methods agree on the same wrong chunk** (correlated errors).
- multi-vector storage even at 4-bit is **N_tokens × bigger** than single-vector — does it actually fit on 64 GB at target corpus size? (needs real measurement)
- Product vs technique: who buys this, what's the moat **beyond "good recall"** (recall is copyable; the air-gapped + provenance + medical-domain combination may be the actual moat).
