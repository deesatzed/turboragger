# RISK_NOTES.md

## Risks

| Risk | Severity | Why It Matters | Mitigation |
|---|---|---|---|
| Building before reproducing the 0.18 nfcorpus baseline | High | Without a reproducible baseline, every later recall claim is ungrounded. | Rebuild Step 0 first: dataset discovery, dependency repair, deterministic scoreboard, saved result artifact. |
| Local dependency drift | High | `sentence_transformers` fails in the current probe and `mlx_lm` cannot access Metal in this session. | Use a project-local environment and separate CPU-compatible benchmark steps from Apple-Silicon MLX steps. |
| Dataset path drift | High | The hardcoded nfcorpus path in the old benchmark was not found in this audit. | Add dataset discovery/config rather than hardcoding `/Volumes/WS4TB/newragcity/UltraRAG-main/datasets/nfcorpus`. |
| Trusting placeholder benchmark artifacts | High | The handoff warns that some newragcity unified benchmark claims are hardcoded placeholders. | Mark artifacts by provenance: reproduced current-run result, historical result, placeholder, or unverified doc claim. |
| Fusing noisy retrievers by RRF can hurt precision | Medium | Recall can rise while nDCG@10 or answer quality falls. | Track Recall@100 and nDCG@10 together; add branch-level ablations. |
| Multi-vector ColBERT storage may exceed practical memory | Medium | Per-token vectors are much larger than one-vector-per-chunk storage. | Prototype storage math and small-corpus turbovec quantization before committing to full architecture. |
| Coverage loop can run indefinitely | Medium | A self-correcting retrieval loop needs deterministic termination. | Add budgets: max subclaims, max retrieval rounds, min marginal gain, and unanswered-claim reporting. |
| Embeddings leaking into deterministic trust paths | Medium | It would weaken the provenance value of Agent_Pidgeon. | Keep embeddings as candidate discovery only; exact resolver/policy/receipt path remains unchanged. |

## Safe Next Step

Create the Step 0 baseline-reproduction harness and environment repair as one small milestone:

1. Locate or redownload BEIR nfcorpus.
2. Repair local imports needed for MiniLM baseline scoring.
3. Run a deterministic Recall@100/nDCG@10 scoreboard.
4. Save the result artifact and only then test new retrieval methodology branches.

