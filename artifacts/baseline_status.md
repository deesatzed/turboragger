# Baseline Status

Generated: 2026-06-08T13:20:59+00:00

Status: reproduced_direct_minilm

Dataset: `/Volumes/WS4TB/WS4TBr/newragcity/UltraRAG-main/datasets/nfcorpus`
Model path: `/Users/o2satz/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf`
Queries tested: 323
Recall@100: 0.31150992401169303
nDCG@10: 0.3160012178022206

This is a MiniLM dense baseline using direct transformers mean pooling through the neutral harness.
It is not an exact reproduction of the historical newragcity integrated artifact.
The sentence_transformers wrapper remains unavailable in this environment.
