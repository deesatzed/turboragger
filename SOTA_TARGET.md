# SOTA Target

Access date: 2026-06-08

## Selected Primary Target

The active SOTA target for this goal is:

- Benchmark: BEIR NFCorpus
- Split: `test`
- Primary metric: `nDCG@10`
- Target system: `voyage-3-m-exp`
- Source: MTEB English Retrieval leaderboard data table
- Reported value: `46.99` nDCG@10
- Normalized local comparison value: `0.4699`
- Required local comparison rule: local `nDCG@10 >= 0.4699`
- Preferred margin: local `nDCG@10 >= 0.4709`

Source URL:

```text
https://huggingface.co/spaces/mteb/leaderboard/blob/059e8089b9db4524ef264b3d04c9468cf8077768/boards_data/en/data_tasks/Retrieval/default.jsonl
```

This target is intentionally strict. The current MTEB table is newer than the official BEIR EvalAI table visible during target discovery, and it reports a higher NFCorpus value. Because the goal is to achieve SOTA, the harder current public target is the correct primary gate.

## Comparator Targets

Official BEIR EvalAI comparator:

- Source: EvalAI BEIR Benchmark Leaderboard
- URL: `https://eval.ai/web/challenges/challenge-page/1897/leaderboard/4475`
- Highest observed NFCorpus value: `0.385`
- System: `ZA+NM+Unicamp (InParsv2)`
- Metric: `nDCG@10` on NFCorpus

Best observed current open-weight MTEB comparator:

- Source: MTEB English Retrieval leaderboard data table
- URL: `https://huggingface.co/spaces/mteb/leaderboard/blob/059e8089b9db4524ef264b3d04c9468cf8077768/boards_data/en/data_tasks/Retrieval/default.jsonl`
- System: `nvidia/NV-Embed-v2`
- Reported value: `45.17`
- Normalized value: `0.4517`

These comparator values are not target-moving fallbacks. They are useful gates:

1. Beat `0.385` to clear the official BEIR EvalAI comparator.
2. Beat `0.4517` to clear the best observed current open-weight comparator.
3. Beat `0.4699` to satisfy the active SOTA target.

## Current Local Baseline Gap

Current local baseline artifact:

```text
artifacts/baseline_minilm_nfcorpus.json
```

Current local score:

- `nDCG@10`: `0.3160012178022206`
- `Recall@100`: `0.31150992401169303`
- Queries: `323`
- Failure count: `0`
- Dataset fingerprint: `79cb102227a4395b63e595fb53535b775d71b9e6e17581f342169eb7f97dc4d2`

Gap from local baseline:

- To official BEIR comparator `0.385`: `+0.0689987821977794`
- To open-weight MTEB comparator `0.4517`: `+0.13569878219777942`
- To selected primary target `0.4699`: `+0.1538987821977794`

## Scope Notes

- The primary target is a hosted/API model entry, not a local open-weight model.
- The project still prefers local/open-weight/offline-compatible models.
- If all feasible local branches fail to reach `0.4699`, the goal must not be marked complete; produce `SOTA_ATTEMPT_REPORT.md` instead.
- All local candidate artifacts must preserve the same NFCorpus test split, metric definitions, qrels, and dataset fingerprint.
