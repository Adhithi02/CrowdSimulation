# Benchmark results

This folder preserves native benchmark CSVs and their accompanying reports, organized by experiment stage. CSV contents and filenames are preserved; the audited Phase 3 results are kept separate from earlier experiments.

## Folders

- `Earlier_Experiments/` — original `PAGP_Exp*.csv` files. In particular, `PAGP_Exp1_Scalability.csv` predates the latest NavMesh fix and was not rerun after that fix. Do not present it as a post-fix or audited Phase 3 result.
- `Phase2_Corrected/` — corrected Phase 2 CSVs.
- `Phase2_1_Validation/` — Phase 2.1 audit CSVs and the validation report, including exact-pair and dense-overflow stress results.
- `Phase3_Final/` — the audited Phase 3 trial and statistics CSVs, plus all preserved Phase 3 report versions and the runtime-validation report.
- `Phase3_1_Audit/` — Phase 3.1 corrected statistics, claim verification, and consistency-audit report.

## Reading the Phase 3 table

`Phase3_Final/Phase3_Statistics.csv` contains the summarized benchmark data; `Phase3_Final/Phase3_Trials.csv` contains the individual trials. The Phase 3 report distinguishes CUDA total-step time from CUDA solve-only time. The total includes work beyond the CUDA solve, so those columns are not interchangeable.

The 5,000-agent CPU brute-force result is recorded as `TIMEOUT` after the 10-second limit. It is not an exact measured runtime. Dense-cluster stress tests also show that spatial-hash capacity can be exceeded and insertions dropped; do not claim correctness at arbitrary density.

The runtime-validation report is evidence for its stated scope only. It reports a managed build and native smoke/layout checks as passing, but Unity benchmark UI flows as unverified and the overall Unity validation as needing sign-off.

## Other Unity profiler output

Existing profiler CSVs and captures remain under `CSIM/Assets/profilerOutputs/`. They were not moved because they are already inside the Unity asset tree and moving them is not required to organize the native experiment archive.

Native benchmark harnesses write result CSVs to their current working directory. After a new run, review the output and place it in the appropriate experiment folder; do not overwrite historical results.
