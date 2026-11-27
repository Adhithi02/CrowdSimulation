# Runtime Validation Report

**Generated:** 2026-11-27T14:00:09+05:30  
**Target:** `Csim-prj-2026/CSIM` Unity project

## Summary

| Step | Status | Exit Code | Details |
|---|---|---:|---|
| Managed build | PASS | 0 | `dotnet build CSIM.sln --no-restore`; 0 errors, 4 assembly-version conflict warnings |
| Native CUDA smoke | PASS | 0 | Existing smoke executable completed at 500, 1,000, 2,500, and 5,000 agents |
| Native layout check | PASS | 0 | `AgentGPU` is 80 bytes, aligned to 8 bytes; expected field offsets verified |
| Unity startup/runtime | PARTIAL | N/A | Play Mode showed 300 live commuters, the PBD CUDA solver, and live telemetry; benchmark lab remained `Idle` |
| Unity performance sample | OBSERVED, INCONSISTENT | N/A | Latest 500 profiler rows: 8.46 ms mean frame, 9.21 ms p95, 9.52 ms max; solver mean 7.69 ms; all `neighbor_ms` values were `NaN`. Live Game view showed about 4–6 FPS despite overlay frame times around 8–11 ms |
| Unity benchmark UI flows | UNVERIFIED | N/A | Existing Experiment 1 and 3 CSVs are present (last modified 11:29), but these flows were not rerun or independently validated in this runtime pass |

**Overall:** NEEDS_SIGNOFF — native checks and a live Unity/CUDA session were observed, but controlled Unity benchmark runs and complete result validation remain outstanding.

## Environment

- Unity Editor: 2022.3.62f3
- CUDA Toolkit: 13.5
- GPU: NVIDIA GeForce RTX 2050 Mobile
- Docker, Node.js, and Playwright: not applicable to this Unity desktop runtime

## Evidence and Limitations

- The profiler sample is observational, not a controlled benchmark. Its scene population and configuration were not recorded, so the FPS values should not be compared directly with other runs.
- The live FPS display (about 4–6) does not agree with the overlay frame-time values (about 8–11 ms) or a reciprocal-FPS calculation from the profiler sample. Treat all Unity FPS/frame-time claims as unverified until their measurement windows and time bases are reconciled.
- Existing Unity Experiment 1 and 3 CSVs predate the latest profiler sample and were not regenerated or validated during this pass.
- The Unity output `../PAGP_Experiment4_PBD_Convergence.csv` is non-monotonic: average penetration is 0.0455 m at 4 iterations and 0.2521 m at 16 iterations. Do not use this CSV as convergence evidence until the run is reproduced and the measurement path is checked.
- This Unity CSV is separate from the native Experiment D results in `Phase3_Final_Performance_Report_Final.md`.
- The managed build emitted four MSB3277 assembly-version conflict warnings in Unity-generated project files; there were no build errors.

## Remaining Validation

1. Run the in-scene comprehensive CPU/CUDA suite and preserve its generated CSV.
2. Run the in-scene LOD-ablation suite and preserve its generated CSV.
3. Re-run the Unity convergence experiment under a recorded, fixed configuration; investigate the non-monotonic output before making convergence claims.
4. Record scene population, solver settings, frame-time window, CUDA timing, and valid neighbor timing for each run.
