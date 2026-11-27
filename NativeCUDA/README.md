# Native CUDA backend

This directory contains the CUDA implementation, native benchmark programs, and helper scripts. The source files are intentionally kept together: the CUDA and C++ sources include `CrowdCommon.h` by its local filename, and the repository does not contain a build project or build script that defines alternate include paths.

## Source map

- `CrowdCommon.h` — shared CUDA/C++ data structures and declarations. Its layouts must remain compatible with the matching structures in `CSIM/Assets/Scripts/Solvers/PBDCudaSolver.cs`.
- `CrowdKernels.cu` — CUDA kernels used by the simulation.
- `CrowdPlugin.cu` — CUDA host-side orchestration and exported C ABI consumed through Unity P/Invoke.
- `run_phase3.cpp` — Phase 3 benchmark harness; writes `Phase3_Trials.csv` and `Phase3_Statistics.csv` to its current working directory.
- `run_benchmarks.cpp` — Phase 2 validation and benchmark harness.
- `compare_ab.cpp`, `test_step.cpp`, `test_clock.cpp`, and `verify_layout.cpp` — native comparison and verification programs.
- `audit_math.py` and `generate_audit_csvs.py` — audit/report helpers.

## Runtime integration

Unity imports `CrowdSimulationCUDA` through `DllImport` in `PBDCudaSolver`. The required Windows 64-bit plugin is kept at:

```text
CSIM/Assets/Plugins/x86_64/CrowdSimulationCUDA.dll
```

Do not move or rename that plugin, alter the exported function names, or change the shared struct layouts without a separately validated integration change. The plugin metadata enables the Win64 target. The `.dll` is a Windows binary; do not assume that the same binary runs on macOS or Linux.

## Build status and requirements

The project metadata and benchmark reports identify CUDA Toolkit 13.5, MSVC 19.42, and NVIDIA compute capability 8.6 (`sm_86`). The checked-in source includes CUDA runtime headers and CUDA kernel code. No `CMakeLists.txt`, Visual Studio native project, Makefile, or reproducible native build script was found, so this repository does not currently document a verified command for rebuilding the DLL. Existing `.exe`, `.lib`, and `.exp` files are local build outputs, not build instructions.

The Unity plugin DLL is deliberately not ignored. A separate copy of the DLL in this directory is retained; the Unity runtime copy remains in `CSIM/Assets/Plugins/x86_64/`.

## Benchmark output paths

The native harnesses write CSV files relative to the process working directory. Running a harness from this directory therefore creates output beside the source files; archive reviewed results under `BenchmarkResults/` without editing their contents.

The audit helper scripts currently contain absolute workspace paths for benchmark inputs and outputs. They are machine-specific and will need a separate portability change before they can be run from a fresh clone. Their source is retained unchanged in this organization-only cleanup.
