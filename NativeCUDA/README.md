# Native CUDA backend

This directory contains the CUDA implementation, native benchmark programs, and helper scripts. The source files are intentionally kept together: the CUDA and C++ sources include `CrowdCommon.h` by its local filename, and `build.bat` compiles them from this folder.

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

## Building

Requirements: CUDA Toolkit (tested 13.5), Visual Studio C++ build tools (tested VS 18 / MSVC v145),
and an NVIDIA GPU (tested RTX 2050 Laptop, `sm_86`). From this folder, in any terminal:

```bat
build.bat            :: builds build\CrowdSimulationCUDA.dll and build\run_phase3.exe
build.bat bench      :: builds, then runs the Phase 3 benchmarks (about 25 s)
build.bat install    :: builds, then copies the DLL into CSIM\Assets\Plugins\x86_64 (close Unity first)
```

The script finds MSVC through `vswhere` automatically. For another GPU, set the architecture
first, e.g. `set CUDA_ARCH=sm_75`. Outputs go to `build\` (git-ignored); the Unity plugin in
`CSIM/Assets/Plugins/x86_64/` is only replaced by `build.bat install`.

The Phase 3 harness runs experiments A–E plus **E2**, a LOD ablation with a real camera frustum
and the runtime's every-4th-step Low cadence, measured against a no-LOD (all High) baseline.
Results of the 2026-11-27 run are archived in `BenchmarkResults/Phase3_Rerun_2026-11-27/`.

## Benchmark output paths

The native harnesses write CSV files relative to the process working directory. Running a harness from this directory therefore creates output beside the source files; archive reviewed results under `BenchmarkResults/` without editing their contents.

The audit helper scripts currently contain absolute workspace paths for benchmark inputs and outputs. They are machine-specific and will need a separate portability change before they can be run from a fresh clone. Their source is retained unchanged in this organization-only cleanup.
