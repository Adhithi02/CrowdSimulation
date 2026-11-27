# Interaction-Aware Adaptive GPU Crowd Simulation

**Metro Station Digital Twin** — a Unity crowd simulation project with a native CUDA Position-Based Dynamics (PBD) backend and benchmark tools.

The repository root contains the Unity project in `CSIM/`, native CUDA source in `NativeCUDA/`, and archived native benchmark results in `BenchmarkResults/`. Historical proposal, literature-review, and earlier report documents are not part of the current implementation repository.

## Project at a glance

- **Unity:** 2022.3.62f3 (`CSIM/ProjectSettings/ProjectVersion.txt`)
- **Native GPU backend:** CUDA C++ plugin called from C# using P/Invoke
- **Benchmark environment recorded in the reports:** CUDA 13.5, MSVC 19.42, NVIDIA RTX 2050 Mobile (4 GB), compute capability 8.6 (`sm_86`)
- **Showcase population:** 300 commuters; native scalability runs include 500, 1,000, 2,500, and 5,000 agents

## Architecture

Unity owns the scene, agents, navigation, and simulation coordination. CPU solver modes provide comparison paths, while `PBDCudaSolver` marshals agent and simulation data to the native `CrowdSimulationCUDA` library. The native plugin coordinates CUDA work and exports a C ABI; `CrowdCommon.h` defines data layouts shared with the C# interop structures.

The CUDA solver includes spatial hashing and importance-based LOD scoring using distance, visibility, interaction, and density signals. Benchmark code and archived results are kept separate from Unity assets.

## Repository layout

```text
CSIM/                         Unity project root
  Assets/                     Scenes, scripts, prefabs, models, materials, plugins
  Packages/                   Unity package manifest and lock file
  ProjectSettings/            Unity editor and project settings
NativeCUDA/                   CUDA/C++ source, native tools, and CUDA README
BenchmarkResults/             Archived experiment CSVs and reports
```

Unity folders and assets were not relocated. The required native runtime library remains at `CSIM/Assets/Plugins/x86_64/CrowdSimulationCUDA.dll`.

## Open and run in Unity

1. Install Unity **2022.3.62f3** and open the `CSIM/` directory as the Unity project (not the repository root).
2. Let Unity resolve the dependencies in `CSIM/Packages/manifest.json`.
3. Open `Assets/Scenes/414/Steering Demo.unity` from the Unity Project window.
4. For the CUDA backend, use a Windows 64-bit target with a compatible NVIDIA GPU and the native plugin at the path above.

The plugin importer currently enables Standalone Win64 but disables Editor, while also marking Linux64 and macOS as enabled. The DLL itself is a Windows binary. Confirm these importer settings against the intended runtime before relying on Editor Play Mode or non-Windows builds; they were not changed during this repository cleanup.

The project has no checked-in native CUDA build project or build script. See [`NativeCUDA/README.md`](NativeCUDA/README.md) for the source map, integration constraints, and current build limitations.

## Audited native benchmark results

The Phase 3 CSVs report means in milliseconds. “CUDA total step” includes more than kernel solving; “CUDA solve” is the solve-only timing.

| Agents | CPU spatial hash | CUDA total step | CUDA solve |
|---:|---:|---:|---:|
| 500 | 5.825 | 0.509 | 0.272 |
| 1,000 | 32.760 | 4.428 | 2.827 |
| 2,500 | 161.945 | 7.266 | 5.690 |
| 5,000 | 2,066.970 | 11.441 | 9.390 |

These are native benchmark results, not Unity frame-time or FPS measurements. The CPU brute-force run at 5,000 agents exceeded its 10-second limit; no exact runtime is reported. The individual trials, summary CSV, and audit reports are in [`BenchmarkResults/`](BenchmarkResults/README.md).

## Limitations and validation status

- Spatial hashing reduced candidate checks in the tested station workloads, but extreme dense-cluster tests exceeded bucket capacity and dropped insertions. Correctness is not claimed for arbitrary crowd densities.
- The original `PAGP_Exp1_Scalability.csv` is retained as historical data and was not rerun after the latest NavMesh fix. It is not a post-fix Phase 3 result.
- The archived runtime-validation report records passing managed-build and native smoke/layout checks, but marks Unity benchmark UI flows as unverified and overall Unity validation as needing sign-off. See `BenchmarkResults/Phase3_Final/runtime-validation-report.md`.
- Native CUDA rebuild instructions are not reproducible from this repository yet: no native build configuration or build script was found. The Unity plugin DLL is preserved.
- No `LICENSE` file is present. Select and add a license before public distribution.

This repository organization work does not change simulation code or claim a new Unity runtime test.
