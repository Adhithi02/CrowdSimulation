# Phase 3 re-run (2026-11-27) — reproducible build

Produced by `NativeCUDA/build.bat bench` (clean `nvcc -O3 -arch=sm_86` build, CUDA 13.5,
MSVC v145, RTX 2050 Laptop GPU, seed 4142026, 2 warm-up + 20 measured trials).
Earlier folders are kept unchanged; these numbers come from the build script in this repo,
so anyone can regenerate them.

## Headline results (means of 20 trials)

| Agents | CPU brute force | CPU spatial hash | CUDA full step | CUDA vs CPU hash | CUDA vs brute force |
|---:|---:|---:|---:|---:|---:|
| 500   | 0.80 ms  | 0.20 ms  | 0.46 ms | 0.4× (CPU wins) | 1.8× |
| 1,000 | 3.08 ms  | 0.60 ms  | 0.59 ms | 1.0× (break-even) | 5.3× |
| 2,500 | 20.90 ms | 3.15 ms  | 1.29 ms | 2.4× | 16× |
| 5,000 | 85.80 ms | 11.01 ms | 2.28 ms | **4.8×** | **38×** |

* **Scaling:** CPU brute force grows ~N² (×107 from 500→5,000); CUDA grows ~linearly (×5).
* **Pipeline (5,000):** solve 1.96 ms of 2.28 ms; PCIe copies 0.22 ms (10%). At 500 agents
  copies + launch overhead dominate, which is why the CPU wins at small N.
* **Memory layout (Exp C):** SoA solve kernel is **2.2× – 2.9×** faster than AoS.
* **PBD convergence (Exp D, 2,500):** average overlap falls 0.331 → 0.208 m from 1 → 16
  iterations while kernel time grows linearly (0.29 → 4.84 ms).
* **LOD (Exp E2, camera at the platform end, Low agents solved every 4th step as in Unity):**
  vs the new no-LOD baseline the full 4-factor LOD performs **30% fewer agent-solves**
  (5,267 vs 7,500) and the solve kernel is **14% faster**; distance-only LOD saves more
  (57% / 29%) but also demotes crowded agents the 4-factor model keeps at full detail.
* **Exp E (no camera)** is kept for transparency: with every agent visible and the focus at the
  crowd centre, almost all agents are High, so LOD saves nothing there. E2 is the realistic case.
