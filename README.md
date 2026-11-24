# Real-Time Crowd Simulation with Density-Driven LOD

A research project exploring GPU-accelerated crowd simulation using Position-Based Dynamics (PBD) and a novel density-driven Level-of-Detail (LOD) system. Built in Unity, the project implements and benchmarks seven interchangeable solver backends ranging from a single-threaded Social Force baseline to a fully GPU-resident PBD solver that allocates physics fidelity based on a real-time crowd density heatmap.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Opening the Test Scene](#opening-the-test-scene)
- [Usage Guide](#usage-guide)
  - [Runtime UI Controls](#runtime-ui-controls)
  - [Solvers](#solvers)
  - [Scene Presets](#scene-presets)
  - [Agent & Physics Parameters](#agent--physics-parameters)
  - [Inspector-Only Settings](#inspector-only-settings)
- [Exporting Performance Data](#exporting-performance-data)
- [Project Structure](#project-structure)

---

## Project Overview

The simulation models crowds of autonomous pedestrians navigating a shared environment. Each agent uses Unity's NavMesh for pathfinding and a physics solver for local collision avoidance. The core research contribution is the **PBDCompute** solver, which runs all constraint solving on the GPU via a compute shader pipeline and determines each agent's physics LOD from a live density heatmap — a dense crowd anywhere in the scene receives high-fidelity collision resolution, while sparse agents cost almost nothing regardless of their distance from the camera.

All seven solvers share the same agent population, NavMesh, and scene geometry, making it straightforward to compare them fairly under controlled conditions.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | DirectX 11 / Metal 2 capable | Dedicated GPU with ≥ 4 GB VRAM |
| CPU | Quad-core, 2.5 GHz | 8-core, 3.5 GHz or faster |
| RAM | 8 GB | 16 GB |
| OS | Windows 10 64-bit / macOS 12 | Windows 11 / macOS 14 |

> **Note:** The `PBDCompute` solver requires a GPU that supports compute shaders (DirectX 11 feature level 11.0 or Metal 2). The remaining CPU-side solvers run on any hardware capable of running Unity 2022.3.

---

## Software Requirements

- **Unity 2022.3.62f1** (exact version — see installation note below)
- **Unity Jobs package** — included via the Unity Package Manager (com.unity.jobs)
- **Burst compiler** — included via the Unity Package Manager (com.unity.burst)
- **TextMeshPro** — included via the Unity Package Manager (com.unity.textmeshpro); required by the in-scene UI

---

## Installation

1. **Clone the repository**
   ```bash
   git clone [<repository-url>](https://github.com/tjong03/Csim-prj-2026.git)
   cd ./Csim-prj-2026
   ```

2. **Install Unity 2022.3.62f1**  
   Open **Unity Hub**, go to *Installs → Add* and select version `2022.3.62f1`. If it does not appear in the list, download it directly from the [Unity download archive](https://unity.com/releases/editor/whats-new/2022.3.62).

3. **Open the project in Unity Hub**  
   Click *Open → Add project from disk* and select the cloned folder. Unity will import all assets and resolve packages automatically on first open — this may take several minutes.

4. **Import TextMeshPro essentials**  
   On first open Unity may display the *TMP Importer* window. Click **Import TMP Essentials**. If the window does not appear automatically, go to *Window → TextMeshPro → Import TMP Essential Resources*.

5. **Verify Burst compilation**  
   Go to *Jobs → Burst → Open Inspector* and confirm that Burst is enabled. The Jobs-based solvers (`ForceJobs`, `PBDJobs`, `PBDCLODJobs`, `PBDCLODCullingJobs`) require Burst for acceptable performance.

---

## Opening the Test Scene

The primary benchmarking scene is located at:

```
Assets/Scenes/414/Steering demo.unity
```

Open it via *File → Open Scene* or double-click it in the Project window. Press **Play** to start the simulation. The runtime control panel appears in the top-left corner of the Game view immediately.

---

## Usage Guide

### Runtime UI Controls

All controls are accessible through the panel in the top-left corner of the Game view while the simulation is running. No pause or restart is required to switch solvers or presets — changes take effect immediately and respawn the crowd automatically.

```
┌─────────────────────────┐
│  Crowd Manager          │
│  Solver: PBDCLODJobs    │
│  Agents: 100            │
├─────────────────────────┤
│  Switch Solver:         │
│  [Force] [F-Jobs] [PBD] │
│  [PBD-Jobs][CLOD][Cmp]  │
│  [Cull]                 │
├─────────────────────────┤
│  Scene Preset:          │
│  [Random][Circle][Cross]│
│  [Bi-Dir][Bottle][D-BN] │
│  [Merge]                │
├─────────────────────────┤
│  ─── Common ───         │
│  Agent Count: 50        │
│  Speed: 5.00            │
│  Agent Radius: 0.50     │
│  Neighbour Dist: 4.0    │
├─────────────────────────┤
│  ─── PBD ───            │  (shown for PBD solvers only)
│  Iterations: 3          │
│  Ksi: 0.050             │
├─────────────────────────┤
│  ─── Force ───          │  (shown for Force solvers only)
│  K: 1.00                │
│  t0: 3.0                │
├─────────────────────────┤
│  [Respawn Crowd]        │
└─────────────────────────┘
```

---

### Solvers

Click a solver button to switch immediately. The active solver is highlighted in amber.

| Button | Solver | Description |
|--------|--------|-------------|
| `Force` | Social Force Model | Single-threaded Helbing force-based avoidance. Baseline reference. |
| `F-Jobs` | ForceJobs | Same algorithm, parallelised with Unity Jobs + Burst. |
| `PBD` | PBD | Single-threaded Position-Based Dynamics (Weiss 2019). |
| `PBD-Jobs` | PBDJobs | PBD parallelised with Unity Jobs + Burst. |
| `CLOD` | PBDCLODJobs | PBD with camera-distance LOD tiers and density-aware promotion. |
| `PBD-Cmp` | PBDCompute | **GPU compute shader PBD with density heatmap LOD.** Primary research solver. |
| `Cull` | PBDCLODCullingJobs | CLOD PBD with frustum culling — disables Renderer and Animator for off-screen agents. |

> **Switching solvers** respawns the agent population. All parameter sliders retain their values across switches.

---

### Scene Presets

Click a preset button to rebuild the scene geometry and respawn agents. The active preset is highlighted in teal.

| Button | Preset | Description |
|--------|--------|-------------|
| `Random` | Random | Agents scatter across the NavMesh and navigate between the four corner goal capsules in the scene. Good general-purpose test. |
| `Circle` | Circle | Agents spawn on a ring and each walks to the antipodal point. All paths cross the centre simultaneously — stresses collision resolution. |
| `Crossing` | Crossing | Single block of agents walks diagonally across open ground. Tests smooth avoidance without geometry constraints. |
| `Bi-Dir` | Bidirectional | Two equal groups walk toward each other inside a corridor. Tests counter-flow lane formation. |
| `Bottleneck` | Bottleneck | All agents must squeeze through a single narrow gap in a wall. Maximum density stress test. |
| `Double-BN` | DoubleBottleneck | Two bottleneck walls in sequence. Tests sustained high-density performance. |
| `MergeLane` | MergeLane | Two lanes converge into one corridor. Tests gradual density build-up and merge behaviour. |

---

### Agent & Physics Parameters

All sliders update live without requiring a respawn.

**Common (all solvers)**

| Slider | Range | Default | Effect |
|--------|-------|---------|--------|
| Agent Count | 5 – 300 | 50 | Number of agents spawned on the next respawn. |
| Speed | 0.5 – 15 | 5.0 | Preferred walking speed (m/s). |
| Agent Radius | 0.1 – 2.0 | 0.5 | Collision radius of each agent (m). |
| Neighbour Dist | 1 – 20 | 4.0 | Perception radius for main-thread grid-based solvers (m). |

**PBD section** (visible when any PBD solver is active)

| Slider | Range | Default | Effect |
|--------|-------|---------|--------|
| Iterations | 1 – 20 | 3 | Number of Jacobi constraint passes per frame. Higher values improve collision resolution quality at increased CPU/GPU cost. |
| Ksi | 0.01 – 1.0 | 0.05 | Velocity blending rate toward preferred velocity. Lower = smoother but slower response. |

**Force section** (visible when Force or F-Jobs is active)

| Slider | Range | Default | Effect |
|--------|-------|---------|--------|
| K | 0.1 – 5.0 | 1.0 | Avoidance force magnitude scaling. |
| t0 | 0.5 – 10.0 | 3.0 | Avoidance time horizon (s). Agents react to predicted collisions this far in the future. |

---

### Inspector-Only Settings

Some settings are not exposed in the runtime UI and must be adjusted in the Unity Inspector while the simulation is stopped (or via the Inspector in Play mode for live effect).

Select the **SimulationManager** GameObject in the Hierarchy to access these:

**CLOD LOD Distances** — relevant for `CLOD` and `Cull` solvers:
- `Clod High Dist` (default 20) — agents closer than this to the focus point receive HIGH-tier physics.
- `Clod Med Dist` (default 50) — agents between the two thresholds receive MED-tier physics.

**PBDCompute density thresholds** — on the `PBDComputeSolver` component:
- `High Density Cutoff` (default 12) — cells with more than this many agents use HIGH LOD.
- `Med Density Cutoff` (default 4) — cells above this but below the high cutoff use MED LOD.

**Benchmark Camera Protocol** — on the SimulationManager component:
- `Enforce Benchmark Camera Protocol` — when enabled, locks the camera to a fixed overhead position every 120 frames, ensuring consistent framing across benchmark runs.
- `Benchmark Camera Height` (default 70) — world-space height of the benchmark camera.
- `Benchmark Pitch Degrees` (default 60°) — downward tilt of the benchmark camera.
- `Benchmark Ortho Size` (default 32) — orthographic size if using an orthographic benchmark camera.

**Debug**
- `Draw Grid` — draws the spatial hash cells in the Scene view, coloured from green (empty) to red (dense). Useful for visualising density heatmap distribution.

---

## Exporting Performance Data

Performance metrics are collected by the `ProfilerManager` component (attached to the same GameObject as `SimulationManager`) and are also visible in Unity's built-in Profiler window.

### Enabling CSV Export

1. Select the **ProfilerManager** GameObject in the Hierarchy (or find the component on the SimulationManager GameObject).
2. In the Inspector, tick **Enable Csv Logging**.
3. Enter Play mode and run the simulation.
4. A file named `profiler_log.csv` is written to `Application.persistentDataPath`:
   - **Windows:** `%APPDATA%\..\LocalLow\cmput414\CSIM\`
   - **macOS:** `~/Library/Application Support/cmput414/CSIM/`

### CSV Format

```
time_unix_ms, frame, frame_ms, neighbor_ms, solver_ms, animation_ms
```

| Column | Description |
|--------|-------------|
| `time_unix_ms` | Wall-clock timestamp in milliseconds since Unix epoch. |
| `frame` | Unity frame number. |
| `frame_ms` | Rolling average duration of the full `FixedUpdate` (ms). |
| `neighbor_ms` | Rolling average duration of the main-thread neighbour grid query (ms). Zero for GPU-side solvers. |
| `solver_ms` | Rolling average duration of the active solver dispatch (ms). |
| `animation_ms` | Rolling average duration of any animation sampling tagged `"Animation"` (ms). |

All rolling averages use a sliding window of the last 300 frames (~5 seconds at 60 fps). The window size is configurable via the `History Size` field on `ProfilerManager`.

### Inspecting Metrics at Runtime

Call `ProfilerManager.Instance.GetAverageMs("Solver")` from any script, or open Unity's **Window → Analysis → Profiler** to see named samples in the CPU timeline. Every `BeginSample` / `EndSample` call also emits a Unity Profiler marker, so samples appear in the Deep Profile view with no extra instrumentation.

### Recommended Benchmark Procedure

1. Enable `Enforce Benchmark Camera Protocol` on `SimulationManager` to fix the camera.
2. Enable `Enable Csv Logging` on `ProfilerManager`.
3. Select a solver and preset, set agent count, then click **Respawn Crowd**.
4. Wait approximately 10 seconds for the rolling averages to stabilise (300 frames at 30 fps).
5. Note the `solver_ms` value in the CSV or Profiler, then switch solver/preset for the next run.
6. Disable Play mode when finished; the CSV is flushed and closed automatically.

---

## Project Structure

```
Assets/
├── Scenes/
│   └── 414/
│       └── Steering demo.unity       # Main test scene
├── Scripts/
│   ├── Agents/
│   │   ├── AgentBase.cs              # Shared agent state
│   │   ├── PBDAgent.cs               # PBD-specific agent component
│   │   └── ForceAgent.cs             # Force-model agent component
│   ├── Solvers/
│   │   ├── PBDSolver.cs              # Single-threaded PBD
│   │   ├── PBDSolverJobs.cs          # Burst-parallel PBD
│   │   ├── CLODPBDSolverJobs.cs      # Camera-distance CLOD PBD
│   │   ├── CLODPBDCullingSolver.cs   # Frustum-culling CLOD PBD
│   │   ├── PBDComputeSolver.cs       # GPU compute PBD (primary solver)
│   │   ├── ForceSolver.cs            # Single-threaded Social Force
│   │   └── ForceSolverJobs.cs        # Burst-parallel Social Force
│   ├── Shaders/
│   │   └── PBDComputeShader.compute  # HLSL compute shader (7 kernels)
│   ├── SimulationManager.cs          # Central coordinator & spawning
│   ├── SimulationUI.cs               # Runtime control panel
│   └── ProfilerManager.cs            # Timing, rolling stats, CSV export
└── Prefabs/
    ├── Agent.prefab                  # Default agent prefab
    └── Protagonist.prefab            # Focus point for CLOD solvers
```
