using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

/// <summary>
/// PAGP Performance Lab: live measurements of the five PAGP concepts.
///
///   0. CUDA data parallelism (CPU vs GPU solver time)
///   1. Spatial hashing vs O(N^2) neighbour search
///   2. PBD Jacobi constraint convergence vs iterations
///   3. 4-factor importance-based simulation LOD (ablation)
///   4. GPU memory layout (AoS vs SoA)
///
/// Every concept can be run on its own (sweeps, CSV export), and the
/// 60-second guided demo measures all five in sequence on the live crowd, then
/// presents every result. All numbers shown are measured in this run unless they
/// are explicitly labelled as archived.
/// </summary>
public class PAGPPerformanceLab : MonoBehaviour
{
    public static PAGPPerformanceLab Instance { get; private set; }

    [Header("Lab Configuration")]
    [Tooltip("Agent counts for the individual sweeps. Kept moderate so animated 3D commuters stay interactive; " +
             "the native CUDA harness (NativeCUDA/run_phase3.cpp) covers 5,000 agents.")]
    public int[] sweepAgentCounts = { 250, 500, 1000 };
    public int experimentAgents = 500;
    public int showcaseAgents = 300;
    public float phaseSeconds = 12f;
    [Tooltip("Tick in Play mode to start the guided demo (same as the panel button).")]
    public bool runGuidedDemoNow;

    [Header("Status (read-only)")]
    public bool isRunning;
    public string currentBenchmarkName = "Idle";
    public float benchmarkProgress;

    // ── Concepts ────────────────────────────────────────────────────────────
    public const int ConceptCount = 5;
    public static readonly string[] ConceptNames =
    {
        "CUDA Data Parallelism",
        "Spatial Hashing vs O(N²)",
        "PBD Jacobi Convergence",
        "4-Factor Simulation LOD",
        "Memory Layout: AoS vs SoA"
    };

    /// <summary>Latest measured result per concept (rich text); null = not measured yet.</summary>
    public readonly string[] ConceptResults = new string[ConceptCount];

    // ── Live demo narration (read by the UI "Live Demo" tab) ─────────────────
    public bool IsShowcase { get; private set; }
    public int PhaseIndex { get; private set; } = -1;   // 0..4 during the demo, 5 = finished
    public string PhaseTitle { get; private set; } = "";
    public string PhaseBody { get; private set; } = "";
    public string PhaseStatus { get; private set; } = "";
    /// <summary>One plain sentence describing what is on screen (shown as a caption over the scene).</summary>
    public string Caption { get; private set; } = "";

    public static readonly string[] PhaseNames =
    {
        "Station overview — CUDA parallel step",
        "Hash-grid heat map — spatial hashing",
        "Fare-gate queue — PBD Jacobi solver",
        "Eye-level LOD rings — 4-factor LOD",
        "Evacuation drill — memory layout"
    };

    /// <summary>Raised whenever status, phase, or a result changes.</summary>
    public event Action OnLabStateChanged;
    /// <summary>Raised when a run (experiment or demo) finishes so the UI can open the Results tab.</summary>
    public event Action OnRunFinished;

    private SimulationManager _mgr;
    private PBDCudaSolver _cuda;
    private Coroutine _routine;
    private float _showcaseStart;

    // Saved configuration so an individual experiment can put the scene back afterwards.
    private SolverMode _savedSolver;
    private ScenePreset _savedPreset;
    private int _savedCount;
    private int _savedIterations;
    private MetroStationManager.CCTVAngle _savedCctv;

    private void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(this); return; }
        Instance = this;
    }

    private void Start()
    {
        _mgr = SimulationManager.Instance;
        _cuda = _mgr != null ? _mgr.GetComponent<PBDCudaSolver>() : null;

        // Experiment 1 reads the "Solver" timing samples. The scene has no
        // ProfilerManager, which made every CPU timing (and speedup) read 0.
        if (ProfilerManager.Instance == null && _mgr != null)
            _mgr.gameObject.AddComponent<ProfilerManager>();
    }

    private void Update()
    {
        if (runGuidedDemoNow)
        {
            runGuidedDemoNow = false;
            if (!isRunning) Run60SecondShowcaseSequence();
        }
        if (IsShowcase && isRunning)
            benchmarkProgress = Mathf.Clamp01((Time.time - _showcaseStart) / (phaseSeconds * ConceptCount));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public entry points (UI buttons)
    // ─────────────────────────────────────────────────────────────────────────
    public void Run60SecondShowcaseSequence() => Begin(ShowcaseRoutine(), "Guided demo", true);
    public void RunComprehensiveSuite()       => Begin(ScalabilityExperiment(), "Exp 1: CPU vs CUDA", false);
    public void RunSpatialHashAnalysis()      => Begin(SpatialHashExperiment(), "Exp 2: Spatial hash", false);
    public void RunPBDConvergenceTest()       => Begin(ConvergenceExperiment(), "Exp 3: PBD convergence", false);
    public void RunLODAblationStudy()         => Begin(LodExperiment(), "Exp 4: LOD ablation", false);
    public void RunMemoryLayoutBenchmark()    => Begin(MemoryExperiment(), "Exp 5: AoS vs SoA", false);

    public void StopBenchmark()
    {
        if (!isRunning) return;
        // StopAllCoroutines also stops nested measurement enumerators
        // (StopCoroutine on the parent alone can leave them running).
        StopAllCoroutines();
        _routine = null;
        RestoreLabOverrides();
        bool wasShowcase = IsShowcase;
        isRunning = false;
        IsShowcase = false;
        currentBenchmarkName = "Stopped by user";
        PhaseStatus = "Stopped.";
        if (wasShowcase) PhaseIndex = -1;
        Changed();
    }

    private void Begin(IEnumerator routine, string name, bool showcase)
    {
        if (isRunning || _mgr == null) return;
        if (_cuda == null) _cuda = _mgr.GetComponent<PBDCudaSolver>();

        isRunning = true;
        IsShowcase = showcase;
        benchmarkProgress = 0f;
        currentBenchmarkName = name;
        Time.timeScale = 1f;

        _savedSolver = _mgr.activeSolver;
        _savedPreset = _mgr.activePreset;
        _savedCount = _mgr.agentCount;
        _savedIterations = _mgr.pbdIterations;
        _savedCctv = MetroStationManager.Instance != null
            ? MetroStationManager.Instance.currentCCTV
            : MetroStationManager.CCTVAngle.CCTV_Platform;

        Changed();
        _routine = StartCoroutine(Wrap(routine));
    }

    private IEnumerator Wrap(IEnumerator body)
    {
        yield return body;
        RestoreLabOverrides();
        isRunning = false;
        IsShowcase = false;
        benchmarkProgress = 1f;
        _routine = null;
        Changed();
        OnRunFinished?.Invoke();
    }

    /// <summary>Undo temporary lab-only settings (ablation mode, iterations, alarm).</summary>
    private void RestoreLabOverrides()
    {
        var viz = _mgr != null ? _mgr.GetComponent<PagpVisualizer>() : null;
        if (viz != null && IsShowcase) { viz.SetHashGrid(false); viz.SetLodRings(false); }
        if (PhaseIndex < ConceptCount) Caption = "";
        if (_cuda != null) _cuda.lodAblationMode = 0;
        if (_mgr != null) _mgr.pbdIterations = _savedIterations;
        var metro = MetroStationManager.Instance;
        if (metro != null && metro.isEvacuationActive) metro.ToggleEmergencyEvacuation();
    }

    private void RestoreSceneConfig()
    {
        _mgr.Configure(_savedSolver, _savedPreset, _savedCount);
        MetroStationManager.Instance?.ApplyCCTVAngle(_savedCctv);
    }

    private void Changed() => OnLabStateChanged?.Invoke();

    // ─────────────────────────────────────────────────────────────────────────
    // Shared helpers
    // ─────────────────────────────────────────────────────────────────────────
    private bool CudaReady => _cuda != null && _cuda.IsInitialized && _mgr.activeSolver == SolverMode.PBDCuda;

    /// <summary>Wait for the respawn to finish, then let the crowd settle.</summary>
    private IEnumerator WaitForCrowd(float settleSeconds)
    {
        while (_mgr.IsRespawning) yield return null;
        yield return new WaitForSeconds(settleSeconds);
    }

    private static string F(float v, string fmt = "F2") => v.ToString(fmt, CultureInfo.InvariantCulture);

    private sealed class StepStats
    {
        public float totalMs, solveMs, h2dMs, d2hMs, hashMs, lodMs, fps;
        public int samples;
    }

    /// <summary>Average CUDA event timings and FPS over a window of rendered frames.</summary>
    private IEnumerator SampleCudaStep(float seconds, StepStats s)
    {
        float elapsed = 0f;
        int frames = 0;
        while (elapsed < seconds)
        {
            yield return null;
            elapsed += Time.unscaledDeltaTime;
            frames++;
            if (!CudaReady) continue;
            var t = _cuda.LastTimings;
            s.totalMs += t.totalMs; s.solveMs += t.solveMs; s.h2dMs += t.h2dMs;
            s.d2hMs += t.d2hMs; s.hashMs += t.hashAgentsMs; s.lodMs += t.lodMs;
            s.samples++;
        }
        if (s.samples > 0)
        {
            float n = s.samples;
            s.totalMs /= n; s.solveMs /= n; s.h2dMs /= n; s.d2hMs /= n; s.hashMs /= n; s.lodMs /= n;
        }
        s.fps = frames / Mathf.Max(elapsed, 0.001f);
    }

    // ── Concept 1 measurement: spatial hash vs brute force ──────────────────
    private struct HashRow
    {
        public int agents, occupiedCells, maxPerCell, overflow;
        public long bruteCandidates, bruteNeighbours, hashCandidatesBound;
        public float prunedPct;
    }

    private bool MeasureSpatialHash(out HashRow row)
    {
        row = default;
        if (!CudaReady) return false;
        if (!_cuda.QueryHashGridStats(out var h)) return false;
        if (!_cuda.BenchmarkBruteForce(_mgr.neighborDist, out long bfCand, out long bfNeigh)) return false;

        int n = _mgr.LiveAgentCount;
        // Candidate pairs the grid search actually tests on this crowd: for every agent,
        // the agents in its own cell and the 8 surrounding cells (same 2D world cells as
        // WorldToCell() in the kernel; every agent counted as High LOD, so a slight overestimate).
        float cs = Mathf.Max(0.5f, _cuda.cellSize);
        var cellCount = new Dictionary<long, int>(n);
        var agentCell = new long[n];
        for (int i = 0; i < n; i++)
        {
            var a = _mgr.GetAgent(i);
            Vector3 p = a != null ? a.transform.position : Vector3.zero;
            long key = ((long)Mathf.FloorToInt(p.x / cs) << 32) ^ (uint)Mathf.FloorToInt(p.z / cs);
            agentCell[i] = key;
            cellCount.TryGetValue(key, out int c);
            cellCount[key] = c + 1;
        }
        long bound = 0;
        for (int i = 0; i < n; i++)
        {
            int cx = (int)(agentCell[i] >> 32), cz = (int)(uint)agentCell[i];
            for (int dx = -1; dx <= 1; dx++)
            for (int dz = -1; dz <= 1; dz++)
                if (cellCount.TryGetValue(((long)(cx + dx) << 32) ^ (uint)(cz + dz), out int c)) bound += c;
            bound -= 1;   // not itself
        }
        row = new HashRow
        {
            agents = n,
            occupiedCells = h.occupiedCells,
            maxPerCell = h.maxOccupancy,
            overflow = h.overflowCount,
            bruteCandidates = bfCand,
            bruteNeighbours = bfNeigh,
            hashCandidatesBound = bound,
            prunedPct = bfCand > 0 ? Mathf.Clamp01(1f - (float)bound / bfCand) * 100f : 0f
        };
        return true;
    }

    // ── Concept 2 measurement: PBD convergence ──────────────────────────────
    private struct ConvRow { public int iterations, constraints; public float avgPen, maxPen, totalPen, solveMs; }

    private IEnumerator MeasureConvergence(int[] iterations, float settle, float window, List<ConvRow> rows)
    {
        // Controlled experiment on the live crowd: freeze a snapshot of every commuter and
        // solve that SAME snapshot with 1, 2, 4, 8 ... Jacobi iterations, then measure the
        // overlap left. Repeated on several snapshots taken a moment apart and averaged.
        // (Comparing iteration counts at different moments mixes in the crowd changing.)
        int k = iterations.Length;
        const int snapshots = 5;
        var sumAvg = new float[k]; var sumTot = new float[k]; var maxPen = new float[k];
        var sumSolve = new float[k]; var sumCons = new long[k]; var cnt = new int[k];
        float gap = Mathf.Max(0.2f, (settle + window) / snapshots);

        for (int sIdx = 0; sIdx < snapshots; sIdx++)
        {
            PhaseStatus = $"Solving crowd snapshot {sIdx + 1}/{snapshots} with {string.Join(", ", iterations)} iterations\u2026";
            currentBenchmarkName = $"PBD convergence: snapshot {sIdx + 1}/{snapshots}";
            if (!IsShowcase) benchmarkProgress = (float)sIdx / snapshots;
            Changed();
            yield return new WaitForSeconds(gap);
            if (!CudaReady) continue;
            var snap = _cuda.SnapshotAgents();
            for (int i = 0; i < k; i++)
            {
                if (!_cuda.SolveSnapshot(snap, iterations[i], out var c, out float ms)) continue;
                sumAvg[i] += c.avgViolationBefore;
                sumTot[i] += c.avgViolationBefore * c.totalConstraints;
                maxPen[i] = Mathf.Max(maxPen[i], c.maxViolationBefore);
                sumCons[i] += c.totalConstraints;
                sumSolve[i] += ms;
                cnt[i]++;
            }
        }
        for (int i = 0; i < k; i++)
            if (cnt[i] > 0)
                rows.Add(new ConvRow
                {
                    iterations = iterations[i], avgPen = sumAvg[i] / cnt[i], totalPen = sumTot[i] / cnt[i],
                    maxPen = maxPen[i], constraints = (int)(sumCons[i] / cnt[i]), solveMs = sumSolve[i] / cnt[i]
                });
        yield break;
    }

    // ── Concept 3 measurement: LOD ablation ─────────────────────────────────
    private struct LodRow { public string label; public float high, med, low, stepMs, workSavedPct; }

    private static readonly string[] LodLabels =
    {
        "A  Distance only",
        "B  + Visibility",
        "C  + Interaction",
        "D  + Density (full)"
    };

    private IEnumerator MeasureLod(float settle, float window, List<LodRow> rows)
    {
        for (int m = 0; m < 4; m++)
        {
            _cuda.lodAblationMode = m + 1;      // 1..4 (see KernelAssignLOD)
            PhaseStatus = $"LOD configuration {LodLabels[m]}…";
            currentBenchmarkName = $"LOD ablation: {LodLabels[m]}";
            if (!IsShowcase) benchmarkProgress = m / 4f;
            Changed();
            yield return new WaitForSeconds(settle);

            float h = 0, md = 0, l = 0, step = 0; int n = 0; float elapsed = 0f;
            while (elapsed < window)
            {
                yield return null;
                elapsed += Time.deltaTime;
                if (!CudaReady) continue;
                _mgr.GetFidelityCounts(out int hi, out int me, out int lo);
                h += hi; md += me; l += lo; step += _cuda.LastTimings.totalMs; n++;
            }
            if (n == 0) continue;
            h /= n; md /= n; l /= n; step /= n;

            // Solver work model taken directly from KernelSolveConstraints:
            //   High = all I iterations, Med = max(1,(I+1)/2) iterations,
            //   Low  = 1 iteration on every 4th frame (0.25 on average).
            int I = Mathf.Max(1, _mgr.pbdIterations);
            float medIters = Mathf.Max(1, (I + 1) / 2);
            float total = h + md + l;
            float work = h * I + md * medIters + l * 0.25f;
            float saved = total > 0 ? (1f - work / (total * I)) * 100f : 0f;
            rows.Add(new LodRow { label = LodLabels[m], high = h, med = md, low = l, stepMs = step, workSavedPct = saved });
        }
        _cuda.lodAblationMode = 0;
    }

    // ── Concept 4 measurement: AoS vs SoA ───────────────────────────────────
    private struct MemRow { public int agents; public float aosMs, soaMs, speedup, vramMB; }

    private bool MeasureMemory(out MemRow row)
    {
        row = default;
        if (!CudaReady) return false;
        if (!_cuda.BenchmarkAoSvsSoA(Mathf.Max(1, _mgr.pbdIterations), out var r)) return false;
        _cuda.QueryGPUMemory(out var mem);
        row = new MemRow
        {
            agents = _mgr.LiveAgentCount,
            aosMs = r.aosSolveMs, soaMs = r.soaSolveMs, speedup = r.speedup,
            vramMB = mem.totalAllocatedBytes / (1024f * 1024f)
        };
        return true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Result formatting (one block per concept, shown in the Results tab)
    // ─────────────────────────────────────────────────────────────────────────
    private const string Accent = "#00E5FF";
    private const string Muted = "#8FA3BF";

    private static string Archived() =>
        $"<color={Muted}>Native benchmark (NativeCUDA/build.bat bench, BenchmarkResults/Phase3_Rerun_2026-11-27), not this run:\n" +
        "  CPU spatial hash vs CUDA step:  1,000: 0.60 vs 0.59 ms (break-even)  |  2,500: 3.15 vs 1.29 ms\n" +
        "  5,000: 11.0 vs 2.28 ms = 4.8× faster (38× vs CPU brute force 85.8 ms)</color>";

    private string FormatCudaLive(int agents, StepStats s)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Live crowd: <b>{agents}</b> commuters, CUDA backend, {s.samples} steps sampled");
        sb.AppendLine($"• Full GPU step: <color={Accent}><b>{F(s.totalMs)} ms</b></color>   (render rate {F(s.fps, "F0")} FPS)");
        sb.AppendLine($"• Constraint kernels: {F(s.solveMs)} ms  • Hash build: {F(s.hashMs)} ms  • LOD: {F(s.lodMs)} ms");
        sb.AppendLine($"• PCIe copies: host→GPU {F(s.h2dMs)} ms, GPU→host {F(s.d2hMs)} ms");
        float copyShare = s.totalMs > 0 ? (s.h2dMs + s.d2hMs) / s.totalMs * 100f : 0f;
        sb.AppendLine($"<i>At this size {F(copyShare, "F0")}% of the step is data transfer: the GPU wins once N is large enough to hide transfer cost.</i>");
        sb.Append(Archived());
        return sb.ToString();
    }

    private static string FormatHash(List<HashRow> rows)
    {
        var sb = new StringBuilder();
        foreach (var r in rows)
        {
            sb.AppendLine($"<b>{r.agents} agents</b>: brute force tests <b>{r.bruteCandidates:N0}</b> pairs \u2192 3\u00D73-cell grid search tests <b>{r.hashCandidatesBound:N0}</b>");
            sb.AppendLine($"   pruned <color={Accent}><b>{F(r.prunedPct, "F1")}%</b></color>  |  {r.occupiedCells} occupied cells, max {r.maxPerCell}/cell, true neighbour pairs {r.bruteNeighbours:N0}, overflow {r.overflow}");
        }
        sb.Append("<i>The grid turns an all-pairs O(N²) search into a local 3×3-cell search (~O(N·k)). " +
                  "Brute-force work grows with N², grid work with N \u00D7 local density, so the saving grows with crowd size.</i>");
        return sb.ToString();
    }

    private static string FormatConvergence(List<ConvRow> rows, int agents)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Same frozen snapshot of the {agents}-commuter station crowd solved with each iteration count (average of 5 snapshots):");
        foreach (var r in rows)
            sb.AppendLine($"• <b>{r.iterations,2} iter</b>: total overlap <color={Accent}><b>{F(r.totalPen, "F2")} m</b></color> over {r.constraints} contacts (avg {F(r.avgPen, "F3")} m, max {F(r.maxPen, "F2")} m), kernels {F(r.solveMs)} ms");
        if (rows.Count >= 2)
        {
            var a = rows[0]; var b = rows[rows.Count - 1];
            string trend = b.totalPen < a.totalPen
                ? $"From {a.iterations} to {b.iterations} iterations total overlap fell {F((1f - b.totalPen / Mathf.Max(a.totalPen, 1e-6f)) * 100f, "F0")}% and the worst overlap fell {F((1f - b.maxPen / Mathf.Max(a.maxPen, 1e-6f)) * 100f, "F0")}%"
                : "Total overlap did not fall on these snapshots";
            sb.AppendLine($"<i>{trend}; kernel time grows ~linearly with iterations. That is the accuracy/cost trade-off PBD exposes.</i>");
        }
        return sb.ToString().TrimEnd();
    }

    private static string FormatLod(List<LodRow> rows, int iterations)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Average agents per tier (High / Med / Low) and solver work saved vs all-High ({iterations} iterations):");
        foreach (var r in rows)
            sb.AppendLine($"• <b>{r.label}</b>: {F(r.high, "F0")} / {F(r.med, "F0")} / {F(r.low, "F0")}  → work saved <color={Accent}><b>{F(r.workSavedPct, "F0")}%</b></color>, step {F(r.stepMs)} ms");
        sb.Append("<i>Work model from the kernel: High = all iterations + 9-cell search, Med = half the iterations, " +
                  "Low = 1 iteration every 4th frame. Each added factor promotes agents that matter (visible, in contact, in dense spots) " +
                  "and demotes the rest.</i>");
        return sb.ToString();
    }

    private static string FormatMemory(List<MemRow> rows)
    {
        var sb = new StringBuilder();
        foreach (var r in rows)
            sb.AppendLine($"• <b>{r.agents} agents</b>: AoS {F(r.aosMs, "F3")} ms vs SoA {F(r.soaMs, "F3")} ms → <color={Accent}><b>{F(r.speedup)}×</b></color>  (VRAM {F(r.vramMB)} MB)");
        sb.Append("<i>AoS: each thread reads an 80-byte AgentGPU struct, so a warp's position loads are strided. " +
                  "SoA: neighbouring threads read neighbouring floats, which coalesce into fewer 128-byte transactions. " +
                  "A speedup near 1× at small N means the kernel is launch/latency-bound, not bandwidth-bound.</i>");
        return sb.ToString();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 60-second guided demo: one concept per 12 s phase, all measured live
    // ─────────────────────────────────────────────────────────────────────────
    private void SetPhase(int index, string caption, string body)
    {
        PhaseIndex = index;
        PhaseTitle = $"Phase {index + 1}/5 — {PhaseNames[index]}";
        Caption = caption;
        PhaseBody = "<b>Watch:</b> " + caption + "\n" + body;
        PhaseStatus = "Moving the camera…";
        currentBenchmarkName = $"Demo {index + 1}/5: {PhaseNames[index]}";
        Changed();
    }

    private IEnumerator HoldUntil(float demoSeconds)
    {
        while (Time.time - _showcaseStart < demoSeconds) yield return null;
    }

    private IEnumerator ShowcaseRoutine()
    {
        var metro = MetroStationManager.Instance;
        var viz = _mgr.GetComponent<PagpVisualizer>();
        float P = phaseSeconds;

        // One spawn for the whole demo: a single, continuous commute through the
        // station. Phases change only the camera, the overlay and what is measured.
        _mgr.Configure(SolverMode.PBDCuda, ScenePreset.StationFlow, showcaseAgents);
        while (_mgr.IsRespawning) yield return null;
        _showcaseStart = Time.time;

        // ── Phase 1: CUDA parallel step ───────────────────────────────────
        SetPhase(0,
            "Commuters enter from the street stairs, tap through the fare gates and wait on the platform for the train.",
            "<b>Concept:</b> every commuter is one CUDA thread. Each physics step copies the crowd to the GPU and runs " +
            "Predict → Hash → LOD → Jacobi solve → Finalize for all of them in parallel, then copies it back.\n" +
            "<b>Measured:</b> GPU step time, split into kernel time and PCIe transfer time.");
        metro?.ApplyCCTVAngle(MetroStationManager.CCTVAngle.CCTV_Concourse);
        yield return new WaitForSeconds(1.0f);
        PhaseStatus = "Sampling CUDA event timers…";
        Changed();
        var step = new StepStats();
        yield return SampleCudaStep(P * 0.55f, step);
        ConceptResults[0] = FormatCudaLive(_mgr.LiveAgentCount, step);
        PhaseStatus = "✓ CUDA timing captured";
        Changed();
        yield return HoldUntil(P);

        // ── Phase 2: spatial hash (heat-map overlay) ──────────────────────
        viz?.SetHashGrid(true);
        SetPhase(1,
            "Floor tiles = the GPU's 4 m hash cells, coloured by how many commuters stand in each (blue few → red crowded).",
            "<b>Concept:</b> each cell is hashed into one of 8,192 GPU buckets. A commuter only tests people in its own tile " +
            "and the 8 tiles around it instead of every other commuter. Busy tiles appear at the gates and the platform doors.\n" +
            "<b>Measured:</b> GPU brute-force pair count vs the hash-grid candidate bound on this exact crowd.");
        metro?.ApplyCCTVAngle(MetroStationManager.CCTVAngle.CCTV_Concourse);
        yield return new WaitForSeconds(3.0f);
        PhaseStatus = "Comparing O(N²) vs grid search on the GPU…";
        Changed();
        if (MeasureSpatialHash(out var hr))
            ConceptResults[1] = FormatHash(new List<HashRow> { hr });
        PhaseStatus = "✓ Neighbour-search comparison captured";
        Changed();
        yield return HoldUntil(2 * P);
        viz?.SetHashGrid(false);

        // ── Phase 3: PBD Jacobi convergence (fare-gate queue) ─────────────
        SetPhase(2,
            "The fare gates squeeze the flow into 1.8 m lanes. Commuters queue and pass without walking through each other.",
            "<b>Concept:</b> Position-Based Dynamics moves positions just enough to remove overlaps. In Jacobi form every GPU " +
            "thread computes its correction independently and all are applied together, so there are no write races.\n" +
            "<b>Measured:</b> a frozen snapshot of this crowd solved with 1, 2, 4 and 8 iterations: overlap left vs GPU time.");
        metro?.ApplyCCTVAngle(MetroStationManager.CCTVAngle.CCTV_Turnstiles);
        yield return new WaitForSeconds(1.0f);
        var conv = new List<ConvRow>();
        yield return MeasureConvergence(new[] { 1, 2, 4, 8 }, 0.8f, 1.4f, conv);
        if (conv.Count > 0) ConceptResults[2] = FormatConvergence(conv, _mgr.LiveAgentCount);
        PhaseStatus = "✓ Convergence curve captured";
        Changed();
        yield return HoldUntil(3 * P);

        // ── Phase 4: 4-factor LOD (rings overlay) ─────────────────────────
        viz?.SetLodRings(true);
        SetPhase(3,
            "Commuter's-eye (VR) view. Ring under each commuter = detail chosen by the GPU: green High, amber Medium, blue Low.",
            "<b>Concept:</b> importance = 0.30·distance + 0.25·on-screen + 0.20·contacts + 0.25·density. " +
            "Near, visible, crowded commuters get full solving; people behind the viewer or far away are solved less often.\n" +
            "<b>Measured:</b> ablation A→D, adding one factor at a time.");
        metro?.ApplyCCTVAngle(MetroStationManager.CCTVAngle.VR_CommuterEyeLevel);
        yield return new WaitForSeconds(1.0f);
        var lod = new List<LodRow>();
        yield return MeasureLod(0.8f, 1.4f, lod);
        if (lod.Count > 0) ConceptResults[3] = FormatLod(lod, _mgr.pbdIterations);
        PhaseStatus = "✓ LOD ablation captured";
        Changed();
        yield return HoldUntil(4 * P);
        viz?.SetLodRings(false);

        // ── Phase 5: evacuation + memory layout ───────────────────────────
        SetPhase(4,
            "Evacuation drill: the alarm sounds and every commuter re-routes to the nearest emergency exit in the end walls.",
            "<b>Concept:</b> GPU memory layout. Array-of-Structures keeps each commuter's 80-byte record together; " +
            "Structure-of-Arrays stores each field contiguously so a warp's 32 loads coalesce.\n" +
            "<b>Measured:</b> the same solve kernel on both layouts, on this crowd.");
        metro?.ApplyCCTVAngle(MetroStationManager.CCTVAngle.CCTV_Concourse);
        if (metro != null && !metro.isEvacuationActive) metro.ToggleEmergencyEvacuation();
        PhaseStatus = "Running AoS and SoA kernels…";
        Changed();
        yield return new WaitForSeconds(1.5f);
        if (MeasureMemory(out var mr))
            ConceptResults[4] = FormatMemory(new List<MemRow> { mr });
        PhaseStatus = "✓ Memory-layout comparison captured";
        Changed();
        yield return HoldUntil(5 * P);

        PhaseIndex = ConceptCount;
        PhaseTitle = "Demo complete";
        Caption = "Demo complete. All five PAGP concepts were measured live. See the Results tab.";
        PhaseBody = "All five PAGP concepts were measured on the live crowd. Their results and explanations are on the Results tab.";
        PhaseStatus = "✓ Finished";
        currentBenchmarkName = "Guided demo complete";
        SaveShowcaseCsv();

        // Back to normal service (turning off the alarm respawns the commuters once).
        if (metro != null && metro.isEvacuationActive) metro.ToggleEmergencyEvacuation();
        metro?.ApplyCCTVAngle(MetroStationManager.CCTVAngle.CCTV_Concourse);
    }

    private void SaveShowcaseCsv()
    {
        var sb = new StringBuilder("Concept,Result\n");
        for (int i = 0; i < ConceptCount; i++)
        {
            string plain = System.Text.RegularExpressions.Regex.Replace(ConceptResults[i] ?? "not measured", "<.*?>", "");
            sb.Append('"').Append(ConceptNames[i]).Append("\",\"").Append(plain.Replace("\"", "'").Replace("\n", " | ")).AppendLine("\"");
        }
        SaveCSV($"PAGP_GuidedDemo_{DateTime.Now:yyyyMMdd_HHmmss}.csv", sb.ToString());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Individual experiments (sweeps + CSV export)
    // ─────────────────────────────────────────────────────────────────────────
    private IEnumerator ScalabilityExperiment()
    {
        SolverMode[] solvers = { SolverMode.PBD, SolverMode.PBDJobs, SolverMode.PBDCuda };
        string[] names = { "CPU 1 thread", "CPU Jobs+Burst", "CUDA GPU" };
        int total = solvers.Length * sweepAgentCounts.Length, done = 0;
        var csv = new StringBuilder("Solver,Agents,SolverStepMs,FPS,CudaTotalMs,CudaKernelMs,CudaH2DMs,CudaD2HMs\n");
        var text = new StringBuilder();
        text.AppendLine("Solver step time per physics step (lower is better), measured on the live scene:");

        foreach (int count in sweepAgentCounts)
        {
            var ms = new float[solvers.Length];
            for (int s = 0; s < solvers.Length; s++)
            {
                currentBenchmarkName = $"Exp 1: {names[s]} @ {count} agents";
                benchmarkProgress = (float)done++ / total;
                Changed();

                _mgr.Configure(solvers[s], ScenePreset.StationFlow, count);
                yield return WaitForCrowd(1.0f);
                ProfilerManager.Instance?.ResetMeasurements();
                var st = new StepStats();
                yield return SampleCudaStep(2.0f, st);

                float avg = ProfilerManager.Instance != null ? ProfilerManager.Instance.GetAverageMs("Solver") : 0f;
                ms[s] = float.IsNaN(avg) ? 0f : avg;
                csv.Append(solvers[s]).Append(',').Append(count).Append(',').Append(F(ms[s], "F3")).Append(',')
                   .Append(F(st.fps, "F1")).Append(',').Append(F(st.totalMs, "F3")).Append(',').Append(F(st.solveMs, "F3"))
                   .Append(',').Append(F(st.h2dMs, "F3")).Append(',').AppendLine(F(st.d2hMs, "F3"));
            }
            float gpu = Mathf.Max(ms[2], 0.001f);
            text.AppendLine($"• <b>{count} agents</b>: 1-thread {F(ms[0])} ms | Jobs {F(ms[1])} ms | CUDA <color={Accent}><b>{F(ms[2])} ms</b></color>" +
                            $"  → {F(ms[0] / gpu, "F1")}× vs 1-thread, {F(ms[1] / gpu, "F1")}× vs Jobs");
        }
        text.AppendLine("<i>CPU cost grows with N·neighbours on few cores; the GPU spreads agents over thousands of threads. " +
                        "At small N the fixed PCIe copy + launch cost dominates, so the gap widens with crowd size.</i>");
        text.Append(Archived());
        ConceptResults[0] = text.ToString();
        SaveCSV($"PAGP_Exp1_CpuVsCuda_{DateTime.Now:yyyyMMdd_HHmmss}.csv", csv.ToString());
        RestoreSceneConfig();
    }

    private IEnumerator SpatialHashExperiment()
    {
        var rows = new List<HashRow>();
        var csv = new StringBuilder("Agents,OccupiedCells,MaxPerCell,Overflow,BruteForcePairs,TrueNeighbourPairs,HashCandidatesUpperBound,PrunedPct\n");
        for (int i = 0; i < sweepAgentCounts.Length; i++)
        {
            int count = sweepAgentCounts[i];
            currentBenchmarkName = $"Exp 2: spatial hash @ {count} agents";
            benchmarkProgress = (float)i / sweepAgentCounts.Length;
            Changed();
            _mgr.Configure(SolverMode.PBDCuda, ScenePreset.StationFlow, count);
            yield return WaitForCrowd(1.5f);
            if (!MeasureSpatialHash(out var r)) continue;
            rows.Add(r);
            csv.Append(r.agents).Append(',').Append(r.occupiedCells).Append(',').Append(r.maxPerCell).Append(',')
               .Append(r.overflow).Append(',').Append(r.bruteCandidates).Append(',').Append(r.bruteNeighbours).Append(',')
               .Append(r.hashCandidatesBound).Append(',').AppendLine(F(r.prunedPct));
        }
        if (rows.Count > 0) ConceptResults[1] = FormatHash(rows);
        else ConceptResults[1] = "<color=#FF6060>CUDA backend not available — could not measure.</color>";
        SaveCSV($"PAGP_Exp2_SpatialHash_{DateTime.Now:yyyyMMdd_HHmmss}.csv", csv.ToString());
        RestoreSceneConfig();
    }

    private IEnumerator ConvergenceExperiment()
    {
        _mgr.Configure(SolverMode.PBDCuda, ScenePreset.StationFlow, experimentAgents);
        yield return WaitForCrowd(1.5f);
        MetroStationManager.Instance?.ApplyCCTVAngle(MetroStationManager.CCTVAngle.CCTV_Turnstiles);
        var rows = new List<ConvRow>();
        yield return MeasureConvergence(new[] { 1, 2, 4, 8, 12 }, 1.0f, 1.5f, rows);

        var csv = new StringBuilder("Iterations,TotalOverlapM,AvgOverlapM,MaxOverlapM,Contacts,KernelMs\n");
        foreach (var r in rows)
            csv.Append(r.iterations).Append(',').Append(F(r.totalPen, "F4")).Append(',').Append(F(r.avgPen, "F4")).Append(',').Append(F(r.maxPen, "F4"))
               .Append(',').Append(r.constraints).Append(',').AppendLine(F(r.solveMs, "F3"));
        ConceptResults[2] = rows.Count > 0 ? FormatConvergence(rows, _mgr.LiveAgentCount)
                                           : "<color=#FF6060>CUDA backend not available — could not measure.</color>";
        SaveCSV($"PAGP_Exp3_PBDConvergence_{DateTime.Now:yyyyMMdd_HHmmss}.csv", csv.ToString());
        RestoreSceneConfig();
    }

    private IEnumerator LodExperiment()
    {
        _mgr.Configure(SolverMode.PBDCuda, ScenePreset.StationFlow, experimentAgents);
        yield return WaitForCrowd(1.5f);
        MetroStationManager.Instance?.ApplyCCTVAngle(MetroStationManager.CCTVAngle.VR_CommuterEyeLevel);
        var rows = new List<LodRow>();
        if (CudaReady) yield return MeasureLod(1.0f, 2.0f, rows);

        var csv = new StringBuilder("Configuration,AvgHigh,AvgMed,AvgLow,StepMs,SolverWorkSavedPct\n");
        foreach (var r in rows)
            csv.Append(r.label.Trim()).Append(',').Append(F(r.high, "F1")).Append(',').Append(F(r.med, "F1")).Append(',')
               .Append(F(r.low, "F1")).Append(',').Append(F(r.stepMs, "F3")).Append(',').AppendLine(F(r.workSavedPct, "F1"));
        ConceptResults[3] = rows.Count > 0 ? FormatLod(rows, _mgr.pbdIterations)
                                           : "<color=#FF6060>CUDA backend not available — could not measure.</color>";
        SaveCSV($"PAGP_Exp4_LODAblation_{DateTime.Now:yyyyMMdd_HHmmss}.csv", csv.ToString());
        RestoreSceneConfig();
    }

    private IEnumerator MemoryExperiment()
    {
        var rows = new List<MemRow>();
        var csv = new StringBuilder("Agents,AoSKernelMs,SoAKernelMs,SoASpeedup,AllocatedVRAM_MB\n");
        for (int i = 0; i < sweepAgentCounts.Length; i++)
        {
            int count = sweepAgentCounts[i];
            currentBenchmarkName = $"Exp 5: AoS vs SoA @ {count} agents";
            benchmarkProgress = (float)i / sweepAgentCounts.Length;
            Changed();
            _mgr.Configure(SolverMode.PBDCuda, ScenePreset.StationFlow, count);
            yield return WaitForCrowd(1.0f);
            if (!MeasureMemory(out var r)) continue;
            rows.Add(r);
            csv.Append(r.agents).Append(',').Append(F(r.aosMs, "F3")).Append(',').Append(F(r.soaMs, "F3")).Append(',')
               .Append(F(r.speedup)).Append(',').AppendLine(F(r.vramMB));
        }
        ConceptResults[4] = rows.Count > 0 ? FormatMemory(rows)
                                           : "<color=#FF6060>CUDA backend not available — could not measure.</color>";
        SaveCSV($"PAGP_Exp5_MemoryLayout_{DateTime.Now:yyyyMMdd_HHmmss}.csv", csv.ToString());
        RestoreSceneConfig();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CSV export: never throws (an exception here used to leave isRunning stuck)
    // ─────────────────────────────────────────────────────────────────────────
    private void SaveCSV(string filename, string content)
    {
        try
        {
            string path = Path.Combine(Application.persistentDataPath, filename);
            File.WriteAllText(path, content, Encoding.UTF8);

            string desktopResults = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "pagp", "BenchmarkResults");
            if (Directory.Exists(desktopResults))
                File.WriteAllText(Path.Combine(desktopResults, filename), content, Encoding.UTF8);

            Debug.Log($"[PAGPPerformanceLab] Exported {filename} to {path}");
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[PAGPPerformanceLab] Could not export {filename}: {ex.Message}");
        }
    }
}
