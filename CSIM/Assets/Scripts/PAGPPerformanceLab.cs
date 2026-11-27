using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

/// <summary>
/// PAGPPerformanceLab: Automated scientific benchmarking suite for CUDA vs CPU crowd simulation.
/// Implements 6 comprehensive experiments:
/// 1. CPU vs CUDA Scaling (500, 1000, 2500, 5000 commuters; SingleThread, Jobs, CUDA)
/// 2. Spatial Hashing vs Brute Force Neighbor Search (distance checks, cell occupancy, overflow)
/// 3. Dynamic LOD Ablation Study (Distance only, +Frustum, +Interaction, +Density)
/// 4. PBD Jacobi Convergence (Error vs Iterations 1, 2, 4, 8, 12, 16)
/// 5. GPU Memory Layout Analysis (AoS vs SoA comparison)
/// 6. Scalability Analysis & CSV Exporter
/// </summary>
public class PAGPPerformanceLab : MonoBehaviour
{
    public static PAGPPerformanceLab Instance { get; private set; }

    [Header("Lab Configuration")]
    public int[] benchmarkAgentCounts = new int[] { 500, 1000, 2500, 5000 };
    public float warmupSeconds = 1.0f;
    public float measurementSeconds = 3.0f;

    [Header("Status")]
    public bool isRunning = false;
    public string currentBenchmarkName = "Idle";
    public float benchmarkProgress = 0f;

    private SimulationManager _mgr;
    private Coroutine _benchmarkRoutine;

    public struct ScalabilityRow
    {
        public string Solver;
        public int AgentCount;
        public float FPS;
        public float Throughput;
        public float FrameAvgMs;
        public float FrameP95Ms;
        public float CudaTotalMs;
        public float CudaKernelSolveMs;
        public float CudaH2DMs;
        public float CudaD2HMs;
        public float SpeedupVsSingleThread;
        public float SpeedupVsJobs;
    }

    public struct SpatialHashRow
    {
        public int AgentCount;
        public int OccupiedCells;
        public int MaxCellOccupancy;
        public int OverflowCount;
        public long BruteForceCandidates;
        public long BruteForceNeighbors;
        public float ChecksEliminatedPct;
    }

    public struct LODAblationRow
    {
        public string Configuration;
        public int HighLODCount;
        public int MedLODCount;
        public int LowLODCount;
        public float StepMs;
        public float FPS;
        public float WorkloadReductionPct;
    }

    public struct PBDConvergenceRow
    {
        public int Iterations;
        public float AvgPenetrationBefore;
        public float MaxPenetrationBefore;
        public float SolverTimeMs;
        public int ActiveConstraints;
    }

    public struct MemoryLayoutRow
    {
        public int AgentCount;
        public float AoSSolveMs;
        public float SoASolveMs;
        public float Speedup;
        public long TotalVRAMBytes;
    }

    private void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(this); return; }
        Instance = this;
    }

    private void Start()
    {
        _mgr = SimulationManager.Instance;
    }

    public void StopBenchmark()
    {
        if (_benchmarkRoutine != null)
        {
            StopCoroutine(_benchmarkRoutine);
            _benchmarkRoutine = null;
        }
        isRunning = false;
        currentBenchmarkName = "Stopped by user";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Experiment 1: CPU vs CUDA Scalability
    // ─────────────────────────────────────────────────────────────────────────
    public void RunComprehensiveSuite(Action onComplete = null)
    {
        if (isRunning) return;
        _benchmarkRoutine = StartCoroutine(ComprehensiveSuiteRoutine(onComplete));
    }

    private IEnumerator ComprehensiveSuiteRoutine(Action onComplete)
    {
        isRunning = true;
        currentBenchmarkName = "Scalability Benchmark";

        var rows = new List<ScalabilityRow>();
        SolverMode[] solvers = new SolverMode[] { SolverMode.PBD, SolverMode.PBDJobs, SolverMode.PBDCuda };
        int total = solvers.Length * benchmarkAgentCounts.Length;
        int current = 0;

        SolverMode origSolver = _mgr.activeSolver;
        int origCount = _mgr.agentCount;
        ScenePreset origPreset = _mgr.activePreset;

        _mgr.SetPreset(ScenePreset.Bidirectional);
        yield return new WaitForSeconds(0.2f);

        for (int c = 0; c < benchmarkAgentCounts.Length; c++)
        {
            int count = benchmarkAgentCounts[c];
            _mgr.agentCount = count;
            float singleThreadMs = 0f;
            float jobsMs = 0f;

            for (int s = 0; s < solvers.Length; s++)
            {
                SolverMode mode = solvers[s];
                current++;
                benchmarkProgress = (float)current / total;
                currentBenchmarkName = $"Scaling: {mode} @ {count} commuters ({current}/{total})";

                _mgr.SetSolverMode(mode);
                _mgr.Respawn();

                yield return new WaitForSeconds(warmupSeconds);
                ProfilerManager.Instance?.ResetMeasurements();

                float elapsed = 0f;
                int frames = 0;
                while (elapsed < measurementSeconds)
                {
                    yield return null;
                    elapsed += Time.unscaledDeltaTime;
                    frames++;
                }

                float fps = frames / Mathf.Max(elapsed, 0.001f);
                float frameAvg = ProfilerManager.Instance != null ? ProfilerManager.Instance.GetAverageMs("Frame") : 0f;
                float frameP95 = ProfilerManager.Instance != null ? ProfilerManager.Instance.GetPercentileMs("Frame", 95f) : 0f;

                if (mode == SolverMode.PBD) singleThreadMs = frameAvg;
                if (mode == SolverMode.PBDJobs) jobsMs = frameAvg;

                var row = new ScalabilityRow
                {
                    Solver = mode.ToString(),
                    AgentCount = count,
                    FPS = fps,
                    Throughput = fps * count,
                    FrameAvgMs = frameAvg,
                    FrameP95Ms = frameP95
                };

                var cuda = _mgr.GetComponent<PBDCudaSolver>();
                if (mode == SolverMode.PBDCuda && cuda != null && cuda.IsInitialized)
                {
                    var timings = cuda.LastTimings;
                    row.CudaTotalMs = timings.totalMs;
                    row.CudaKernelSolveMs = timings.solveMs;
                    row.CudaH2DMs = timings.h2dMs;
                    row.CudaD2HMs = timings.d2hMs;

                    if (singleThreadMs > 0.001f) row.SpeedupVsSingleThread = singleThreadMs / Mathf.Max(timings.totalMs, 0.001f);
                    if (jobsMs > 0.001f) row.SpeedupVsJobs = jobsMs / Mathf.Max(timings.totalMs, 0.001f);
                }

                rows.Add(row);
            }
        }

        _mgr.SetPreset(origPreset);
        _mgr.agentCount = origCount;
        _mgr.SetSolverMode(origSolver);
        _mgr.Respawn();

        // Export to CSV
        var sb = new StringBuilder();
        sb.AppendLine("Solver,AgentCount,FPS,ThroughputAgentsPerSec,FrameAvgMs,FrameP95Ms,CudaTotalMs,CudaKernelSolveMs,CudaH2DMs,CudaD2HMs,SpeedupVsSingleThread,SpeedupVsJobs");
        foreach (var r in rows)
        {
            sb.Append(r.Solver).Append(',')
              .Append(r.AgentCount).Append(',')
              .Append(r.FPS.ToString("F1", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.Throughput.ToString("F0", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.FrameAvgMs.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.FrameP95Ms.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.CudaTotalMs.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.CudaKernelSolveMs.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.CudaH2DMs.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.CudaD2HMs.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.SpeedupVsSingleThread.ToString("F2", CultureInfo.InvariantCulture)).Append(',')
              .AppendLine(r.SpeedupVsJobs.ToString("F2", CultureInfo.InvariantCulture));
        }
        string filename = $"PAGP_Experiment1_Scalability_{DateTime.UtcNow:yyyyMMdd_HHmmss_fffffff}.csv";
        SaveCSV(filename, sb.ToString());

        isRunning = false;
        currentBenchmarkName = "Scalability Benchmark Complete";
        benchmarkProgress = 1f;
        onComplete?.Invoke();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Experiment 2: Spatial Hashing Analysis
    // ─────────────────────────────────────────────────────────────────────────
    public void RunSpatialHashAnalysis(Action onComplete = null)
    {
        if (isRunning) return;
        _benchmarkRoutine = StartCoroutine(SpatialHashAnalysisRoutine(onComplete));
    }

    private IEnumerator SpatialHashAnalysisRoutine(Action onComplete)
    {
        isRunning = true;
        currentBenchmarkName = "Spatial Hash vs Brute Force";

        var cuda = _mgr.GetComponent<PBDCudaSolver>();
        if (cuda == null || !cuda.IsInitialized)
        {
            _mgr.SetSolverMode(SolverMode.PBDCuda);
            yield return new WaitForSeconds(0.5f);
            cuda = _mgr.GetComponent<PBDCudaSolver>();
        }

        int[] testCounts = new int[] { 500, 1000, 2500, 5000 };
        var rows = new List<SpatialHashRow>();

        for (int i = 0; i < testCounts.Length; i++)
        {
            int count = testCounts[i];
            benchmarkProgress = (float)(i + 1) / testCounts.Length;
            currentBenchmarkName = $"Spatial Hash @ {count} commuters";

            _mgr.agentCount = count;
            _mgr.Respawn();
            yield return new WaitForSeconds(1.5f);

            if (cuda != null && cuda.IsInitialized)
            {
                cuda.QueryHashGridStats(out var hStats);
                cuda.BenchmarkBruteForce(_mgr.neighborDist, out long bfCandidates, out long bfNeighbors);

                long hashCandidates = (long)count * 9 * Mathf.Min(hStats.maxOccupancy, 16);
                float savedPct = bfCandidates > 0 ? (1f - (float)hashCandidates / bfCandidates) * 100f : 0f;

                rows.Add(new SpatialHashRow
                {
                    AgentCount = count,
                    OccupiedCells = hStats.occupiedCells,
                    MaxCellOccupancy = hStats.maxOccupancy,
                    OverflowCount = hStats.overflowCount,
                    BruteForceCandidates = bfCandidates,
                    BruteForceNeighbors = bfNeighbors,
                    ChecksEliminatedPct = savedPct
                });
            }
        }

        var sb = new StringBuilder();
        sb.AppendLine("AgentCount,OccupiedCells,MaxCellOccupancy,BucketOverflowCount,BruteForceCandidates,BruteForceNeighbors,ChecksEliminatedPct");
        foreach (var r in rows)
        {
            sb.Append(r.AgentCount).Append(',')
              .Append(r.OccupiedCells).Append(',')
              .Append(r.MaxCellOccupancy).Append(',')
              .Append(r.OverflowCount).Append(',')
              .Append(r.BruteForceCandidates).Append(',')
              .Append(r.BruteForceNeighbors).Append(',')
              .AppendLine(r.ChecksEliminatedPct.ToString("F2", CultureInfo.InvariantCulture));
        }
        SaveCSV("PAGP_Experiment2_SpatialHash.csv", sb.ToString());

        isRunning = false;
        currentBenchmarkName = "Spatial Hash Analysis Complete";
        benchmarkProgress = 1f;
        onComplete?.Invoke();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Experiment 3: Dynamic LOD Ablation Study
    // ─────────────────────────────────────────────────────────────────────────
    public void RunLODAblationStudy(Action onComplete = null)
    {
        if (isRunning) return;
        _benchmarkRoutine = StartCoroutine(LODAblationRoutine(onComplete));
    }

    private IEnumerator LODAblationRoutine(Action onComplete)
    {
        isRunning = true;
        currentBenchmarkName = "Dynamic LOD Ablation Study";

        var cuda = _mgr.GetComponent<PBDCudaSolver>();
        _mgr.SetSolverMode(SolverMode.PBDCuda);
        _mgr.agentCount = 2500;
        _mgr.Respawn();
        yield return new WaitForSeconds(1.0f);

        // 4 Ablation modes:
        // 1: Distance Only
        // 2: Distance + Visibility
        // 3: Distance + Visibility + Interaction
        // 4: All 4 Factors (Distance + Visibility + Interaction + Density)
        string[] labels = new string[]
        {
            "A: Distance Only",
            "B: Distance + Visibility",
            "C: Distance + Visibility + Interaction",
            "D: Full 4-Factor (Distance+Vis+Inter+Density)"
        };
        int[] modes = new int[] { 1, 2, 3, 4 };
        var rows = new List<LODAblationRow>();

        float baseStepMs = 0f;

        for (int m = 0; m < modes.Length; m++)
        {
            benchmarkProgress = (float)(m + 1) / modes.Length;
            currentBenchmarkName = $"LOD Ablation: {labels[m]}";

            cuda.lodAblationMode = modes[m];
            yield return new WaitForSeconds(1.0f);

            float totalStepMs = 0f;
            int frames = 0;
            while (frames < 90)
            {
                yield return null;
                totalStepMs += cuda.LastTimings.totalMs;
                frames++;
            }
            float avgStep = totalStepMs / frames;
            if (m == 0) baseStepMs = avgStep;

            _mgr.GetFidelityCounts(out int high, out int med, out int low);
            float workloadReduction = (1f - (avgStep / Mathf.Max(baseStepMs, 0.001f))) * 100f;

            rows.Add(new LODAblationRow
            {
                Configuration = labels[m],
                HighLODCount = high,
                MedLODCount = med,
                LowLODCount = low,
                StepMs = avgStep,
                FPS = 1000f / Mathf.Max(avgStep, 0.001f),
                WorkloadReductionPct = workloadReduction
            });
        }

        cuda.lodAblationMode = 0; // restore default full mode

        var sb = new StringBuilder();
        sb.AppendLine("Configuration,HighLODCount,MedLODCount,LowLODCount,StepMs,FPS,WorkloadReductionVsConfigAPct");
        foreach (var r in rows)
        {
            sb.Append(r.Configuration).Append(',')
              .Append(r.HighLODCount).Append(',')
              .Append(r.MedLODCount).Append(',')
              .Append(r.LowLODCount).Append(',')
              .Append(r.StepMs.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.FPS.ToString("F1", CultureInfo.InvariantCulture)).Append(',')
              .AppendLine(r.WorkloadReductionPct.ToString("F2", CultureInfo.InvariantCulture));
        }
        SaveCSV("PAGP_Experiment3_LOD_Ablation.csv", sb.ToString());

        isRunning = false;
        currentBenchmarkName = "LOD Ablation Study Complete";
        benchmarkProgress = 1f;
        onComplete?.Invoke();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Experiment 4: PBD Convergence Analysis
    // ─────────────────────────────────────────────────────────────────────────
    public void RunPBDConvergenceTest(Action onComplete = null)
    {
        if (isRunning) return;
        _benchmarkRoutine = StartCoroutine(PBDConvergenceRoutine(onComplete));
    }

    private IEnumerator PBDConvergenceRoutine(Action onComplete)
    {
        isRunning = true;
        currentBenchmarkName = "PBD Convergence vs Iterations";

        var cuda = _mgr.GetComponent<PBDCudaSolver>();
        _mgr.SetSolverMode(SolverMode.PBDCuda);
        _mgr.agentCount = 2500;
        _mgr.Respawn();
        yield return new WaitForSeconds(1.0f);

        int[] iterList = new int[] { 1, 2, 4, 8, 12, 16 };
        var rows = new List<PBDConvergenceRow>();

        int origIter = _mgr.pbdIterations;

        for (int i = 0; i < iterList.Length; i++)
        {
            int it = iterList[i];
            benchmarkProgress = (float)(i + 1) / iterList.Length;
            currentBenchmarkName = $"PBD Convergence @ {it} iterations";

            _mgr.pbdIterations = it;
            yield return new WaitForSeconds(1.0f);

            cuda.QueryConstraintViolations(out var cStats);

            rows.Add(new PBDConvergenceRow
            {
                Iterations = it,
                AvgPenetrationBefore = cStats.avgViolationBefore,
                MaxPenetrationBefore = cStats.maxViolationBefore,
                SolverTimeMs = cuda.LastTimings.solveMs,
                ActiveConstraints = cStats.totalConstraints
            });
        }

        _mgr.pbdIterations = origIter;

        var sb = new StringBuilder();
        sb.AppendLine("Iterations,AvgPenetrationMeters,MaxPenetrationMeters,SolverKernelTimeMs,ActiveConstraintsDetected");
        foreach (var r in rows)
        {
            sb.Append(r.Iterations).Append(',')
              .Append(r.AvgPenetrationBefore.ToString("F4", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.MaxPenetrationBefore.ToString("F4", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.SolverTimeMs.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
              .AppendLine(r.ActiveConstraints.ToString());
        }
        SaveCSV("PAGP_Experiment4_PBD_Convergence.csv", sb.ToString());

        isRunning = false;
        currentBenchmarkName = "PBD Convergence Test Complete";
        benchmarkProgress = 1f;
        onComplete?.Invoke();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Experiment 5: AoS vs SoA Memory Layout
    // ─────────────────────────────────────────────────────────────────────────
    public void RunMemoryLayoutBenchmark(Action onComplete = null)
    {
        if (isRunning) return;
        _benchmarkRoutine = StartCoroutine(MemoryLayoutRoutine(onComplete));
    }

    private IEnumerator MemoryLayoutRoutine(Action onComplete)
    {
        isRunning = true;
        currentBenchmarkName = "AoS vs SoA Memory Layout Comparison";

        var cuda = _mgr.GetComponent<PBDCudaSolver>();
        _mgr.SetSolverMode(SolverMode.PBDCuda);

        int[] counts = new int[] { 500, 1000, 2500, 5000 };
        var rows = new List<MemoryLayoutRow>();

        for (int i = 0; i < counts.Length; i++)
        {
            int count = counts[i];
            benchmarkProgress = (float)(i + 1) / counts.Length;
            currentBenchmarkName = $"Memory Benchmark @ {count} commuters";

            _mgr.agentCount = count;
            _mgr.Respawn();
            yield return new WaitForSeconds(1.5f);

            if (cuda != null && cuda.BenchmarkAoSvsSoA(_mgr.pbdIterations, out var res))
            {
                cuda.QueryGPUMemory(out var mem);
                rows.Add(new MemoryLayoutRow
                {
                    AgentCount = count,
                    AoSSolveMs = res.aosSolveMs,
                    SoASolveMs = res.soaSolveMs,
                    Speedup = res.speedup,
                    TotalVRAMBytes = mem.totalAllocatedBytes
                });
            }
        }

        var sb = new StringBuilder();
        sb.AppendLine("AgentCount,AoSSolveKernelMs,SoASolveKernelMs,SoASpeedupRatio,AllocatedVRAMBytes");
        foreach (var r in rows)
        {
            sb.Append(r.AgentCount).Append(',')
              .Append(r.AoSSolveMs.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.SoASolveMs.ToString("F3", CultureInfo.InvariantCulture)).Append(',')
              .Append(r.Speedup.ToString("F2", CultureInfo.InvariantCulture)).Append(',')
              .AppendLine(r.TotalVRAMBytes.ToString());
        }
        SaveCSV("PAGP_Experiment5_Memory_Layout.csv", sb.ToString());

        isRunning = false;
        currentBenchmarkName = "AoS vs SoA Benchmark Complete";
        benchmarkProgress = 1f;
        onComplete?.Invoke();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CSV File Exporter
    // ─────────────────────────────────────────────────────────────────────────
private void SaveCSV(string filename, string content)
    {
        string desktop = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
        string resultsDir = Path.Combine(desktop, "pagp", "BenchmarkResults");
        Directory.CreateDirectory(resultsDir);

        string stem = Path.GetFileNameWithoutExtension(filename);
        string extension = Path.GetExtension(filename);
        byte[] bytes = Encoding.UTF8.GetBytes(content);

        for (int attempt = 0; attempt < 100; attempt++)
        {
            string uniqueFilename = attempt == 0 ? filename : $"{stem}_{attempt}{extension}";
            string persistentPath = Path.Combine(Application.persistentDataPath, uniqueFilename);
            string resultsPath = Path.Combine(resultsDir, uniqueFilename);
            bool persistentCreated = false;

            FileStream persistentStream = null;
            FileStream resultsStream = null;
            try
            {
                persistentStream = new FileStream(persistentPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                persistentCreated = true;
                resultsStream = new FileStream(resultsPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                persistentStream.Write(bytes, 0, bytes.Length);
                resultsStream.Write(bytes, 0, bytes.Length);
                Debug.Log($"[PAGPPerformanceLab] Exported {uniqueFilename} to {persistentPath} and {resultsPath}");
                return;
            }
            catch (IOException)
            {
                persistentStream?.Dispose();
                persistentStream = null;
                resultsStream?.Dispose();
                resultsStream = null;

                // Remove only this incomplete candidate before trying another unique name.
                if (persistentCreated && !File.Exists(resultsPath))
                    File.Delete(persistentPath);
            }
            finally
            {
                persistentStream?.Dispose();
                resultsStream?.Dispose();
            }
        }

        throw new IOException($"Could not create unique benchmark output files for {filename}.");
    }
}
