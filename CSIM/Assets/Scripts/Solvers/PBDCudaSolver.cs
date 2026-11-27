// ============================================================================
//  PBDCudaSolver.cs
//  Native NVIDIA CUDA C++ backend for Position-Based Dynamics crowd simulation.
//  Uses CrowdSimulationCUDA.dll compiled via NVCC.
// ============================================================================

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Profiling;
using UnityEngine;

public class PBDCudaSolver : MonoBehaviour
{
    private static readonly ProfilerMarker CudaSolveMarker = new ProfilerMarker("PBDCuda.Solve");

    [Header("Spatial Grid")]
    public float cellSize = 4f;

    [Header("Density LOD Thresholds")]
    public int highDensityCutoff = 12;
    public int medDensityCutoff  = 4;

    [Header("Collision Physics")]
    [Range(0f, 1f)] public float kShortRange    = 1.0f;
    [Range(0f, 1f)] public float kLongRange     = 1.0f;
    [Range(0f, 1f)] public float muStatic       = 0.21f;
    [Range(0f, 1f)] public float muKinematic    = 0.10f;
    [Range(0f, 2f)] public float avgCoefficient = 0.25f;
    [Range(0f, 5f)] public float lrcLookAhead   = 2.0f;

    [Header("Importance Scoring & Ablation")]
    public Camera activeCamera;
    public float weightDistance = 0.30f;
    public float weightVisibility = 0.25f;
    public float weightInteraction = 0.20f;
    public float weightDensity = 0.25f;
    public float highImportanceThreshold = 0.60f;
    public float medImportanceThreshold = 0.30f;
    public float maxImportanceDistance = 60f;
    public int maxInteractionContacts = 6;
    [Tooltip("0=All 4 Factors, 1=Distance Only, 2=Distance+Visibility, 3=Distance+Vis+Contacts, 4=All 4 Factors")]
    public int lodAblationMode = 0;

    // ── Native CUDA Struct Layouts (Must match CrowdCommon.h exactly) ─────────
    [StructLayout(LayoutKind.Sequential)]
    public struct AgentGPU
    {
        public Vector3 position;            // 12 bytes
        public float   ksi;                 //  4 bytes -> 16
        public Vector3 predictedPosition;   // 12 bytes
        public float   radius;              //  4 bytes -> 32
        public Vector3 velocity;            // 12 bytes
        public float   mass;                //  4 bytes -> 48
        public Vector3 preferredVelocity;   // 12 bytes
        public int     lodLevel;            //  4 bytes -> 64
        public float   importanceScore;     //  4 bytes
        public int     lastContactCount;    //  4 bytes
        public Vector2 _padding;            //  8 bytes -> 80
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SimParams
    {
        public int   agentCount;
        public float deltaTime;
        public float cellSize;

        public int   highDensityCutoff;
        public int   medDensityCutoff;

        public Vector3 focusPoint;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 6)]
        public Vector4[] frustumPlanes;
        public int    hasFrustumPlanes;
        public float  weightDistance;
        public float  weightVisibility;
        public float  weightInteraction;
        public float  weightDensity;
        public float  highImportanceThresh;
        public float  medImportanceThresh;
        public float  maxImportanceDist;
        public int    maxInteractionContacts;
        public int    currentIteration;
        public int    iterationCount;
        public int    solveLowThisFrame;

        public float kShortRange;
        public float kLongRange;
        public float muStatic;
        public float muKinematic;
        public float avgCoefficient;
        public float lrcLookAhead;

        public int   lodAblationMode;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUDATimingData
    {
        public float h2dMs;
        public float predictMs;
        public float clearHashMs;
        public float hashAgentsMs;
        public float lodMs;
        public float solveMs;
        public float applyMs;
        public float finalizeMs;
        public float d2hMs;
        public float totalMs;
        public float streamSyncMs;
        public int   iterationCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct HashGridStats
    {
        public int occupiedCells;
        public int maxOccupancy;
        public int totalInserted;
        public int overflowCount;
        public long totalCandidates;
        public long actualNeighbors;
        public long distanceChecks;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ConstraintStats
    {
        public float avgViolationBefore;
        public float maxViolationBefore;
        public float avgViolationAfter;
        public float maxViolationAfter;
        public float avgCorrectionMag;
        public float maxCorrectionMag;
        public int   totalConstraints;
        public int   activeAgents;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct GPUMemoryInfo
    {
        public long agentBufferBytes;
        public long correctionBufferBytes;
        public long hashGridBytes;
        public long hashGridCountBytes;
        public long totalAllocatedBytes;
        public long deviceFreeBytes;
        public long deviceTotalBytes;
        public int  agentStructSize;
        public int  agentStructAlignment;
        public int  allocatedAgentCapacity;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SoABenchmarkResult
    {
        public float aosSolveMs;
        public float soaSolveMs;
        public float speedup;
    }

    // ── Native P/Invoke Bindings ──────────────────────────────────────────────
    private const string DLL_NAME = "CrowdSimulationCUDA";

    [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
    private static extern bool InitCUDA(int maxAgents);

    [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
    private static extern bool StepCUDA(
        [In, Out] AgentGPU[] hostAgents,
        int agentCount,
        ref SimParams parameters,
        int iterations);

    [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
    private static extern void ShutdownCUDA();

    [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
    private static extern bool GetLastTimings(out CUDATimingData outTimings);

    [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
    private static extern bool GetGPUMemoryInfo(out GPUMemoryInfo outInfo);

    [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
    private static extern bool RunBruteForceNeighborBenchmark(
        [In] AgentGPU[] hostAgents,
        int agentCount,
        float interactionRadius,
        out long outTotalCandidates,
        out long outTotalNeighbors);

    [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
    private static extern bool GetHashGridStats(out HashGridStats outStats);

    [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
    private static extern bool MeasureConstraintViolations(
        [In] AgentGPU[] hostAgents,
        int agentCount,
        ref SimParams parameters,
        out ConstraintStats outStats);

    [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
    private static extern bool RunAoSvsSoABenchmark(
        [In] AgentGPU[] hostAgents,
        int agentCount,
        ref SimParams parameters,
        int iterations,
        out SoABenchmarkResult outResult);

    // ── Runtime buffers ───────────────────────────────────────────────────────
    private AgentGPU[] _hostAgents;
    private SimParams  _params;
    private bool       _initialized;
    private CUDATimingData _currentTimings;

    public CUDATimingData LastTimings => _currentTimings;
    public bool IsInitialized => _initialized;

    public void Init(List<AgentBase> agents)
    {
        if (agents == null || agents.Count == 0) return;

        int count = agents.Count;
        _hostAgents = new AgentGPU[count];
        _params.frustumPlanes = new Vector4[6];

        try
        {
            _initialized = InitCUDA(count + 128);
            if (!_initialized)
            {
                Debug.LogError("[PBDCudaSolver] Failed to initialize CUDA device memory!");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[PBDCudaSolver] Could not load {DLL_NAME}.dll: {ex.Message}");
            _initialized = false;
        }
    }

    public void Solve(List<AgentBase> agents, float dt, int iterations, Vector3 focusPoint)
    {
        if (!_initialized || agents == null || agents.Count == 0) return;

        int n = Mathf.Min(agents.Count, _hostAgents.Length);

        // 1. Gather host agent state
        for (int i = 0; i < n; i++)
        {
            var pbd = agents[i] as PBDAgent;
            if (pbd == null) continue;

            Vector3 vPref = Vector3.zero;
            if (pbd._agent != null && pbd._agent.hasPath)
            {
                Vector3 dir = pbd._agent.steeringTarget - pbd.transform.position;
                dir.y = 0f;
                if (dir.sqrMagnitude > 1e-4f)
                    vPref = dir.normalized * pbd.prefSpeed;
            }

            _hostAgents[i] = new AgentGPU
            {
                position          = pbd.transform.position,
                predictedPosition = pbd.predictedPosition,
                velocity          = pbd.velocity,
                preferredVelocity = vPref,
                ksi               = pbd.ksi,
                radius            = pbd.radius,
                mass              = pbd.mass,
                lodLevel          = pbd.fidelityLevel,
                importanceScore   = pbd.importanceScore,
                lastContactCount  = _hostAgents[i].lastContactCount
            };
        }

        // 2. Setup parameters
        _params.agentCount             = n;
        _params.deltaTime              = dt;
        _params.cellSize               = cellSize;
        _params.highDensityCutoff      = highDensityCutoff;
        _params.medDensityCutoff       = medDensityCutoff;
        _params.focusPoint             = focusPoint;
        _params.weightDistance         = weightDistance;
        _params.weightVisibility       = weightVisibility;
        _params.weightInteraction      = weightInteraction;
        _params.weightDensity          = weightDensity;
        _params.highImportanceThresh   = highImportanceThreshold;
        _params.medImportanceThresh    = medImportanceThreshold;
        _params.maxImportanceDist      = Mathf.Max(maxImportanceDistance, 1f);
        _params.maxInteractionContacts = Mathf.Max(maxInteractionContacts, 1);
        _params.iterationCount         = Mathf.Max(iterations, 1);
        _params.solveLowThisFrame      = (Time.frameCount % 4 == 0) ? 1 : 0;
        _params.kShortRange            = kShortRange;
        _params.kLongRange             = kLongRange;
        _params.muStatic               = muStatic;
        _params.muKinematic            = muKinematic;
        _params.avgCoefficient         = avgCoefficient;
        _params.lrcLookAhead           = lrcLookAhead;
        _params.lodAblationMode        = lodAblationMode;

        Camera cam = activeCamera != null ? activeCamera : Camera.main;
        if (cam != null)
        {
            Plane[] planes = GeometryUtility.CalculateFrustumPlanes(cam);
            for (int p = 0; p < 6; p++)
                _params.frustumPlanes[p] = new Vector4(planes[p].normal.x, planes[p].normal.y, planes[p].normal.z, planes[p].distance);
            _params.hasFrustumPlanes = 1;
        }
        else
        {
            _params.hasFrustumPlanes = 0;
        }

        // 3. Dispatch CUDA solver
        CudaSolveMarker.Begin();
        bool ok = StepCUDA(_hostAgents, n, ref _params, iterations);
        CudaSolveMarker.End();

        if (!ok) return;

        // Fetch detailed hardware event timings
        if (GetLastTimings(out _currentTimings))
        {
            ProfilerManager.Instance?.AddSampleMs("CUDA.Total", _currentTimings.totalMs);
            ProfilerManager.Instance?.AddSampleMs("CUDA.KernelSolve", _currentTimings.solveMs);
            ProfilerManager.Instance?.AddSampleMs("CUDA.H2D", _currentTimings.h2dMs);
            ProfilerManager.Instance?.AddSampleMs("CUDA.D2H", _currentTimings.d2hMs);
        }

        // 4. Update agent transforms and properties
        for (int i = 0; i < n; i++)
        {
            AgentGPU g = _hostAgents[i];
            var pbd = agents[i] as PBDAgent;
            if (pbd == null) continue;

            pbd.velocity          = g.velocity;
            pbd.predictedPosition = g.predictedPosition;
            pbd.importanceScore   = g.importanceScore;
            pbd.fidelityLevel     = g.lodLevel;

            pbd.transform.position = g.position;
            if (g.velocity.sqrMagnitude > 0.01f)
            {
                Quaternion targetRot = Quaternion.LookRotation(g.velocity, Vector3.up);
                pbd.transform.rotation = Quaternion.Slerp(
                    pbd.transform.rotation, targetRot,
                    Time.fixedDeltaTime * 10f);
            }

            if (pbd._agent != null && g.lodLevel < 2)
                pbd._agent.nextPosition = g.position;
        }
    }

    // ── Public PAGP Instrumentation Helpers ─────────────────────────────────
    public bool QueryGPUMemory(out GPUMemoryInfo info)
    {
        if (!_initialized)
        {
            info = default;
            return false;
        }
        return GetGPUMemoryInfo(out info);
    }

    public bool QueryHashGridStats(out HashGridStats stats)
    {
        if (!_initialized)
        {
            stats = default;
            return false;
        }
        return GetHashGridStats(out stats);
    }

    public bool BenchmarkBruteForce(float radius, out long candidates, out long neighbors)
    {
        if (!_initialized || _hostAgents == null || _hostAgents.Length == 0)
        {
            candidates = neighbors = 0;
            return false;
        }
        return RunBruteForceNeighborBenchmark(_hostAgents, _hostAgents.Length, radius, out candidates, out neighbors);
    }

    public bool QueryConstraintViolations(out ConstraintStats stats)
    {
        if (!_initialized || _hostAgents == null || _hostAgents.Length == 0)
        {
            stats = default;
            return false;
        }
        return MeasureConstraintViolations(_hostAgents, _hostAgents.Length, ref _params, out stats);
    }

    public bool BenchmarkAoSvsSoA(int iterations, out SoABenchmarkResult result)
    {
        if (!_initialized || _hostAgents == null || _hostAgents.Length == 0)
        {
            result = default;
            return false;
        }
        return RunAoSvsSoABenchmark(_hostAgents, _hostAgents.Length, ref _params, iterations, out result);
    }

    private void OnDisable()
    {
        if (_initialized)
        {
            ShutdownCUDA();
            _initialized = false;
        }
    }

    private void OnDestroy()
    {
        if (_initialized)
        {
            ShutdownCUDA();
            _initialized = false;
        }
    }
}
