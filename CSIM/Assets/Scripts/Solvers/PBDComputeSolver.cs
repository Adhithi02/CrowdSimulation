// ============================================================================
//  PBDComputeSolver.cs
//  Manages GPU-side PBD computation via Unity's ComputeShader API.
//
//  Architecture
//  ─────────────
//  • CPU uploads per-frame state (preferred velocities, positions) once via
//    ComputeBuffer.SetData — the only CPU→GPU transfer per frame.
//  • Seven compute kernels run fully on the GPU:
//      Predict → Clear → Hash → AssignLOD → (Solve → Apply) × N → Finalize
//  • AsyncGPUReadback returns results to the main thread 1-2 frames later,
//    then syncs NavMeshAgent.nextPosition and Transforms.
//  • A version counter guards against stale readback callbacks after
//    Respawn() replaces the agent list.
//
//  Heatmap LOD system
//  ───────────────────
//  After KernelHashAgents, _HashGridCount[bucket] holds the number of agents
//  in each spatial cell.  KernelAssignLOD reads this value for each agent's
//  own cell and assigns a physics LOD tier:
//
//    density > HighDensityCutoff (12)  →  HIGH: 3×3 search, ≤12 contacts
//    density > MedDensityCutoff  ( 4)  →  MED:  centre cell, ≤4 contacts
//    density ≤ MedDensityCutoff        →  LOW:  centre cell, ≤1 contact
//
//  No camera distance, no hysteresis, no frame-offset skipping.
//  LOD is recalculated from live density data every frame.
//
//  Requires Unity 2022.3+ (AsyncGPUReadback, ComputeShader).
// ============================================================================

using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

public class PBDComputeSolver : MonoBehaviour
{
    // ── Inspector ─────────────────────────────────────────────────────────────

    [Header("Shader")]
    [Tooltip("Assign the PBDComputeShader.compute asset here.")]
    public ComputeShader computeShader;

    [Header("Spatial Grid")]
    [Tooltip("World-space side length of one hash cell. Should roughly match agent diameter × 2.")]
    public float cellSize = 4f;

    [Header("Density LOD Thresholds")]
    [Tooltip("Agents per cell above this → HIGH LOD (3×3 search, ≤12 contacts).")]
    public int highDensityCutoff = 12;
    [Tooltip("Agents per cell above this → MED LOD (centre cell, ≤4 contacts). " +
             "At or below → LOW LOD (centre cell, ≤1 contact).")]
    public int medDensityCutoff  = 4;

    [Header("Collision Physics")]
    [Range(0f, 1f)] public float kShortRange    = 1.0f;
    [Range(0f, 1f)] public float kLongRange     = 1.0f;
    [Range(0f, 1f)] public float muStatic       = 0.21f;
    [Range(0f, 1f)] public float muKinematic    = 0.10f;
    [Range(0f, 2f)] public float avgCoefficient = 0.25f;

    [Header("Long-Range Collision")]
    [Tooltip("Anticipatory look-ahead in seconds. 0 = disabled. Applied to HIGH-tier agents only.")]
    [Range(0f, 5f)]
    public float lrcLookAhead = 2f;

    // ── Constants — must mirror #define values in PBDComputeShader.compute ──

    private const int HASH_BUCKET_COUNT  = 8192;
    private const int MAX_PER_BUCKET     = 16;
    private const int THREAD_GROUP_SIZE  = 64;

    // Stride sizes must match HLSL struct layouts exactly.
    // AgentGPU:      5 × 16 = 80 bytes
    // CorrectionData: float3 + int = 16 bytes
    private const int AGENT_STRIDE      = 80;
    private const int CORRECTION_STRIDE = 16;

    [Header("Importance Scoring (Synced by SimulationManager)")]
    public Camera activeCamera;
    public float weightDistance = 0.30f;
    public float weightVisibility = 0.25f;
    public float weightInteraction = 0.20f;
    public float weightDensity = 0.25f;
    public float highImportanceThreshold = 0.60f;
    public float medImportanceThreshold = 0.30f;
    public float maxImportanceDistance = 60f;
    public int maxInteractionContacts = 6;

    // ── GPU-side struct mirroring HLSL AgentGPU ───────────────────────────────

    [StructLayout(LayoutKind.Sequential)]
    private struct AgentGPU
    {
        public Vector3 position;
        public float   ksi;
        public Vector3 predictedPosition;
        public float   radius;
        public Vector3 velocity;
        public float   mass;
        public Vector3 preferredVelocity;
        public int     lodLevel;          // 0=HIGH  1=MED  2=LOW
        public float   importanceScore;   // [0, 1]
        public int     lastContactCount;
        public Vector2 _padding;
    }

    // ── Kernel IDs ────────────────────────────────────────────────────────────

    private int  _kPredict, _kClear, _kHash, _kAssignLOD, _kSolve, _kApply, _kFinalize;
    private bool _kernelsReady;

    // ── GPU buffers ───────────────────────────────────────────────────────────

    private ComputeBuffer _agentsBuffer;
    private ComputeBuffer _correctionsBuffer;
    private ComputeBuffer _hashGridBuffer;
    private ComputeBuffer _hashGridCountBuffer;   // also serves as density map

    // ── CPU shadow arrays ─────────────────────────────────────────────────────

    private AgentGPU[]      _agentUpload;   // written each frame, sent to GPU
    private AgentGPU[]      _agentReadback; // received from AsyncGPUReadback
    private List<AgentBase> _agents;
    private readonly Vector4[] _frustumPlaneVectors = new Vector4[6];

    // ── Async readback guard ──────────────────────────────────────────────────

    private bool _readbackPending;
    private int  _version;   // incremented on Init; guards stale callbacks

    // ── Cached dispatch counts ────────────────────────────────────────────────

    private int _agentGroups;
    private int _clearGroups;

    private bool _initialized;

    // =========================================================================
    //  Public API
    // =========================================================================

    /// <summary>
    /// Allocates GPU buffers for the supplied agent list.
    /// Call from SimulationManager.Respawn().
    /// </summary>
    public void Init(List<AgentBase> agents)
    {
        DisposeBuffers();
        _version++;
        _agents = agents;

        int n = agents.Count;
        if (n == 0) return;

        _agentUpload   = new AgentGPU[n];
        _agentReadback = new AgentGPU[n];

        _agentsBuffer        = new ComputeBuffer(n, AGENT_STRIDE);
        _correctionsBuffer   = new ComputeBuffer(n, CORRECTION_STRIDE);
        _hashGridBuffer      = new ComputeBuffer(HASH_BUCKET_COUNT * MAX_PER_BUCKET, 4);
        _hashGridCountBuffer = new ComputeBuffer(HASH_BUCKET_COUNT, 4);

        CacheKernels();
        BindBuffers();

        _agentGroups = (n + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        // Clear kernel covers the larger of the two hash buffers
        _clearGroups = (HASH_BUCKET_COUNT * MAX_PER_BUCKET + THREAD_GROUP_SIZE - 1)
                       / THREAD_GROUP_SIZE;

        _initialized = true;
    }

    /// <summary>
    /// Main solve — call from SimulationManager.FixedUpdate.
    /// focusPoint is accepted for API compatibility with other solvers but
    /// is not used; LOD derives entirely from per-cell density.
    /// </summary>
    public void Solve(List<AgentBase> agents, float dt, int iterations, Vector3 focusPoint)
    {
        if (computeShader == null) return;
        int n = agents.Count;
        if (n == 0 || dt <= 0f) return;

        if (!_initialized || _agentUpload == null || _agentUpload.Length != n)
            Init(agents);

        // 1 ── Upload per-frame CPU data ──────────────────────────────────────
        GatherAgentData(agents);
        _agentsBuffer.SetData(_agentUpload);

        // 2 ── Per-frame uniforms ──────────────────────────────────────────────
        SetUniforms(n, dt, iterations, focusPoint);

        // 3 ── Predict (velocity blend + position integration) ────────────────
        computeShader.Dispatch(_kPredict, _agentGroups, 1, 1);

        // 4 ── Build density heatmap (Clear → Hash → AssignLOD) ───────────────
        //  These three kernels form the heatmap pipeline.  They run once per
        //  frame (not per iteration) — density is stable across Jacobi passes.
        computeShader.Dispatch(_kClear,     _clearGroups,  1, 1);
        computeShader.Dispatch(_kHash,      _agentGroups,  1, 1);
        computeShader.Dispatch(_kAssignLOD, _agentGroups,  1, 1);

        // 5 ── Constraint iterations (Solve → Apply) ──────────────────────────
        //  The hash grid is already built; we reuse it every iteration.
        for (int iter = 0; iter < iterations; iter++)
        {
            computeShader.SetInt("_CurrentIteration", iter);
            computeShader.Dispatch(_kSolve, _agentGroups, 1, 1);
            computeShader.Dispatch(_kApply, _agentGroups, 1, 1);
        }

        // 6 ── Finalize (commit velocity + position) ──────────────────────────
        computeShader.Dispatch(_kFinalize, _agentGroups, 1, 1);

        // 7 ── Async readback (1-2 frame lag — acceptable for NavMesh/Transforms)
        if (!_readbackPending)
        {
            _readbackPending = true;
            int capturedVersion = _version;
            AsyncGPUReadback.Request(_agentsBuffer, req =>
                OnReadbackComplete(req, capturedVersion));
        }
    }

    // =========================================================================
    //  Private helpers
    // =========================================================================

    private void GatherAgentData(List<AgentBase> agents)
    {
        for (int i = 0; i < agents.Count; i++)
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

            int prevContacts = (_agentReadback != null && i < _agentReadback.Length)
                ? _agentReadback[i].lastContactCount
                : 0;

            _agentUpload[i] = new AgentGPU
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
                lastContactCount  = prevContacts
            };
        }
    }

    private void SetUniforms(int agentCount, float dt, int iterations, Vector3 focusPoint)
    {
        computeShader.SetInt  ("_AgentCount",         agentCount);
        computeShader.SetFloat("_DeltaTime",          dt);
        computeShader.SetFloat("_CellSize",           cellSize);

        // Density LOD thresholds
        computeShader.SetInt  ("_HighDensityCutoff",  highDensityCutoff);
        computeShader.SetInt  ("_MedDensityCutoff",   medDensityCutoff);

        // Importance-driven dynamic fidelity uniforms
        computeShader.SetVector("_FocusPoint",            focusPoint);
        computeShader.SetFloat ("_WeightDistance",        weightDistance);
        computeShader.SetFloat ("_WeightVisibility",      weightVisibility);
        computeShader.SetFloat ("_WeightInteraction",     weightInteraction);
        computeShader.SetFloat ("_WeightDensity",         weightDensity);
        computeShader.SetFloat ("_HighImportanceThresh",  highImportanceThreshold);
        computeShader.SetFloat ("_MedImportanceThresh",   medImportanceThreshold);
        computeShader.SetFloat ("_MaxImportanceDist",     Mathf.Max(maxImportanceDistance, 1f));
        computeShader.SetInt   ("_MaxInteractionContacts", Mathf.Max(maxInteractionContacts, 1));
        computeShader.SetInt   ("_IterationCount",        Mathf.Max(iterations, 1));
        computeShader.SetInt   ("_SolveLowThisFrame",     (Time.frameCount % 4 == 0) ? 1 : 0);

        Camera cam = activeCamera != null ? activeCamera : Camera.main;
        if (cam != null)
        {
            Plane[] planes = GeometryUtility.CalculateFrustumPlanes(cam);
            for (int p = 0; p < 6; p++)
                _frustumPlaneVectors[p] = new Vector4(planes[p].normal.x, planes[p].normal.y, planes[p].normal.z, planes[p].distance);
            computeShader.SetInt("_HasFrustumPlanes", 1);
            computeShader.SetVectorArray("_FrustumPlanes", _frustumPlaneVectors);
        }
        else
        {
            computeShader.SetInt("_HasFrustumPlanes", 0);
        }

        // Collision parameters
        computeShader.SetFloat("_KShortRange",        kShortRange);
        computeShader.SetFloat("_KLongRange",         kLongRange);
        computeShader.SetFloat("_MuStatic",           muStatic);
        computeShader.SetFloat("_MuKinematic",        muKinematic);
        computeShader.SetFloat("_AvgCoefficient",     avgCoefficient);
        computeShader.SetFloat("_LRCLookAhead",       lrcLookAhead);
    }

    private void OnReadbackComplete(AsyncGPUReadbackRequest req, int capturedVersion)
    {
        _readbackPending = false;

        if (req.hasError || _agents == null || _version != capturedVersion) return;

        var data = req.GetData<AgentGPU>();
        int n    = Mathf.Min(data.Length, _agents.Count);

        for (int i = 0; i < n; i++)
        {
            AgentGPU g   = data[i];
            var      pbd = _agents[i] as PBDAgent;
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

        for (int i = 0; i < n; i++)
            _agentReadback[i] = data[i];
    }

    // ── Kernel caching ────────────────────────────────────────────────────────

    private void CacheKernels()
    {
        if (_kernelsReady || computeShader == null) return;
        _kPredict   = computeShader.FindKernel("KernelPredict");
        _kClear     = computeShader.FindKernel("KernelClearHashGrid");
        _kHash      = computeShader.FindKernel("KernelHashAgents");
        _kAssignLOD = computeShader.FindKernel("KernelAssignLOD");
        _kSolve     = computeShader.FindKernel("KernelSolveConstraints");
        _kApply     = computeShader.FindKernel("KernelApplyCorrections");
        _kFinalize  = computeShader.FindKernel("KernelFinalize");
        _kernelsReady = true;
    }

    private void BindBuffers()
    {
        int[] kernels = { _kPredict, _kClear, _kHash, _kAssignLOD,
                          _kSolve, _kApply, _kFinalize };
        foreach (int k in kernels)
        {
            computeShader.SetBuffer(k, "_Agents",          _agentsBuffer);
            computeShader.SetBuffer(k, "_Corrections",     _correctionsBuffer);
            computeShader.SetBuffer(k, "_HashGrid",        _hashGridBuffer);
            computeShader.SetBuffer(k, "_HashGridCount",   _hashGridCountBuffer);
        }
    }

    // ── Buffer lifecycle ──────────────────────────────────────────────────────

    private void DisposeBuffers()
    {
        _agentsBuffer?.Release();
        _correctionsBuffer?.Release();
        _hashGridBuffer?.Release();
        _hashGridCountBuffer?.Release();
        _agentsBuffer = _correctionsBuffer = _hashGridBuffer = _hashGridCountBuffer = null;
        _initialized     = false;
        _readbackPending = false;
        _kernelsReady    = false;
    }

    void OnDisable() => DisposeBuffers();
    void OnDestroy() => DisposeBuffers();
}
