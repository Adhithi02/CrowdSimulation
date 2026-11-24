using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Jobs;

// ---------------------------------------------------------------------------
// Agent data blob — lives entirely inside NativeArrays for Jobs/Burst.
// lastLodLevel persists between frames for hysteresis.
// ---------------------------------------------------------------------------
public struct CLODAgentData
{
    public float3 position;
    public float3 predictedPosition;
    public float3 velocity;
    public float3 preferredVelocity;
    public float  ksi;
    public float  radius;
    public float  mass;
    public float  importanceScore; // [0, 1] computed from distance, visibility, interaction, density
    public int    lastLodLevel;    // 0=HIGH  1=MED  2=LOW  — kept across frames
}

// ---------------------------------------------------------------------------
// CLOD PBD Solver
//
// LOD tiers:
//   HIGH  — full 3×3 neighbourhood, up to 12 neighbours, every iteration
//   MED   — centre cell only,       up to 4  neighbours, every iteration
//   LOW   — centre cell only,       1        neighbour,  every 4th frame
//
// All agents are hashed into the spatial map each iteration, so HIGH/MED
// agents can still push away from LOW LOD agents even on frames where LOW
// doesn't resolve collisions itself.
// ---------------------------------------------------------------------------
public class CLODPBDSolverJobs : MonoBehaviour
{
    private NativeArray<CLODAgentData>              _agentDataArray;
    private NativeArray<float3>                     _corrections;
    private NativeArray<int>                        _correctionCounts;
    private NativeArray<float4>                     _frustumPlanes;
    private TransformAccessArray                    _transformArray;
    private NativeParallelMultiHashMap<int, int>    _spatialHashMap;

    private NativeList<int> _highIndices;
    private NativeList<int> _medIndices;
    private NativeList<int> _lowIndices;

    private bool _initialized;

    // Inspector-friendly fields (written by SimulationManager)
    public float gridCellSize    = 4f;
    public float highLodDistance = 20f;
    public float medLodDistance  = 50f;

    // Camera data for frustum-aware importance (set by SimulationManager each frame)
    public Camera activeCamera;
    public float3 cameraPosition = new float3(0, 10, 0);
    public float3 cameraForward  = new float3(0, -1, 0); // default: top-down

    // Importance scoring weights & thresholds (synced by SimulationManager)
    public float weightDistance          = 0.30f;
    public float weightVisibility        = 0.25f;
    public float weightInteraction       = 0.20f;
    public float weightDensity           = 0.25f;
    public float highImportanceThreshold = 0.60f;
    public float medImportanceThreshold  = 0.30f;
    public float importanceHysteresis    = 0.05f;
    public float maxImportanceDistance   = 60f;
    public int   maxInteractionContacts  = 6;
    public int   maxImportanceDensity    = 12;

    // Adaptive LOD: scale importance thresholds when FPS drops below target
    public bool  adaptiveLOD = true;
    public float targetFPS   = 60f;

    // -----------------------------------------------------------------------
    // Init / Dispose
    // -----------------------------------------------------------------------

    public void Init(List<AgentBase> agents)
    {
        if (_initialized) DisposeData();
        int count = agents.Count;

        _agentDataArray  = new NativeArray<CLODAgentData>(count, Allocator.Persistent);
        _corrections     = new NativeArray<float3>(count,        Allocator.Persistent);
        _correctionCounts= new NativeArray<int>(count,           Allocator.Persistent);
        _frustumPlanes   = new NativeArray<float4>(6,            Allocator.Persistent);

        // 2× over-provision to keep fill-factor ≤ 50 % and reduce hash collisions.
        // All three tiers are hashed now, so worst case is count entries total.
        _spatialHashMap  = new NativeParallelMultiHashMap<int, int>(count * 2, Allocator.Persistent);

        _highIndices = new NativeList<int>(count, Allocator.Persistent);
        _medIndices  = new NativeList<int>(count, Allocator.Persistent);
        _lowIndices  = new NativeList<int>(count, Allocator.Persistent);

        var transforms = new Transform[count];
        for (int i = 0; i < count; i++) transforms[i] = agents[i].transform;
        _transformArray = new TransformAccessArray(transforms);

        _initialized = true;
    }

    private bool RefreshFrustumPlanes()
    {
        Camera cam = activeCamera != null ? activeCamera : Camera.main;
        if (cam == null) return false;
        Plane[] planes = GeometryUtility.CalculateFrustumPlanes(cam);
        for (int p = 0; p < 6; p++)
            _frustumPlanes[p] = new float4(planes[p].normal.x, planes[p].normal.y, planes[p].normal.z, planes[p].distance);
        return true;
    }

    // -----------------------------------------------------------------------
    // Main solve — called by SimulationManager.FixedUpdate
    // -----------------------------------------------------------------------

    public void Solve(List<AgentBase> agents, float dt, int iterations, float3 focusPoint)
    {
        int count = agents.Count;
        if (count == 0 || dt <= 0f) return;
        if (!_initialized || _agentDataArray.Length != count) Init(agents);

        bool hasFrustum = RefreshFrustumPlanes();

        // ── 1. Populate NativeArray from main thread ────────────────────────
        for (int i = 0; i < count; i++)
        {
            var pbd   = agents[i] as PBDAgent;
            var vPref = float3.zero;
            if (pbd._agent != null && pbd._agent.hasPath)
                vPref = math.normalizesafe(
                    (float3)pbd._agent.steeringTarget - (float3)pbd.transform.position)
                    * pbd.prefSpeed;

            // Preserve lastLodLevel & importanceScore from the previous frame for hysteresis.
            int prevLod = _agentDataArray[i].lastLodLevel;
            float prevImp = _agentDataArray[i].importanceScore;

            _agentDataArray[i] = new CLODAgentData
            {
                position          = pbd.transform.position,
                predictedPosition = pbd.predictedPosition,
                velocity          = pbd.velocity,
                preferredVelocity = vPref,
                ksi               = pbd.ksi,
                radius            = pbd.radius,
                mass              = pbd.mass,
                importanceScore   = prevImp,
                lastLodLevel      = prevLod
            };
        }

        // ── 2. Predict step — all agents ────────────────────────────────────
        JobHandle handle = new CLODPredictJob { Agents = _agentDataArray, dt = dt }
            .Schedule(count, 64);

        // ── 3. Build spatial hashmap ONCE per frame before scoring ──────────
        // All agents are hashed so CategorizeLODJob can read live local crowd density.
        handle = new CLODClearHashMapJob { HashMap = _spatialHashMap }.Schedule(handle);
        handle = new CLODHashAllPositionsJob
        {
            Agents   = _agentDataArray,
            HashMap  = _spatialHashMap.AsParallelWriter(),
            CellSize = gridCellSize
        }.Schedule(count, 64, handle);

        // ── 4. Compute importance score & assign fidelity tiers ─────────────
        // Adaptive: when FPS drops, raise thresholds slightly so more agents move to MED/LOW.
        float fps   = 1f / math.max(Time.unscaledDeltaTime, 1e-4f);
        float scale = adaptiveLOD ? math.clamp(targetFPS / math.max(fps, 1f), 1.0f, 1.35f) : 1.0f;
        float adaptedHigh = math.min(0.95f, highImportanceThreshold * scale);
        float adaptedMed  = math.min(adaptedHigh - 0.05f, medImportanceThreshold * scale);

        handle = new CategorizeLODJob
        {
            Agents                 = _agentDataArray,
            FocusPoint             = focusPoint,
            CameraPos              = cameraPosition,
            CameraForward          = math.normalizesafe(cameraForward),
            HasFrustumPlanes       = hasFrustum ? 1 : 0,
            FrustumPlanes          = _frustumPlanes,
            HashMap                = _spatialHashMap,
            CellSize               = gridCellSize,
            PrevCorrectionCounts   = _correctionCounts,
            WeightDistance         = weightDistance,
            WeightVisibility       = weightVisibility,
            WeightInteraction      = weightInteraction,
            WeightDensity          = weightDensity,
            HighImportanceThresh   = adaptedHigh,
            MedImportanceThresh    = adaptedMed,
            Hysteresis             = importanceHysteresis,
            MaxImportanceDist      = math.max(maxImportanceDistance, 1f),
            MaxInteractionContacts = math.max(maxInteractionContacts, 1),
            MaxImportanceDensity   = math.max(maxImportanceDensity, 1),
            HighIndices            = _highIndices,
            MedIndices             = _medIndices,
            LowIndices             = _lowIndices
        }.Schedule(handle);

        // LOW LOD resolves every 4th frame (cheap, minimal visual impact far away)
        bool resolveLowThisFrame = Time.frameCount % 4 == 0;
        int medIterationCap = math.max(1, (iterations + 1) / 2);

        // ── 5. Constraint iterations (allocated by fidelity tier) ───────────
        for (int iter = 0; iter < iterations; iter++)
        {
            // HIGH: 3×3 grid, ≤12 neighbours, every iteration
            handle = new HighDetailConstraintJob
            {
                Indices            = _highIndices.AsDeferredJobArray(),
                AgentsIn           = _agentDataArray,
                HashMap            = _spatialHashMap,
                PositionCorrections= _corrections,
                CorrectionCounts   = _correctionCounts,
                CellSize           = gridCellSize
            }.Schedule(_highIndices, 64, handle);

            // MED: centre cell, ≤4 neighbours, half-budget iterations
            if (iter < medIterationCap)
            {
                handle = new MedDetailConstraintJob
                {
                    Indices            = _medIndices.AsDeferredJobArray(),
                    AgentsIn           = _agentDataArray,
                    HashMap            = _spatialHashMap,
                    PositionCorrections= _corrections,
                    CorrectionCounts   = _correctionCounts,
                    CellSize           = gridCellSize
                }.Schedule(_medIndices, 64, handle);
            }

            // LOW: centre cell, ≤1 neighbour, first iteration only, every 4th frame
            if (iter == 0 && resolveLowThisFrame)
            {
                handle = new LowDetailConstraintJob
                {
                    Indices            = _lowIndices.AsDeferredJobArray(),
                    AgentsIn           = _agentDataArray,
                    HashMap            = _spatialHashMap,
                    PositionCorrections= _corrections,
                    CorrectionCounts   = _correctionCounts,
                    CellSize           = gridCellSize
                }.Schedule(_lowIndices, 64, handle);
            }

            // ── Apply corrections ───────────────────────────────────────────

            handle = new CLODApplyConstraintsJob
            {
                Indices             = _highIndices.AsDeferredJobArray(),
                Agents              = _agentDataArray,
                PositionCorrections = _corrections,
                CorrectionCounts    = _correctionCounts
            }.Schedule(_highIndices, 64, handle);

            if (iter < medIterationCap)
            {
                handle = new CLODApplyConstraintsJob
                {
                    Indices             = _medIndices.AsDeferredJobArray(),
                    Agents              = _agentDataArray,
                    PositionCorrections = _corrections,
                    CorrectionCounts    = _correctionCounts
                }.Schedule(_medIndices, 64, handle);
            }

            if (iter == 0 && resolveLowThisFrame)
            {
                handle = new CLODApplyConstraintsJob
                {
                    Indices             = _lowIndices.AsDeferredJobArray(),
                    Agents              = _agentDataArray,
                    PositionCorrections = _corrections,
                    CorrectionCounts    = _correctionCounts
                }.Schedule(_lowIndices, 64, handle);
            }
        }

        // ── 6. Finalise velocity + position, write to transforms — all agents ─
        handle = new CLODUpdateJob    { Agents = _agentDataArray, dt = dt }.Schedule(count, 64, handle);
        handle = new CLODApplyTransformJob { Agents = _agentDataArray, dt = dt }.Schedule(_transformArray, handle);
        handle.Complete();

        // ── 7. Sync back to main thread (NavMesh, C# velocity & importance) ─
        // Skip NavMesh nextPosition for LOW LOD agents — they're less relevant,
        // a frame-stale navmesh position is imperceptible and saves N writes/frame.
        for (int i = 0; i < count; i++)
        {
            var pbd  = agents[i] as PBDAgent;
            var data = _agentDataArray[i];
            pbd.velocity          = data.velocity;
            pbd.predictedPosition = data.predictedPosition;
            pbd.importanceScore   = data.importanceScore;
            pbd.fidelityLevel     = data.lastLodLevel;
            if (pbd._agent != null && data.lastLodLevel < 2)   // skip LOW
                pbd._agent.nextPosition = data.position;
        }
    }

    public void DisposeData()
    {
        if (_agentDataArray.IsCreated)  _agentDataArray.Dispose();
        if (_corrections.IsCreated)     _corrections.Dispose();
        if (_correctionCounts.IsCreated)_correctionCounts.Dispose();
        if (_frustumPlanes.IsCreated)   _frustumPlanes.Dispose();
        if (_transformArray.isCreated)  _transformArray.Dispose();
        if (_spatialHashMap.IsCreated)  _spatialHashMap.Dispose();
        if (_highIndices.IsCreated)     _highIndices.Dispose();
        if (_medIndices.IsCreated)      _medIndices.Dispose();
        if (_lowIndices.IsCreated)      _lowIndices.Dispose();
        _initialized = false;
    }

    void OnDisable()
    {
        DisposeData();
    }
}

// =============================================================================
// BURST JOBS
// =============================================================================

// ---------------------------------------------------------------------------
// Importance-driven LOD categorisation — runs once per frame.
// Computes normalized importance score [0, 1] from:
//   1. Distance    (proximity to focus point / camera)
//   2. Visibility  (camera frustum planes or forward half-space)
//   3. Interaction (active collision contacts from previous frame)
//   4. Density     (live local agent count in spatial hash cell)
// Assigns fidelity tier (0=HIGH, 1=MED, 2=LOW) with hysteresis.
// ---------------------------------------------------------------------------
[BurstCompile]
public struct CategorizeLODJob : IJob
{
    public  NativeArray<CLODAgentData> Agents;
    public  float3 FocusPoint;
    public  float3 CameraPos;
    public  float3 CameraForward;   // must be normalised before scheduling
    public  int    HasFrustumPlanes;
    [ReadOnly] public NativeArray<float4> FrustumPlanes;
    [ReadOnly] public NativeParallelMultiHashMap<int, int> HashMap;
    public  float  CellSize;

    // Interaction from previous frame
    [ReadOnly] public NativeArray<int> PrevCorrectionCounts;

    // Importance weights & thresholds
    public float WeightDistance;
    public float WeightVisibility;
    public float WeightInteraction;
    public float WeightDensity;
    public float HighImportanceThresh;
    public float MedImportanceThresh;
    public float Hysteresis;
    public float MaxImportanceDist;
    public int   MaxInteractionContacts;
    public int   MaxImportanceDensity;

    public NativeList<int> HighIndices;
    public NativeList<int> MedIndices;
    public NativeList<int> LowIndices;

    public void Execute()
    {
        HighIndices.Clear();
        MedIndices.Clear();
        LowIndices.Clear();

        float totalWeight = math.max(WeightDistance + WeightVisibility + WeightInteraction + WeightDensity, 1e-5f);

        for (int i = 0; i < Agents.Length; i++)
        {
            CLODAgentData agent = Agents[i];
            int prev = agent.lastLodLevel;

            // 1. Distance score [0, 1]
            float dist = math.distance(agent.position, FocusPoint);
            float distScore = math.saturate(1f - dist / MaxImportanceDist);

            // 2. Visibility score [0, 1]
            bool visible = true;
            if (HasFrustumPlanes == 1)
            {
                for (int p = 0; p < 6; p++)
                {
                    if (math.dot(FrustumPlanes[p].xyz, agent.position) + FrustumPlanes[p].w < 0f)
                    {
                        visible = false;
                        break;
                    }
                }
            }
            else
            {
                visible = math.dot(agent.position - CameraPos, CameraForward) >= 0f;
            }
            float visScore = visible ? 1f : 0f;

            // 3. Interaction score [0, 1]
            float interScore = math.saturate((float)PrevCorrectionCounts[i] / MaxInteractionContacts);

            // 4. Local crowd density score [0, 1]
            int2 cell = new int2((int)math.floor(agent.predictedPosition.x / CellSize),
                                 (int)math.floor(agent.predictedPosition.z / CellSize));
            int hash = (cell.x * 73856093) ^ (cell.y * 19349663);
            int localDensity = 0;
            if (HashMap.TryGetFirstValue(hash, out int nb, out var it))
            {
                do { localDensity++; }
                while (HashMap.TryGetNextValue(out nb, ref it));
            }
            float densityScore = math.saturate((float)localDensity / MaxImportanceDensity);

            // Combined importance score
            float importance = math.saturate(
                (WeightDistance * distScore +
                 WeightVisibility * visScore +
                 WeightInteraction * interScore +
                 WeightDensity * densityScore) / totalWeight);

            // Dynamic fidelity assignment with hysteresis
            int next;
            if (prev == 0)
            {
                if      (importance >= HighImportanceThresh - Hysteresis) next = 0;
                else if (importance >= MedImportanceThresh  - Hysteresis) next = 1;
                else                                                      next = 2;
            }
            else if (prev == 1)
            {
                if      (importance >= HighImportanceThresh + Hysteresis) next = 0;
                else if (importance >= MedImportanceThresh  - Hysteresis) next = 1;
                else                                                      next = 2;
            }
            else
            {
                if      (importance >= HighImportanceThresh + Hysteresis) next = 0;
                else if (importance >= MedImportanceThresh  + Hysteresis) next = 1;
                else                                                      next = 2;
            }

            agent.importanceScore = importance;
            agent.lastLodLevel = next;
            Agents[i] = agent;

            if      (next == 0) HighIndices.Add(i);
            else if (next == 1) MedIndices.Add(i);
            else                LowIndices.Add(i);
        }
    }
}

// ---------------------------------------------------------------------------
// Predict step — all agents, parallel
// ---------------------------------------------------------------------------
[BurstCompile]
public struct CLODPredictJob : IJobParallelFor
{
    public NativeArray<CLODAgentData> Agents;
    public float dt;

    public void Execute(int index)
    {
        CLODAgentData a = Agents[index];
        a.velocity          = math.lerp(a.velocity, a.preferredVelocity, a.ksi);
        a.predictedPosition = a.position + a.velocity * dt;
        a.predictedPosition.y = a.position.y;   // no vertical drift
        Agents[index] = a;
    }
}

// ---------------------------------------------------------------------------
// Clear the spatial hashmap — now Burst-compiled
// ---------------------------------------------------------------------------
[BurstCompile]
public struct CLODClearHashMapJob : IJob
{
    public NativeParallelMultiHashMap<int, int> HashMap;
    public void Execute() { HashMap.Clear(); }
}

// ---------------------------------------------------------------------------
// Hash all agents in parallel into the spatial map
// ---------------------------------------------------------------------------
[BurstCompile]
public struct CLODHashAllPositionsJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<CLODAgentData> Agents;
    public NativeParallelMultiHashMap<int, int>.ParallelWriter HashMap;
    public float CellSize;

    public void Execute(int idx)
    {
        float3 pos  = Agents[idx].predictedPosition;
        int2   cell = new int2((int)math.floor(pos.x / CellSize),
                               (int)math.floor(pos.z / CellSize));
        int hash = (cell.x * 73856093) ^ (cell.y * 19349663);
        HashMap.Add(hash, idx);
    }
}

// ---------------------------------------------------------------------------
// Hash a subset of agents (by index list) into the spatial map
// ---------------------------------------------------------------------------
[BurstCompile]
public struct CLODHashPositionsJob : IJobParallelForDefer
{
    [ReadOnly] public NativeArray<int>         Indices;
    [ReadOnly] public NativeArray<CLODAgentData> Agents;
    public NativeParallelMultiHashMap<int, int>.ParallelWriter HashMap;
    public float CellSize;

    public void Execute(int loopIndex)
    {
        int    idx  = Indices[loopIndex];
        float3 pos  = Agents[idx].predictedPosition;
        int2   cell = new int2((int)math.floor(pos.x / CellSize),
                               (int)math.floor(pos.z / CellSize));
        int hash = (cell.x * 73856093) ^ (cell.y * 19349663);
        HashMap.Add(hash, idx);
    }
}

// ---------------------------------------------------------------------------
// HIGH LOD constraint — 3×3 neighbourhood, ≤12 neighbours
// ---------------------------------------------------------------------------
[BurstCompile]
public struct HighDetailConstraintJob : IJobParallelForDefer
{
    [ReadOnly] public NativeArray<int>           Indices;
    [ReadOnly] public NativeArray<CLODAgentData> AgentsIn;
    [ReadOnly] public NativeParallelMultiHashMap<int, int> HashMap;
    [NativeDisableParallelForRestriction] public NativeArray<float3> PositionCorrections;
    [NativeDisableParallelForRestriction] public NativeArray<int>    CorrectionCounts;
    public float CellSize;

    public void Execute(int loopIndex)
    {
        int   index  = Indices[loopIndex];
        var   agent  = AgentsIn[index];
        float3 total = float3.zero;
        int   count  = 0;
        int2  center = new int2((int)math.floor(agent.predictedPosition.x / CellSize),
                                (int)math.floor(agent.predictedPosition.z / CellSize));

        for (int i = -1; i <= 1 && count < 12; i++)
        for (int j = -1; j <= 1 && count < 12; j++)
        {
            int hash = ((center.x + i) * 73856093) ^ ((center.y + j) * 19349663);
            if (!HashMap.TryGetFirstValue(hash, out int nb, out var it)) continue;
            do
            {
                if (nb == index || count >= 12) continue;
                total += ComputeCorrection(agent, AgentsIn[nb], ref count);
            }
            while (HashMap.TryGetNextValue(out nb, ref it));
        }

        PositionCorrections[index] = total;
        CorrectionCounts[index]    = count;
    }

    static float3 ComputeCorrection(CLODAgentData a, CLODAgentData b, ref int count)
    {
        float3 diff   = a.predictedPosition - b.predictedPosition;
        diff.y        = 0f;
        float  distSq = math.lengthsq(diff);
        float  margin = (a.radius > 0 ? a.radius : 0.5f) + (b.radius > 0 ? b.radius : 0.5f);
        if (distSq >= margin * margin || distSq < 1e-6f) return float3.zero;

        float  dist  = math.sqrt(distSq);
        float  pen   = margin - dist;
        float  wI    = 1f / math.max(a.mass, 1e-4f);
        float  wJ    = 1f / math.max(b.mass, 1e-4f);
        count++;
        return (pen / (wI + wJ)) * wI * (diff / dist);
    }
}

// ---------------------------------------------------------------------------
// MED LOD constraint — centre cell only, ≤4 neighbours
// (runs every iteration — time-slicing removed to eliminate jitter)
// ---------------------------------------------------------------------------
[BurstCompile]
public struct MedDetailConstraintJob : IJobParallelForDefer
{
    [ReadOnly] public NativeArray<int>           Indices;
    [ReadOnly] public NativeArray<CLODAgentData> AgentsIn;
    [ReadOnly] public NativeParallelMultiHashMap<int, int> HashMap;
    [NativeDisableParallelForRestriction] public NativeArray<float3> PositionCorrections;
    [NativeDisableParallelForRestriction] public NativeArray<int>    CorrectionCounts;
    public float CellSize;

    public void Execute(int loopIndex)
    {
        int   index  = Indices[loopIndex];
        var   agent  = AgentsIn[index];
        float3 total = float3.zero;
        int   count  = 0;
        int2  center = new int2((int)math.floor(agent.predictedPosition.x / CellSize),
                                (int)math.floor(agent.predictedPosition.z / CellSize));

        int hash = (center.x * 73856093) ^ (center.y * 19349663);
        if (HashMap.TryGetFirstValue(hash, out int nb, out var it))
        do
        {
            if (nb == index || count >= 4) continue;
            float3 diff   = agent.predictedPosition - AgentsIn[nb].predictedPosition;
            diff.y        = 0f;
            float  distSq = math.lengthsq(diff);
            float  margin = (agent.radius > 0 ? agent.radius : 0.5f) +
                            (AgentsIn[nb].radius > 0 ? AgentsIn[nb].radius : 0.5f);
            if (distSq < margin * margin && distSq > 1e-6f)
            {
                float dist = math.sqrt(distSq);
                // Simplified mass (assume equal) — faster, acceptable at medium distance
                total += ((margin - dist) * 0.5f) * (diff / dist);
                count++;
            }
        }
        while (HashMap.TryGetNextValue(out nb, ref it));

        PositionCorrections[index] = total;
        CorrectionCounts[index]    = count;
    }
}

// ---------------------------------------------------------------------------
// LOW LOD constraint — centre cell only, ≤1 neighbour, every 4th frame
// Gives low-LOD agents just enough push to prevent complete interpenetration.
// ---------------------------------------------------------------------------
[BurstCompile]
public struct LowDetailConstraintJob : IJobParallelForDefer
{
    [ReadOnly] public NativeArray<int>           Indices;
    [ReadOnly] public NativeArray<CLODAgentData> AgentsIn;
    [ReadOnly] public NativeParallelMultiHashMap<int, int> HashMap;
    [NativeDisableParallelForRestriction] public NativeArray<float3> PositionCorrections;
    [NativeDisableParallelForRestriction] public NativeArray<int>    CorrectionCounts;
    public float CellSize;

    public void Execute(int loopIndex)
    {
        int   index  = Indices[loopIndex];
        var   agent  = AgentsIn[index];
        float3 total = float3.zero;
        int   count  = 0;
        int2  center = new int2((int)math.floor(agent.predictedPosition.x / CellSize),
                                (int)math.floor(agent.predictedPosition.z / CellSize));

        int hash = (center.x * 73856093) ^ (center.y * 19349663);
        if (HashMap.TryGetFirstValue(hash, out int nb, out var it))
        do
        {
            if (nb == index || count >= 1) continue;
            float3 diff   = agent.predictedPosition - AgentsIn[nb].predictedPosition;
            diff.y        = 0f;
            float  distSq = math.lengthsq(diff);
            float  margin = (agent.radius > 0 ? agent.radius : 0.5f) +
                            (AgentsIn[nb].radius > 0 ? AgentsIn[nb].radius : 0.5f);
            if (distSq < margin * margin && distSq > 1e-6f)
            {
                float dist = math.sqrt(distSq);
                total += ((margin - dist) * 0.5f) * (diff / dist);
                count++;
            }
        }
        while (HashMap.TryGetNextValue(out nb, ref it));

        PositionCorrections[index] = total;
        CorrectionCounts[index]    = count;
    }
}

// ---------------------------------------------------------------------------
// Apply corrections — apply the full correction sum (no averaging).
// Averaging was dividing out the effect and causing under-correction / penetration.
// ---------------------------------------------------------------------------
[BurstCompile]
public struct CLODApplyConstraintsJob : IJobParallelForDefer
{
    [ReadOnly] public NativeArray<int> Indices;
    [NativeDisableParallelForRestriction] public NativeArray<CLODAgentData> Agents;
    [ReadOnly] public NativeArray<float3> PositionCorrections;
    [ReadOnly] public NativeArray<int>    CorrectionCounts;

    public void Execute(int loopIndex)
    {
        int index = Indices[loopIndex];
        if (CorrectionCounts[index] <= 0) return;

        CLODAgentData a = Agents[index];
        a.predictedPosition += PositionCorrections[index]; // full sum, not averaged
        Agents[index] = a;
    }
}

// ---------------------------------------------------------------------------
// Finalise velocity and position — all agents
// ---------------------------------------------------------------------------
[BurstCompile]
public struct CLODUpdateJob : IJobParallelFor
{
    public NativeArray<CLODAgentData> Agents;
    public float dt;

    public void Execute(int index)
    {
        CLODAgentData a = Agents[index];
        float3 rawVelocity = (a.predictedPosition - a.position) / math.max(dt, 1e-5f);
        float preferredSpeed = math.length(a.preferredVelocity);
        float maxSpeed = math.max(preferredSpeed * 1.15f, 0.75f);
        float rawSpeed = math.length(rawVelocity);
        if (rawSpeed > maxSpeed)
            rawVelocity *= maxSpeed / rawSpeed;

        if (preferredSpeed > 0.01f)
        {
            float3 prefDir = a.preferredVelocity / preferredSpeed;
            float backward = math.dot(rawVelocity, -prefDir);
            if (backward > 0f)
                rawVelocity += prefDir * backward * 0.85f;
        }

        a.velocity = rawVelocity;
        a.position = a.predictedPosition;
        Agents[index] = a;
    }
}

// ---------------------------------------------------------------------------
// Write final positions / rotations to Unity Transforms — all agents
// ---------------------------------------------------------------------------
[BurstCompile]
public struct CLODApplyTransformJob : IJobParallelForTransform
{
    [ReadOnly] public NativeArray<CLODAgentData> Agents;
    public float dt;

    public void Execute(int index, TransformAccess transform)
    {
        CLODAgentData a = Agents[index];
        transform.position = a.position;
        if (math.lengthsq(a.velocity) > 0.01f)
        {
            quaternion targetRot = quaternion.LookRotationSafe(math.normalize(a.velocity), math.up());
            transform.rotation  = math.slerp(transform.rotation, targetRot, dt * 10f);
        }
    }
}
