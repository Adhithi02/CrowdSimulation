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
    public int    lastLodLevel;  // 0=HIGH  1=MED  2=LOW  — kept across frames
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

    // Camera data for frustum-aware LOD (set by SimulationManager each frame)
    public float3 cameraPosition = new float3(0, 10, 0);
    public float3 cameraForward  = new float3(0, -1, 0); // default: top-down

    // Adaptive LOD: scale distances down when FPS drops below target
    public bool  adaptiveLOD = true;
    public float targetFPS   = 60f;

    // 10 % hysteresis band — prevents agents flickering at tier boundaries
    private const float HYSTERESIS       = 0.10f;
    // Corrections/frame needed to promote an agent one tier (density promotion)
    private const int   DENSITY_THRESHOLD = 5;

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

    // -----------------------------------------------------------------------
    // Main solve — called by SimulationManager.FixedUpdate
    // -----------------------------------------------------------------------

    public void Solve(List<AgentBase> agents, float dt, int iterations, float3 focusPoint)
    {
        int count = agents.Count;
        if (count == 0 || dt <= 0f) return;
        if (!_initialized || _agentDataArray.Length != count) Init(agents);

        // ── 1. Populate NativeArray from main thread ────────────────────────
        for (int i = 0; i < count; i++)
        {
            var pbd   = agents[i] as PBDAgent;
            var vPref = float3.zero;
            if (pbd._agent != null && pbd._agent.hasPath)
                vPref = math.normalizesafe(
                    (float3)pbd._agent.steeringTarget - (float3)pbd.transform.position)
                    * pbd.prefSpeed;

            // Preserve lastLodLevel from the previous frame for hysteresis.
            int prevLod = _agentDataArray[i].lastLodLevel;

            _agentDataArray[i] = new CLODAgentData
            {
                position          = pbd.transform.position,
                predictedPosition = pbd.predictedPosition,
                velocity          = pbd.velocity,
                preferredVelocity = vPref,
                ksi               = pbd.ksi,
                radius            = pbd.radius,
                mass              = pbd.mass,
                lastLodLevel      = prevLod
            };
        }

        // ── 2. Categorise agents into LOD tiers ─────────────────────────────
        // Adaptive: when FPS drops, shrink thresholds so more agents go to lower tiers.
        float fps   = 1f / math.max(Time.unscaledDeltaTime, 1e-4f);
        float scale = adaptiveLOD ? math.clamp(fps / targetFPS, 0.4f, 1.0f) : 1.0f;
        float adaptedHigh = highLodDistance * scale;
        float adaptedMed  = medLodDistance  * scale;

        JobHandle handle = new CategorizeLODJob
        {
            Agents               = _agentDataArray,
            FocusPoint           = focusPoint,
            HighLodDistSq        = adaptedHigh * adaptedHigh,
            MedLodDistSq         = adaptedMed  * adaptedMed,
            Hysteresis           = HYSTERESIS,
            CameraPos            = cameraPosition,
            CameraForward        = math.normalizesafe(cameraForward),
            PrevCorrectionCounts = _correctionCounts,   // values from last frame
            DensityThreshold     = DENSITY_THRESHOLD,
            HighIndices          = _highIndices,
            MedIndices           = _medIndices,
            LowIndices           = _lowIndices
        }.Schedule();

        // ── 3. Predict step — all agents ────────────────────────────────────
        handle = new CLODPredictJob { Agents = _agentDataArray, dt = dt }
            .Schedule(count, 64, handle);

        // LOW LOD resolves every 4th frame (cheap, minimal visual impact far away)
        bool resolveLowThisFrame = Time.frameCount % 4 == 0;

        // ── 4. Build spatial hashmap ONCE per frame (not per iteration) ──────
        // Predicted positions change only slightly between iterations, so
        // rebuilding once gives ~same quality at 1/iterations the hashing cost.
        handle = new CLODClearHashMapJob { HashMap = _spatialHashMap }.Schedule(handle);

        handle = new CLODHashPositionsJob
        {
            Indices  = _highIndices.AsDeferredJobArray(),
            Agents   = _agentDataArray,
            HashMap  = _spatialHashMap.AsParallelWriter(),
            CellSize = gridCellSize
        }.Schedule(_highIndices, 64, handle);

        handle = new CLODHashPositionsJob
        {
            Indices  = _medIndices.AsDeferredJobArray(),
            Agents   = _agentDataArray,
            HashMap  = _spatialHashMap.AsParallelWriter(),
            CellSize = gridCellSize
        }.Schedule(_medIndices, 64, handle);

        handle = new CLODHashPositionsJob
        {
            Indices  = _lowIndices.AsDeferredJobArray(),
            Agents   = _agentDataArray,
            HashMap  = _spatialHashMap.AsParallelWriter(),
            CellSize = gridCellSize
        }.Schedule(_lowIndices, 64, handle);

        // ── 5. Constraint iterations ─────────────────────────────────────────
        for (int iter = 0; iter < iterations; iter++)
        {

            // ── Constraint solve ────────────────────────────────────────────

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

            // MED: centre cell, ≤4 neighbours, every iteration (no time-slicing)
            handle = new MedDetailConstraintJob
            {
                Indices            = _medIndices.AsDeferredJobArray(),
                AgentsIn           = _agentDataArray,
                HashMap            = _spatialHashMap,
                PositionCorrections= _corrections,
                CorrectionCounts   = _correctionCounts,
                CellSize           = gridCellSize
            }.Schedule(_medIndices, 64, handle);

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

            handle = new CLODApplyConstraintsJob
            {
                Indices             = _medIndices.AsDeferredJobArray(),
                Agents              = _agentDataArray,
                PositionCorrections = _corrections,
                CorrectionCounts    = _correctionCounts
            }.Schedule(_medIndices, 64, handle);

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

        // ── 7. Sync back to main thread (NavMesh, C# velocity cache) ────────
        // Skip NavMesh nextPosition for LOW LOD agents — they're far from camera,
        // a frame-stale navmesh position is imperceptible and saves N writes/frame.
        for (int i = 0; i < count; i++)
        {
            var pbd  = agents[i] as PBDAgent;
            var data = _agentDataArray[i];
            pbd.velocity          = data.velocity;
            pbd.predictedPosition = data.predictedPosition;
            if (pbd._agent != null && data.lastLodLevel < 2)   // skip LOW
                pbd._agent.nextPosition = data.position;
        }
    }

    public void DisposeData()
    {
        if (_agentDataArray.IsCreated)  _agentDataArray.Dispose();
        if (_corrections.IsCreated)     _corrections.Dispose();
        if (_correctionCounts.IsCreated)_correctionCounts.Dispose();
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
// LOD categorisation — runs once per frame, single-threaded.
// Uses a 10 % hysteresis band per tier so agents don't flicker at boundaries.
// Writes lastLodLevel back into the agent data for next-frame hysteresis.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// LOD categorisation with three enhancements:
//   1. Frustum-aware  — agents behind the camera are forced to LOW
//   2. Adaptive       — distances pre-scaled by caller based on FPS
//   3. Density-aware  — agents with many collisions last frame are promoted
// ---------------------------------------------------------------------------
[BurstCompile]
public struct CategorizeLODJob : IJob
{
    public  NativeArray<CLODAgentData> Agents;
    public  float3 FocusPoint;
    public  float  HighLodDistSq;
    public  float  MedLodDistSq;
    public  float  Hysteresis;

    // Frustum culling
    public float3 CameraPos;
    public float3 CameraForward;   // must be normalised before scheduling

    // Density-aware promotion
    [ReadOnly] public NativeArray<int> PrevCorrectionCounts;
    public int DensityThreshold;

    public NativeList<int> HighIndices;
    public NativeList<int> MedIndices;
    public NativeList<int> LowIndices;

    public void Execute()
    {
        HighIndices.Clear();
        MedIndices.Clear();
        LowIndices.Clear();

        float hOuter = 1f + Hysteresis;
        float hInner = 1f - Hysteresis;

        for (int i = 0; i < Agents.Length; i++)
        {
            CLODAgentData agent = Agents[i];
            int prev = agent.lastLodLevel;
            int next;

            // ── 1. Frustum cull ────────────────────────────────────────────
            // Agent is in the half-space behind the camera → always LOW.
            // Avoids wasting HIGH/MED compute on invisible agents.
            float3 toAgent = agent.position - CameraPos;
            bool behindCamera = math.dot(toAgent, CameraForward) < 0f;

            if (behindCamera)
            {
                next = 2;
            }
            else
            {
                // ── 2. Distance-based categorisation (with hysteresis) ─────
                float distSq = math.lengthsq(toAgent);

                if (prev == 0)       // was HIGH — hold until clearly past boundary
                {
                    float outerHighSq = HighLodDistSq * hOuter * hOuter;
                    float outerMedSq  = MedLodDistSq  * hOuter * hOuter;
                    if      (distSq < outerHighSq) next = 0;
                    else if (distSq < outerMedSq)  next = 1;
                    else                           next = 2;
                }
                else if (prev == 1)  // was MED
                {
                    float innerHighSq = HighLodDistSq * hInner * hInner;
                    float outerMedSq  = MedLodDistSq  * hOuter * hOuter;
                    if      (distSq < innerHighSq) next = 0;
                    else if (distSq < outerMedSq)  next = 1;
                    else                           next = 2;
                }
                else                 // was LOW
                {
                    float innerMedSq = MedLodDistSq * hInner * hInner;
                    if      (distSq < HighLodDistSq) next = 0;
                    else if (distSq < innerMedSq)    next = 1;
                    else                             next = 2;
                }

                // ── 3. Density-aware promotion ─────────────────────────────
                // If the agent had many collisions last frame it's in a dense
                // cluster — promote it one tier to get better collision resolution.
                if (next > 0 && PrevCorrectionCounts[i] >= DensityThreshold)
                    next--;
            }

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
