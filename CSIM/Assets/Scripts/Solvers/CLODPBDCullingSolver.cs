using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Jobs;

// =============================================================================
//  CLODPBDCullingSolver.cs - PREDICTIVE HEATMAP EDITION
//
//  Physics LOD is driven entirely by local crowd density, ensuring tight 
//  collision resolution in dense packs regardless of distance from the camera.
//  Visibility (Renderers/Animators) remains driven by frustum culling.
// =============================================================================

public struct CullAgentData
{
    public float3 position;
    public float3 predictedPosition;
    public float3 velocity;
    public float3 preferredVelocity;
    public float ksi;
    public float radius;
    public float mass;
    public float importanceScore;
    public int lastLodLevel;
    public int isVisible; // 1 = true, 0 = false
}

public struct HeatmapCell
{
    public int count;
    public float3 sumVelocity;
    public float3 centerPos;
}

public class CLODPBDCullingSolver : MonoBehaviour
{
    // ── Inspector ─────────────────────────────────────────────────────────────

    [Header("Protagonist Camera (Visibility Only)")]
    public Camera protagonistCamera;
    public float frustumMargin = 1.5f;

    [HideInInspector] public float highLodDistance = 20f;
    [HideInInspector] public float medLodDistance = 50f;

    [Header("Predictive Heatmap LOD")]
    [Tooltip("How often (in seconds) the base heatmap is fully recalculated. Between updates, cells are advected by their average velocity.")]
    public float heatmapUpdateInterval = 0.25f;
    [Tooltip("Size of the density measurement cells. Larger = smoother crowd detection.")]
    public float heatmapCellSize = 8f;

    [Header("Density Cutoffs")]
    public int highDensityCutoff = 12;
    public int medDensityCutoff = 4;
    [Tooltip("Prevents agents from flickering between LODs if their density dances on the cutoff line.")]
    public int densityHysteresis = 2;

    [Header("Spatial Collision Grid")]
    public float gridCellSize = 4f;

    [Header("Importance Scoring (Synced by SimulationManager)")]
    public float weightDistance = 0.30f;
    public float weightVisibility = 0.25f;
    public float weightInteraction = 0.20f;
    public float weightDensity = 0.25f;
    public float highImportanceThreshold = 0.60f;
    public float medImportanceThreshold = 0.30f;
    public float importanceHysteresis = 0.05f;
    public float maxImportanceDistance = 60f;
    public int maxInteractionContacts = 6;
    public int maxImportanceDensity = 12;

    // ── Native collections ────────────────────────────────────────────────────

    private NativeArray<CullAgentData> _agentData;
    private NativeArray<float3> _corrections;
    private NativeArray<int> _correctionCounts;
    private NativeParallelMultiHashMap<int, int> _hashMap;
    private TransformAccessArray _transforms;
    private NativeArray<float4> _frustumPlanes;

    // Heatmap structures
    private NativeParallelMultiHashMap<int, int> _baseGridMap;
    private NativeParallelHashMap<int, HeatmapCell> _baseHeatmap;
    private NativeParallelMultiHashMap<int, int> _predictedHeatmap;

    private NativeList<int> _highIdx;
    private NativeList<int> _medIdx;
    private NativeList<int> _lowIdx;

    private Renderer[][] _renderers;
    private Animator[][] _animators;
    private bool[] _wasVisible;

    private bool _initialized;
    private float _timeSinceHeatmapUpdate;

    // =========================================================================

    public void Init(List<AgentBase> agents)
    {
        DisposeData();
        int n = agents.Count;
        if (n == 0) return;

        _agentData = new NativeArray<CullAgentData>(n, Allocator.Persistent);
        _corrections = new NativeArray<float3>(n, Allocator.Persistent);
        _correctionCounts = new NativeArray<int>(n, Allocator.Persistent);
        _hashMap = new NativeParallelMultiHashMap<int, int>(n * 2, Allocator.Persistent);
        _frustumPlanes = new NativeArray<float4>(6, Allocator.Persistent);

        _baseGridMap = new NativeParallelMultiHashMap<int, int>(n, Allocator.Persistent);
        _baseHeatmap = new NativeParallelHashMap<int, HeatmapCell>(n, Allocator.Persistent);
        _predictedHeatmap = new NativeParallelMultiHashMap<int, int>(n, Allocator.Persistent);

        _highIdx = new NativeList<int>(n, Allocator.Persistent);
        _medIdx = new NativeList<int>(n, Allocator.Persistent);
        _lowIdx = new NativeList<int>(n, Allocator.Persistent);

        _renderers = new Renderer[n][];
        _animators = new Animator[n][];
        _wasVisible = new bool[n];

        var xforms = new Transform[n];
        for (int i = 0; i < n; i++)
        {
            xforms[i] = agents[i].transform;
            _renderers[i] = agents[i].GetComponentsInChildren<Renderer>(true);
            _animators[i] = agents[i].GetComponentsInChildren<Animator>(true);
            _wasVisible[i] = true;
        }
        _transforms = new TransformAccessArray(xforms);
        _initialized = true;

        // Force immediate heatmap generation on frame 1
        _timeSinceHeatmapUpdate = heatmapUpdateInterval;
    }

    public void Solve(List<AgentBase> agents, float dt, int iterations, Vector3 focusPoint)
    {
        int n = agents.Count;
        if (n == 0 || dt <= 0f) return;
        if (!_initialized || _agentData.Length != n) Init(agents);

        RefreshFrustumPlanes();

        // 1. Upload state
        for (int i = 0; i < n; i++)
        {
            var pbd = agents[i] as PBDAgent;
            var vPref = float3.zero;
            if (pbd._agent != null && pbd._agent.hasPath)
                vPref = math.normalizesafe((float3)pbd._agent.steeringTarget - (float3)pbd.transform.position) * pbd.prefSpeed;

            _agentData[i] = new CullAgentData
            {
                position = pbd.transform.position,
                predictedPosition = pbd.predictedPosition,
                velocity = pbd.velocity,
                preferredVelocity = vPref,
                ksi = pbd.ksi,
                radius = pbd.radius,
                mass = pbd.mass,
                importanceScore = _agentData[i].importanceScore,
                lastLodLevel = _agentData[i].lastLodLevel
            };
        }

        JobHandle handle = default;

        // 2. Base Heatmap generation (Run every X seconds)
        _timeSinceHeatmapUpdate += dt;
        if (_timeSinceHeatmapUpdate >= heatmapUpdateInterval)
        {
            handle = new HeatmapClearGridJob { GridMap = _baseGridMap }.Schedule(handle);
            handle = new HeatmapHashAgentsJob
            {
                Agents = _agentData,
                GridMap = _baseGridMap.AsParallelWriter(),
                CellSize = heatmapCellSize
            }.Schedule(n, 64, handle);

            handle = new HeatmapCompileBaseJob
            {
                Agents = _agentData,
                GridMap = _baseGridMap,
                BaseHeatmap = _baseHeatmap
            }.Schedule(handle);

            _timeSinceHeatmapUpdate = 0f;
        }

        // 3. Predictive Heatmap Shift (Run every frame)
        // Advects the dense cells forward based on crowd velocity to maintain LOD accuracy between base updates.
        handle = new HeatmapPredictJob
        {
            BaseHeatmap = _baseHeatmap,
            PredictedHeatmap = _predictedHeatmap,
            TimeSinceUpdate = _timeSinceHeatmapUpdate,
            CellSize = heatmapCellSize
        }.Schedule(handle);

        // 4. Categorise & Cull (Importance-Driven Dynamic Fidelity)
        handle = new CategorizeAndCullJob
        {
            Agents = _agentData,
            FrustumPlanes = _frustumPlanes,
            PredictedHeatmap = _predictedHeatmap,
            PrevCorrectionCounts = _correctionCounts,
            FocusPoint = focusPoint,
            HeatmapCellSize = heatmapCellSize,
            WeightDistance = weightDistance,
            WeightVisibility = weightVisibility,
            WeightInteraction = weightInteraction,
            WeightDensity = weightDensity,
            HighImportanceThresh = highImportanceThreshold,
            MedImportanceThresh = medImportanceThreshold,
            ImportanceHysteresis = importanceHysteresis,
            MaxImportanceDist = math.max(maxImportanceDistance, 1f),
            MaxInteractionContacts = math.max(maxInteractionContacts, 1),
            MaxImportanceDensity = math.max(maxImportanceDensity, 1),
            HighIndices = _highIdx,
            MedIndices = _medIdx,
            LowIndices = _lowIdx
        }.Schedule(handle);

        // 5. Predict positions
        handle = new CullPredictJob { Agents = _agentData, dt = dt }.Schedule(n, 64, handle);

        // 6. Build Spatial Hash for Collision (All agents)
        handle = new CullClearHashJob { HashMap = _hashMap }.Schedule(handle);
        handle = ScheduleHash(_highIdx, handle);
        handle = ScheduleHash(_medIdx, handle);
        handle = ScheduleHash(_lowIdx, handle);

        bool resolveLowThisFrame = Time.frameCount % 4 == 0;
        int medIterationCap = math.max(1, (iterations + 1) / 2);

        // 7. Constraint Iterations
        for (int iter = 0; iter < iterations; iter++)
        {
            // HIGH - Full local grid
            handle = new CullHighConstraintJob
            {
                Indices = _highIdx.AsDeferredJobArray(),
                AgentsIn = _agentData,
                HashMap = _hashMap,
                PositionCorrections = _corrections,
                CorrectionCounts = _correctionCounts,
                CellSize = gridCellSize
            }.Schedule(_highIdx, 64, handle);

            // MED - Single Cell, half-budget iterations
            if (iter < medIterationCap)
            {
                handle = new CullMedConstraintJob
                {
                    Indices = _medIdx.AsDeferredJobArray(),
                    AgentsIn = _agentData,
                    HashMap = _hashMap,
                    PositionCorrections = _corrections,
                    CorrectionCounts = _correctionCounts,
                    CellSize = gridCellSize
                }.Schedule(_medIdx, 64, handle);
            }

            // LOW - Single Iteration, Single Cell, every 4th frame
            if (iter == 0 && resolveLowThisFrame)
            {
                handle = new CullLowConstraintJob
                {
                    Indices = _lowIdx.AsDeferredJobArray(),
                    AgentsIn = _agentData,
                    HashMap = _hashMap,
                    PositionCorrections = _corrections,
                    CorrectionCounts = _correctionCounts,
                    CellSize = gridCellSize,
                    MaxContacts = 1
                }.Schedule(_lowIdx, 64, handle);
            }

            handle = ScheduleApply(_highIdx, handle);
            if (iter < medIterationCap) handle = ScheduleApply(_medIdx, handle);
            if (iter == 0 && resolveLowThisFrame) handle = ScheduleApply(_lowIdx, handle);
        }

        // 8. Finalise and Apply
        handle = new CullUpdateJob { Agents = _agentData, dt = dt }.Schedule(n, 64, handle);
        handle = new CullApplyTransformJob { Agents = _agentData, dt = dt }.Schedule(_transforms, handle);

        handle.Complete();

        // 9. Synchronize Renderers
        SyncVisibilityAndNavMesh(agents, n);
    }

    // =========================================================================

    private JobHandle ScheduleHash(NativeList<int> list, JobHandle dep) =>
        new CullHashJob { Indices = list.AsDeferredJobArray(), Agents = _agentData, HashMap = _hashMap.AsParallelWriter(), CellSize = gridCellSize }.Schedule(list, 64, dep);

    private JobHandle ScheduleApply(NativeList<int> list, JobHandle dep) =>
        new CullApplyJob { Indices = list.AsDeferredJobArray(), Agents = _agentData, PositionCorrections = _corrections, CorrectionCounts = _correctionCounts }.Schedule(list, 64, dep);

    private void RefreshFrustumPlanes()
    {
        Camera cam = protagonistCamera != null ? protagonistCamera : Camera.main;
        if (cam == null) return;
        Plane[] planes = GeometryUtility.CalculateFrustumPlanes(cam);
        for (int p = 0; p < 6; p++)
            _frustumPlanes[p] = new float4(planes[p].normal.x, planes[p].normal.y, planes[p].normal.z, planes[p].distance - frustumMargin);
    }

    private void SyncVisibilityAndNavMesh(List<AgentBase> agents, int n)
    {
        for (int i = 0; i < n; i++)
        {
            CullAgentData data = _agentData[i];
            bool visible = data.isVisible == 1;

            if (visible != _wasVisible[i])
            {
                if (_renderers[i] != null)
                    for (int r = 0; r < _renderers[i].Length; r++)
                        if (_renderers[i][r] != null) _renderers[i][r].enabled = visible;

                if (_animators[i] != null)
                    for (int a = 0; a < _animators[i].Length; a++)
                        if (_animators[i][a] != null) _animators[i][a].enabled = visible;

                _wasVisible[i] = visible;
            }

            var pbd = agents[i] as PBDAgent;
            if (pbd == null) continue;
            pbd.velocity = data.velocity;
            pbd.predictedPosition = data.predictedPosition;
            pbd.importanceScore = data.importanceScore;
            pbd.fidelityLevel = data.lastLodLevel;
            if (pbd._agent != null && data.lastLodLevel < 2)
                pbd._agent.nextPosition = data.position;
        }
    }

    public void DisposeData()
    {
        if (_agentData.IsCreated) _agentData.Dispose();
        if (_corrections.IsCreated) _corrections.Dispose();
        if (_correctionCounts.IsCreated) _correctionCounts.Dispose();
        if (_hashMap.IsCreated) _hashMap.Dispose();
        if (_frustumPlanes.IsCreated) _frustumPlanes.Dispose();
        if (_transforms.isCreated) _transforms.Dispose();
        if (_highIdx.IsCreated) _highIdx.Dispose();
        if (_medIdx.IsCreated) _medIdx.Dispose();
        if (_lowIdx.IsCreated) _lowIdx.Dispose();

        if (_baseGridMap.IsCreated) _baseGridMap.Dispose();
        if (_baseHeatmap.IsCreated) _baseHeatmap.Dispose();
        if (_predictedHeatmap.IsCreated) _predictedHeatmap.Dispose();

        _initialized = false;
    }

    void OnDisable() => DisposeData();
    void OnDestroy() => DisposeData();
}

// =============================================================================
//  BURST JOBS
// =============================================================================

[BurstCompile]
public struct HeatmapClearGridJob : IJob
{
    public NativeParallelMultiHashMap<int, int> GridMap;
    public void Execute() => GridMap.Clear();
}

[BurstCompile]
public struct HeatmapHashAgentsJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<CullAgentData> Agents;
    public NativeParallelMultiHashMap<int, int>.ParallelWriter GridMap;
    public float CellSize;

    public void Execute(int i)
    {
        float3 pos = Agents[i].position;
        int cx = (int)math.floor(pos.x / CellSize);
        int cz = (int)math.floor(pos.z / CellSize);
        GridMap.Add((cx * 73856093) ^ (cz * 19349663), i);
    }
}

[BurstCompile]
public struct HeatmapCompileBaseJob : IJob
{
    [ReadOnly] public NativeArray<CullAgentData> Agents;
    [ReadOnly] public NativeParallelMultiHashMap<int, int> GridMap;
    public NativeParallelHashMap<int, HeatmapCell> BaseHeatmap;

    public void Execute()
    {
        BaseHeatmap.Clear();

        // FIX: Extract the array (Item1) and length (Item2) from the Tuple
        var keysTuple = GridMap.GetUniqueKeyArray(Allocator.Temp);
        NativeArray<int> keys = keysTuple.Item1;
        int uniqueCount = keysTuple.Item2;

        for (int i = 0; i < uniqueCount; i++)
        {
            int hash = keys[i];
            int count = 0;
            float3 sumVel = float3.zero;
            float3 sumPos = float3.zero;

            if (GridMap.TryGetFirstValue(hash, out int idx, out var it))
            {
                do
                {
                    count++;
                    sumVel += Agents[idx].velocity;
                    sumPos += Agents[idx].position;
                } while (GridMap.TryGetNextValue(out idx, ref it));
            }

            BaseHeatmap.TryAdd(hash, new HeatmapCell
            {
                count = count,
                sumVelocity = sumVel,
                centerPos = sumPos / count
            });
        }

        // Prevent memory leaks
        keys.Dispose();
    }
}

[BurstCompile]
public struct HeatmapPredictJob : IJob
{
    [ReadOnly] public NativeParallelHashMap<int, HeatmapCell> BaseHeatmap;
    public NativeParallelMultiHashMap<int, int> PredictedHeatmap;
    public float TimeSinceUpdate;
    public float CellSize;

    public void Execute()
    {
        PredictedHeatmap.Clear();
        var keys = BaseHeatmap.GetKeyArray(Allocator.Temp);
        for (int i = 0; i < keys.Length; i++)
        {
            var cell = BaseHeatmap[keys[i]];
            float3 avgVel = cell.sumVelocity / cell.count;

            // Advect the density cell forward based on elapsed time
            float3 predPos = cell.centerPos + avgVel * TimeSinceUpdate;

            int cx = (int)math.floor(predPos.x / CellSize);
            int cz = (int)math.floor(predPos.z / CellSize);
            PredictedHeatmap.Add((cx * 73856093) ^ (cz * 19349663), cell.count);
        }
    }
}

[BurstCompile]
public struct CategorizeAndCullJob : IJob
{
    public NativeArray<CullAgentData> Agents;
    [ReadOnly] public NativeArray<float4> FrustumPlanes;
    [ReadOnly] public NativeParallelMultiHashMap<int, int> PredictedHeatmap;
    [ReadOnly] public NativeArray<int> PrevCorrectionCounts;

    public float3 FocusPoint;
    public float HeatmapCellSize;
    public float WeightDistance;
    public float WeightVisibility;
    public float WeightInteraction;
    public float WeightDensity;
    public float HighImportanceThresh;
    public float MedImportanceThresh;
    public float ImportanceHysteresis;
    public float MaxImportanceDist;
    public int MaxInteractionContacts;
    public int MaxImportanceDensity;

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
            CullAgentData a = Agents[i];

            // 1. Visibility (Frustum test)
            bool visible = true;
            for (int p = 0; p < 6; p++)
            {
                if (math.dot(FrustumPlanes[p].xyz, a.position) + FrustumPlanes[p].w < 0f)
                {
                    visible = false;
                    break;
                }
            }
            a.isVisible = visible ? 1 : 0;
            float visScore = visible ? 1f : 0f;

            // 2. Distance score
            float dist = math.distance(a.position, FocusPoint);
            float distScore = math.saturate(1f - dist / MaxImportanceDist);

            // 3. Interaction score (previous frame contact count)
            float interScore = math.saturate((float)PrevCorrectionCounts[i] / MaxInteractionContacts);

            // 4. Local Predictive Density score
            int cx = (int)math.floor(a.position.x / HeatmapCellSize);
            int cz = (int)math.floor(a.position.z / HeatmapCellSize);
            int hash = (cx * 73856093) ^ (cz * 19349663);

            int localDensity = 0;
            if (PredictedHeatmap.TryGetFirstValue(hash, out int d, out var it))
            {
                do { localDensity += d; }
                while (PredictedHeatmap.TryGetNextValue(out d, ref it));
            }
            float densityScore = math.saturate((float)localDensity / MaxImportanceDensity);

            // 5. Combined Importance Score [0, 1]
            float importance = math.saturate(
                (WeightDistance * distScore +
                 WeightVisibility * visScore +
                 WeightInteraction * interScore +
                 WeightDensity * densityScore) / totalWeight);

            // 6. Assign Physics Fidelity based on Importance (with hysteresis)
            int prev = a.lastLodLevel;
            int next;
            if (prev == 0)
            {
                if      (importance >= HighImportanceThresh - ImportanceHysteresis) next = 0;
                else if (importance >= MedImportanceThresh  - ImportanceHysteresis) next = 1;
                else                                                              next = 2;
            }
            else if (prev == 1)
            {
                if      (importance >= HighImportanceThresh + ImportanceHysteresis) next = 0;
                else if (importance >= MedImportanceThresh  - ImportanceHysteresis) next = 1;
                else                                                              next = 2;
            }
            else
            {
                if      (importance >= HighImportanceThresh + ImportanceHysteresis) next = 0;
                else if (importance >= MedImportanceThresh  + ImportanceHysteresis) next = 1;
                else                                                              next = 2;
            }

            a.importanceScore = importance;
            a.lastLodLevel = next;
            Agents[i] = a;

            if (next == 0) HighIndices.Add(i);
            else if (next == 1) MedIndices.Add(i);
            else LowIndices.Add(i);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// STANDARD PBD JOBS (Unchanged Physics)
// ─────────────────────────────────────────────────────────────────────────────

[BurstCompile]
public struct CullPredictJob : IJobParallelFor
{
    public NativeArray<CullAgentData> Agents;
    public float dt;

    public void Execute(int index)
    {
        CullAgentData a = Agents[index];
        a.velocity = math.lerp(a.velocity, a.preferredVelocity, a.ksi);
        a.predictedPosition = a.position + a.velocity * dt;
        a.predictedPosition.y = a.position.y;
        Agents[index] = a;
    }
}

[BurstCompile]
public struct CullClearHashJob : IJob { public NativeParallelMultiHashMap<int, int> HashMap; public void Execute() => HashMap.Clear(); }

[BurstCompile]
public struct CullHashJob : IJobParallelForDefer
{
    [ReadOnly] public NativeArray<int> Indices;
    [ReadOnly] public NativeArray<CullAgentData> Agents;
    public NativeParallelMultiHashMap<int, int>.ParallelWriter HashMap;
    public float CellSize;
    public void Execute(int loopIndex)
    {
        int idx = Indices[loopIndex];
        float3 pos = Agents[idx].predictedPosition;
        int cx = (int)math.floor(pos.x / CellSize);
        int cz = (int)math.floor(pos.z / CellSize);
        HashMap.Add((cx * 73856093) ^ (cz * 19349663), idx);
    }
}

[BurstCompile]
public struct CullHighConstraintJob : IJobParallelForDefer
{
    [ReadOnly] public NativeArray<int> Indices;
    [ReadOnly] public NativeArray<CullAgentData> AgentsIn;
    [ReadOnly] public NativeParallelMultiHashMap<int, int> HashMap;
    [NativeDisableParallelForRestriction] public NativeArray<float3> PositionCorrections;
    [NativeDisableParallelForRestriction] public NativeArray<int> CorrectionCounts;
    public float CellSize;

    public void Execute(int loopIndex)
    {
        int idx = Indices[loopIndex];
        CullAgentData agent = AgentsIn[idx];
        float3 total = float3.zero;
        int count = 0;
        int cx = (int)math.floor(agent.predictedPosition.x / CellSize);
        int cz = (int)math.floor(agent.predictedPosition.z / CellSize);

        for (int i = -1; i <= 1 && count < 12; i++)
            for (int j = -1; j <= 1 && count < 12; j++)
            {
                int hash = ((cx + i) * 73856093) ^ ((cz + j) * 19349663);
                if (!HashMap.TryGetFirstValue(hash, out int nb, out var it)) continue;
                do
                {
                    if (nb == idx || count >= 12) continue;
                    CullAgentData b = AgentsIn[nb];
                    float3 diff = agent.predictedPosition - b.predictedPosition;
                    diff.y = 0f;
                    float dSq = math.lengthsq(diff);
                    float margin = math.max(agent.radius, 0.3f) + math.max(b.radius, 0.3f);
                    if (dSq < margin * margin && dSq > 1e-6f)
                    {
                        float d = math.sqrt(dSq);
                        float wI = 1f / math.max(agent.mass, 1e-4f);
                        float wJ = 1f / math.max(b.mass, 1e-4f);
                        total += ((margin - d) / (wI + wJ)) * wI * (diff / d);
                        count++;
                    }
                } while (HashMap.TryGetNextValue(out nb, ref it));
            }
        PositionCorrections[idx] = total;
        CorrectionCounts[idx] = count;
    }
}

[BurstCompile]
public struct CullMedConstraintJob : IJobParallelForDefer
{
    [ReadOnly] public NativeArray<int> Indices;
    [ReadOnly] public NativeArray<CullAgentData> AgentsIn;
    [ReadOnly] public NativeParallelMultiHashMap<int, int> HashMap;
    [NativeDisableParallelForRestriction] public NativeArray<float3> PositionCorrections;
    [NativeDisableParallelForRestriction] public NativeArray<int> CorrectionCounts;
    public float CellSize;

    public void Execute(int loopIndex)
    {
        int idx = Indices[loopIndex];
        CullAgentData agent = AgentsIn[idx];
        float3 total = float3.zero;
        int count = 0;
        int cx = (int)math.floor(agent.predictedPosition.x / CellSize);
        int cz = (int)math.floor(agent.predictedPosition.z / CellSize);
        int hash = (cx * 73856093) ^ (cz * 19349663);

        if (HashMap.TryGetFirstValue(hash, out int nb, out var it))
        {
            do
            {
                if (nb == idx || count >= 4) continue;
                CullAgentData b = AgentsIn[nb];
                float3 diff = agent.predictedPosition - b.predictedPosition;
                diff.y = 0f;
                float dSq = math.lengthsq(diff);
                float margin = math.max(agent.radius, 0.3f) + math.max(b.radius, 0.3f);
                if (dSq < margin * margin && dSq > 1e-6f)
                {
                    float d = math.sqrt(dSq);
                    total += ((margin - d) * 0.5f) * (diff / d);
                    count++;
                }
            } while (HashMap.TryGetNextValue(out nb, ref it));
        }
        PositionCorrections[idx] = total;
        CorrectionCounts[idx] = count;
    }
}

[BurstCompile]
public struct CullLowConstraintJob : IJobParallelForDefer
{
    [ReadOnly] public NativeArray<int> Indices;
    [ReadOnly] public NativeArray<CullAgentData> AgentsIn;
    [ReadOnly] public NativeParallelMultiHashMap<int, int> HashMap;
    [NativeDisableParallelForRestriction] public NativeArray<float3> PositionCorrections;
    [NativeDisableParallelForRestriction] public NativeArray<int> CorrectionCounts;
    public float CellSize;
    public int MaxContacts;

    public void Execute(int loopIndex)
    {
        int idx = Indices[loopIndex];
        CullAgentData agent = AgentsIn[idx];
        float3 total = float3.zero;
        int count = 0;
        int cx = (int)math.floor(agent.predictedPosition.x / CellSize);
        int cz = (int)math.floor(agent.predictedPosition.z / CellSize);
        int hash = (cx * 73856093) ^ (cz * 19349663);

        if (HashMap.TryGetFirstValue(hash, out int nb, out var it))
        {
            do
            {
                if (nb == idx || count >= MaxContacts) continue;
                CullAgentData b = AgentsIn[nb];
                float3 diff = agent.predictedPosition - b.predictedPosition;
                diff.y = 0f;
                float dSq = math.lengthsq(diff);
                float margin = math.max(agent.radius, 0.3f) + math.max(b.radius, 0.3f);
                if (dSq < margin * margin && dSq > 1e-6f)
                {
                    float d = math.sqrt(dSq);
                    total += ((margin - d) * 0.5f) * (diff / d);
                    count++;
                }
            } while (HashMap.TryGetNextValue(out nb, ref it));
        }
        PositionCorrections[idx] = total;
        CorrectionCounts[idx] = count;
    }
}

[BurstCompile]
public struct CullApplyJob : IJobParallelForDefer
{
    [ReadOnly] public NativeArray<int> Indices;
    [NativeDisableParallelForRestriction] public NativeArray<CullAgentData> Agents;
    [ReadOnly] public NativeArray<float3> PositionCorrections;
    [ReadOnly] public NativeArray<int> CorrectionCounts;

    public void Execute(int loopIndex)
    {
        int idx = Indices[loopIndex];
        if (CorrectionCounts[idx] <= 0) return;
        CullAgentData a = Agents[idx];
        a.predictedPosition += PositionCorrections[idx];
        Agents[idx] = a;
    }
}

[BurstCompile]
public struct CullUpdateJob : IJobParallelFor
{
    public NativeArray<CullAgentData> Agents;
    public float dt;

    public void Execute(int index)
    {
        CullAgentData a = Agents[index];
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

[BurstCompile]
public struct CullApplyTransformJob : IJobParallelForTransform
{
    [ReadOnly] public NativeArray<CullAgentData> Agents;
    public float dt;

    public void Execute(int index, TransformAccess transform)
    {
        CullAgentData a = Agents[index];
        transform.position = a.position;

        if (a.isVisible == 1 && math.lengthsq(a.velocity) > 0.01f)
        {
            quaternion targetRot = quaternion.LookRotationSafe(math.normalize(a.velocity), math.up());
            transform.rotation = math.slerp(transform.rotation, targetRot, dt * 10f);
        }
    }
}