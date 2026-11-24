using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Jobs;

public struct PBDAgentData
{
    public float3 position;
    public float3 predictedPosition;
    public float3 velocity;
    public float3 preferredVelocity;
    public float ksi;
    public float radius;
    public float mass;
}


[BurstCompile]
public struct PBDPredictJob : IJobParallelFor
{
    public NativeArray<PBDAgentData> Agents;
    public float dt;

    public void Execute(int index)
    {
        PBDAgentData agent = Agents[index];

        // Velocity interpolation
        agent.velocity = math.lerp(agent.velocity, agent.preferredVelocity, agent.ksi);

        // Predict position
        agent.predictedPosition = agent.position + agent.velocity * dt;
        agent.predictedPosition.y = agent.position.y;

        Agents[index] = agent;
    }
}

public struct ClearHashMapJob : IJob
{
    public NativeParallelMultiHashMap<int, int> HashMap;

    public void Execute()
    {
        HashMap.Clear();
    }
}

[BurstCompile]
public struct HashPositionsJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<PBDAgentData> Agents;
    public NativeParallelMultiHashMap<int, int>.ParallelWriter HashMap;
    public float CellSize;

    public void Execute(int index)
    {
        float3 pos = Agents[index].predictedPosition;

        int2 cell = new int2((int)math.floor(pos.x / CellSize), (int)math.floor(pos.z / CellSize));

        int hash = (cell.x * 73856093) ^ (cell.y * 19349663);

        HashMap.Add(hash, index);
    }
}

[BurstCompile]
public struct PBDConstraintJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<PBDAgentData> AgentsIn;
    [ReadOnly] public NativeParallelMultiHashMap<int, int> HashMap;
    public NativeArray<float3> PositionCorrections;
    public NativeArray<int> CorrectionCounts;
    public float CellSize;

    public void Execute(int index)
    {
        PBDAgentData agent = AgentsIn[index];
        float3 totalCorrection = float3.zero;
        int count = 0;

        float3 pos = agent.predictedPosition;
        int2 centerCell = new int2((int)math.floor(pos.x / CellSize), (int)math.floor(pos.z / CellSize));

        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                int2 currentCell = centerCell + new int2(i, j);
                int hash = (currentCell.x * 73856093) ^ (currentCell.y * 19349663);

                if (HashMap.TryGetFirstValue(hash, out int neighborIndex, out NativeParallelMultiHashMapIterator<int> it))
                {
                    do
                    {
                        if (neighborIndex == index) continue;

                        PBDAgentData neighbor = AgentsIn[neighborIndex];
                        float3 diff = agent.predictedPosition - neighbor.predictedPosition;
                        diff.y = 0;
                        float distSq = math.lengthsq(diff);

                        float margin = (agent.radius > 0 ? agent.radius : 0.5f) +
                                       (neighbor.radius > 0 ? neighbor.radius : 0.5f);

                        if (distSq < margin * margin && distSq > 0.000001f)
                        {
                            float distance = math.sqrt(distSq);
                            float penetration = margin - distance;

                            float3 contactNormal = diff / distance;

                            float wI = 1f / (agent.mass > 0 ? agent.mass : 1f);
                            float wJ = 1f / (neighbor.mass > 0 ? neighbor.mass : 1f);
                            float wSum = wI + wJ;

                            float3 correction = (penetration / wSum) * contactNormal;
                            totalCorrection += correction * wI;
                            count++;
                        }
                    } while (HashMap.TryGetNextValue(out neighborIndex, ref it));
                }
            }
        }

        PositionCorrections[index] = totalCorrection;
        CorrectionCounts[index] = count;
    }
}

[BurstCompile]
public struct PBDApplyConstraintsJob : IJobParallelFor
{
    public NativeArray<PBDAgentData> Agents;
    [ReadOnly] public NativeArray<float3> PositionCorrections;
    [ReadOnly] public NativeArray<int> CorrectionCounts;

    public void Execute(int index)
    {
        if (CorrectionCounts[index] > 0)
        {
            PBDAgentData agent = Agents[index];

            agent.predictedPosition += PositionCorrections[index] / math.max(1, CorrectionCounts[index]);
            Agents[index] = agent;
        }
    }
}

[BurstCompile]
public struct PBDUpdateJob : IJobParallelFor
{
    public NativeArray<PBDAgentData> Agents;
    public float dt;

    public void Execute(int index)
    {
        PBDAgentData agent = Agents[index];
        float3 rawVelocity = (agent.predictedPosition - agent.position) / math.max(dt, 1e-5f);
        float preferredSpeed = math.length(agent.preferredVelocity);
        float maxSpeed = math.max(preferredSpeed * 1.15f, 0.75f);
        float rawSpeed = math.length(rawVelocity);
        if (rawSpeed > maxSpeed)
            rawVelocity *= maxSpeed / rawSpeed;

        if (preferredSpeed > 0.01f)
        {
            float3 prefDir = agent.preferredVelocity / preferredSpeed;
            float backward = math.dot(rawVelocity, -prefDir);
            if (backward > 0f)
                rawVelocity += prefDir * backward * 0.85f;
        }

        agent.velocity = rawVelocity;
        agent.position = agent.predictedPosition;
        Agents[index] = agent;
    }
}

[BurstCompile]
public struct PBDApplyTransformJob : IJobParallelForTransform
{
    [ReadOnly] public NativeArray<PBDAgentData> Agents;
    public float dt;

    public void Execute(int index, TransformAccess transform)
    {
        PBDAgentData agent = Agents[index];
        transform.position = agent.position;

        if (math.lengthsq(agent.velocity) > 0.01f)
        {
            quaternion targetRot = quaternion.LookRotationSafe(agent.velocity, math.up());
            transform.rotation = math.slerp(transform.rotation, targetRot, dt * 10f);
        }
    }
}

public class PBDSolverJobs : MonoBehaviour
{
    private NativeArray<PBDAgentData> _agentDataArray;
    private NativeArray<float3> _corrections;
    private NativeArray<int> _correctionCounts;
    private NativeParallelMultiHashMap<int, int> _spatialHashMap;
    private TransformAccessArray _transformArray;
    private bool _initialized = false;

    public float gridCellSize = 4.0f;

    public void Init(List<AgentBase> agents)
    {
        if (_initialized) DisposeData();

        int count = agents.Count;
        _agentDataArray = new NativeArray<PBDAgentData>(count, Allocator.Persistent);
        _corrections = new NativeArray<float3>(count, Allocator.Persistent);
        _correctionCounts = new NativeArray<int>(count, Allocator.Persistent);

        _spatialHashMap = new NativeParallelMultiHashMap<int, int>(count, Allocator.Persistent);

        Transform[] transforms = new Transform[count];
        for (int i = 0; i < count; i++)
        {
            transforms[i] = agents[i].transform;
        }
        _transformArray = new TransformAccessArray(transforms);
        _initialized = true;
    }

    public void Solve(List<AgentBase> agents, float dt, int iterations)
    {
        int count = agents.Count;
        if (count == 0 || dt <= 0f) return;

        if (!_initialized || _agentDataArray.Length != count)
        {
            Init(agents);
        }

        // 1. Gather remaining single-thread data (vPref)
        for (int i = 0; i < count; i++)
        {
            PBDAgent pbdAgent = agents[i] as PBDAgent;
            float3 vPref = float3.zero;

            if (pbdAgent._agent != null && pbdAgent._agent.hasPath)
            {
                float3 desiredDirection = math.normalize((float3)pbdAgent._agent.steeringTarget - (float3)pbdAgent.transform.position);
                vPref = desiredDirection * pbdAgent.prefSpeed;
            }

            _agentDataArray[i] = new PBDAgentData
            {
                position = pbdAgent.transform.position,
                predictedPosition = pbdAgent.predictedPosition,
                velocity = pbdAgent.velocity,
                preferredVelocity = vPref,
                ksi = pbdAgent.ksi,
                radius = pbdAgent.radius,
                mass = pbdAgent.mass
            };
        }

        // 2. Predict step updates positions once for the frame
        JobHandle handle = new PBDPredictJob { Agents = _agentDataArray, dt = dt }.Schedule(count, 64);

        for (int iter = 0; iter < iterations; iter++)
        {
            handle = new ClearHashMapJob
            {
                HashMap = _spatialHashMap
            }.Schedule(handle);

            handle = new HashPositionsJob
            {
                Agents = _agentDataArray,
                HashMap = _spatialHashMap.AsParallelWriter(),
                CellSize = gridCellSize
            }.Schedule(count, 64, handle);

            handle = new PBDConstraintJob
            {
                AgentsIn = _agentDataArray,
                HashMap = _spatialHashMap,
                PositionCorrections = _corrections,
                CorrectionCounts = _correctionCounts,
                CellSize = gridCellSize
            }.Schedule(count, 64, handle);

            handle = new PBDApplyConstraintsJob
            {
                Agents = _agentDataArray,
                PositionCorrections = _corrections,
                CorrectionCounts = _correctionCounts
            }.Schedule(count, 64, handle);
        }

        handle = new PBDUpdateJob { Agents = _agentDataArray, dt = dt }.Schedule(count, 64, handle);

        // Schedule parallel Transform modification
        handle = new PBDApplyTransformJob { Agents = _agentDataArray, dt = dt }.Schedule(_transformArray, handle);

        // Wait for complete
        handle.Complete();

        // Sync NavMesh parameters that are restricted to main thread
        for (int i = 0; i < count; i++)
        {
            PBDAgent pbdAgent = agents[i] as PBDAgent;
            PBDAgentData data = _agentDataArray[i];

            pbdAgent.velocity = data.velocity;
            pbdAgent.predictedPosition = data.predictedPosition;

            if (pbdAgent._agent != null)
            {
                pbdAgent._agent.nextPosition = data.position;
            }
        }
    }

    private void DisposeData()
    {
        if (_agentDataArray.IsCreated) _agentDataArray.Dispose();
        if (_corrections.IsCreated) _corrections.Dispose();
        if (_correctionCounts.IsCreated) _correctionCounts.Dispose();
        if (_transformArray.isCreated) _transformArray.Dispose();
        if (_spatialHashMap.IsCreated) _spatialHashMap.Dispose();
        _initialized = false;
    }

    void OnEnable()
    {
    }

    void OnDisable()
    {
        DisposeData();
    }
}