using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

public struct ForceAgentData
{
    public float3 position;
    public float3 velocity;
    public float3 goalPosition;
    public float radius;
    public float maxAccel;
    public float prefSpeed;
    public float goalRadius;
    public float ksi;
    public float k;
    public float m;
    public float t0;
}

[BurstCompile]
public struct ForceHashPositionsJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<ForceAgentData> Agents;
    public NativeParallelMultiHashMap<int, int>.ParallelWriter HashMap;
    public float CellSize;

    public void Execute(int index)
    {
        float3 pos = Agents[index].position;

        int2 cell = new int2((int)math.floor(pos.x / CellSize), (int)math.floor(pos.z / CellSize));
        int hash = (cell.x * 73856093) ^ (cell.y * 19349663);

        HashMap.Add(hash, index);
    }
}

[BurstCompile]
public struct ForceCalculationJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<ForceAgentData> Agents;
    [ReadOnly] public NativeParallelMultiHashMap<int, int> HashMap;
    public NativeArray<float3> Forces;
    public float CellSize;
    public float deltaTime;
    public float epsilon;

    public void Execute(int index)
    {
        ForceAgentData agent = Agents[index];
        float3 vPref = float3.zero;

        // Goal force
        float3 diff = agent.goalPosition - agent.position;
        diff.y = 0;
        float distSq = math.lengthsq(diff);

        // (agent.goalRadius * agent.goalRadius)
        if (distSq > agent.goalRadius * agent.goalRadius)
        {
            vPref = math.normalize(diff) * agent.prefSpeed;
        }
        else
        {
            vPref = float3.zero;
        }

        float3 force = (vPref - agent.velocity) / agent.ksi;

        int2 centerCell = new int2((int)math.floor(agent.position.x / CellSize), (int)math.floor(agent.position.z / CellSize));

        // 2. Avoidance forces (Local Hash Grid Search)
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

                        ForceAgentData other = Agents[neighborIndex];
                        float3 w = other.position - agent.position;
                        w.y = 0;
                        float3 v = agent.velocity - other.velocity;
                        v.y = 0;

                        float distanceSq = math.lengthsq(w);
                        float combinedRadius = other.radius + agent.radius;
                        float rSq = combinedRadius * combinedRadius;

                        if (distanceSq < rSq)
                        {
                            rSq = math.pow(combinedRadius - math.sqrt(distanceSq), 2);
                        }

                        float a = math.dot(v, v);
                        float b = math.dot(w, v);
                        float c = math.dot(w, w) - rSq;
                        float discr = b * b - a * c;

                        if (discr > 0.0f && (a < -epsilon || a > epsilon))
                        {
                            discr = math.sqrt(discr);
                            float t = (b - discr) / a;
                            if (t > 0)
                            {
                                float3 forcePart = (v - (b * v - a * w) / discr) / (a * math.pow(t, agent.m)) * (agent.m / t + 1 / agent.t0);
                                force += -agent.k * math.exp(-t / agent.t0) * forcePart;
                            }
                        }
                    } while (HashMap.TryGetNextValue(out neighborIndex, ref it));
                }
            }
        }

        force.y = 0;

        // 3. Clamp acceleration
        if (math.lengthsq(force) > agent.maxAccel * agent.maxAccel)
        {
            force = math.normalize(force) * agent.maxAccel;
        }

        Forces[index] = force;
    }
}

public class ForceSolverJobs : MonoBehaviour
{
    private NativeArray<ForceAgentData> agentDataArray;
    private NativeArray<float3> forcesArray;
    private NativeParallelMultiHashMap<int, int> _spatialHashMap;
    private bool _initialized = false;
    public float gridCellSize = 4.0f;

    public void Init(List<AgentBase> agents)
    {
        if (_initialized) DisposeData();

        int count = agents.Count;
        agentDataArray = new NativeArray<ForceAgentData>(count, Allocator.Persistent);
        forcesArray = new NativeArray<float3>(count, Allocator.Persistent);
        _spatialHashMap = new NativeParallelMultiHashMap<int, int>(count, Allocator.Persistent);

        _initialized = true;
    }

    private void DisposeData()
    {
        if (agentDataArray.IsCreated) agentDataArray.Dispose();
        if (forcesArray.IsCreated) forcesArray.Dispose();
        if (_spatialHashMap.IsCreated) _spatialHashMap.Dispose();
        _initialized = false;
    }

    private void OnDestroy()
    {
        DisposeData();
    }

    public void Solve(List<AgentBase> agents)
    {
        int count = agents.Count;
        if (count == 0) return;

        if (!_initialized || agentDataArray.Length != count)
        {
            Init(agents);
        }

        for (int i = 0; i < count; i++)
        {
            ForceAgent fa = agents[i] as ForceAgent;
            agentDataArray[i] = new ForceAgentData
            {
                position = fa.transform.position,
                velocity = fa.velocity,
                goalPosition = fa.goal != null ? fa.goal.position : fa.transform.position,
                radius = fa.radius,
                maxAccel = fa.maxAccel,
                prefSpeed = fa.prefSpeed,
                goalRadius = fa.goalRadius,
                ksi = fa.ksi,
                k = fa.k,
                m = fa.m,
                t0 = fa.t0
            };
        }

        JobHandle handle = new ClearHashMapJob { HashMap = _spatialHashMap }.Schedule();

        handle = new ForceHashPositionsJob
        {
            Agents = agentDataArray,
            HashMap = _spatialHashMap.AsParallelWriter(),
            CellSize = gridCellSize
        }.Schedule(count, 64, handle);

        ForceCalculationJob job = new ForceCalculationJob
        {
            Agents = agentDataArray,
            HashMap = _spatialHashMap,
            Forces = forcesArray,
            CellSize = gridCellSize,
            deltaTime = Time.fixedDeltaTime,
            epsilon = 0.0001f
        };

        handle = job.Schedule(count, 64, handle);
        handle.Complete();

        for (int i = 0; i < count; i++)
        {
            ForceAgent fa = agents[i] as ForceAgent;

            if (fa.goal != null)
            {
                Vector3 diff = fa.goal.position - fa.transform.position;
                diff.y = 0;

                if (diff.sqrMagnitude < fa.goalRadius * fa.goalRadius)
                {
                    fa.velocity = Vector3.zero;
                    fa.currentGoal = fa.GetNextGoalIndex();
                    fa.goal = fa.goals[fa.currentGoal];
                }
            }

            fa.F = forcesArray[i];

            // Wall repulsion — applied on main thread since Burst can't access managed Bounds list.
            // Same exponential decay model as agent-agent avoidance, weighted heavier for walls.
            var walls = SimulationManager.Instance?.WallBounds;
            if (walls != null)
            {
                Vector3 pos = fa.transform.position;
                foreach (var wall in walls)
                {
                    Vector3 closest = wall.ClosestPoint(pos);
                    Vector3 away    = pos - closest;
                    away.y = 0;
                    float dist = away.magnitude;
                    if (dist < 0.001f || dist > fa.t0 * 4f) continue;
                    fa.F += away.normalized * (fa.k * 2f * Mathf.Exp(-dist / fa.t0));
                }
            }

            fa.UpdateAgent();
        }
    }
}
