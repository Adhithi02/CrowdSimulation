using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.AI;

// Manages the flock. Mirrors Flock.cpp + KDTree spatial queries.
// Now fully integrated with Position-Based Dynamics (PBD) per Tomer Weiss (2019).
public class FlockManager : MonoBehaviour
{
    [Header("Prefabs")]
    public GameObject boidPrefab;     // KinematicSteering is required on this
    public GameObject predatorPrefab; // optional; null = reuse boidPrefab

    [Header("Flock Size")]
    public int flockSize = 20;
    public int predatorCount = 0;

    [Header("Goal")]
    public Transform[] goals;

    [Header("Spawn Area (XZ plane)")]
    public float spawnWidth = 20f;
    public float spawnDepth = 20f;

    [Header("Different Solvers")]
    public bool usePBD;
    public bool useMultiThreading = true; // Add this toggle

    [Header("PBD Solver Settings")]
    [Tooltip("Number of iterations to resolve collisions. Higher = stiffer crowds but costs more CPU.")]
    public int pbdIterations = 3;

    [Tooltip("Velocity interpolation factor (Ksi). Default is 0.05 per Weiss paper.")]
    public float ksi = 0.05f;

    // All boids tracked by this manager
    private readonly List<AgentBase> boids = new();

    // Spatial grid — replaces KDTree for O(1)-per-cell neighbour queries
    private readonly Dictionary<(int, int), List<AgentBase>> grid = new();
    private float cellSize = 4f;

    public enum PBDSolverType
    {
        SingleThreaded,
        MultiThreaded_Standard,
        MultiThreaded_CLOD
    }
    [Header("Different Solvers")]
    public PBDSolverType activeSolver = PBDSolverType.MultiThreaded_CLOD;

    private PBDSolver _pbdSolver;
    private PBDSolverJobs _standardJobsSolver; // Add this
    private CLODPBDSolverJobs _clodJobsSolver = new CLODPBDSolverJobs(); private ForceSolver _forceSolver;
    private ForceSolverJobs _forceSolverJobs; // Add this
    private Transform protag; // Add this for the protagonist's position

    // -------------------------------------------------------------------------
    void Start()
    {
        goals = GameObject.FindGameObjectWithTag("Goals")
            .GetComponentsInChildren<Transform>()
            .Where(t => t != GameObject.FindGameObjectWithTag("Goals").transform)
            .ToArray();
        _pbdSolver = GetComponent<PBDSolver>();
        _standardJobsSolver = GetComponent<PBDSolverJobs>();

        _forceSolver = GetComponent<ForceSolver>();
        _forceSolverJobs = GetComponent<ForceSolverJobs>();
        SpawnFlock();
        _standardJobsSolver.Init(boids);
        _forceSolverJobs.Init(boids);


        // Protag position
        protag = predatorPrefab.transform;
    }

    void FixedUpdate()
    {
        if (ProfilerManager.Instance != null)
            ProfilerManager.Instance.BeginSample("Frame");

        if (ProfilerManager.Instance != null)
            ProfilerManager.Instance.BeginSample("NeighborQuery");
        // Phase 1: spatial queries
        RebuildGrid();
        foreach (var boid in boids)
        {
            QueryNeighbours(boid);
        }

        if (ProfilerManager.Instance != null)
            ProfilerManager.Instance.EndSample("NeighborQuery");

        if (ProfilerManager.Instance != null)
            ProfilerManager.Instance.BeginSample("Solver");

        Vector3 focusPoint = protag.position;
        // Phase 2: Force base or PBD-style separation (position projection)
        if (!usePBD)
        {
            if (useMultiThreading)
                _forceSolverJobs.Solve(boids); // Multithreaded Force
            else
                _forceSolver.Solve(boids); // Original Singlethreaded
        }
        else
        {
            float dt = Time.fixedDeltaTime;

            if (activeSolver == PBDSolverType.MultiThreaded_Standard)
            {
                // Uses your original PBDSolverJobs.cs
                _standardJobsSolver.Solve(boids, dt, pbdIterations);
            }
            else if (activeSolver == PBDSolverType.MultiThreaded_CLOD)
            {
                // Uses the new CLOD solver
                _clodJobsSolver.Solve(boids, dt, pbdIterations, focusPoint);
            }
            else if (activeSolver == PBDSolverType.SingleThreaded)
            {
                // Fallback to original single threaded logic if needed
                _pbdSolver.Solve(boids, Time.fixedDeltaTime, pbdIterations); // Original Singlethreaded
            }
        }

        if (ProfilerManager.Instance != null)
            ProfilerManager.Instance.EndSample("Solver");

        if (ProfilerManager.Instance != null)
        {
            ProfilerManager.Instance.EndSample("Frame");

            // Log the frame's data to the CSV (requires frameCount)
            // ProfilerManager.Instance.LogFrameData(Time.frameCount);
        }
    }

    // -------------------------------------------------------------------------
    // Spawning
    // -------------------------------------------------------------------------
    void SpawnFlock()
    {
        for (int i = 0; i < flockSize; i++)
            SpawnBoid(boidPrefab, isPredator: false);


        // for (int i = 0; i < predatorCount; i++)
        // {
        //     var prefab = predatorPrefab != null ? predatorPrefab : boidPrefab;
        //     SpawnBoid(prefab, isPredator: true);
        // }

        foreach (var b in boids)
            cellSize = Mathf.Max(cellSize, b.perceptionRadius);
    }

    void SpawnBoid(GameObject prefab, bool isPredator)
    {
        Vector3 spawnPos = RandomNavMeshPoint();
        Quaternion rot = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
        var go = Instantiate(prefab, spawnPos, rot, transform);

        // Re-added, but fixed: Assign scene goals dynamically to the instantiated prefab

        if (usePBD)
        {
            var pbdAgent = go.AddComponent<PBDAgent>();
            pbdAgent.PbdIterations = pbdIterations;
            pbdAgent.goals = goals;
        }
        else
        {
            go.AddComponent<ForceAgent>().goals = goals;
        }
        var boid = go.GetComponent<AgentBase>();
        boids.Add(boid);
    }

    Vector3 RandomNavMeshPoint()
    {
        for (int attempt = 0; attempt < 10; attempt++)
        {
            Vector3 candidate = transform.position + new Vector3(
                Random.Range(-spawnWidth * 0.5f, spawnWidth * 0.5f),
                0f,
                Random.Range(-spawnDepth * 0.5f, spawnDepth * 0.5f)
            );
            if (NavMesh.SamplePosition(candidate, out var hit, 2f, NavMesh.AllAreas))
                return hit.position;
        }
        return transform.position;
    }

    // -------------------------------------------------------------------------
    // Spatial Grid 
    // -------------------------------------------------------------------------
    (int, int) GridCell(Vector3 pos) => (Mathf.FloorToInt(pos.x / cellSize), Mathf.FloorToInt(pos.z / cellSize));

    void RebuildGrid()
    {
        foreach (var list in grid.Values) list.Clear();

        foreach (var boid in boids)
        {
            var cell = GridCell(boid.transform.position);
            if (!grid.TryGetValue(cell, out var list))
            {
                list = new List<AgentBase>();
                grid[cell] = list;
            }
            list.Add(boid);
        }
    }

    void QueryNeighbours(AgentBase boid)
    {
        boid.neighbours.Clear();
        float r = boid.perceptionRadius;
        float rSq = r * r;
        Vector3 pos = boid.transform.position;

        int minCx = Mathf.FloorToInt((pos.x - r) / cellSize);
        int maxCx = Mathf.FloorToInt((pos.x + r) / cellSize);
        int minCz = Mathf.FloorToInt((pos.z - r) / cellSize);
        int maxCz = Mathf.FloorToInt((pos.z + r) / cellSize);

        for (int x = minCx; x <= maxCx; x++)
        {
            for (int z = minCz; z <= maxCz; z++)
            {
                if (grid.TryGetValue((x, z), out var cellBoids))
                {
                    foreach (var other in cellBoids)
                    {
                        if (other == boid) continue;
                        if ((other.transform.position - pos).sqrMagnitude < rSq)
                        {
                            boid.neighbours.Add(other);
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    [Header("Debug")]
    public bool drawGrid = true;

    void OnDrawGizmos()
    {
        // Spawn area outline
        Gizmos.color = new Color(1f, 1f, 0f, 0.3f);
        Gizmos.DrawWireCube(transform.position, new Vector3(spawnWidth, 0.5f, spawnDepth));

        if (!drawGrid || !Application.isPlaying || grid.Count == 0) return;

        foreach (var kvp in grid)
        {
            int count = kvp.Value.Count;
            if (count == 0) continue;

            var (cx, cz) = kvp.Key;
            Vector3 center = new Vector3(
                (cx + 0.5f) * cellSize,
                0.1f,
                (cz + 0.5f) * cellSize
            );
            Vector3 size = new Vector3(cellSize, 0.05f, cellSize);

            float t = Mathf.Clamp01(count / 8f);
            Gizmos.color = Color.Lerp(new Color(0, 1, 0, 0.1f), new Color(1, 0, 0, 0.5f), t);
            Gizmos.DrawCube(center, size);
            Gizmos.color = new Color(0, 0, 0, 0.2f);
            Gizmos.DrawWireCube(center, size);
        }
    }
}
