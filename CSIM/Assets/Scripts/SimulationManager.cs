using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.AI;

// ---------------------------------------------------------------------------
// Test-scene presets
// ---------------------------------------------------------------------------
public enum ScenePreset
{
    Random,         // random spawn using Goals tag (default)
    Circle,         // ring of agents — all walk to centre (convergence test)
    Crossing,       // block south-west → goal north-east (diagonal open-field walk)
    Bidirectional,  // block at west end of corridor → goal east end (corridor squeeze)
    Bottleneck,     // block south of wall → goal north, squeeze through gap
    DoubleBottleneck, // two bottlenecks in sequence
    MergeLane       // two lanes merge into one corridor
}

// ---------------------------------------------------------------------------
// All available solver modes in one enum
// ---------------------------------------------------------------------------
public enum SolverMode
{
    Force,              // Helbing-style anticipatory forces, single-threaded
    ForceJobs,          // Same, parallelised with Unity Jobs + Burst
    PBD,                // Position-Based Dynamics (Weiss 2019), single-threaded
    PBDJobs,            // PBD, parallelised with Unity Jobs + Burst
    PBDCLODJobs,        // PBD + Camera-LOD: high/med/low quality tiers by distance
    PBDCompute,         // GPU compute-shader solver with built-in LOD
    PBDCLODCullingJobs, // CLOD + frustum culling: disables Renderer+Animator for off-screen agents
    PBDCuda             // Native NVIDIA CUDA C++ backend
}

/// <summary>
/// Central simulation manager for the crowd.
/// Replaces FlockManager — attach this (and its sibling solver components)
/// to the same GameObject.
///
/// Required sibling components: PBDSolver, PBDSolverJobs, CLODPBDSolverJobs,
///                               ForceSolver, ForceSolverJobs
/// </summary>
[DefaultExecutionOrder(-10)]
public class SimulationManager : MonoBehaviour
{
    public static SimulationManager Instance { get; private set; }

    // -----------------------------------------------------------------------
    // Inspector — Prefabs & Scene
    // -----------------------------------------------------------------------

    [Header("Prefabs")]
    public GameObject agentPrefab;
    public GameObject protagonistPrefab;   // used as focus point for CLOD LOD

    [Header("Spawn")]
    public int agentCount = 50;
    public float spawnWidth = 20f;
    public float spawnDepth = 20f;

    [Header("Scene Preset")]
    public ScenePreset activePreset = ScenePreset.Random;

    // -----------------------------------------------------------------------
    // Inspector — Solver Selection
    // -----------------------------------------------------------------------

    [Header("Solver")]
    public SolverMode activeSolver = SolverMode.PBDCuda;

    // -----------------------------------------------------------------------
    // Inspector — Shared Agent Parameters (live-synced every frame)
    // -----------------------------------------------------------------------

    [Header("Shared Agent Parameters")]
    public float agentSpeed = 5f;
    public float agentRadius = 0.5f;
    public float neighborDist = 4f;
    public float agentMass = 1f;
    public float maxAccel = 5f;

    // -----------------------------------------------------------------------
    // Inspector — PBD Parameters
    // -----------------------------------------------------------------------

    [Header("PBD Parameters")]
    [Range(1, 20)] public int pbdIterations = 3;
    [Range(0.01f, 1f)] public float pbdKsi = 0.05f;

    // -----------------------------------------------------------------------
    // Inspector — Force Parameters
    // -----------------------------------------------------------------------

    [Header("Force Parameters")]
    public float forceK = 1f;
    [Range(0.01f, 1f)] public float forceKsi = 0.05f;
    public float forceM = 2f;
    public float forceT0 = 3f;

    // -----------------------------------------------------------------------
    // Inspector — CLOD Distances
    // -----------------------------------------------------------------------

    [Header("CLOD LOD Distances")]
    public float clodHighDist = 20f;
    public float clodMedDist = 50f;

    // -----------------------------------------------------------------------
    // Inspector — Importance-Driven Dynamic Fidelity
    // -----------------------------------------------------------------------

    [Header("Importance Scoring Weights")]
    [Range(0f, 1f)] public float weightDistance = 0.30f;
    [Range(0f, 1f)] public float weightVisibility = 0.25f;
    [Range(0f, 1f)] public float weightInteraction = 0.20f;
    [Range(0f, 1f)] public float weightDensity = 0.25f;

    [Header("Importance Fidelity Thresholds")]
    [Tooltip("Importance >= this assigns HIGH fidelity (0).")]
    [Range(0f, 1f)] public float highImportanceThreshold = 0.60f;
    [Tooltip("Importance >= this (and < high) assigns MED fidelity (1); below assigns LOW (2).")]
    [Range(0f, 1f)] public float medImportanceThreshold = 0.30f;
    [Tooltip("Hysteresis band around importance thresholds to prevent tier flickering.")]
    [Range(0f, 0.2f)] public float importanceHysteresis = 0.05f;
    [Tooltip("Distance at which distance-importance decays to 0.")]
    public float maxImportanceDistance = 60f;
    [Tooltip("Interaction contact count that saturates interaction-importance to 1.")]
    public int maxInteractionContacts = 6;
    [Tooltip("Local cell density that saturates density-importance to 1.")]
    public int maxImportanceDensity = 12;

    // -----------------------------------------------------------------------
    // Inspector — Debug
    // -----------------------------------------------------------------------

    [Header("Debug & Metro Visualization")]
    public bool drawGrid = true;
    [Tooltip("Dynamically tints agents based on live GPU/Job fidelity tier (Green=High, Yellow=Med, Blue=Low)")]
    public bool showFidelityColors = false;

    public enum CameraViewMode
    {
        VRCommuterEyeLevel,   // Immersive 1.7m eye-level perspective inside the crowd
        StationOverheadView   // 60-degree angled overhead benchmark view
    }

    [Header("VR & Station Camera Mode")]
    public CameraViewMode cameraViewMode = CameraViewMode.VRCommuterEyeLevel;
    public bool enforceBenchmarkCameraProtocol = false;
    public Camera benchmarkCamera;
    public bool forceAngledTopBenchmarkView = true;
    [Range(45f, 85f)] public float benchmarkPitchDegrees = 60f;
    public float benchmarkYawDegrees = 180f;
    public float benchmarkCameraHeight = 70f;
    public float benchmarkOrthoSize = 32f;
    public int benchmarkCameraRecheckEveryNFrames = 120;

    // -----------------------------------------------------------------------
    // Private state
    // -----------------------------------------------------------------------

    private readonly List<AgentBase> agents = new();
    private readonly Dictionary<(int, int), List<AgentBase>> grid = new();
    private float cellSize = 4f;
    private Transform[] goals;

    // Temporary goal GameObjects created by scene presets (destroyed on next Respawn)
    private readonly List<GameObject> _tempGoals = new();

    // Wall/obstacle GameObjects created by geometry presets (destroyed on next Respawn)
    private readonly List<GameObject> _geometryObjects = new();

    // World-space bounds of every wall, used for post-solver obstacle repulsion.
    // Treated like stationary agents: any real agent inside or too close gets pushed out.
    private readonly List<Bounds> _wallBounds = new();

    /// <summary>Read-only view of current wall bounds — used by solvers for obstacle forces.</summary>
    public IReadOnlyList<Bounds> WallBounds => _wallBounds;

    // Cached spawn-area centre — computed ONCE per Respawn on the clean NavMesh
    // BEFORE any walls are built.  Avoids goal/wall misalignment caused by
    // NavMesh carving shifting the nearest-point result between calls.
    private Vector3 _spawnCenter;

    // Protagonist / focus point for CLOD
    private Transform focusPoint;

    // Solver components — expected as siblings on the same GameObject
    private PBDSolver _pbdSolver;
    private PBDSolverJobs _pbdJobsSolver;
    // private CLODPBDSolverJobs _clodSolver = new();
    private CLODPBDSolverJobs _clodSolver;
    private ForceSolver _forceSolver;
    private ForceSolverJobs _forceJobsSolver;
    private PBDComputeSolver _computeSolver;
    private PBDCudaSolver _cudaSolver;
    private CLODPBDCullingSolver _cullingSolver;

    // Change detection
    private SolverMode _lastSolver;

    // -----------------------------------------------------------------------
    // Unity lifecycle
    // -----------------------------------------------------------------------

    void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    void Start()
    {
        if (GetComponent<MetroStationManager>() == null)
            gameObject.AddComponent<MetroStationManager>();

        CacheGoals();
        CacheSolvers();
        SetupFocusPoint();
        ResolveBenchmarkCamera();
        SetupPresentationLod();
        _lastSolver = activeSolver;
        Respawn();
    }

    void Update()
    {
        if (enforceBenchmarkCameraProtocol &&
            benchmarkCameraRecheckEveryNFrames > 0 &&
            Time.frameCount % benchmarkCameraRecheckEveryNFrames == 0)
            EnforceBenchmarkCameraProtocol();

        // Detect Inspector solver changes during Play mode
        if (activeSolver != _lastSolver)
        {
            _lastSolver = activeSolver;
            Respawn();
        }

        if (showFidelityColors && Time.frameCount % 4 == 0)
        {
            UpdateFidelityVisuals();
        }

        ProfilerManager.Instance?.LogFrameData(Time.frameCount);
    }

    void UpdateFidelityVisuals()
    {
        for (int i = 0; i < agents.Count; i++)
        {
            var a = agents[i];
            if (a == null) continue;
            Color col = a.fidelityLevel switch
            {
                0 => new Color(0.2f, 1f, 0.35f), // Neon Green for HIGH (foveated / dense bottleneck)
                1 => new Color(1f, 0.85f, 0.1f), // Amber Yellow for MED
                _ => new Color(0.3f, 0.65f, 1f)  // Cyan Blue for LOW (peripheral / sparse)
            };
            ColorAgent(a, col);
        }
    }

    void FixedUpdate()
    {
        if (agents.Count == 0) return;

        var pm = ProfilerManager.Instance;
        pm?.BeginSample("Frame");

        pm?.BeginSample("SyncParams");
        SyncParams();
        pm?.EndSample("SyncParams");

        pm?.BeginSample("RefreshDestinations");
        RefreshDestinations();
        pm?.EndSample("RefreshDestinations");

        // PBDJobs, PBDCLODJobs, PBDCompute, and PBDCLODCullingJobs all build their own
        // internal spatial hashmaps inside the Job/GPU pipeline — the main-thread
        // neighbour query is wasted work for those modes. Force, ForceJobs, and PBD
        // (single-threaded) still need the main-thread grid because their solvers read
        // agent.neighbours.
        bool needsNeighbourQuery = activeSolver == SolverMode.Force ||
                                   activeSolver == SolverMode.ForceJobs ||
                                   activeSolver == SolverMode.PBD;

        if (needsNeighbourQuery)
        {
            pm?.BeginSample("NeighborQuery");
            RebuildGrid();
            foreach (var a in agents) QueryNeighbours(a);
            pm?.EndSample("NeighborQuery");
        }

        pm?.BeginSample("Solver");
        DispatchSolver();
        pm?.EndSample("Solver");

        // Post-step: push any agent that overlaps or grazes a wall back outside it.
        // Run 3 passes — PBD-Jobs accumulates wall-bound velocity over its iterations,
        // so a single pass can be overwhelmed by crowd pressure on the next frame.
        if (_wallBounds.Count > 0)
            for (int wallPass = 0; wallPass < 3; wallPass++)
                PushAgentsFromWalls();

        pm?.EndSample("Frame");
    }

    // -----------------------------------------------------------------------
    // Spawning
    // -----------------------------------------------------------------------

    [ContextMenu("Respawn")]
    public void Respawn()
    {
        StopAllCoroutines();
        StartCoroutine(DoRespawn());
    }

    IEnumerator DoRespawn()
    {
        var pm = ProfilerManager.Instance;
        pm?.BeginSample("Respawn");

        // Destroy old agents
        foreach (var a in agents)
            if (a != null) Destroy(a.gameObject);
        agents.Clear();
        grid.Clear();

        // Destroy previous temporary goals and geometry
        foreach (var g in _tempGoals)       if (g != null) Destroy(g);
        foreach (var g in _geometryObjects) if (g != null) Destroy(g);
        _tempGoals.Clear();
        _geometryObjects.Clear();
        _wallBounds.Clear();

        // Cache the spawn centre NOW on the clean NavMesh — before walls are added.
        // Every spawn/geometry method uses _spawnCenter so they're all consistent.
        _spawnCenter = Center();
        if (enforceBenchmarkCameraProtocol)
        {
            ApplyBenchmarkCameraView();
            EnforceBenchmarkCameraProtocol();
        }

        // Build preset geometry (walls/obstacles) first
        BuildPresetGeometry();
        MetroStationManager.Instance?.BuildMetroStation3D();

        // Give the NavMesh obstacle carver a few frames to cut the holes.
        // One frame is rarely enough on large meshes.
        if (_geometryObjects.Count > 0)
        {
            yield return null;
            yield return null;
            yield return new WaitForSeconds(0.1f);   // thicker walls need a bit more carving time
        }

        // Now spawn agents into the (updated) NavMesh
        // The 4 corner goal capsules in the scene are only relevant for Random.
        // Hide them for all other presets so they don't clutter obstacle scenes.
        bool useCornerGoals = activePreset == ScenePreset.Random;
        if (goals != null)
            foreach (var g in goals)
                if (g != null) g.gameObject.SetActive(useCornerGoals);

        switch (activePreset)
        {
            case ScenePreset.Circle:        SpawnCircle();        break;
            case ScenePreset.Crossing:      SpawnCrossing();      break;
            case ScenePreset.Bidirectional: SpawnBidirectional(); break;
            case ScenePreset.Bottleneck:    SpawnBottleneck();    break;
            case ScenePreset.DoubleBottleneck: SpawnDoubleBottleneck(); break;
            case ScenePreset.MergeLane:     SpawnMergeLane();     break;
            default:                        SpawnRandom();        break;
        }

        cellSize = Mathf.Max(4f, neighborDist);

        _pbdJobsSolver?.Init(agents);
        _forceJobsSolver?.Init(agents);
        _clodSolver?.Init(agents);
        _computeSolver?.Init(agents);
        _cudaSolver?.Init(agents);
        _cullingSolver?.Init(agents);

        FindObjectOfType<FocusPointSphere>()?.RefreshAgents();

        // MetroStationManager.Start can run before this coroutine has spawned agents.
        // Re-apply the selected view here so CCTV follows the live crowd bounds.
        if (MetroStationManager.Instance != null)
            MetroStationManager.Instance.ApplyCCTVAngle(MetroStationManager.Instance.currentCCTV);

        GetComponent<PBDCudaVisualLodController>()?.RefreshAgents();
        pm?.EndSample("Respawn");
    }

    // -----------------------------------------------------------------------
    // Preset geometry — walls / obstacles built before agents spawn
    // -----------------------------------------------------------------------

    void BuildPresetGeometry()
    {
        switch (activePreset)
        {
            case ScenePreset.Bottleneck:    BuildBottleneckWalls();    break;
            case ScenePreset.Bidirectional: BuildCorridorWalls();      break;
            case ScenePreset.DoubleBottleneck: BuildDoubleBottleneckWalls(); break;
            case ScenePreset.MergeLane:     BuildMergeLaneWalls();     break;
            // Circle / Crossing / Random — open space, no geometry needed
        }
    }

    // Wall barrier running along the X axis at Z=center, gap at X=center.
    // Agents spawn south of the wall and their goal is north — everyone squeezes
    // through the single gap.
    // Metro Turnstiles & Fare-Gate Barriers
    void BuildBottleneckWalls()
    {
        Vector3 c = _spawnCenter;

        float gap      = Mathf.Max(spawnWidth * 0.15f, agentRadius * 8f);
        float wallH    = 6f;
        float wallThick = 8f;
        float wingLen   = 500f;

        MakeWall("BN_Wall_L",
            c + new Vector3(-gap * 0.5f - wingLen * 0.5f, wallH * 0.5f, 0f),
            new Vector3(wingLen, wallH, wallThick));

        MakeWall("BN_Wall_R",
            c + new Vector3( gap * 0.5f + wingLen * 0.5f, wallH * 0.5f, 0f),
            new Vector3(wingLen, wallH, wallThick));

        // ── Metro Props: Turnstile Glass Gates, Pedestals, Overhead Line Sign ──
        MakeStationProp("Metro_Overhead_Sign_BN", c + new Vector3(0f, 7.5f, 0f), new Vector3(gap + 6f, 1.4f, 0.4f), new Color(0.1f, 0.65f, 0.25f));
        MakeStationProp("Metro_Turnstile_Floor_Line", c + new Vector3(0f, 0.02f, -2.5f), new Vector3(gap + 4f, 0.02f, 0.8f), new Color(0.96f, 0.82f, 0.05f));
        MakeStationProp("Metro_Pedestal_L", c + new Vector3(-gap * 0.5f, 1.2f, 0f), new Vector3(0.6f, 2.4f, 2.5f), new Color(0.18f, 0.2f, 0.22f), true);
        MakeStationProp("Metro_Pedestal_R", c + new Vector3( gap * 0.5f, 1.2f, 0f), new Vector3(0.6f, 2.4f, 2.5f), new Color(0.18f, 0.2f, 0.22f), true);
        MakeStationProp("Metro_GlassFlap_L", c + new Vector3(-gap * 0.25f, 1.1f, 0f), new Vector3(gap * 0.45f, 1.3f, 0.12f), new Color(0.25f, 0.85f, 0.95f));
        MakeStationProp("Metro_GlassFlap_R", c + new Vector3( gap * 0.25f, 1.1f, 0f), new Vector3(gap * 0.45f, 1.3f, 0.12f), new Color(0.25f, 0.85f, 0.95f));
    }

    // Metro Subway Platform & Train Boarding
    void BuildCorridorWalls()
    {
        Vector3 c = _spawnCenter;
        float corridorW = Mathf.Max(agentRadius * 30f, 14f);
        float wallH     = 4f;
        float wallThick = 2f;
        float wallLen   = 500f;

        // North barrier: Subway Train Car body with Red Line livery
        MakeWall("BD_Wall_N",
            c + new Vector3(0f, wallH * 0.5f,  corridorW * 0.5f + wallThick * 0.5f),
            new Vector3(wallLen, wallH, wallThick));

        // South barrier: Station concourse wall
        MakeWall("BD_Wall_S",
            c + new Vector3(0f, wallH * 0.5f, -corridorW * 0.5f - wallThick * 0.5f),
            new Vector3(wallLen, wallH, wallThick));

        // ── Metro Props: Yellow Safety Edge, Train Roof, Platform Overhead Signs ──
        MakeStationProp("Metro_Platform_Safety_Line", c + new Vector3(0f, 0.02f, corridorW * 0.5f - 0.5f), new Vector3(wallLen, 0.02f, 0.5f), new Color(0.96f, 0.82f, 0.05f));
        MakeStationProp("Metro_Train_Roof", c + new Vector3(0f, wallH + 0.6f, corridorW * 0.5f + wallThick * 0.5f), new Vector3(wallLen, 0.8f, wallThick + 2.5f), new Color(0.22f, 0.25f, 0.28f));
        MakeStationProp("Metro_Train_RedStripe", c + new Vector3(0f, wallH * 0.7f, corridorW * 0.5f - 0.05f), new Vector3(wallLen, 0.6f, 0.1f), new Color(0.85f, 0.15f, 0.15f));
        MakeStationProp("Metro_Platform_Sign_1", c + new Vector3(-25f, 5.5f, 0f), new Vector3(14f, 1.3f, 0.4f), new Color(0.12f, 0.2f, 0.35f));
        MakeStationProp("Metro_Platform_Sign_2", c + new Vector3( 25f, 5.5f, 0f), new Vector3(14f, 1.3f, 0.4f), new Color(0.12f, 0.2f, 0.35f));
    }

    void BuildDoubleBottleneckWalls()
    {
        Vector3 c = _spawnCenter;
        float gap = Mathf.Max(spawnWidth * 0.16f, agentRadius * 8f);
        float wallH = 6f;
        float wallThick = 8f;
        float wingLen = 500f;

        MakeGapBarrier("DBN_A", c.z - 10f, c.x - gap * 0.45f, gap, wallH, wallThick, wingLen);
        MakeGapBarrier("DBN_B", c.z + 10f, c.x + gap * 0.45f, gap, wallH, wallThick, wingLen);

        MakeStationProp("Metro_Security_Sign_A", c + new Vector3(c.x - gap * 0.45f, 7.5f, c.z - 10f), new Vector3(gap + 4f, 1.2f, 0.4f), new Color(0.15f, 0.45f, 0.75f));
        MakeStationProp("Metro_Security_Sign_B", c + new Vector3(c.x + gap * 0.45f, 7.5f, c.z + 10f), new Vector3(gap + 4f, 1.2f, 0.4f), new Color(0.75f, 0.25f, 0.15f));
    }

    void BuildMergeLaneWalls()
    {
        Vector3 c = _spawnCenter;
        float wallH = 6f;
        float wallThick = 2f;
        float laneLen = 500f;
        float corridorHalf = Mathf.Max(agentRadius * 11f, 6f);

        MakeWall("MG_Outer_W",
            c + new Vector3(-corridorHalf - wallThick * 0.5f, wallH * 0.5f, 0f),
            new Vector3(wallThick, wallH, laneLen));
        MakeWall("MG_Outer_E",
            c + new Vector3(corridorHalf + wallThick * 0.5f, wallH * 0.5f, 0f),
            new Vector3(wallThick, wallH, laneLen));

        float dividerLen = 90f;
        float dividerCenterZ = c.z - 60f;
        MakeWall("MG_Divider",
            new Vector3(c.x, wallH * 0.5f, dividerCenterZ),
            new Vector3(wallThick, wallH, dividerLen));

        // ── Metro Props: Concourse Staircase Divider Signs & Floor Merge Line ──
        MakeStationProp("Metro_Stairs_Sign", new Vector3(c.x, 6.2f, dividerCenterZ + dividerLen * 0.4f), new Vector3(12f, 1.3f, 0.4f), new Color(0.15f, 0.45f, 0.75f));
        MakeStationProp("Metro_Merge_Floor_Line", new Vector3(c.x, 0.02f, c.z - 10f), new Vector3(0.5f, 0.02f, 25f), new Color(0.96f, 0.82f, 0.05f));
    }

    void MakeGapBarrier(string prefix, float centerZ, float gapCenterX, float gapWidth, float wallH, float wallThick, float wingLen)
    {
        MakeWall($"{prefix}_L",
            new Vector3(gapCenterX - gapWidth * 0.5f - wingLen * 0.5f, wallH * 0.5f, centerZ),
            new Vector3(wingLen, wallH, wallThick));
        MakeWall($"{prefix}_R",
            new Vector3(gapCenterX + gapWidth * 0.5f + wingLen * 0.5f, wallH * 0.5f, centerZ),
            new Vector3(wingLen, wallH, wallThick));
    }

    // Creates a visible cube + NavMeshObstacle (carving) at the given position/size.
    void MakeWall(string name, Vector3 pos, Vector3 size)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.SetParent(transform);
        go.transform.position   = pos;
        go.transform.localScale = size;

        // Metro-themed station architecture coloring
        var mr = go.GetComponent<MeshRenderer>();
        if (mr != null)
        {
            if (name.Contains("BN") || name.Contains("DBN"))
                mr.material.color = new Color(0.25f, 0.35f, 0.45f); // Turnstiles / Fare-Gate barriers
            else if (name.Contains("BD"))
                mr.material.color = new Color(0.75f, 0.22f, 0.18f); // Subway Train & Platform Edge (Red Line)
            else if (name.Contains("MG"))
                mr.material.color = new Color(0.2f, 0.45f, 0.7f);   // Concourse Stairs & Escalator divider
            else
                mr.material.color = new Color(0.85f, 0.85f, 0.85f);
        }

        // Carves a hole in the NavMesh so agents path around / through the gap
        var obs = go.AddComponent<NavMeshObstacle>();
        obs.carving = true;
        obs.shape   = NavMeshObstacleShape.Box;
        obs.size    = Vector3.one;   // matches localScale

        _geometryObjects.Add(go);

        // Register world-space bounds so PushAgentsFromWalls() can treat this
        // wall as a stationary obstacle in the same way solvers treat agents.
        _wallBounds.Add(new Bounds(pos, size));
    }

    // Creates metro station props (signs, platform edges, turnstile glass, ticket kiosks)
    void MakeStationProp(string name, Vector3 pos, Vector3 size, Color color, bool carveNavMesh = false)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.SetParent(transform);
        go.transform.position   = pos;
        go.transform.localScale = size;

        var mr = go.GetComponent<MeshRenderer>();
        if (mr != null) mr.material.color = color;

        if (carveNavMesh)
        {
            var obs = go.AddComponent<NavMeshObstacle>();
            obs.carving = true;
            obs.shape   = NavMeshObstacleShape.Box;
            obs.size    = Vector3.one;
            _wallBounds.Add(new Bounds(pos, size));
        }
        else
        {
            var col = go.GetComponent<Collider>();
            if (col != null) Destroy(col);
        }

        _geometryObjects.Add(go);
    }

    // -----------------------------------------------------------------------
    // Preset spawn patterns
    // -----------------------------------------------------------------------
    //
    //  ALL presets follow the same simple model:
    //    • One block of agents on one side (south / west / ring)
    //    • One goal on the other side (north / east / centre)
    //    • Scene GEOMETRY (walls) forces the interesting crowd behaviour
    //  No multi-group spawning — the walls create the scenario, not the spawn.
    // -----------------------------------------------------------------------

    float AgentSpacing() => Mathf.Max(agentRadius * 2.5f, 1.2f);

    // Scatter agents randomly, navigate to the 4 existing scene corner goals
    // (the white capsules already in the scene tagged "Goals").
    // No geometry, no spawned goal spheres.
    void SpawnRandom()
    {
        for (int i = 0; i < agentCount; i++)
        {
            var agent = SpawnAgent(SampleNavMesh());
            if (goals != null && goals.Length > 0) agent.Goals = goals;
        }
    }

    // Agents on a ring — each walks to its ANTIPODAL point (directly opposite).
    // They all cross through the centre simultaneously.
    // Per-agent invisible goals (no cluttered goal spheres).
    void SpawnCircle()
    {
        float   radius = Mathf.Min(spawnWidth, spawnDepth) * 0.42f;
        Vector3 c      = _spawnCenter;
        Color   col    = new Color(0.9f, 0.55f, 0.1f);

        for (int i = 0; i < agentCount; i++)
        {
            float   angle = i * Mathf.PI * 2f / agentCount;
            Vector3 dir   = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));

            // Invisible goal at antipodal point — just a plain Transform
            var goalGO = new GameObject($"CGoal_{i}");
            goalGO.transform.SetParent(transform);
            goalGO.transform.position = c - dir * radius;
            _tempGoals.Add(goalGO);

            var agent = SpawnAgent(SnapNavMesh(c + dir * radius));
            agent.Goals = new[] { goalGO.transform };
            ColorAgent(agent, col);
        }
    }

    // Block south-west, goal north-east — diagonal open-field walk.
    // No geometry needed; the diagonal path creates natural density variation.
    void SpawnCrossing()
    {
        Vector3 c    = _spawnCenter;
        float   r    = Mathf.Min(spawnWidth, spawnDepth) * 0.40f;
        var     goal = MakeTempGoal("CrossGoal", c + new Vector3(r, 0, r));
        SpawnBlock(c + new Vector3(-r, 0, -r), agentCount, new[] { goal },
                   new Color(0.9f, 0.3f, 0.3f));
    }

    // Equal groups at OPPOSITE corridor ends — organic counter-flow, no pre-assigned lanes.
    // Groups are WIDE (full corridor Z) and SHALLOW (few rows in X) so they look like
    // real crowds and let the solver create natural lane formation as they cross.
    void SpawnBidirectional()
    {
        Vector3 c  = _spawnCenter;
        float   sp = AgentSpacing();

        // Corridor half-width (must match BuildCorridorWalls)
        float corridorHalf = Mathf.Max(agentRadius * 15f, 7f);
        // Stay clear of wall faces and keep a small gap between agents on each side
        float usable       = (corridorHalf - agentRadius * 2f) * 2f; // full usable width
        // How many agents fit ACROSS the corridor — wide and shallow is the goal
        int   zCols = Mathf.Max(1, Mathf.FloorToInt(usable / sp));
        float halfZ = (zCols - 1) * sp * 0.5f;

        // Groups start at ±startX, rows grow OUTWARD so agents nearest the gap go first
        float startX = Mathf.Max(spawnWidth * 0.2f, 6f);
        float goalX  = Mathf.Max(spawnWidth * 0.45f, 20f);

        var goalE = MakeTempGoal("BD_GoalE", c + new Vector3( goalX, 0, 0));
        var goalW = MakeTempGoal("BD_GoalW", c + new Vector3(-goalX, 0, 0));

        int half  = agentCount / 2;
        int rowsA = Mathf.CeilToInt((float)half              / zCols);
        int rowsB = Mathf.CeilToInt((float)(agentCount-half) / zCols);

        // ── Group A (blue) — west end, heading east ────────────────────────
        for (int row = 0, n = 0; row < rowsA && n < half; row++)
            for (int zi = 0; zi < zCols && n < half; zi++, n++)
            {
                // Small Z jitter so rows don't lock into rigid columns when they meet
                float jitter = Random.Range(-sp * 0.15f, sp * 0.15f);
                Vector3 pos  = c + new Vector3(-(startX + row * sp), 0f,
                                               zi * sp - halfZ + jitter);
                var a = SpawnAgent(SnapNavMesh(pos));
                a.Goals = new[] { goalE };
                ColorAgent(a, new Color(0.25f, 0.5f, 1f));
            }

        // ── Group B (red) — east end, heading west ─────────────────────────
        for (int row = 0, n = 0; row < rowsB && n < agentCount - half; row++)
            for (int zi = 0; zi < zCols && n < agentCount - half; zi++, n++)
            {
                float jitter = Random.Range(-sp * 0.15f, sp * 0.15f);
                Vector3 pos  = c + new Vector3(startX + row * sp, 0f,
                                               zi * sp - halfZ + jitter);
                var a = SpawnAgent(SnapNavMesh(pos));
                a.Goals = new[] { goalW };
                ColorAgent(a, new Color(1f, 0.3f, 0.3f));
            }
    }

    // Block on SOUTH side, single goal on NORTH side past the wall.
    // The wall wings are 500 units long so agents CANNOT go around — only through the gap.
    void SpawnBottleneck()
    {
        Vector3 c  = _spawnCenter;
        float   sp = AgentSpacing();

        // Spawn block: wide shallow rectangle well south of the wall
        int   xCols = Mathf.Max(1, Mathf.RoundToInt(Mathf.Sqrt(agentCount * 2f)));
        int   rows  = Mathf.CeilToInt((float)agentCount / xCols);

        // Keep spawn block well clear of the wall (half-thickness = 4 + buffer).
        // wallThick is 8, so wall face is at Z = ±4. Add clearance beyond that.
        float wallHalfThick = 4f;
        float clearance     = Mathf.Max(agentRadius * 4f, 6f);
        float northEdgeZ    = -(wallHalfThick + clearance);   // e.g. ≈ -10 from wall centre

        // Goal north of the wall — half-thickness plus generous clearance
        float goalZ = wallHalfThick + clearance + Mathf.Max(spawnDepth * 0.15f, 12f);
        var   goal  = MakeTempGoal("BNGoal", c + new Vector3(0, 0, goalZ));

        float halfX = (xCols - 1) * sp * 0.5f;
        int   n = 0;
        for (int row = 0; row < rows && n < agentCount; row++)
            for (int ci = 0; ci < xCols && n < agentCount; ci++, n++)
            {
                Vector3 pos = c + new Vector3(ci * sp - halfX, 0f,
                                              northEdgeZ - row * sp);
                var agent = SpawnAgent(SnapNavMesh(pos));
                agent.Goals = new[] { goal };
                ColorAgent(agent, new Color(0.6f, 0.2f, 0.85f));
            }
    }

    void SpawnDoubleBottleneck()
    {
        Vector3 c = _spawnCenter;
        float sp = AgentSpacing();

        int xCols = Mathf.Max(1, Mathf.RoundToInt(Mathf.Sqrt(agentCount * 2f)));
        int rows = Mathf.CeilToInt((float)agentCount / xCols);

        float startNorthEdgeZ = -24f;
        float goalZ = 26f;
        var goal = MakeTempGoal("DBNGoal", c + new Vector3(0f, 0f, goalZ));

        float halfX = (xCols - 1) * sp * 0.5f;
        int n = 0;
        for (int row = 0; row < rows && n < agentCount; row++)
            for (int ci = 0; ci < xCols && n < agentCount; ci++, n++)
            {
                Vector3 pos = c + new Vector3(ci * sp - halfX, 0f, startNorthEdgeZ - row * sp);
                var agent = SpawnAgent(SnapNavMesh(pos));
                agent.Goals = new[] { goal };
                ColorAgent(agent, new Color(0.35f, 0.6f, 0.95f));
            }
    }

    void SpawnMergeLane()
    {
        Vector3 c = _spawnCenter;
        float sp = AgentSpacing();

        float corridorHalf = Mathf.Max(agentRadius * 11f, 6f);
        float groupX = corridorHalf * 0.52f;
        float southZ = -24f;
        float goalZ = 26f;
        var goal = MakeTempGoal("MGGoal", c + new Vector3(0f, 0f, goalZ));

        int leftCount = agentCount / 2;
        int rightCount = agentCount - leftCount;

        int colsPerGroup = Mathf.Max(1, Mathf.RoundToInt(Mathf.Sqrt(leftCount)));
        int leftRows = Mathf.CeilToInt((float)leftCount / colsPerGroup);
        int rightRows = Mathf.CeilToInt((float)rightCount / colsPerGroup);

        float halfW = (colsPerGroup - 1) * sp * 0.5f;

        for (int row = 0, n = 0; row < leftRows && n < leftCount; row++)
            for (int ci = 0; ci < colsPerGroup && n < leftCount; ci++, n++)
            {
                Vector3 pos = c + new Vector3(-groupX + (ci * sp - halfW), 0f, southZ - row * sp);
                var agent = SpawnAgent(SnapNavMesh(pos));
                agent.Goals = new[] { goal };
                ColorAgent(agent, new Color(0.2f, 0.65f, 1f));
            }

        for (int row = 0, n = 0; row < rightRows && n < rightCount; row++)
            for (int ci = 0; ci < colsPerGroup && n < rightCount; ci++, n++)
            {
                Vector3 pos = c + new Vector3(groupX + (ci * sp - halfW), 0f, southZ - row * sp);
                var agent = SpawnAgent(SnapNavMesh(pos));
                agent.Goals = new[] { goal };
                ColorAgent(agent, new Color(1f, 0.45f, 0.35f));
            }
    }

    // ── Shared block helper ──────────────────────────────────────────────────
    // Spawns `count` agents in a compact rectangular block centred at `centre`.
    // Wide and shallow (landscape): more columns than rows for a clean front.
    void SpawnBlock(Vector3 centre, int count, Transform[] groupGoals, Color groupCol)
    {
        float sp    = AgentSpacing();
        int   xCols = Mathf.Max(1, Mathf.RoundToInt(Mathf.Sqrt(count * 2f))); // 2:1 ratio
        int   rows  = Mathf.CeilToInt((float)count / xCols);
        float halfX = (xCols - 1) * sp * 0.5f;
        float halfZ = (rows  - 1) * sp * 0.5f;
        int   n     = 0;
        for (int row = 0; row < rows && n < count; row++)
            for (int ci = 0; ci < xCols && n < count; ci++, n++)
            {
                Vector3 pos = centre + new Vector3(ci * sp - halfX, 0f, row * sp - halfZ);
                var agent   = SpawnAgent(SnapNavMesh(pos));
                agent.Goals = groupGoals;
                ColorAgent(agent, groupCol);
            }
    }

    // -----------------------------------------------------------------------
    // Spawn helpers
    // -----------------------------------------------------------------------

    // Lays out `count` agents in a tidy rectangular grid centred at `centre`.
    // xDir / zDir are orthogonal unit vectors defining the two grid axes.
    // spacing = distance between agent centres.
    // maxCols = how many agents per row (rows grow in zDir direction).
    // groupColor = tint applied to every agent in this group.
    void SpawnGrid(Vector3 centre, int count,
                   Vector3 xDir, Vector3 zDir,
                   float spacing, int maxCols,
                   Transform[] groupGoals,
                   Color groupColor = default)
    {
        int   cols  = Mathf.Clamp(maxCols, 1, count);
        int   rows  = Mathf.CeilToInt((float)count / cols);
        float halfX = (cols - 1) * spacing * 0.5f;
        float halfZ = (rows - 1) * spacing * 0.5f;
        int   n     = 0;
        for (int r = 0; r < rows && n < count; r++)
            for (int c = 0; c < cols && n < count; c++, n++)
            {
                Vector3 pos   = centre
                              + xDir * (c * spacing - halfX)
                              + zDir * (r * spacing - halfZ);
                var agent = SpawnAgent(SnapNavMesh(pos));
                agent.Goals = groupGoals;
                if (groupColor != default) ColorAgent(agent, groupColor);
            }
    }

    // Tints the main visible mesh(es) on an agent without touching the foveated sphere.
    void ColorAgent(AgentBase agent, Color color)
    {
        if (agent == null) return;
        foreach (var r in agent.GetComponentsInChildren<Renderer>())
        {
            // Skip the foveated focus sphere (named "Sphere" or is a plain sphere child)
            if (r.gameObject.name.ToLower().Contains("sphere")) continue;
            if (r is MeshRenderer || r is SkinnedMeshRenderer)
                r.material.color = color;
        }
    }

    AgentBase SpawnAgent(Vector3 pos)
    {
        bool isPBD = activeSolver != SolverMode.Force && activeSolver != SolverMode.ForceJobs;
        var rot = Quaternion.Euler(0, Random.Range(0f, 360f), 0);
        var go  = Instantiate(agentPrefab, pos, rot, transform);

        AgentBase agent = isPBD
            ? go.AddComponent<PBDAgent>()
            : (AgentBase)go.AddComponent<ForceAgent>();

        ApplyAgentParams(agent);

        // Natural commuter movement speed variation (+/- 15%)
        agent.prefSpeed *= Random.Range(0.85f, 1.15f);

        // Natural clothing color tint variation using MaterialPropertyBlock
        Color[] commuterTints = new Color[]
        {
            new Color(0.25f, 0.35f, 0.45f), // Charcoal Navy
            new Color(0.55f, 0.25f, 0.20f), // Burgundy Rust
            new Color(0.20f, 0.45f, 0.35f), // Forest Green
            new Color(0.60f, 0.55f, 0.50f), // Trench Beige
            new Color(0.18f, 0.18f, 0.20f), // Dark Slate
            new Color(0.70f, 0.60f, 0.35f)  // Ochre Mustard
        };
        Color randomClothing = commuterTints[Random.Range(0, commuterTints.Length)];
        MaterialPropertyBlock mpb = new MaterialPropertyBlock();
        mpb.SetColor("_Color", randomClothing);
        foreach (var r in go.GetComponentsInChildren<Renderer>())
        {
            if (r != null && !r.gameObject.name.ToLower().Contains("sphere"))
                r.SetPropertyBlock(mpb);
        }

        agents.Add(agent);
        return agent;
    }

    Transform MakeTempGoal(string name, Vector3 pos)
    {
        // Visible sphere so you can see exactly where the goal is placed
        var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.name = name;
        go.transform.SetParent(transform);
        go.transform.position   = pos;
        go.transform.localScale = Vector3.one * 1.5f;

        var mr = go.GetComponent<MeshRenderer>();
        if (mr != null) mr.material.color = new Color(0.05f, 1f, 0.2f); // bright green

        // No collider — agents should walk through the goal marker, not bump into it
        var col = go.GetComponent<Collider>();
        if (col != null) Destroy(col);

        _tempGoals.Add(go);
        return go.transform;
    }

    // Returns the NavMesh floor height at a given XZ position, ignoring any
    // vertical offset on the manager GameObject itself.
    public Vector3 Center()
    {
        var xz = new Vector3(transform.position.x, 0f, transform.position.z);
        return NavMesh.SamplePosition(xz, out var hit, 200f, NavMesh.AllAreas)
            ? hit.position : xz;
    }

    // Snaps pos to the nearest NavMesh point.  Uses a generous radius so it
    // succeeds even when pos has a large Y offset.
    Vector3 SnapNavMesh(Vector3 pos)
    {
        if (NavMesh.SamplePosition(pos, out var hit, 10f, NavMesh.AllAreas))
            return hit.position;
        // Retry from floor level in case pos.y is far from the NavMesh
        var flat = new Vector3(pos.x, 0f, pos.z);
        return NavMesh.SamplePosition(flat, out hit, 10f, NavMesh.AllAreas)
            ? hit.position : flat;
    }

    void ApplyAgentParams(AgentBase a)
    {
        a.radius = agentRadius;
        a.neighborDist = neighborDist;
        a.maxAccel = maxAccel;
        a.mass = agentMass;
        a.prefSpeed = agentSpeed;

        if (a is PBDAgent pbd)
        {
            pbd.PbdIterations = pbdIterations;
            pbd.ksi = pbdKsi;
            if (pbd._agent != null) pbd._agent.speed = agentSpeed;
        }
        else
        {
            a.k = forceK;
            a.ksi = forceKsi;
            a.m = forceM;
            a.t0 = forceT0;
        }
    }

    Vector3 SampleNavMesh()
    {
        Vector3 origin = Center();
        for (int i = 0; i < 20; i++)
        {
            Vector3 candidate = origin + new Vector3(
                Random.Range(-spawnWidth * .5f, spawnWidth * .5f), 0f,
                Random.Range(-spawnDepth * .5f, spawnDepth * .5f));
            if (NavMesh.SamplePosition(candidate, out var hit, 10f, NavMesh.AllAreas))
                return hit.position;
        }
        return origin;
    }

    // -----------------------------------------------------------------------
    // Solver dispatch
    // -----------------------------------------------------------------------

    void DispatchSolver()
    {
        float dt = Time.fixedDeltaTime;
        Vector3 focus = focusPoint != null ? focusPoint.position : transform.position;

        switch (_lastSolver)
        {
            case SolverMode.Force:
                _forceSolver?.Solve(agents);
                break;
            case SolverMode.ForceJobs:
                _forceJobsSolver?.Solve(agents);
                break;
            case SolverMode.PBD:
                _pbdSolver?.Solve(agents, dt, pbdIterations);
                break;
            case SolverMode.PBDJobs:
                _pbdJobsSolver?.Solve(agents, dt, pbdIterations);
                break;
            case SolverMode.PBDCLODJobs:
                UpdateCLODDistances();
                _clodSolver?.Solve(agents, dt, pbdIterations, focus);
                break;
            case SolverMode.PBDCompute:
                UpdateComputeImportanceParams();
                _computeSolver?.Solve(agents, dt, pbdIterations, focus);
                break;
            case SolverMode.PBDCuda:
                UpdateCudaImportanceParams();
                _cudaSolver?.Solve(agents, dt, pbdIterations, focus);
                break;
            case SolverMode.PBDCLODCullingJobs:
                UpdateCLODDistances();
                _cullingSolver?.Solve(agents, dt, pbdIterations, focus);
                break;

        }
    }

    /// <summary>
    /// After the solver moves agents, eject any that have clipped into or are too close
    /// to a wall, then cancel the wall-penetrating component of their stored velocity so
    /// they don't immediately re-enter the wall next frame (critical for PBD-Jobs which
    /// carries velocity across frames).
    /// </summary>
    void PushAgentsFromWalls()
    {
        // Use a slightly larger radius as the detection zone so we catch agents
        // before they physically touch the wall surface.
        float r      = agentRadius * 1.4f;
        float rHard  = agentRadius;           // hard contact distance (no penetration allowed)

        foreach (var agent in agents)
        {
            if (agent == null) continue;
            Vector3 pos        = agent.transform.position;
            bool    moved      = false;
            Vector3 pushNormal = Vector3.zero; // accumulated outward normal from all walls hit

            foreach (var wall in _wallBounds)
            {
                Vector3 closest = wall.ClosestPoint(pos);
                Vector3 delta   = pos - closest;
                float   dist    = delta.magnitude;

                if (dist < 0.001f)
                {
                    // Agent is fully inside the wall — eject through the thinnest horizontal face.
                    float penX = Mathf.Min(Mathf.Abs(pos.x - wall.min.x),
                                           Mathf.Abs(pos.x - wall.max.x));
                    float penZ = Mathf.Min(Mathf.Abs(pos.z - wall.min.z),
                                           Mathf.Abs(pos.z - wall.max.z));

                    Vector3 ejectDir;
                    if (penZ <= penX)
                    {
                        pos.z = (pos.z < wall.center.z)
                              ? wall.min.z - rHard
                              : wall.max.z + rHard;
                        ejectDir = new Vector3(0, 0, pos.z < wall.center.z ? -1f : 1f);
                    }
                    else
                    {
                        pos.x = (pos.x < wall.center.x)
                              ? wall.min.x - rHard
                              : wall.max.x + rHard;
                        ejectDir = new Vector3(pos.x < wall.center.x ? -1f : 1f, 0, 0);
                    }
                    pushNormal += ejectDir;
                    moved       = true;
                }
                else if (dist < r)
                {
                    // Agent is within the soft detection zone — push outward.
                    Vector3 dir = delta.normalized;
                    pos        += dir * (r - dist);
                    pushNormal += dir;
                    moved       = true;
                }
            }

            if (moved)
            {
                agent.transform.position = pos;
                // Wall ejection changes the Transform outside NavMeshAgent. Keep
                // pathing attached to the same nearby floor before its next route update.
                TryRecoverAgentOnNavMesh(agent);

                // Cancel the velocity component pointing INTO any wall that was hit.
                // Without this, PBD-Jobs re-accelerates the agent back into the wall
                // on the very next frame using the carried-over velocity.
                if (pushNormal.sqrMagnitude > 0.001f)
                {
                    Vector3 n   = pushNormal.normalized;
                    float   vIn = Vector3.Dot(agent.velocity, -n); // how much velocity points into wall
                    if (vIn > 0f)
                        agent.velocity += n * vIn;   // remove the penetrating component
                }
            }
        }
    }

    void UpdateCLODDistances()
    {
        var cam = benchmarkCamera != null ? benchmarkCamera : Camera.main;

        if (_clodSolver != null)
        {
            _clodSolver.highLodDistance = clodHighDist;
            _clodSolver.medLodDistance  = clodMedDist;
            _clodSolver.weightDistance = weightDistance;
            _clodSolver.weightVisibility = weightVisibility;
            _clodSolver.weightInteraction = weightInteraction;
            _clodSolver.weightDensity = weightDensity;
            _clodSolver.highImportanceThreshold = highImportanceThreshold;
            _clodSolver.medImportanceThreshold = medImportanceThreshold;
            _clodSolver.importanceHysteresis = importanceHysteresis;
            _clodSolver.maxImportanceDistance = maxImportanceDistance;
            _clodSolver.maxInteractionContacts = maxInteractionContacts;
            _clodSolver.maxImportanceDensity = maxImportanceDensity;

            if (cam != null)
            {
                _clodSolver.cameraPosition = cam.transform.position;
                _clodSolver.cameraForward  = cam.transform.forward;
                _clodSolver.activeCamera   = cam;
            }
        }

        if (_cullingSolver != null)
        {
            _cullingSolver.highLodDistance = clodHighDist;
            _cullingSolver.medLodDistance  = clodMedDist;
            _cullingSolver.weightDistance = weightDistance;
            _cullingSolver.weightVisibility = weightVisibility;
            _cullingSolver.weightInteraction = weightInteraction;
            _cullingSolver.weightDensity = weightDensity;
            _cullingSolver.highImportanceThreshold = highImportanceThreshold;
            _cullingSolver.medImportanceThreshold = medImportanceThreshold;
            _cullingSolver.importanceHysteresis = importanceHysteresis;
            _cullingSolver.maxImportanceDistance = maxImportanceDistance;
            _cullingSolver.maxInteractionContacts = maxInteractionContacts;
            _cullingSolver.maxImportanceDensity = maxImportanceDensity;
            if (cam != null &&
                (enforceBenchmarkCameraProtocol || _cullingSolver.protagonistCamera == null))
                _cullingSolver.protagonistCamera = cam;
        }
    }

    void UpdateComputeImportanceParams()
    {
        if (_computeSolver == null) return;
        var cam = benchmarkCamera != null ? benchmarkCamera : Camera.main;
        _computeSolver.weightDistance = weightDistance;
        _computeSolver.weightVisibility = weightVisibility;
        _computeSolver.weightInteraction = weightInteraction;
        _computeSolver.weightDensity = weightDensity;
        _computeSolver.highImportanceThreshold = highImportanceThreshold;
        _computeSolver.medImportanceThreshold = medImportanceThreshold;
        _computeSolver.maxImportanceDistance = maxImportanceDistance;
        _computeSolver.maxInteractionContacts = maxInteractionContacts;
        _computeSolver.highDensityCutoff = maxImportanceDensity;
        if (cam != null)
            _computeSolver.activeCamera = cam;
    }

    void UpdateCudaImportanceParams()
    {
        if (_cudaSolver == null) return;
        var cam = benchmarkCamera != null ? benchmarkCamera : Camera.main;
        _cudaSolver.weightDistance = weightDistance;
        _cudaSolver.weightVisibility = weightVisibility;
        _cudaSolver.weightInteraction = weightInteraction;
        _cudaSolver.weightDensity = weightDensity;
        _cudaSolver.highImportanceThreshold = highImportanceThreshold;
        _cudaSolver.medImportanceThreshold = medImportanceThreshold;
        _cudaSolver.maxImportanceDistance = maxImportanceDistance;
        _cudaSolver.maxInteractionContacts = maxInteractionContacts;
        _cudaSolver.highDensityCutoff = maxImportanceDensity;
        if (cam != null)
            _cudaSolver.activeCamera = cam;
    }

    // -----------------------------------------------------------------------
    // Live parameter sync (called every FixedUpdate)
    // -----------------------------------------------------------------------

    void SyncParams()
    {
        bool isPBD = activeSolver != SolverMode.Force && activeSolver != SolverMode.ForceJobs;
        cellSize = Mathf.Max(4f, neighborDist);

        foreach (var a in agents)
        {
            a.radius = agentRadius;
            a.neighborDist = neighborDist;
            a.maxAccel = maxAccel;
            a.mass = agentMass;
            a.prefSpeed = agentSpeed;

            if (isPBD && a is PBDAgent pbd)
            {
                pbd.PbdIterations = pbdIterations;
                pbd.ksi = pbdKsi;
                if (pbd._agent != null) pbd._agent.speed = agentSpeed;
            }
            else if (!isPBD)
            {
                a.k = forceK;
                a.ksi = forceKsi;
                a.m = forceM;
                a.t0 = forceT0;
            }
        }
    }

    // Repath agents when goals are moved in the Scene view
void RefreshDestinations()
    {
        const float sqrThresh = 0.01f;
        foreach (var a in agents)
        {
            if (a == null || !a.enabled || !a.gameObject.activeInHierarchy ||
                a.goal == null || a._agent == null || !a._agent.enabled)
                continue;

            // PBD and wall correction update Transforms independently of the
            // NavMeshAgent. Recover locally before reading/writing its path.
            if (!TryRecoverAgentOnNavMesh(a))
                continue;

            if ((a.goal.position - a._agent.destination).sqrMagnitude > sqrThresh)
                TrySetAgentDestination(a, a.goal.position);
        }
    }

    // -----------------------------------------------------------------------
    // Spatial hash-grid
    // -----------------------------------------------------------------------

    (int, int) Cell(Vector3 p) =>
        (Mathf.FloorToInt(p.x / cellSize), Mathf.FloorToInt(p.z / cellSize));

    void RebuildGrid()
    {
        foreach (var l in grid.Values) l.Clear();
        foreach (var a in agents)
        {
            var c = Cell(a.transform.position);
            if (!grid.TryGetValue(c, out var l)) { l = new List<AgentBase>(); grid[c] = l; }
            l.Add(a);
        }
    }

    void QueryNeighbours(AgentBase a)
    {
        a.neighbours.Clear();
        float r = a.perceptionRadius, rSq = r * r;
        Vector3 p = a.transform.position;

        int x0 = Mathf.FloorToInt((p.x - r) / cellSize), x1 = Mathf.FloorToInt((p.x + r) / cellSize);
        int z0 = Mathf.FloorToInt((p.z - r) / cellSize), z1 = Mathf.FloorToInt((p.z + r) / cellSize);

        for (int x = x0; x <= x1; x++)
            for (int z = z0; z <= z1; z++)
                if (grid.TryGetValue((x, z), out var l))
                    foreach (var b in l)
                        if (b != a && (b.transform.position - p).sqrMagnitude < rSq)
                            a.neighbours.Add(b);
    }

    // -----------------------------------------------------------------------
    // Init helpers
    // -----------------------------------------------------------------------

    void CacheGoals()
    {
        var parent = GameObject.FindGameObjectWithTag("Goals");
        if (parent == null)
        {
            Debug.LogWarning("[SimulationManager] No GameObject tagged 'Goals'.");
            return;
        }
        goals = parent.GetComponentsInChildren<Transform>()
                      .Where(t => t != parent.transform).ToArray();
    }

    void CacheSolvers()
    {
        _pbdSolver = GetComponent<PBDSolver>();
        _pbdJobsSolver = GetComponent<PBDSolverJobs>();

        _forceSolver = GetComponent<ForceSolver>();
        _forceJobsSolver = GetComponent<ForceSolverJobs>();

        _computeSolver = GetComponent<PBDComputeSolver>();
        _cudaSolver = GetComponent<PBDCudaSolver>();
        if (_cudaSolver == null) _cudaSolver = gameObject.AddComponent<PBDCudaSolver>();
        _clodSolver = GetComponent<CLODPBDSolverJobs>();
        _cullingSolver = GetComponent<CLODPBDCullingSolver>();
        // CLODPBDSolverJobs is now MonoBehaviour to manage memory better.
        // CLODPBDSolverJobs is a plain class, not a MonoBehaviour — already newed above
    }

void SetupPresentationLod()
    {
        var presentation = GetComponent<PBDCudaVisualLodController>();
        if (presentation == null)
            presentation = gameObject.AddComponent<PBDCudaVisualLodController>();

        // Showcase is clean by default; F8 exposes optional visual-LOD diagnostics.
        presentation.cleanShowcaseMode = true;
        showFidelityColors = false;
        drawGrid = false;
    }


    void SetupFocusPoint()
    {
        if (protagonistPrefab != null)
            focusPoint = protagonistPrefab.transform;
    }

    void ResolveBenchmarkCamera()
    {
        if (benchmarkCamera != null) return;
        benchmarkCamera = Camera.main;
        if (benchmarkCamera != null) return;
        var go = GameObject.Find("Main Camera");
        if (go != null) benchmarkCamera = go.GetComponent<Camera>();
    }

    void EnforceBenchmarkCameraProtocol()
    {
        ResolveBenchmarkCamera();
        if (benchmarkCamera == null) return;

        foreach (var cam in FindObjectsOfType<Camera>(true))
        {
            bool keep = cam == benchmarkCamera;
            if (cam.enabled != keep) cam.enabled = keep;

            if (!keep)
            {
                var listener = cam.GetComponent<AudioListener>();
                if (listener != null && listener.enabled) listener.enabled = false;
                continue;
            }

            cam.depth = 10f;
            cam.targetDisplay = 0;
            var mainListener = cam.GetComponent<AudioListener>();
            if (mainListener != null && !mainListener.enabled) mainListener.enabled = true;
        }
    }

    void ApplyBenchmarkCameraView()
    {
        ResolveBenchmarkCamera();
        if (benchmarkCamera == null) return;

        var t = benchmarkCamera.transform;
        t.localScale = Vector3.one;

        if (cameraViewMode == CameraViewMode.VRCommuterEyeLevel)
        {
            // First-person VR commuter perspective at human eye height (1.7m)
            benchmarkCamera.orthographic = false;
            benchmarkCamera.fieldOfView = 75f;
            benchmarkCamera.nearClipPlane = 0.1f;
            benchmarkCamera.farClipPlane = 500f;
            t.position = _spawnCenter + new Vector3(0f, 1.7f, -14f);
            t.rotation = Quaternion.Euler(6f, 0f, 0f);
        }
        else
        {
            // 60-degree angled overhead benchmark view
            Quaternion rot = Quaternion.Euler(benchmarkPitchDegrees, benchmarkYawDegrees, 0f);
            Vector3 forward = rot * Vector3.forward;
            float down = Mathf.Max(0.05f, -forward.y);
            float dist = benchmarkCameraHeight / down;

            t.rotation = rot;
            t.position = _spawnCenter - forward * dist;

            benchmarkCamera.orthographic = true;
            benchmarkCamera.orthographicSize = benchmarkOrthoSize;
            benchmarkCamera.nearClipPlane = 0.3f;
            benchmarkCamera.farClipPlane = 1000f;
        }
    }

    // -----------------------------------------------------------------------
    // Public API used by SimulationUI
    // -----------------------------------------------------------------------

    public int LiveAgentCount => agents.Count;
    public float CurrentFPS => 1f / Mathf.Max(Time.unscaledDeltaTime, 1e-4f);
    public AgentBase GetAgent(int index) => (index >= 0 && index < agents.Count) ? agents[index] : null;

    public void GetFidelityCounts(out int high, out int med, out int low)
    {
        high = med = low = 0;
        for (int i = 0; i < agents.Count; i++)
        {
            int tier = agents[i].fidelityLevel;
            if (tier == 0) high++;
            else if (tier == 1) med++;
            else low++;
        }
    }

    public void SetSolverMode(SolverMode mode)
    {
        if (activeSolver == mode) return;
        activeSolver = mode;
        _lastSolver = mode;
        Respawn();
    }

    public void ToggleCameraMode()
    {
        cameraViewMode = (cameraViewMode == CameraViewMode.VRCommuterEyeLevel)
            ? CameraViewMode.StationOverheadView
            : CameraViewMode.VRCommuterEyeLevel;
        ApplyBenchmarkCameraView();
    }

    public void SetPreset(ScenePreset preset)
    {
        if (activePreset == preset) return;
        activePreset = preset;
        Respawn();
    }

    // -----------------------------------------------------------------------
    // Gizmos
    // -----------------------------------------------------------------------

    void OnDrawGizmos()
    {
        Gizmos.color = new Color(1f, 1f, 0f, 0.3f);
        Gizmos.DrawWireCube(transform.position, new Vector3(spawnWidth, 0.5f, spawnDepth));

        if (!drawGrid || !Application.isPlaying || grid.Count == 0) return;

        foreach (var kvp in grid)
        {
            int cnt = kvp.Value.Count;
            if (cnt == 0) continue;
            var (cx, cz) = kvp.Key;
            Vector3 center = new Vector3((cx + .5f) * cellSize, 0.1f, (cz + .5f) * cellSize);
            Vector3 size = new Vector3(cellSize, 0.05f, cellSize);
            Gizmos.color = Color.Lerp(new Color(0, 1, 0, .15f), new Color(1, 0, 0, .5f), Mathf.Clamp01(cnt / 8f));
            Gizmos.DrawCube(center, size);
            Gizmos.color = new Color(1, 1, 1, 0.2f);
            Gizmos.DrawWireCube(center, size);
        }
    }


/// <summary>
    /// Restores a displaced active agent to the closest local NavMesh point.
    /// The small radius prevents a recovery from jumping to another station area.
    /// </summary>
    public bool TryRecoverAgentOnNavMesh(AgentBase agent, float maxRecoveryDistance = 1.5f)
    {
        if (agent == null || !agent.enabled || !agent.gameObject.activeInHierarchy)
            return false;

        var navAgent = agent._agent;
        if (navAgent == null || !navAgent.enabled || !navAgent.gameObject.activeInHierarchy)
            return false;

        if (navAgent.isOnNavMesh)
            return true;

        Vector3 origin = agent.transform.position;
        if (!NavMesh.SamplePosition(origin, out var hit, maxRecoveryDistance, NavMesh.AllAreas))
            return false;

        // Keep recovery local: do not move an agent across the station merely to
        // regain a path.  SamplePosition includes vertical distance too.
        if ((hit.position - origin).sqrMagnitude > maxRecoveryDistance * maxRecoveryDistance)
            return false;

        agent.transform.position = hit.position;
        agent.predictedPosition = hit.position;

        if (!navAgent.Warp(hit.position))
            return false;

        navAgent.nextPosition = hit.position;
        return navAgent.isOnNavMesh;
    }

    public bool TrySetAgentDestination(AgentBase agent, Vector3 destination)
    {
        if (agent == null || !agent.enabled || !agent.gameObject.activeInHierarchy ||
            agent._agent == null || !agent._agent.enabled)
            return false;

        if (!TryRecoverAgentOnNavMesh(agent))
            return false;

        agent._agent.SetDestination(destination);
        return true;
    }
}