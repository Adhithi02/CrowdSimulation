using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Builds and operates the metro station "digital twin" (the CGVR side):
///  • a realistic, calm station: street stairs → fare gates → platform → train
///  • platform screen doors that slide open while a 3-car train dwells
///  • an automatic train timetable (arrive → doors open → depart → next train)
///  • readable 3D signage, CCTV camera presets with smooth glides
///  • fare-gate card-tap slowdown and the emergency evacuation drill
/// Layout values come from <see cref="StationLayout"/> so the visuals line up
/// with the walkable obstacles and the commuter routes.
/// </summary>
public class MetroStationManager : MonoBehaviour
{
    public static MetroStationManager Instance { get; private set; }

    public enum CCTVAngle
    {
        CCTV_Concourse = 0,      // station overview
        CCTV_Turnstiles = 1,     // fare gates
        CCTV_Platform = 2,       // along the platform
        VR_CommuterEyeLevel = 3  // 1.7 m commuter view
    }

    [Header("Cameras")]
    public CCTVAngle currentCCTV = CCTVAngle.CCTV_Concourse;
    public Camera mainCamera;
    [Range(0.1f, 2f)] public float cameraRefreshInterval = 0.5f;
    [Tooltip("How quickly the camera glides to a new CCTV preset (higher = faster).")]
    public float cameraGlideSpeed = 2.2f;

    [Header("Train service")]
    public bool isTrainAtPlatform;
    public float doorsOpenSeconds = 11f;
    public float headwaySeconds = 4f;

    [Header("Emergency evacuation")]
    public bool isEvacuationActive;
    public float evacuationTimer;
    public int remainingEvacuees;

    [Header("Fare gates")]
    public float turnstileCardTapDelay = 0.6f;

    /// <summary>True while the train is docked and the platform doors are open.</summary>
    public bool DoorsOpen { get; private set; }
    /// <summary>One-line description of the train service for the UI.</summary>
    public string ServiceStatus { get; private set; } = "Train approaching";

    private readonly List<GameObject> _stationObjects = new();
    private readonly List<Transform> _evacuationExits = new();
    private readonly List<(Transform left, Transform right, float doorX)> _doorLeaves = new();
    private readonly List<Renderer> _doorLamps = new();
    private readonly Dictionary<int, float> _tapTimers = new();
    private Vector3 _builtCentre;
    private bool _built;
    private Font _font;

    private GameObject _train;
    private Vector3 _trainDockPos;
    private Coroutine _service;
    private bool _departNow;
    private float _doorOpenAmount;      // 0 closed … 1 open (animated)
    private Light _alarmLight;
    private Material _lampOpen, _lampClosed;

    private float _nextCameraRefresh;
    private Vector3 _camGoalPos;
    private Quaternion _camGoalRot;
    private bool _camHasGoal;

    // ─────────────────────────────────────────────────────────────────────────
    private void Awake() => Instance = this;
    private void OnEnable() { if (Instance == null) Instance = this; }   // survives a script reload during Play

    private void Start()
    {
        if (mainCamera == null) mainCamera = Camera.main;
        BuildMetroStation3D();
        ApplyCCTVAngle(currentCCTV, true);
    }

    private void Update()
    {
        if (mainCamera != null && Time.unscaledTime >= _nextCameraRefresh)
        {
            _nextCameraRefresh = Time.unscaledTime + cameraRefreshInterval;
            ApplyCCTVAngle(currentCCTV);
        }
        GlideCamera();

        // Platform screen doors slide with the service state.
        float targetOpen = DoorsOpen ? 1f : 0f;
        if (!Mathf.Approximately(_doorOpenAmount, targetOpen))
        {
            _doorOpenAmount = Mathf.MoveTowards(_doorOpenAmount, targetOpen, Time.deltaTime / 1.2f);
            foreach (var (l, r, x) in _doorLeaves)
            {
                float off = Mathf.Lerp(0.48f, 1.45f, Mathf.SmoothStep(0f, 1f, _doorOpenAmount));
                l.localPosition = new Vector3(x - off, l.localPosition.y, l.localPosition.z);
                r.localPosition = new Vector3(x + off, r.localPosition.y, r.localPosition.z);
            }
        }

        if (isEvacuationActive)
        {
            evacuationTimer += Time.deltaTime;
            UpdateEvacuationCount();
            if (_alarmLight != null) _alarmLight.intensity = 0.4f + Mathf.PingPong(Time.time * 2.5f, 1.6f);
        }

        ProcessTurnstileQueuing();
    }

    // =========================================================================
    //  1. Station builder
    // =========================================================================
    public void BuildMetroStation3D()
    {
        var sim = SimulationManager.Instance;
        Vector3 c = sim != null ? sim.Center() : Vector3.zero;

        // The station is static: rebuild only if its centre moved. (Rebuilding on every
        // respawn recreated every mesh/light and destroyed the train mid-animation.)
        if (_built && (c - _builtCentre).sqrMagnitude < 0.01f) return;

        if (_service != null) { StopCoroutine(_service); _service = null; }
        foreach (var o in _stationObjects) if (o != null) Destroy(o);
        _stationObjects.Clear();
        _evacuationExits.Clear();
        _doorLeaves.Clear();
        _doorLamps.Clear();
        _builtCentre = c;
        _built = true;
        _font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

        float L = StationLayout.HalfLength;
        float gz = StationLayout.GateZ;
        float h = StationLayout.WallHeight;

        // Calm, realistic palette
        Color concourseFloor = new Color(0.66f, 0.64f, 0.60f);
        Color platformFloor  = new Color(0.55f, 0.56f, 0.58f);
        Color wallTile       = new Color(0.86f, 0.86f, 0.84f);
        Color darkTrim       = new Color(0.18f, 0.20f, 0.23f);
        Color steel          = new Color(0.55f, 0.57f, 0.60f);
        Color signNavy       = new Color(0.08f, 0.16f, 0.32f);
        Color glass          = new Color(0.62f, 0.74f, 0.80f);
        Color safetyYellow   = new Color(0.93f, 0.78f, 0.12f);

        // ── Floors ─────────────────────────────────────────────────────────
        float concDepth = gz - StationLayout.FloorMinZ;
        Box("Floor_Concourse", c + new Vector3(0f, -0.05f, (gz + StationLayout.FloorMinZ) * 0.5f), new Vector3(2 * L, 0.1f, concDepth), concourseFloor, 0.25f);
        float platDepth = StationLayout.PsdZ - gz;
        Box("Floor_Platform", c + new Vector3(0f, -0.05f, (gz + StationLayout.PsdZ) * 0.5f), new Vector3(2 * L, 0.1f, platDepth), platformFloor, 0.35f);
        Box("Platform_Edge_Tactile", c + new Vector3(0f, 0.01f, StationLayout.PsdZ - 0.55f), new Vector3(2 * L, 0.02f, 0.45f), safetyYellow, 0.1f);
        Box("Track_Bed", c + new Vector3(0f, -0.4f, StationLayout.TrackZ), new Vector3(2 * L, 0.2f, 6f), new Color(0.12f, 0.12f, 0.13f), 0.05f);
        Box("Rail_N", c + new Vector3(0f, -0.22f, StationLayout.TrackZ + 0.75f), new Vector3(2 * L, 0.15f, 0.12f), steel, 0.8f);
        Box("Rail_S", c + new Vector3(0f, -0.22f, StationLayout.TrackZ - 0.75f), new Vector3(2 * L, 0.15f, 0.12f), steel, 0.8f);

        // ── Walls ──────────────────────────────────────────────────────────
        Box("Wall_South", c + new Vector3(0f, h * 0.5f, StationLayout.FloorMinZ - 0.3f), new Vector3(2 * L + 1.2f, h, 0.6f), wallTile, 0.3f);
        float wallDepth = 14f - StationLayout.FloorMinZ;
        float wallMidZ = (14f + StationLayout.FloorMinZ) * 0.5f;
        Box("Wall_West", c + new Vector3(-L - 0.3f, h * 0.5f, wallMidZ), new Vector3(0.6f, h, wallDepth), wallTile, 0.3f);
        Box("Wall_East", c + new Vector3(L + 0.3f, h * 0.5f, wallMidZ), new Vector3(0.6f, h, wallDepth), wallTile, 0.3f);
        Box("Wall_Track", c + new Vector3(0f, 4.75f, 14f), new Vector3(2 * L, 9.5f, 0.6f), new Color(0.30f, 0.32f, 0.36f), 0.2f);
        Box("Tunnel_W", c + new Vector3(-L + 0.2f, 2.2f, StationLayout.TrackZ), new Vector3(0.5f, 4.4f, 5.5f), new Color(0.03f, 0.03f, 0.04f), 0f);
        Box("Tunnel_E", c + new Vector3(L - 0.2f, 2.2f, StationLayout.TrackZ), new Vector3(0.5f, 4.4f, 5.5f), new Color(0.03f, 0.03f, 0.04f), 0f);
        for (int i = 0; i < 4; i++)   // muted advertising panels on the track wall
        {
            float x = -27f + i * 18f;
            Box("AdPanel", c + new Vector3(x, 2.6f, 13.65f), new Vector3(7f, 2.4f, 0.08f),
                Color.Lerp(new Color(0.20f, 0.30f, 0.42f), new Color(0.45f, 0.30f, 0.22f), i / 3f), 0.6f);
        }
        Sign("CENTRAL STATION", c + new Vector3(0f, 7.2f, 13.6f), true, 1.2f, new Color(0.95f, 0.95f, 0.95f), signNavy, 22f);
        Sign("PLATFORM 1  ·  City Line  →  Airport", c + new Vector3(0f, 5.6f, 13.6f), true, 0.55f, Color.white, new Color(0.10f, 0.40f, 0.30f), 22f);

        // Street stairways in the south wall
        foreach (float sx in StationLayout.StreetX)
        {
            Box("Stair_Portal", c + new Vector3(sx, 1.6f, StationLayout.FloorMinZ - 0.02f), new Vector3(4.6f, 3.2f, 0.66f), new Color(0.08f, 0.08f, 0.09f), 0f);
            Sign("STREET  ↑", c + new Vector3(sx, 3.8f, StationLayout.FloorMinZ + 0.07f), false, 0.45f, Color.white, signNavy, 4.6f);
        }

        // ── Fare gates (6 cabinets → 3 entry lanes + 2 exit lanes) ───────────
        float innerX = StationLayout.GateCabinetX[StationLayout.GateCabinetX.Length - 1] + 0.59f;
        float wing = L - innerX;
        Box("GateLine_W", c + new Vector3(-(innerX + wing * 0.5f), 0.55f, gz), new Vector3(wing, 1.1f, 0.12f), glass, 0.8f);
        Box("GateLine_E", c + new Vector3(innerX + wing * 0.5f, 0.55f, gz), new Vector3(wing, 1.1f, 0.12f), glass, 0.8f);
        Box("GateRail_W", c + new Vector3(-(innerX + wing * 0.5f), 1.12f, gz), new Vector3(wing, 0.06f, 0.16f), steel, 0.8f);
        Box("GateRail_E", c + new Vector3(innerX + wing * 0.5f, 1.12f, gz), new Vector3(wing, 0.06f, 0.16f), steel, 0.8f);
        for (int i = 0; i < StationLayout.GateCabinetX.Length; i++)
        {
            float x = StationLayout.GateCabinetX[i];
            var gate = Model("Metro_TurnstileGate", c + new Vector3(x, 0f, gz), Quaternion.identity, new Color(0.30f, 0.33f, 0.37f));
            if (gate == null) Box("GateCabinet", c + new Vector3(x, 0.6f, gz), new Vector3(1.18f, 1.2f, 1.6f), new Color(0.30f, 0.33f, 0.37f), 0.6f);
            bool entrySide = x < 0.1f;   // cabinets on the left bound the entry lanes
            Box("TapReader", c + new Vector3(x, 1.2f, gz - 0.45f), new Vector3(0.28f, 0.04f, 0.3f),
                entrySide ? new Color(0.15f, 0.85f, 0.35f) : new Color(0.95f, 0.55f, 0.15f), 0.9f);
        }
        // Gate sign board: readable from the concourse (south) and the platform (north)
        Box("GateSignBoard", c + new Vector3(0f, 3.1f, gz), new Vector3(17f, 0.9f, 0.2f), signNavy, 0.4f);
        Text("ENTRY  ↑  tap card", c + new Vector3(-3f, 3.1f, gz - 0.12f), true, 0.42f, new Color(0.55f, 1f, 0.65f));
        Text("EXIT ONLY", c + new Vector3(4.5f, 3.1f, gz - 0.12f), true, 0.42f, new Color(1f, 0.75f, 0.45f));
        Text("EXIT  ↑  to street", c + new Vector3(4.5f, 3.1f, gz + 0.12f), false, 0.42f, new Color(1f, 0.85f, 0.55f));
        Text("NO EXIT", c + new Vector3(-3f, 3.1f, gz + 0.12f), false, 0.42f, new Color(0.8f, 0.8f, 0.8f));

        // Ticket machines in the unpaid concourse
        for (float x = -18f; x <= -14f; x += 2f)
            Model("Metro_TicketKiosk", c + new Vector3(x, 0f, -16f), Quaternion.identity, new Color(0.18f, 0.36f, 0.52f));
        Sign("TICKETS", c + new Vector3(-16f, 2.9f, -16.5f), true, 0.4f, Color.white, signNavy, 5f);
        Model("Metro_SecurityCCTV", c + new Vector3(9.5f, 4.8f, gz - 1.5f), Quaternion.Euler(20f, -135f, 0f), darkTrim);

        // ── Platform furniture ──────────────────────────────────────────────
        foreach (float x in StationLayout.PillarX)
            Model("Metro_PlatformPillar", c + new Vector3(x, 0f, StationLayout.PillarZ), Quaternion.identity, new Color(0.78f, 0.78f, 0.76f));
        foreach (float x in StationLayout.BenchX)
            Model("Metro_PlatformBench", c + new Vector3(x, 0f, StationLayout.PillarZ), Quaternion.Euler(0f, 180f, 0f), new Color(0.45f, 0.36f, 0.26f));

        // ── Platform screen doors (fixed glass panels + sliding door leaves) ──
        BuildScreenDoors(c, glass, darkTrim);

        // ── Lighting (few lights, no shadows: protects frame time) ──────────
        for (int i = 0; i < 5; i++)
        {
            float x = -32f + i * 16f;
            Box("LightStrip", c + new Vector3(x, 5.9f, 1.5f), new Vector3(10f, 0.08f, 0.35f), new Color(1f, 0.98f, 0.92f), 0.9f);
            var lgo = new GameObject("StationLight");
            lgo.transform.SetParent(transform, false);
            lgo.transform.position = c + new Vector3(x, 5.5f, -4f);
            var l = lgo.AddComponent<Light>();
            l.type = LightType.Point;
            l.color = new Color(1f, 0.97f, 0.90f);
            l.range = 24f;
            l.intensity = 1.1f;
            l.shadows = LightShadows.None;
            // Vertex/SH lighting: in forward rendering every per-pixel light redraws all
            // 300 skinned commuters, so these fill lights must not be pixel lights.
            l.renderMode = LightRenderMode.ForceVertex;
            _stationObjects.Add(lgo);
        }

        // ── Train (3 cars) ───────────────────────────────────────────────────
        BuildTrain(c);

        // ── Emergency exits in the end walls ────────────────────────────────
        foreach (float side in new[] { -1f, 1f })
        {
            Box("EmergencyExitDoor", c + new Vector3(side * (L - 0.05f), 1.4f, -12f), new Vector3(0.12f, 2.8f, 3f), new Color(0.12f, 0.45f, 0.22f), 0.5f);
            var exitSign = Box("EmergencyExitSign", c + new Vector3(side * (L - 0.1f), 3.2f, -12f), new Vector3(0.1f, 0.5f, 2.2f), new Color(0.10f, 0.70f, 0.30f), 0.5f);
            Text("EXIT", c + new Vector3(side * (L - 0.18f), 3.2f, -12f), side > 0 ? Quaternion.Euler(0f, 90f, 0f) : Quaternion.Euler(0f, -90f, 0f), 0.35f, Color.white);
            var t = new GameObject("EvacuationPoint").transform;
            t.SetParent(transform, false);
            t.position = c + new Vector3(side * (L - 2.5f), 0f, -12f);
            _evacuationExits.Add(t);
            _stationObjects.Add(t.gameObject);
        }

        var alarm = new GameObject("Station_Alarm_Light");
        alarm.transform.SetParent(transform, false);
        alarm.transform.position = c + new Vector3(0f, 5.5f, -8f);
        _alarmLight = alarm.AddComponent<Light>();
        _alarmLight.type = LightType.Point;
        _alarmLight.color = new Color(1f, 0.15f, 0.1f);
        _alarmLight.range = 60f;
        _alarmLight.intensity = 0f;
        _alarmLight.enabled = false;   // a disabled light costs nothing until the drill starts
        _stationObjects.Add(alarm);

        _service = StartCoroutine(TrainService());
    }

    private void BuildScreenDoors(Vector3 c, Color glass, Color trim)
    {
        float L = StationLayout.HalfLength;
        float z = StationLayout.PsdZ;
        const float half = 0.95f;   // door opening half-width (1.9 m opening)

        // Fixed panels fill the gaps between door openings
        var edges = new List<float> { -L };
        foreach (float d in StationLayout.DoorX) { edges.Add(d - half); edges.Add(d + half); }
        edges.Add(L);
        for (int i = 0; i < edges.Count; i += 2)
        {
            float a = edges[i], b = edges[i + 1];
            if (b - a < 0.05f) continue;
            Box("PSD_Panel", c + new Vector3((a + b) * 0.5f, 1.15f, z), new Vector3(b - a, 2.3f, 0.08f), glass, 0.9f);
        }
        Box("PSD_Header", c + new Vector3(0f, 2.55f, z), new Vector3(2 * L, 0.5f, 0.3f), trim, 0.4f);

        _lampOpen = new Material(Shader.Find("Standard")) { color = new Color(0.2f, 0.95f, 0.4f) };
        _lampOpen.EnableKeyword("_EMISSION");
        _lampOpen.SetColor("_EmissionColor", new Color(0.1f, 0.6f, 0.2f));
        _lampClosed = new Material(Shader.Find("Standard")) { color = new Color(0.55f, 0.12f, 0.1f) };

        var leafRoot = new GameObject("PSD_DoorLeaves").transform;
        leafRoot.SetParent(transform, false);
        leafRoot.position = c;
        _stationObjects.Add(leafRoot.gameObject);
        foreach (float d in StationLayout.DoorX)
        {
            var left = Box("PSD_Leaf", Vector3.zero, new Vector3(0.95f, 2.25f, 0.06f), new Color(0.70f, 0.80f, 0.86f), 0.9f).transform;
            var right = Box("PSD_Leaf", Vector3.zero, new Vector3(0.95f, 2.25f, 0.06f), new Color(0.70f, 0.80f, 0.86f), 0.9f).transform;
            left.SetParent(leafRoot, false); right.SetParent(leafRoot, false);
            left.localPosition = new Vector3(d - 0.48f, 1.13f, z - 0.08f);
            right.localPosition = new Vector3(d + 0.48f, 1.13f, z - 0.08f);
            _doorLeaves.Add((left, right, d));

            var lamp = Box("PSD_Lamp", c + new Vector3(d, 2.55f, z - 0.17f), new Vector3(0.6f, 0.12f, 0.04f), Color.white, 0.5f);
            var mr = lamp.GetComponent<Renderer>();
            mr.sharedMaterial = _lampClosed;
            _doorLamps.Add(mr);
        }
        _doorOpenAmount = 0f;
    }

    private void BuildTrain(Vector3 c)
    {
        _trainDockPos = c + new Vector3(0f, 0.15f, StationLayout.TrackZ);
        _train = new GameObject("Metro_Train");
        _train.transform.SetParent(transform, false);
        _train.transform.position = _trainDockPos + new Vector3(-120f, 0f, 0f);
        _stationObjects.Add(_train);

        Color body = new Color(0.80f, 0.82f, 0.85f);
        Color stripe = new Color(0.10f, 0.40f, 0.30f);
        foreach (float x in StationLayout.CarX)
        {
            var car = Model("Metro_TrainCar", Vector3.zero, Quaternion.identity, body, false);
            if (car == null) car = Box("TrainCar", Vector3.zero, new Vector3(24f, 3.6f, 3.2f), body, 0.6f);
            car.transform.SetParent(_train.transform, false);
            car.transform.localPosition = new Vector3(x, 0f, 0f);
            var s = Box("TrainStripe", Vector3.zero, new Vector3(23.6f, 0.35f, 0.05f), stripe, 0.6f, false);
            s.transform.SetParent(_train.transform, false);
            s.transform.localPosition = new Vector3(x, 1.0f, -1.63f);
        }
        var head = new GameObject("Train_Headlight");
        head.transform.SetParent(_train.transform, false);
        head.transform.localPosition = new Vector3(37.5f, 1.6f, 0f);
        head.transform.localRotation = Quaternion.Euler(8f, 90f, 0f);
        var hl = head.AddComponent<Light>();
        hl.type = LightType.Spot; hl.range = 30f; hl.spotAngle = 50f; hl.intensity = 2.5f;
        hl.color = new Color(1f, 0.95f, 0.85f); hl.shadows = LightShadows.None;
    }

    // =========================================================================
    //  2. Train timetable: arrive → doors open → close → depart → headway
    // =========================================================================
    private IEnumerator TrainService()
    {
        yield return new WaitForSeconds(1.5f);
        while (true)
        {
            ServiceStatus = "Train arriving";
            if (_train != null) _train.SetActive(true);
            yield return MoveTrain(_trainDockPos + new Vector3(-120f, 0f, 0f), _trainDockPos, 5f, false);
            isTrainAtPlatform = true;

            DoorsOpen = true;
            SetLamps(true);
            float t = 0f;
            _departNow = false;
            while (t < doorsOpenSeconds && !_departNow)
            {
                t += Time.deltaTime;
                ServiceStatus = $"Doors open — boarding ({Mathf.CeilToInt(doorsOpenSeconds - t)} s)";
                yield return null;
            }
            DoorsOpen = false;
            SetLamps(false);
            ServiceStatus = "Doors closing";
            yield return new WaitForSeconds(1.5f);

            isTrainAtPlatform = false;
            ServiceStatus = "Train departing";
            yield return MoveTrain(_trainDockPos, _trainDockPos + new Vector3(120f, 0f, 0f), 5f, true);
            if (_train != null) _train.SetActive(false);   // out of sight between services

            float w = 0f;
            while (w < headwaySeconds) { w += Time.deltaTime; ServiceStatus = $"Next train in {Mathf.CeilToInt(headwaySeconds - w + 5f)} s"; yield return null; }
        }
    }

    private IEnumerator MoveTrain(Vector3 from, Vector3 to, float seconds, bool accelerate)
    {
        float e = 0f;
        while (e < seconds)
        {
            e += Time.deltaTime;
            float u = Mathf.Clamp01(e / seconds);
            float k = accelerate ? u * u : 1f - (1f - u) * (1f - u);   // speed up leaving, brake arriving
            if (_train != null) _train.transform.position = Vector3.LerpUnclamped(from, to, k);
            yield return null;
        }
        if (_train != null) _train.transform.position = to;
    }

    private void SetLamps(bool open)
    {
        foreach (var r in _doorLamps) if (r != null) r.sharedMaterial = open ? _lampOpen : _lampClosed;
    }

    /// <summary>"Cycle Train" button: if a train is dwelling, close the doors and send it now.</summary>
    public void TriggerTrainCycle() => _departNow = true;

    // =========================================================================
    //  3. CCTV cameras
    // =========================================================================
    public void CycleCCTVCamera()
    {
        currentCCTV = (CCTVAngle)(((int)currentCCTV + 1) % 4);
        ApplyCCTVAngle(currentCCTV);
    }

    public void ApplyCCTVAngle(CCTVAngle angle) => ApplyCCTVAngle(angle, false);

    public void ApplyCCTVAngle(CCTVAngle angle, bool snap)
    {
        currentCCTV = angle;
        if (mainCamera == null) mainCamera = Camera.main;
        if (mainCamera == null) return;

        var sim = SimulationManager.Instance;
        Vector3 c = sim != null ? sim.Center() : Vector3.zero;
        mainCamera.orthographic = false;
        mainCamera.fieldOfView = 60f;
        mainCamera.nearClipPlane = 0.25f;
        mainCamera.farClipPlane = 220f;

        Vector3 pos, look;
        if (sim == null || sim.activePreset == ScenePreset.StationFlow)
        {
            // Fixed, readable views of the real station layout
            switch (angle)
            {
                case CCTVAngle.CCTV_Concourse:  pos = c + new Vector3(0f, 22f, -30f);   look = c + new Vector3(0f, 0f, -4f); break;
                case CCTVAngle.CCTV_Turnstiles: pos = c + new Vector3(9f, 5.5f, -20f);  look = c + new Vector3(-1f, 0.8f, -6f); break;
                case CCTVAngle.CCTV_Platform:   pos = c + new Vector3(-36f, 5.5f, -4f); look = c + new Vector3(0f, 1.2f, 4f); break;
                default:                        pos = c + new Vector3(-20f, 1.7f, -1.5f); look = c + new Vector3(6f, 1.5f, 3.5f); break;
            }
        }
        else
        {
            // Synthetic test scenarios: follow the live crowd
            Vector3 cc = GetLiveCrowdCentre(sim, c);
            switch (angle)
            {
                case CCTVAngle.CCTV_Concourse:  pos = cc + new Vector3(0f, 14f, -26f); break;
                case CCTVAngle.CCTV_Turnstiles: pos = cc + new Vector3(18f, 7f, -16f); break;
                case CCTVAngle.CCTV_Platform:   pos = cc + new Vector3(-18f, 7f, -16f); break;
                default:                        pos = cc + new Vector3(0f, 1.7f, -13f); break;
            }
            look = cc + new Vector3(0f, angle == CCTVAngle.VR_CommuterEyeLevel ? 1.7f : 1.2f, 0f);
        }

        _camGoalPos = pos;
        _camGoalRot = Quaternion.LookRotation((look - pos).normalized, Vector3.up);
        if (snap || !_camHasGoal)
            mainCamera.transform.SetPositionAndRotation(_camGoalPos, _camGoalRot);
        _camHasGoal = true;
    }

    private void GlideCamera()
    {
        if (!_camHasGoal || mainCamera == null) return;
        float k = 1f - Mathf.Exp(-cameraGlideSpeed * Time.unscaledDeltaTime);
        var t = mainCamera.transform;
        t.SetPositionAndRotation(Vector3.Lerp(t.position, _camGoalPos, k), Quaternion.Slerp(t.rotation, _camGoalRot, k));
    }

    private static Vector3 GetLiveCrowdCentre(SimulationManager sim, Vector3 fallback)
    {
        if (sim == null || sim.LiveAgentCount == 0) return fallback;
        Vector3 sum = Vector3.zero; int n = 0;
        for (int i = 0; i < sim.LiveAgentCount; i++)
        {
            var a = sim.GetAgent(i);
            if (a == null || a.transform.position.y < fallback.y - 3f) continue;
            sum += a.transform.position; n++;
        }
        return n > 0 ? sum / n : fallback;
    }

    public string GetCCTVLabel() => currentCCTV switch
    {
        CCTVAngle.CCTV_Concourse => "CCTV 01 · Station overview",
        CCTVAngle.CCTV_Turnstiles => "CCTV 02 · Fare gates",
        CCTVAngle.CCTV_Platform => "CCTV 03 · Platform 1",
        _ => "Commuter eye level (1.7 m)"
    };

    // =========================================================================
    //  4. Fare-gate card tap: commuters slow down briefly inside a gate lane
    // =========================================================================
    private void ProcessTurnstileQueuing()
    {
        var sim = SimulationManager.Instance;
        if (sim == null || !_built) return;
        float gateZ = _builtCentre.z + StationLayout.GateZ;
        float gateHalf = StationLayout.GateCabinetX[StationLayout.GateCabinetX.Length - 1] + 0.6f;

        for (int i = 0; i < sim.LiveAgentCount; i++)
        {
            var a = sim.GetAgent(i);
            if (a == null) continue;
            Vector3 p = a.transform.position;
            bool inLane = Mathf.Abs(p.z - gateZ) < 1.0f && Mathf.Abs(p.x - _builtCentre.x) <= gateHalf;
            int id = a.GetInstanceID();
            if (inLane)
            {
                if (!_tapTimers.TryGetValue(id, out float left)) left = turnstileCardTapDelay;
                left -= Time.deltaTime;
                _tapTimers[id] = left;
                a.speedScale = left > 0f ? 0.3f : 1f;   // read by SimulationManager.SyncParams → solver
            }
            else
            {
                if (Mathf.Abs(p.z - gateZ) > 2.5f) _tapTimers.Remove(id);
                a.speedScale = 1f;
            }
        }
    }

    // =========================================================================
    //  5. Emergency evacuation
    // =========================================================================
    public void ToggleEmergencyEvacuation()
    {
        isEvacuationActive = !isEvacuationActive;
        var sim = SimulationManager.Instance;

        if (isEvacuationActive)
        {
            evacuationTimer = 0f;
            if (_alarmLight != null) _alarmLight.enabled = true;
            DoorsOpen = false;
            SetLamps(false);
            if (sim != null && _evacuationExits.Count > 0)
            {
                for (int i = 0; i < sim.LiveAgentCount; i++)
                {
                    var a = sim.GetAgent(i);
                    if (a == null || a.transform.position.y < _builtCentre.y - 3f) continue;   // riding the train
                    Transform exit = NearestExit(a.transform.position);
                    a.Goals = new[] { exit };
                    a.goal = exit;   // RefreshDestinations re-paths to a.goal, so it must be the exit
                    sim.TrySetAgentDestination(a, exit.position);
                }
            }
        }
        else
        {
            if (_alarmLight != null) { _alarmLight.intensity = 0f; _alarmLight.enabled = false; }
            sim?.Respawn();
        }
    }

    private Transform NearestExit(Vector3 p)
    {
        Transform best = _evacuationExits[0];
        foreach (var e in _evacuationExits)
            if ((e.position - p).sqrMagnitude < (best.position - p).sqrMagnitude) best = e;
        return best;
    }

    private void UpdateEvacuationCount()
    {
        var sim = SimulationManager.Instance;
        if (sim == null || _evacuationExits.Count == 0) return;
        int remaining = 0;
        for (int i = 0; i < sim.LiveAgentCount; i++)
        {
            var a = sim.GetAgent(i);
            if (a == null || a.transform.position.y < _builtCentre.y - 3f) continue;
            if ((NearestExit(a.transform.position).position - a.transform.position).sqrMagnitude > 9f) remaining++;
        }
        remainingEvacuees = remaining;
    }

    // =========================================================================
    //  Helpers
    // =========================================================================
    private GameObject Box(string name, Vector3 pos, Vector3 size, Color col, float gloss, bool register = true)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.SetParent(transform, false);
        go.transform.position = pos;
        go.transform.localScale = size;
        Destroy(go.GetComponent<Collider>());
        var mr = go.GetComponent<MeshRenderer>();
        mr.material.color = col;
        mr.material.SetFloat("_Glossiness", gloss);
        mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        if (register) _stationObjects.Add(go);
        return go;
    }

    private GameObject Model(string modelName, Vector3 pos, Quaternion rot, Color col, bool register = true)
    {
        var prefab = Resources.Load<GameObject>($"Models/Metro/{modelName}");
        if (prefab == null) return null;
        var go = Instantiate(prefab, pos, rot, transform);
        go.name = modelName;
        foreach (var mr in go.GetComponentsInChildren<MeshRenderer>())
        {
            mr.material.color = col;
            mr.material.SetFloat("_Glossiness", 0.35f);
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        }
        if (register) _stationObjects.Add(go);
        return go;
    }

    /// <summary>A sign board with text. southFacing = readable from the concourse/camera side (looking north).</summary>
    private void Sign(string text, Vector3 pos, bool southFacing, float textHeight, Color textCol, Color boardCol, float boardWidth)
    {
        Box("SignBoard", pos, new Vector3(boardWidth, textHeight * 1.9f, 0.12f), boardCol, 0.3f);
        // South-facing text sits on the south face; north-facing text on the north face, turned around.
        Text(text, pos + new Vector3(0f, 0f, southFacing ? -0.08f : 0.08f), southFacing, textHeight, textCol);
    }

    private void Text(string text, Vector3 pos, bool southFacing, float height, Color col) =>
        Text(text, pos, southFacing ? Quaternion.identity : Quaternion.Euler(0f, 180f, 0f), height, col);

    private Material _textMat;

    /// <summary>Font material that respects depth, so a board hides the text on its back.</summary>
    private Material TextMaterial()
    {
        if (_textMat != null) return _textMat;
        var shader = Resources.Load<Shader>("Shaders/StationText3D");
        if (shader == null) shader = Shader.Find("Station/Text3D");
        _textMat = shader != null ? new Material(shader) { mainTexture = _font.material.mainTexture } : _font.material;
        return _textMat;
    }

    private void Text(string text, Vector3 pos, Quaternion rot, float height, Color col)
    {
        var go = new GameObject("Sign_" + text);
        go.transform.SetParent(transform, false);
        go.transform.SetPositionAndRotation(pos, rot);
        var tm = go.AddComponent<TextMesh>();
        tm.font = _font;
        tm.text = text;
        tm.fontSize = 96;
        tm.characterSize = height * 10f / 96f;   // ≈ capital-letter height in metres
        tm.anchor = TextAnchor.MiddleCenter;
        tm.alignment = TextAlignment.Center;
        tm.color = col;
        go.GetComponent<MeshRenderer>().sharedMaterial = TextMaterial();
        _stationObjects.Add(go);
    }
}
