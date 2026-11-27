using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class MetroStationManager : MonoBehaviour
{
    public static MetroStationManager Instance { get; private set; }

    public enum CCTVAngle
    {
        CCTV_Concourse = 0,
        CCTV_Turnstiles = 1,
        CCTV_Platform = 2,
        VR_CommuterEyeLevel = 3
    }

    [Header("Cinematic CCTV Cameras")]
    public CCTVAngle currentCCTV = CCTVAngle.CCTV_Platform;
    public Camera mainCamera;

    [Header("Train Cycle State")]
    public bool isTrainAtPlatform = true;
    public float trainSpeed = 14f;
    private GameObject _trainCar;
    private Vector3 _trainArrivedPos;
    private Vector3 _trainDepartedPos;
    private Coroutine _trainRoutine;
    private Light _trainHeadlightL;
    private Light _trainHeadlightR;
    private Light _trainInteriorLight;

    [Header("Presentation Camera")]
    [Tooltip("Keeps the selected showcase view outside the moving crowd bounds.")]
    [Range(0.1f, 2f)] public float cameraRefreshInterval = 0.5f;
    private float _nextCameraRefresh;


    [Header("Emergency Evacuation System")]
    public bool isEvacuationActive = false;
    public float evacuationTimer = 0f;
    public int remainingEvacuees = 0;
    private Light _alarmLight;
    private List<Transform> _evacuationExits = new();

    [Header("Turnstile Queuing")]
    public float turnstileCardTapDelay = 0.5f;
    private readonly Dictionary<int, float> _agentTapDelays = new();

    // Spawned station 3D objects
    private readonly List<GameObject> _stationObjects = new();

    private void Awake()
    {
        Instance = this;
    }

    private void Start()
    {
        if (mainCamera == null)
            mainCamera = Camera.main;

        BuildMetroStation3D();
        ApplyCCTVAngle(currentCCTV);
    }

    private void Update()
    {
        if (mainCamera != null && Time.unscaledTime >= _nextCameraRefresh)
        {
            _nextCameraRefresh = Time.unscaledTime + cameraRefreshInterval;
            ApplyCCTVAngle(currentCCTV);
        }

        if (isEvacuationActive)
        {
            evacuationTimer += Time.deltaTime;
            UpdateEvacuationCount();
            if (_alarmLight != null)
            {
                _alarmLight.intensity = Mathf.PingPong(Time.time * 5f, 3.5f);
            }
        }

        ProcessTurnstileQueuing();
    }

    // =========================================================================
    //  1. 3D Metro Station Environment Builder (PBR & Lighting)
    // =========================================================================

    public void BuildMetroStation3D()
    {
        foreach (var obj in _stationObjects)
            if (obj != null) Destroy(obj);
        _stationObjects.Clear();
        _evacuationExits.Clear();

        var sim = SimulationManager.Instance;
        Vector3 c = sim != null ? sim.Center() : Vector3.zero;

        // ── 0. Station Platform & Track Bed ──
        // Concourse & Platform floor (polished transit stone)
        var platformFloor = CreateBoxProp("Metro_Platform_Floor", c + new Vector3(0f, -0.05f, -8f), new Vector3(88f, 0.1f, 34f), new Color(0.24f, 0.25f, 0.27f));
        _stationObjects.Add(platformFloor);

        // Recessed track ballast bed
        var trackBed = CreateBoxProp("Metro_Track_Bed", c + new Vector3(0f, -0.3f, 10f), new Vector3(88f, 0.2f, 7.5f), new Color(0.10f, 0.10f, 0.11f));
        _stationObjects.Add(trackBed);

        // Track back wall behind train with modern dark charcoal architectural panels
        var trackWall = CreateBoxProp("Metro_Track_BackWall", c + new Vector3(0f, 4.5f, 14.0f), new Vector3(88f, 9.5f, 1.2f), new Color(0.14f, 0.15f, 0.18f));
        _stationObjects.Add(trackWall);

        // Architectural station billboards
        for (float x = -30f; x <= 30f; x += 20f)
        {
            var adBoard = CreateBoxProp($"AdBoard_{x}", c + new Vector3(x, 4.5f, 13.3f), new Vector3(12f, 4.0f, 0.15f), new Color(0.16f, 0.38f, 0.62f));
            _stationObjects.Add(adBoard);
        }

        // Tunnel portals (East & West)
        var tunnelEast = CreateBoxProp("Tunnel_Portal_East", c + new Vector3(44f, 4f, 10f), new Vector3(2.5f, 8.5f, 7.5f), new Color(0.03f, 0.03f, 0.04f));
        var tunnelWest = CreateBoxProp("Tunnel_Portal_West", c + new Vector3(-44f, 4f, 10f), new Vector3(2.5f, 8.5f, 7.5f), new Color(0.03f, 0.03f, 0.04f));
        _stationObjects.Add(tunnelEast);
        _stationObjects.Add(tunnelWest);

        // ── A. Subway Train & Tracks ──
        _trainArrivedPos = c + new Vector3(0f, 0.25f, 10f);
        _trainDepartedPos = _trainArrivedPos + new Vector3(85f, 0f, 0f);

        _trainCar = InstantiateMetroModel("Metro_TrainCar", _trainArrivedPos, Quaternion.Euler(0, 0, 0));
        if (_trainCar != null)
        {
            SetColorRecursively(_trainCar, new Color(0.72f, 0.75f, 0.80f));
            _stationObjects.Add(_trainCar);

            // Add dual high-power front headlights to the train
            var hlLeft = new GameObject("Train_Headlight_L");
            hlLeft.transform.SetParent(_trainCar.transform);
            hlLeft.transform.localPosition = new Vector3(-12.5f, 1.8f, 1.2f);
            hlLeft.transform.localRotation = Quaternion.Euler(10f, -90f, 0f);
            _trainHeadlightL = hlLeft.AddComponent<Light>();
            _trainHeadlightL.type = LightType.Spot;
            _trainHeadlightL.color = new Color(1f, 0.95f, 0.8f);
            _trainHeadlightL.range = 35f;
            _trainHeadlightL.spotAngle = 45f;
            _trainHeadlightL.intensity = 3.5f;

            var hlRight = new GameObject("Train_Headlight_R");
            hlRight.transform.SetParent(_trainCar.transform);
            hlRight.transform.localPosition = new Vector3(-12.5f, 1.8f, -1.2f);
            hlRight.transform.localRotation = Quaternion.Euler(10f, -90f, 0f);
            _trainHeadlightR = hlRight.AddComponent<Light>();
            _trainHeadlightR.type = LightType.Spot;
            _trainHeadlightR.color = new Color(1f, 0.95f, 0.8f);
            _trainHeadlightR.range = 35f;
            _trainHeadlightR.spotAngle = 45f;
            _trainHeadlightR.intensity = 3.5f;

            // Warm interior lighting glowing through train passenger windows
            var intLight = new GameObject("Train_Interior_Light");
            intLight.transform.SetParent(_trainCar.transform);
            intLight.transform.localPosition = new Vector3(0f, 2.0f, 0f);
            _trainInteriorLight = intLight.AddComponent<Light>();
            _trainInteriorLight.type = LightType.Point;
            _trainInteriorLight.color = new Color(1f, 0.92f, 0.75f);
            _trainInteriorLight.range = 22f;
            _trainInteriorLight.intensity = 2.0f;
        }

        // Tactile yellow safety line along platform edge
        var yellowLine = CreateBoxProp("Metro_Safety_Line", c + new Vector3(0f, 0.03f, 7.3f), new Vector3(88f, 0.02f, 0.65f), new Color(0.96f, 0.82f, 0.05f));
        _stationObjects.Add(yellowLine);

        // Platform Screen Doors (PSD) with transparent glazing
        for (float x = -35f; x <= 35f; x += 10f)
        {
            if (Mathf.Abs(x % 10f) < 0.1f && Mathf.Abs(x) != 0f && Mathf.Abs(x) != 20f)
            {
                var screen = CreateBoxProp($"Platform_Screen_{x}", c + new Vector3(x, 1.25f, 7.1f), new Vector3(4.8f, 2.3f, 0.12f), new Color(0.25f, 0.4f, 0.55f, 0.65f));
                _stationObjects.Add(screen);
            }
        }

        // Metallic rails
        var railL = CreateBoxProp("Rail_L", c + new Vector3(0f, 0.05f, 9.2f), new Vector3(88f, 0.12f, 0.16f), new Color(0.35f, 0.35f, 0.38f));
        var railR = CreateBoxProp("Rail_R", c + new Vector3(0f, 0.05f, 10.8f), new Vector3(88f, 0.12f, 0.16f), new Color(0.35f, 0.35f, 0.38f));
        _stationObjects.Add(railL);
        _stationObjects.Add(railR);

        // ── B. Platform Structural Pillars & Benches ──
        for (float x = -28f; x <= 28f; x += 14f)
        {
            var pillar = InstantiateMetroModel("Metro_PlatformPillar", c + new Vector3(x, 0f, 2f), Quaternion.identity);
            if (pillar != null)
            {
                SetColorRecursively(pillar, new Color(0.28f, 0.30f, 0.34f));
                _stationObjects.Add(pillar);
            }

            if (Mathf.Abs(x) > 5f)
            {
                var bench = InstantiateMetroModel("Metro_PlatformBench", c + new Vector3(x + 5f, 0f, 2f), Quaternion.Euler(0, 180, 0));
                if (bench != null)
                {
                    SetColorRecursively(bench, new Color(0.42f, 0.45f, 0.48f));
                    _stationObjects.Add(bench);
                }
            }
        }

        // ── C. Turnstile Fare Gate Bank ──
        float turnstileZ = -8f;
        for (float x = -6f; x <= 6f; x += 3f)
        {
            var gate = InstantiateMetroModel("Metro_TurnstileGate", c + new Vector3(x, 0f, turnstileZ), Quaternion.identity);
            if (gate != null)
            {
                SetColorRecursively(gate, new Color(0.22f, 0.25f, 0.28f));
                _stationObjects.Add(gate);

                // Contactless smartcard reader status indicator
                var tapLight = CreateBoxProp($"TapReader_{x}", c + new Vector3(x, 1.05f, turnstileZ), new Vector3(0.3f, 0.05f, 0.4f), new Color(0.1f, 0.95f, 0.35f));
                _stationObjects.Add(tapLight);
            }
        }

        // Concourse side safety railings
        var railWest = CreateBoxProp("Turnstile_Rail_W", c + new Vector3(-18f, 1.0f, turnstileZ), new Vector3(22f, 1.8f, 0.2f), new Color(0.32f, 0.34f, 0.38f));
        var railEast = CreateBoxProp("Turnstile_Rail_E", c + new Vector3(18f, 1.0f, turnstileZ), new Vector3(22f, 1.8f, 0.2f), new Color(0.32f, 0.34f, 0.38f));
        _stationObjects.Add(railWest);
        _stationObjects.Add(railEast);

        // ── D. Ticket Vending Machine Kiosks ──
        for (float x = -18f; x <= -14f; x += 2f)
        {
            var kiosk = InstantiateMetroModel("Metro_TicketKiosk", c + new Vector3(x, 0f, turnstileZ - 8f), Quaternion.Euler(0, 0, 0));
            if (kiosk != null)
            {
                SetColorRecursively(kiosk, new Color(0.18f, 0.36f, 0.52f));
                _stationObjects.Add(kiosk);
            }
        }

        // ── E. Overhead Suspended LED Lighting & Architecture ──
        for (float x = -35f; x <= 35f; x += 17.5f)
        {
            var beam = CreateBoxProp($"CeilingBeam_{x}", c + new Vector3(x, 6.8f, 0f), new Vector3(1.2f, 0.8f, 34f), new Color(0.18f, 0.20f, 0.22f));
            _stationObjects.Add(beam);

            // Overhead platform cool-white linear LED fixtures with actual Unity Point Light
            var fixture = CreateBoxProp($"LightFixture_{x}", c + new Vector3(x, 6.2f, 2f), new Vector3(0.5f, 0.15f, 8f), new Color(0.95f, 0.98f, 1f));
            _stationObjects.Add(fixture);

            var ceilingLight = new GameObject($"Ceiling_PointLight_{x}");
            ceilingLight.transform.position = c + new Vector3(x, 5.8f, 2f);
            var pl = ceilingLight.AddComponent<Light>();
            pl.type = LightType.Point;
            pl.color = new Color(0.92f, 0.96f, 1.0f);
            pl.range = 26f;
            pl.intensity = 1.6f;
            _stationObjects.Add(ceilingLight);
        }

        var signPlat = InstantiateMetroModel("Metro_OverheadSign", c + new Vector3(0f, 5.2f, 4f), Quaternion.identity);
        if (signPlat != null)
        {
            SetColorRecursively(signPlat, new Color(0.12f, 0.28f, 0.55f));
            _stationObjects.Add(signPlat);
        }

        var signGate = InstantiateMetroModel("Metro_OverheadSign", c + new Vector3(0f, 5.4f, turnstileZ), Quaternion.identity);
        if (signGate != null)
        {
            SetColorRecursively(signGate, new Color(0.15f, 0.55f, 0.25f));
            _stationObjects.Add(signGate);
        }

        // High-definition CCTV security camera dome
        var cctv = InstantiateMetroModel("Metro_SecurityCCTV", c + new Vector3(7.5f, 4.6f, turnstileZ - 2f), Quaternion.Euler(20, -135, 0));
        if (cctv != null)
        {
            SetColorRecursively(cctv, new Color(0.15f, 0.15f, 0.15f));
            _stationObjects.Add(cctv);
        }

        // ── F. Emergency Exit Portals (East & West) ──
        var exitEast = CreateBoxProp("Metro_Emergency_Exit_East", c + new Vector3(38f, 2.2f, -12f), new Vector3(2.5f, 4.4f, 6.5f), new Color(0.08f, 0.85f, 0.25f));
        var exitWest = CreateBoxProp("Metro_Emergency_Exit_West", c + new Vector3(-38f, 2.2f, -12f), new Vector3(2.5f, 4.4f, 6.5f), new Color(0.08f, 0.85f, 0.25f));
        _evacuationExits.Add(exitEast.transform);
        _evacuationExits.Add(exitWest.transform);
        _stationObjects.Add(exitEast);
        _stationObjects.Add(exitWest);

        // Emergency flashing alarm beacon
        var alarmGO = new GameObject("Station_Alarm_Light");
        alarmGO.transform.position = c + new Vector3(0f, 6.2f, 0f);
        _alarmLight = alarmGO.AddComponent<Light>();
        _alarmLight.type = LightType.Point;
        _alarmLight.color = Color.red;
        _alarmLight.range = 55f;
        _alarmLight.intensity = 0f;
        _stationObjects.Add(alarmGO);
    }

    private GameObject InstantiateMetroModel(string modelName, Vector3 pos, Quaternion rot)
    {
        string path = $"Models/Metro/{modelName}";
        var prefab = Resources.Load<GameObject>(path);
        if (prefab == null)
        {
            #if UNITY_EDITOR
            prefab = UnityEditor.AssetDatabase.LoadAssetAtPath<GameObject>($"Assets/Models/Metro/{modelName}.obj");
            #endif
        }

        if (prefab != null)
        {
            var go = Instantiate(prefab, pos, rot, transform);
            go.name = modelName;
            return go;
        }

        var fallback = GameObject.CreatePrimitive(PrimitiveType.Cube);
        fallback.name = modelName;
        fallback.transform.position = pos;
        fallback.transform.rotation = rot;
        fallback.transform.SetParent(transform);
        return fallback;
    }

    private GameObject CreateBoxProp(string name, Vector3 pos, Vector3 size, Color col)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.SetParent(transform);
        go.transform.position = pos;
        go.transform.localScale = size;
        var mr = go.GetComponent<MeshRenderer>();
        if (mr != null) mr.material.color = col;
        var colComp = go.GetComponent<Collider>();
        if (colComp != null) Destroy(colComp);
        return go;
    }

    private void SetColorRecursively(GameObject root, Color c)
    {
        foreach (var mr in root.GetComponentsInChildren<MeshRenderer>())
            mr.material.color = c;
    }

    // =========================================================================
    //  2. Cinematic CCTV & Security Camera System
    // =========================================================================

    public void CycleCCTVCamera()
    {
        currentCCTV = (CCTVAngle)(((int)currentCCTV + 1) % 4);
        ApplyCCTVAngle(currentCCTV);
    }

public void ApplyCCTVAngle(CCTVAngle angle)
    {
        currentCCTV = angle;
        if (mainCamera == null) mainCamera = Camera.main;
        if (mainCamera == null) return;

        var sim = SimulationManager.Instance;
        Vector3 stationCentre = sim != null ? sim.Center() : Vector3.zero;
        Vector3 crowdCentre = GetLiveCrowdCentre(sim, stationCentre);

        mainCamera.orthographic = false;
        mainCamera.fieldOfView = 62f;
        mainCamera.nearClipPlane = 0.25f;
        mainCamera.farClipPlane = 160f;

        // Use the live crowd centre rather than an assumed spawn centre: presets can
        // place the PBD crowd away from the station origin.  These stand-off views
        // avoid both commuter clipping and platform/roof intersections.
        switch (angle)
        {
            case CCTVAngle.CCTV_Concourse:
                AimCamera(mainCamera, crowdCentre + new Vector3(0f, 4.0f, 18f),
                    crowdCentre + new Vector3(0f, 1.5f, 0f));
                break;

            case CCTVAngle.CCTV_Turnstiles:
                AimCamera(mainCamera, crowdCentre + new Vector3(18f, 6.0f, 16f),
                    crowdCentre + new Vector3(0f, 1.5f, 0f));
                break;

            case CCTVAngle.CCTV_Platform:
                AimCamera(mainCamera, crowdCentre + new Vector3(-18f, 7.0f, 16f),
                    crowdCentre + new Vector3(0f, 1.5f, 0f));
                break;

            case CCTVAngle.VR_CommuterEyeLevel:
                // Believable 1.7 m eye level with a clear two-metre stand-off
                // outside the crowd bound, rather than inside an occupied agent.
                AimCamera(mainCamera, crowdCentre + new Vector3(0f, 1.7f, 13f),
                    crowdCentre + new Vector3(0f, 1.7f, 0f));
                break;
        }
    }

private static void AimCamera(Camera camera, Vector3 position, Vector3 target)
    {
        Vector3 direction = target - position;
        if (direction.sqrMagnitude < 0.001f) return;
        camera.transform.SetPositionAndRotation(position,
            Quaternion.LookRotation(direction.normalized, Vector3.up));
    }

private static Vector3 GetLiveCrowdCentre(SimulationManager sim, Vector3 fallback)
    {
        if (sim == null || sim.LiveAgentCount == 0) return fallback;

        AgentBase first = sim.GetAgent(0);
        if (first == null) return fallback;

        var bounds = new Bounds(first.transform.position, Vector3.zero);
        for (int i = 1; i < sim.LiveAgentCount; i++)
        {
            AgentBase agent = sim.GetAgent(i);
            if (agent != null) bounds.Encapsulate(agent.transform.position);
        }
        return bounds.center;
    }



    public string GetCCTVLabel()
    {
        return currentCCTV switch
        {
            CCTVAngle.CCTV_Concourse => "[REC] CCTV 01 - CONCOURSE",
            CCTVAngle.CCTV_Turnstiles => "[REC] CCTV 02 - FARE GATES",
            CCTVAngle.CCTV_Platform => "[REC] CCTV 03 - PLATFORM BOARDING",
            _ => "VIEW: VR COMMUTER (1.7M)"
        };
    }

    // =========================================================================
    //  3. Train Arrival & Departure Presentation
    // =========================================================================

    public void TriggerTrainCycle()
    {
        if (_trainRoutine != null) StopCoroutine(_trainRoutine);
        _trainRoutine = StartCoroutine(TrainCycleRoutine());
    }

    private IEnumerator TrainCycleRoutine()
    {
        if (_trainCar == null) yield break;

        // 1. Train accelerates and departs platform
        isTrainAtPlatform = false;
        float elapsed = 0f;
        Vector3 start = _trainCar.transform.position;
        while (elapsed < 3.2f)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.SmoothStep(0f, 1f, elapsed / 3.2f);
            _trainCar.transform.position = Vector3.Lerp(start, _trainDepartedPos, t);
            yield return null;
        }

        // 2. Station empty wait interval
        yield return new WaitForSeconds(2.0f);

        // 3. New Train arrives with smooth deceleration and braking
        _trainCar.transform.position = _trainArrivedPos - new Vector3(70f, 0f, 0f);
        start = _trainCar.transform.position;
        elapsed = 0f;
        while (elapsed < 4.0f)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.SmoothStep(0f, 1f, elapsed / 4.0f);
            _trainCar.transform.position = Vector3.Lerp(start, _trainArrivedPos, t);
            yield return null;
        }

        _trainCar.transform.position = _trainArrivedPos;
        isTrainAtPlatform = true;

        // Boarding / alighting scenario refresh
        SimulationManager.Instance?.SetPreset(ScenePreset.Bidirectional);
    }

    // =========================================================================
    //  4. Turnstile Queuing
    // =========================================================================

    private void ProcessTurnstileQueuing()
    {
        var sim = SimulationManager.Instance;
        if (sim == null) return;

        Vector3 c = sim.Center();
        float gateZ = c.z - 8f;

        for (int i = 0; i < sim.LiveAgentCount; i++)
        {
            var agent = sim.GetAgent(i);
            if (agent == null || agent._agent == null) continue;

            Vector3 pos = agent.transform.position;
            if (Mathf.Abs(pos.z - gateZ) < 1.5f && Mathf.Abs(pos.x - c.x) <= 8f)
            {
                if (!_agentTapDelays.ContainsKey(i))
                    _agentTapDelays[i] = turnstileCardTapDelay;

                if (_agentTapDelays[i] > 0f)
                {
                    _agentTapDelays[i] -= Time.deltaTime;
                    agent._agent.speed = 0.45f;
                }
                else
                {
                    agent._agent.speed = sim.agentSpeed;
                }
            }
            else
            {
                if (_agentTapDelays.ContainsKey(i) && Mathf.Abs(pos.z - gateZ) > 3f)
                    _agentTapDelays.Remove(i);
                if (agent._agent.speed < sim.agentSpeed)
                    agent._agent.speed = sim.agentSpeed;
            }
        }
    }

    // =========================================================================
    //  5. Emergency Evacuation
    // =========================================================================

    public void ToggleEmergencyEvacuation()
    {
        isEvacuationActive = !isEvacuationActive;

        if (isEvacuationActive)
        {
            evacuationTimer = 0f;
            if (_alarmLight != null) _alarmLight.intensity = 3f;

            var sim = SimulationManager.Instance;
            if (sim != null && _evacuationExits.Count > 0)
            {
                Transform exitA = _evacuationExits[0];
                Transform exitB = _evacuationExits.Count > 1 ? _evacuationExits[1] : exitA;

                for (int i = 0; i < sim.LiveAgentCount; i++)
                {
                    Transform chosenExit = (i % 2 == 0) ? exitA : exitB;
                    var agent = sim.GetAgent(i);
                    if (agent != null)
                    {
                        agent.Goals = new[] { chosenExit };
                        // Use the manager's local NavMesh recovery path.  A PBD
                        // correction may have moved this Transform since spawning.
                        sim.TrySetAgentDestination(agent, chosenExit.position);
                    }
                }
            }
        }
        else
        {
            if (_alarmLight != null) _alarmLight.intensity = 0f;
            SimulationManager.Instance?.Respawn();
        }
    }

    private void UpdateEvacuationCount()
    {
        var sim = SimulationManager.Instance;
        if (sim == null) return;

        int remaining = 0;
        for (int i = 0; i < sim.LiveAgentCount; i++)
        {
            var agent = sim.GetAgent(i);
            if (agent != null && agent.transform.position.y > -5f)
            {
                float distA = Vector3.Distance(agent.transform.position, _evacuationExits[0].position);
                float distB = _evacuationExits.Count > 1 ? Vector3.Distance(agent.transform.position, _evacuationExits[1].position) : distA;
                if (Mathf.Min(distA, distB) > 3f) remaining++;
            }
        }
        remainingEvacuees = remaining;
    }
}
