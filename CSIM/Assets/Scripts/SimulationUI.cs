using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

/// <summary>
/// Metro OCC Digital Twin UI.
///
///  LEFT  (top-left)  : Operations Control Center: live status plus three tabs
///                      (Solver &amp; Scene / Tuning / Station Ops). Locked while a
///                      PAGP run is in progress so clicks cannot disturb measurements.
///  RIGHT (top-right) : PAGP Performance Lab: guided demo, five experiments, and a
///                      scrollable card with Concepts / Live Demo / Results tabs.
///
/// The Visual-LOD diagnostics overlay draws at the bottom centre, so nothing overlaps.
/// </summary>
public class SimulationUI : MonoBehaviour
{
    [Header("Appearance")]
    public float leftPanelWidth = 400f;
    public float rightPanelWidth = 520f;
    public float resultsCardHeight = 470f;
    public int fontSize = 16;
    [Tooltip("Upper bound of the commuter slider. Each commuter is an animated 3D character, so very large counts hurt frame rate.")]
    public int maxCommuterSlider = 1500;

    // Palette
    private static readonly Color ColorBgPanel     = new Color(0.043f, 0.075f, 0.125f, 0.94f);
    private static readonly Color ColorCard        = new Color(0.06f, 0.09f, 0.14f, 0.97f);
    private static readonly Color ColorCyanAccent  = new Color(0.0f, 0.90f, 1.0f);
    private static readonly Color ColorActiveBtn   = new Color(0.0f, 0.70f, 0.85f);
    private static readonly Color ColorPresetOn    = new Color(0.1f, 0.75f, 0.6f);
    private static readonly Color ColorInactiveBtn = new Color(0.12f, 0.17f, 0.24f);
    private static readonly Color ColorHeaderTxt   = new Color(0.85f, 0.92f, 1.0f);
    private static readonly Color ColorSubTxt      = new Color(0.55f, 0.65f, 0.78f);
    private static readonly Color ColorOrange      = new Color(1.0f, 0.65f, 0.15f);
    private static readonly Color ColorTabActive   = new Color(0.08f, 0.45f, 0.58f);

    private static readonly SolverMode[] Modes =
    {
        SolverMode.Force, SolverMode.ForceJobs, SolverMode.PBD,
        SolverMode.PBDJobs, SolverMode.PBDCLODJobs, SolverMode.PBDCuda,
        SolverMode.PBDCLODCullingJobs
    };
    private static readonly string[] ModeLabels = { "Force", "Force Jobs", "PBD", "PBD Jobs", "CLOD", "CUDA (GPU)", "Cull" };
    private static readonly string[] ModeInfo =
    {
        "Anticipatory social forces, 1 CPU thread.",
        "Social forces on all CPU cores (Unity Jobs + Burst).",
        "Position-Based Dynamics, 1 CPU thread.",
        "PBD on all CPU cores (Unity Jobs + Burst).",
        "PBD Jobs + camera-distance LOD tiers.",
        "Native CUDA PBD: GPU spatial hash + 4-factor LOD.",
        "CLOD + frustum culling of off-screen commuters."
    };

    private static readonly ScenePreset[] Presets =
    {
        ScenePreset.StationFlow,
        ScenePreset.Random, ScenePreset.Circle, ScenePreset.Crossing, ScenePreset.Bidirectional,
        ScenePreset.Bottleneck, ScenePreset.DoubleBottleneck, ScenePreset.MergeLane
    };
    private static readonly string[] PresetLabels = { "Metro Commute", "Open Concourse", "Circle", "Crossing", "Corridor", "Single Gap", "Double Gap", "Lane Merge" };
    private static readonly string[] PresetInfo =
    {
        "Realistic: street → fare gates → platform → train, on a live timetable.",
        "Test: commuters wander between open-floor goals.",
        "Test: a ring converges on one point (dense contacts).",
        "Test: one block walks diagonally across open floor.",
        "Test: two counter-flows share one walled corridor.",
        "Test: the whole crowd squeezes through one gap.",
        "Test: two gaps in sequence.",
        "Test: two lanes merge into one corridor."
    };

    private SimulationManager _mgr;
    private PAGPPerformanceLab _lab;
    private Font _font;

    // Left panel
    private GameObject _leftBody;
    private Text _leftToggleLbl, _statusLbl, _telemetryLbl, _lockLbl, _solverInfoLbl, _presetInfoLbl;
    private CanvasGroup _leftPagesGroup;
    private readonly GameObject[] _leftPages = new GameObject[3];
    private readonly Image[] _leftTabImgs = new Image[3];
    private readonly Text[] _leftTabTxts = new Text[3];
    private Image[] _solverImgs; private Text[] _solverTxts;
    private Image[] _presetImgs; private Text[] _presetTxts;
    private SolverMode _shownSolver = (SolverMode)(-1);
    private ScenePreset _shownPreset = (ScenePreset)(-1);

    private Slider _agentSld, _speedSld, _radiusSld, _neighSld, _pbdIterSld, _pbdKsiSld, _forceKSld, _forceT0Sld;
    private Text _agentLbl, _speedLbl, _radiusLbl, _neighLbl, _pbdIterLbl, _pbdKsiLbl, _forceKLbl, _forceT0Lbl;
    private GameObject _pbdSec, _forceSec;
    private Text _evacLbl, _pauseLbl, _visualLodLbl, _gridLbl, _ringLbl, _fovLbl;
    private FocusPointSphere _fov;
    private FocusPointSphere Fov => _fov != null ? _fov : (_fov = FindObjectOfType<FocusPointSphere>());
    private GameObject _captionBox;
    private Text _captionTxt;
    private PagpVisualizer Viz => _mgr != null ? _mgr.GetComponent<PagpVisualizer>() : null;

    // Right panel
    private GameObject _rightBody;
    private Text _rightToggleLbl, _labStatusLbl, _cardHeaderLbl, _cardTxt;
    private RectTransform _progressFill;
    private Button[] _runButtons;
    private ScrollRect _scroll;
    private readonly Image[] _tabImgs = new Image[3];
    private readonly Text[] _tabTxts = new Text[3];
    private int _tab;            // 0 Concepts, 1 Live Demo, 2 Results
    private bool _contentDirty = true;
    private float _nextUiTick;

    // ─────────────────────────────────────────────────────────────────────────
    void Start()
    {
        _mgr = SimulationManager.Instance;
        if (_mgr == null) { enabled = false; return; }

        if (PAGPPerformanceLab.Instance == null) gameObject.AddComponent<PAGPPerformanceLab>();
        _lab = PAGPPerformanceLab.Instance;
        if (_lab != null)
        {
            _lab.OnLabStateChanged += MarkDirty;
            _lab.OnRunFinished += HandleRunFinished;
        }

        // Buttons and scrolling need an EventSystem; create one if the scene lacks it.
        if (EventSystem.current == null && FindObjectOfType<EventSystem>() == null)
        {
            var es = new GameObject("EventSystem");
            es.AddComponent<EventSystem>();
            es.AddComponent<StandaloneInputModule>();
        }

        _font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        BuildCanvas();
    }

    void OnDestroy()
    {
        if (_lab != null)
        {
            _lab.OnLabStateChanged -= MarkDirty;
            _lab.OnRunFinished -= HandleRunFinished;
        }
    }

    private void MarkDirty() => _contentDirty = true;
    private void HandleRunFinished() => SetTab(2);

    // ─────────────────────────────────────────────────────────────────────────
    void Update()
    {
        if (_mgr == null) return;

        // Button highlights: only when the state actually changes.
        if (_mgr.activeSolver != _shownSolver)
        {
            _shownSolver = _mgr.activeSolver;
            for (int i = 0; i < Modes.Length; i++)
            {
                bool on = Modes[i] == _shownSolver;
                _solverImgs[i].color = on ? ColorActiveBtn : ColorInactiveBtn;
                _solverTxts[i].color = on ? Color.black : ColorHeaderTxt;
                if (on) _solverInfoLbl.text = ModeInfo[i];
            }
            bool isPBD = _shownSolver != SolverMode.Force && _shownSolver != SolverMode.ForceJobs;
            _pbdSec.SetActive(isPBD);
            _forceSec.SetActive(!isPBD);
        }
        if (_mgr.activePreset != _shownPreset)
        {
            _shownPreset = _mgr.activePreset;
            for (int i = 0; i < Presets.Length; i++)
            {
                bool on = Presets[i] == _shownPreset;
                _presetImgs[i].color = on ? ColorPresetOn : ColorInactiveBtn;
                _presetTxts[i].color = on ? Color.black : ColorHeaderTxt;
                if (on) _presetInfoLbl.text = PresetInfo[i];
            }
        }

        // Everything else refreshes at ~8 Hz (string formatting + layout rebuilds are not free).
        if (Time.unscaledTime < _nextUiTick) return;
        _nextUiTick = Time.unscaledTime + 0.12f;

        bool running = _lab != null && _lab.isRunning;
        RefreshStatus();
        RefreshLab(running);

        _leftPagesGroup.interactable = !running;
        _leftPagesGroup.alpha = running ? 0.55f : 1f;
        _lockLbl.gameObject.SetActive(running);

        // Live tab shows live numbers; other tabs only re-render when something changed.
        if (_contentDirty || (_tab == 1 && running)) RefreshCard();

        SetLbl(_agentLbl, $"Commuters: {(int)_agentSld.value}  (press Apply)");
        SetLbl(_speedLbl, $"Walking speed: {_speedSld.value:F2} m/s");
        SetLbl(_radiusLbl, $"Body radius: {_radiusSld.value:F2} m");
        SetLbl(_neighLbl, $"Perception distance: {_neighSld.value:F1} m");
        SetLbl(_pbdIterLbl, $"PBD iterations: {(int)_pbdIterSld.value}");
        SetLbl(_pbdKsiLbl, $"Relaxation (ksi): {_pbdKsiSld.value:F3}");
        SetLbl(_forceKLbl, $"Stiffness (k): {_forceKSld.value:F2}");
        SetLbl(_forceT0Lbl, $"Look-ahead (t0): {_forceT0Sld.value:F1} s");
        if ((int)_pbdIterSld.value != _mgr.pbdIterations)
            _pbdIterSld.SetValueWithoutNotify(_mgr.pbdIterations);

        var metro = MetroStationManager.Instance;
        SetLbl(_evacLbl, metro != null && metro.isEvacuationActive ? "Stop Evacuation Alarm" : "Trigger Evacuation Alarm");
        SetLbl(_pauseLbl, Time.timeScale > 0f ? "Pause" : "Resume");
        var vl = _mgr.GetComponent<PBDCudaVisualLodController>();
        SetLbl(_visualLodLbl, vl != null && vl.showDiagnostics ? "Hide Render LOD Stats" : "Render LOD Stats");
        var viz = Viz;
        SetLbl(_gridLbl, viz != null && viz.ShowHashGrid ? "Hide Heat Map" : "Hash-Grid Heat Map");
        SetLbl(_ringLbl, viz != null && viz.ShowLodRings ? "Hide LOD Rings" : "LOD Rings");
        SetLbl(_fovLbl, Fov != null && Fov.DisplayFoveation ? "Hide Foveated Animation" : "Foveated Animation");
        RefreshCaption(running, viz);
    }

    private void RefreshStatus()
    {
        float fps = 1.0f / Mathf.Max(Time.smoothDeltaTime, 1e-4f);
        _mgr.GetFidelityCounts(out int high, out int med, out int low);
        var metro = MetroStationManager.Instance;

        var flow = _mgr.GetComponent<StationFlowController>();
        string s = $"<b>{_mgr.activeSolver}</b>  |  <color=#00E5FF>{fps:F0} FPS</color>  |  Commuters: <b>{_mgr.LiveAgentCount}</b>\n" +
                   $"Sim LOD  High {high} / Med {med} / Low {low}\n" +
                   (metro != null ? $"{metro.GetCCTVLabel()}  |  {metro.ServiceStatus}" : "");
        if (flow != null && flow.Active)
            s += $"\nWaiting on platform {flow.WaitingCount}  |  On train {flow.OnTrainCount}  |  Boarded {flow.TotalBoarded}";
        if (metro != null && metro.isEvacuationActive)
            s += $"\n<color=#FF3B30><b>EVACUATION</b> {metro.evacuationTimer:F1}s, remaining {metro.remainingEvacuees}</color>";
        _statusLbl.text = s;

        var cuda = _mgr.GetComponent<PBDCudaSolver>();
        if (cuda != null && cuda.IsInitialized && _mgr.activeSolver == SolverMode.PBDCuda)
        {
            var t = cuda.LastTimings;
            _telemetryLbl.text = $"GPU step <b>{t.totalMs:F2} ms</b>  |  kernels {t.solveMs:F2}  |  hash {t.hashAgentsMs:F2}\n" +
                                 $"PCIe host→GPU {t.h2dMs:F2} ms  |  GPU→host {t.d2hMs:F2} ms";
        }
        else
        {
            float solverMs = ProfilerManager.Instance != null ? ProfilerManager.Instance.GetAverageMs("Solver") : float.NaN;
            _telemetryLbl.text = float.IsNaN(solverMs)
                ? "CPU solver timing: collecting…"
                : $"CPU solver step <b>{solverMs:F2} ms</b> (rolling average)";
        }
    }

    private void RefreshLab(bool running)
    {
        if (_lab == null) return;
        _labStatusLbl.text = running
            ? $"<color=#FFA500><b>RUNNING</b></color>  {_lab.currentBenchmarkName}"
            : $"<color=#00E5FF><b>READY</b></color>  {_lab.currentBenchmarkName}";
        _progressFill.anchorMax = new Vector2(running ? Mathf.Clamp01(_lab.benchmarkProgress) : (_lab.benchmarkProgress >= 1f ? 1f : 0f), 1f);
        for (int i = 0; i < _runButtons.Length; i++) _runButtons[i].interactable = !running;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Canvas
    // ─────────────────────────────────────────────────────────────────────────
    void BuildCanvas()
    {
        var cGO = new GameObject("OCC_Canvas");
        cGO.transform.SetParent(transform);
        var cv = cGO.AddComponent<Canvas>();
        cv.renderMode = RenderMode.ScreenSpaceOverlay;
        cv.sortingOrder = 100;
        var sc = cGO.AddComponent<CanvasScaler>();
        sc.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        sc.referenceResolution = new Vector2(1920, 1080);
        sc.matchWidthOrHeight = 0.5f;
        cGO.AddComponent<GraphicRaycaster>();

        BuildLeftPanel(cGO.transform);
        BuildRightPanel(cGO.transform);
        BuildCaption(cGO.transform);
        SetLeftTab(0);
        SetTab(0);
    }

    // ── CAPTION: one plain sentence about what is on screen ──────────────────
    void BuildCaption(Transform root)
    {
        _captionBox = Box(root, "Caption", new Color(0.02f, 0.05f, 0.09f, 0.82f));
        var rt = _captionBox.GetComponent<RectTransform>();
        rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0.5f, 1f);
        rt.anchoredPosition = new Vector2((leftPanelWidth - rightPanelWidth) * 0.5f, -12f);
        rt.sizeDelta = new Vector2(1920f - leftPanelWidth - rightPanelWidth - 72f, 10f);
        var vl = _captionBox.AddComponent<VerticalLayoutGroup>();
        vl.padding = new RectOffset(14, 14, 8, 8);
        vl.childControlWidth = vl.childControlHeight = true;
        vl.childForceExpandHeight = false;
        _captionBox.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;

        var go = new GameObject("CaptionText", typeof(RectTransform));
        go.transform.SetParent(_captionBox.transform, false);
        _captionTxt = go.AddComponent<Text>();
        _captionTxt.font = _font; _captionTxt.fontSize = fontSize; _captionTxt.color = Color.white;
        _captionTxt.alignment = TextAnchor.MiddleCenter; _captionTxt.supportRichText = true;
        _captionTxt.horizontalOverflow = HorizontalWrapMode.Wrap; _captionTxt.raycastTarget = false;
        _captionBox.SetActive(false);
    }

    void RefreshCaption(bool running, PagpVisualizer viz)
    {
        string text = "";
        if (running && _lab != null && !string.IsNullOrEmpty(_lab.Caption))
            text = (_lab.IsShowcase ? $"<color=#00E5FF><b>{_lab.PhaseIndex + 1}/5</b></color>  " : "") + _lab.Caption;

        if (viz != null && viz.ShowHashGrid)
            text += (text.Length > 0 ? "\n" : "") +
                    $"<size={fontSize - 3}><color=#33CCFF>■</color> 1-2   <color=#33FF66>■</color> 3-5   <color=#FFCC1A>■</color> 6-9   " +
                    $"<color=#FF4D33>■</color> 10+ commuters per {viz.CellSize:F0} m cell  ·  {viz.OccupiedCells} occupied, max {viz.MaxPerCell}</size>";
        if (viz != null && viz.ShowLodRings)
            text += (text.Length > 0 ? "\n" : "") +
                    $"<size={fontSize - 3}><color=#33F266>●</color> High detail   <color=#FFBF1A>●</color> Medium   " +
                    "<color=#4D99FF>●</color> Low (solved every 4th step)</size>";
        if (Fov != null && Fov.DisplayFoveation)
            text += (text.Length > 0 ? "\n" : "") +
                    $"<size={fontSize - 3}>Foveated animation: <color=#33F266>●</color> full rate (centre of view)   " +
                    "<color=#FF4D4D>●</color> \u2192 <color=#9A9A9A>●</color> fewer animation updates further out   no sphere = paused</size>";

        bool show = text.Length > 0;
        if (_captionBox.activeSelf != show) _captionBox.SetActive(show);
        SetLbl(_captionTxt, text);
    }

    // ── LEFT: Operations Control Center ──────────────────────────────────────
    void BuildLeftPanel(Transform root)
    {
        var p = Panel(root, "OCC_LeftPanel", new Vector2(0, 1), new Vector2(12, -12), leftPanelWidth);

        var header = HRow(p, 4, 30);
        Lbl(header, "OPERATIONS CONTROL CENTER", fontSize, FontStyle.Bold, ColorCyanAccent, 28).GetComponent<LayoutElement>().flexibleWidth = 1;
        var tg = MkBtn(header, "Hide", fontSize - 3, () => Toggle(_leftBody, _leftToggleLbl));
        tg.GetComponent<LayoutElement>().preferredWidth = 64;
        _leftToggleLbl = tg.GetComponentInChildren<Text>();

        _leftBody = SecGO(p, "LeftBody");
        var body = _leftBody.transform;

        _statusLbl = Lbl(body, "", fontSize - 3, FontStyle.Normal, ColorHeaderTxt, 92);
        var tel = Box(body, "Telemetry", new Color(0.08f, 0.12f, 0.18f, 0.9f));
        tel.AddComponent<LayoutElement>().preferredHeight = 38;
        var telVL = tel.AddComponent<VerticalLayoutGroup>();
        telVL.padding = new RectOffset(6, 6, 3, 3);
        telVL.childControlWidth = telVL.childControlHeight = true;
        _telemetryLbl = Lbl(tel.transform, "", fontSize - 3, FontStyle.Normal, ColorCyanAccent, 32);

        var tabs = HRow(body, 3, 26);
        string[] names = { "Solver & Scene", "Tuning", "Station Ops" };
        for (int i = 0; i < 3; i++)
        {
            int idx = i;
            var b = MkBtn(tabs, names[i], fontSize - 3, () => SetLeftTab(idx));
            _leftTabImgs[i] = b.GetComponent<Image>();
            _leftTabTxts[i] = b.GetComponentInChildren<Text>();
        }

        _lockLbl = Lbl(body, "<i>Locked while a PAGP run is in progress (press Stop to unlock).</i>", fontSize - 4, FontStyle.Normal, ColorOrange, 16);
        _lockLbl.gameObject.SetActive(false);

        var pages = SecGO(body, "Pages");
        _leftPagesGroup = pages.AddComponent<CanvasGroup>();
        for (int i = 0; i < 3; i++) _leftPages[i] = SecGO(pages.transform, "Page" + i);

        BuildSolverScenePage(_leftPages[0].transform);
        BuildTuningPage(_leftPages[1].transform);
        BuildStationOpsPage(_leftPages[2].transform);
    }

    void BuildSolverScenePage(Transform page)
    {
        Lbl(page, "PARALLEL SOLVER", fontSize - 2, FontStyle.Bold, ColorCyanAccent, 18);
        _solverImgs = new Image[Modes.Length];
        _solverTxts = new Text[Modes.Length];
        Transform row = null;
        for (int i = 0; i < Modes.Length; i++)
        {
            if (i % 3 == 0) row = HRow(page, 3, 26);
            int idx = i;
            var b = MkBtn(row, ModeLabels[i], fontSize - 3, () => _mgr.SetSolverMode(Modes[idx]));
            _solverImgs[i] = b.GetComponent<Image>();
            _solverTxts[i] = b.GetComponentInChildren<Text>();
        }
        PadRow(row, Modes.Length);
        _solverInfoLbl = Lbl(page, "", fontSize - 4, FontStyle.Italic, ColorSubTxt, 18);

        HSep(page);
        Lbl(page, "SCENARIO  (Metro Commute = realistic; others = synthetic stress tests)", fontSize - 3, FontStyle.Bold, ColorCyanAccent, 18);
        _presetImgs = new Image[Presets.Length];
        _presetTxts = new Text[Presets.Length];
        for (int i = 0; i < Presets.Length; i++)
        {
            if (i % 3 == 0) row = HRow(page, 3, 26);
            int idx = i;
            var b = MkBtn(row, PresetLabels[i], fontSize - 3, () => _mgr.SetPreset(Presets[idx]));
            _presetImgs[i] = b.GetComponent<Image>();
            _presetTxts[i] = b.GetComponentInChildren<Text>();
        }
        PadRow(row, Presets.Length);
        _presetInfoLbl = Lbl(page, "", fontSize - 4, FontStyle.Italic, ColorSubTxt, 18);
    }

    void BuildTuningPage(Transform page)
    {
        Lbl(page, "CROWD", fontSize - 2, FontStyle.Bold, ColorCyanAccent, 18);
        (_agentSld, _agentLbl) = MkSlider(page, Mathf.Min(_mgr.agentCount, maxCommuterSlider), 10, maxCommuterSlider, true, v => _mgr.agentCount = (int)v);
        (_speedSld, _speedLbl) = MkSlider(page, _mgr.agentSpeed, 0.5f, 15f, false, v => _mgr.agentSpeed = v);
        (_radiusSld, _radiusLbl) = MkSlider(page, _mgr.agentRadius, 0.1f, 2f, false, v => _mgr.agentRadius = v);
        (_neighSld, _neighLbl) = MkSlider(page, _mgr.neighborDist, 1f, 20f, false, v => _mgr.neighborDist = v);

        _pbdSec = SecGO(page, "PBDSec");
        Lbl(_pbdSec.transform, "PBD SOLVER", fontSize - 2, FontStyle.Bold, ColorCyanAccent, 18);
        (_pbdIterSld, _pbdIterLbl) = MkSlider(_pbdSec.transform, _mgr.pbdIterations, 1, 20, true, v => _mgr.pbdIterations = (int)v);
        (_pbdKsiSld, _pbdKsiLbl) = MkSlider(_pbdSec.transform, _mgr.pbdKsi, 0.01f, 1f, false, v => _mgr.pbdKsi = v);

        _forceSec = SecGO(page, "ForceSec");
        Lbl(_forceSec.transform, "FORCE SOLVER", fontSize - 2, FontStyle.Bold, ColorOrange, 18);
        (_forceKSld, _forceKLbl) = MkSlider(_forceSec.transform, _mgr.forceK, 0.1f, 5f, false, v => _mgr.forceK = v);
        (_forceT0Sld, _forceT0Lbl) = MkSlider(_forceSec.transform, _mgr.forceT0, 0.5f, 10f, false, v => _mgr.forceT0 = v);

        var apply = MkBtn(page, "Apply Commuter Count & Respawn", fontSize - 2, () => _mgr.Respawn());
        apply.GetComponent<Image>().color = new Color(0.12f, 0.45f, 0.25f);
        Lbl(page, "<i>Speed, radius and solver values apply live. The commuter count needs a respawn.</i>", fontSize - 4, FontStyle.Normal, ColorSubTxt, 30);
    }

    void BuildStationOpsPage(Transform page)
    {
        Lbl(page, "CAMERAS & TRANSIT", fontSize - 2, FontStyle.Bold, ColorCyanAccent, 18);
        var r1 = HRow(page, 4, 28);
        MkBtn(r1, "Next CCTV Angle", fontSize - 3, () => MetroStationManager.Instance?.CycleCCTVCamera());
        var train = MkBtn(r1, "Cycle Train", fontSize - 3, () => MetroStationManager.Instance?.TriggerTrainCycle());
        train.GetComponent<Image>().color = new Color(0.12f, 0.38f, 0.32f);

        HSep(page);
        Lbl(page, "EMERGENCY", fontSize - 2, FontStyle.Bold, ColorCyanAccent, 18);
        var evac = MkBtn(page, "Trigger Evacuation Alarm", fontSize - 2, () => MetroStationManager.Instance?.ToggleEmergencyEvacuation());
        evac.GetComponent<Image>().color = new Color(0.55f, 0.15f, 0.15f);
        evac.GetComponent<LayoutElement>().preferredHeight = 28;
        _evacLbl = evac.GetComponentInChildren<Text>();

        HSep(page);
        Lbl(page, "PAGP OVERLAYS (explain what the GPU is doing)", fontSize - 2, FontStyle.Bold, ColorCyanAccent, 18);
        var ov = HRow(page, 4, 28);
        _gridLbl = MkBtn(ov, "Hash-Grid Heat Map", fontSize - 3, () => { var v = Viz; if (v != null) v.SetHashGrid(!v.ShowHashGrid); }).GetComponentInChildren<Text>();
        _ringLbl = MkBtn(ov, "LOD Rings", fontSize - 3, () => { var v = Viz; if (v != null) v.SetLodRings(!v.ShowLodRings); }).GetComponentInChildren<Text>();
        Lbl(page, "CGVR OVERLAY (animation detail follows the viewer's gaze)", fontSize - 3, FontStyle.Bold, ColorCyanAccent, 18);
        var ov2 = HRow(page, 4, 28);
        _fovLbl = MkBtn(ov2, "Foveated Animation", fontSize - 3, () => { var f = Fov; if (f != null) f.DisplayFoveation = !f.DisplayFoveation; }).GetComponentInChildren<Text>();

        HSep(page);
        Lbl(page, "SIMULATION", fontSize - 2, FontStyle.Bold, ColorCyanAccent, 18);
        var r2 = HRow(page, 4, 28);
        _pauseLbl = MkBtn(r2, "Pause", fontSize - 3, TogglePause).GetComponentInChildren<Text>();
        var respawn = MkBtn(r2, "Respawn Crowd", fontSize - 3, () => _mgr.Respawn());
        respawn.GetComponent<Image>().color = new Color(0.12f, 0.45f, 0.25f);
        var r3 = HRow(page, 4, 28);
        _visualLodLbl = MkBtn(r3, "Render LOD Stats", fontSize - 3, ToggleVisualLodDiagnostics).GetComponentInChildren<Text>();
        var reset = MkBtn(r3, "Reset Scene", fontSize - 3, ResetScene);
        reset.GetComponent<Image>().color = new Color(0.16f, 0.38f, 0.30f);
        Lbl(page, "<i>Reset: CUDA solver, Metro Commute, 300 commuters, alarm off.</i>", fontSize - 4, FontStyle.Normal, ColorSubTxt, 16);
    }

    // ── RIGHT: PAGP Performance Lab ──────────────────────────────────────────
    void BuildRightPanel(Transform root)
    {
        var p = Panel(root, "PAGP_RightPanel", new Vector2(1, 1), new Vector2(-12, -12), rightPanelWidth);

        var header = HRow(p, 4, 30);
        Lbl(header, "PAGP PERFORMANCE LAB", fontSize, FontStyle.Bold, ColorCyanAccent, 28).GetComponent<LayoutElement>().flexibleWidth = 1;
        var tg = MkBtn(header, "Hide", fontSize - 3, () => Toggle(_rightBody, _rightToggleLbl));
        tg.GetComponent<LayoutElement>().preferredWidth = 64;
        _rightToggleLbl = tg.GetComponentInChildren<Text>();

        _rightBody = SecGO(p, "RightBody");
        var body = _rightBody.transform;

        _labStatusLbl = Lbl(body, "", fontSize - 3, FontStyle.Normal, ColorHeaderTxt, 18);
        var bar = Box(body, "Progress", new Color(0.10f, 0.14f, 0.20f));
        bar.AddComponent<LayoutElement>().preferredHeight = 6;
        var fill = Box(bar.transform, "Fill", ColorCyanAccent);
        _progressFill = fill.GetComponent<RectTransform>();
        _progressFill.anchorMin = Vector2.zero;
        _progressFill.anchorMax = new Vector2(0, 1);
        _progressFill.offsetMin = _progressFill.offsetMax = Vector2.zero;

        var demo = MkBtn(body, "▶  RUN GUIDED DEMO  (60 s, all 5 concepts measured live)", fontSize - 1, () =>
        {
            _lab?.Run60SecondShowcaseSequence();
            SetTab(1);
        });
        demo.GetComponent<Image>().color = new Color(0.05f, 0.50f, 0.45f);
        demo.GetComponent<LayoutElement>().preferredHeight = 34;

        Lbl(body, "Or run one experiment as a sweep (CSV export, ~15-35 s each):", fontSize - 4, FontStyle.Normal, ColorSubTxt, 16);
        var e1 = HRow(body, 4, 26);
        var b1 = MkBtn(e1, "1 CPU vs CUDA", fontSize - 3, () => RunExperiment(() => _lab.RunComprehensiveSuite()));
        var b2 = MkBtn(e1, "2 Spatial Hash", fontSize - 3, () => RunExperiment(() => _lab.RunSpatialHashAnalysis()));
        var b3 = MkBtn(e1, "3 PBD Converge", fontSize - 3, () => RunExperiment(() => _lab.RunPBDConvergenceTest()));
        var e2 = HRow(body, 4, 26);
        var b4 = MkBtn(e2, "4 LOD Ablation", fontSize - 3, () => RunExperiment(() => _lab.RunLODAblationStudy()));
        var b5 = MkBtn(e2, "5 AoS vs SoA", fontSize - 3, () => RunExperiment(() => _lab.RunMemoryLayoutBenchmark()));
        var stop = MkBtn(e2, "■ Stop", fontSize - 3, () => _lab?.StopBenchmark());
        stop.GetComponent<Image>().color = new Color(0.50f, 0.15f, 0.15f);
        _runButtons = new[] { demo, b1, b2, b3, b4, b5 };
        Color[] tints = { new Color(0.08f, 0.30f, 0.45f), new Color(0.18f, 0.28f, 0.42f), new Color(0.35f, 0.22f, 0.40f),
                          new Color(0.12f, 0.35f, 0.28f), new Color(0.20f, 0.35f, 0.45f) };
        for (int i = 0; i < 5; i++) _runButtons[i + 1].GetComponent<Image>().color = tints[i];

        HSep(body);
        var tabRow = HRow(body, 3, 28);
        string[] names = { "Concepts", "Live Demo", "Results" };
        for (int i = 0; i < 3; i++)
        {
            int idx = i;
            var b = MkBtn(tabRow, names[i], fontSize - 2, () => SetTab(idx));
            _tabImgs[i] = b.GetComponent<Image>();
            _tabTxts[i] = b.GetComponentInChildren<Text>();
        }

        _cardHeaderLbl = Lbl(body, "", fontSize - 1, FontStyle.Bold, ColorCyanAccent, 22);
        BuildScrollCard(body);
    }

    void BuildScrollCard(Transform parent)
    {
        var card = Box(parent, "ResultsCard", ColorCard);
        card.AddComponent<LayoutElement>().preferredHeight = resultsCardHeight;

        var viewport = new GameObject("Viewport", typeof(RectTransform), typeof(RectMask2D));
        viewport.transform.SetParent(card.transform, false);
        var vRT = viewport.GetComponent<RectTransform>();
        vRT.anchorMin = Vector2.zero; vRT.anchorMax = Vector2.one;
        vRT.offsetMin = new Vector2(10, 8); vRT.offsetMax = new Vector2(-18, -8);

        var txtGO = new GameObject("Content", typeof(RectTransform));
        txtGO.transform.SetParent(viewport.transform, false);
        var cRT = txtGO.GetComponent<RectTransform>();
        cRT.anchorMin = new Vector2(0, 1); cRT.anchorMax = new Vector2(1, 1); cRT.pivot = new Vector2(0.5f, 1);
        cRT.offsetMin = cRT.offsetMax = Vector2.zero;
        _cardTxt = txtGO.AddComponent<Text>();
        _cardTxt.font = _font; _cardTxt.fontSize = fontSize - 3; _cardTxt.color = ColorHeaderTxt;
        _cardTxt.supportRichText = true; _cardTxt.alignment = TextAnchor.UpperLeft;
        _cardTxt.horizontalOverflow = HorizontalWrapMode.Wrap;
        _cardTxt.verticalOverflow = VerticalWrapMode.Overflow;
        _cardTxt.lineSpacing = 1.1f;
        _cardTxt.raycastTarget = false;
        txtGO.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;

        // Slim vertical scrollbar
        var sbGO = Box(card.transform, "Scrollbar", new Color(0.10f, 0.14f, 0.20f));
        var sbRT = sbGO.GetComponent<RectTransform>();
        sbRT.anchorMin = new Vector2(1, 0); sbRT.anchorMax = Vector2.one; sbRT.pivot = new Vector2(1, 0.5f);
        sbRT.sizeDelta = new Vector2(8, -16); sbRT.anchoredPosition = new Vector2(-4, 0);
        var area = new GameObject("Area", typeof(RectTransform));
        area.transform.SetParent(sbGO.transform, false);
        var aRT = area.GetComponent<RectTransform>();
        aRT.anchorMin = Vector2.zero; aRT.anchorMax = Vector2.one; aRT.offsetMin = aRT.offsetMax = Vector2.zero;
        var handle = Box(area.transform, "Handle", new Color(0.0f, 0.70f, 0.85f, 0.8f));
        var hRT = handle.GetComponent<RectTransform>();
        hRT.offsetMin = hRT.offsetMax = Vector2.zero;
        var sb = sbGO.AddComponent<Scrollbar>();
        sb.handleRect = hRT; sb.targetGraphic = handle.GetComponent<Image>();
        sb.direction = Scrollbar.Direction.BottomToTop;

        _scroll = card.AddComponent<ScrollRect>();
        _scroll.viewport = vRT;
        _scroll.content = cRT;
        _scroll.horizontal = false;
        _scroll.vertical = true;
        _scroll.movementType = ScrollRect.MovementType.Clamped;
        _scroll.scrollSensitivity = 30f;
        _scroll.verticalScrollbar = sb;
        _scroll.verticalScrollbarVisibility = ScrollRect.ScrollbarVisibility.AutoHide;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tabs & content
    // ─────────────────────────────────────────────────────────────────────────
    void RunExperiment(System.Action start)
    {
        if (_lab == null || _lab.isRunning) return;
        start();
        SetTab(2);
    }

    void SetLeftTab(int index)
    {
        for (int i = 0; i < 3; i++)
        {
            _leftPages[i].SetActive(i == index);
            _leftTabImgs[i].color = i == index ? ColorTabActive : ColorInactiveBtn;
            _leftTabTxts[i].color = i == index ? ColorCyanAccent : ColorHeaderTxt;
        }
    }

    void SetTab(int index)
    {
        _tab = index;
        for (int i = 0; i < 3; i++)
        {
            _tabImgs[i].color = i == index ? ColorTabActive : ColorInactiveBtn;
            _tabTxts[i].color = i == index ? ColorCyanAccent : ColorHeaderTxt;
        }
        RefreshCard();
        if (_scroll != null) _scroll.verticalNormalizedPosition = 1f;
    }

    void RefreshCard()
    {
        _contentDirty = false;
        if (_cardTxt == null) return;
        switch (_tab)
        {
            case 0: _cardHeaderLbl.text = "The five PAGP concepts in this simulator"; _cardTxt.text = ConceptsText; break;
            case 1: _cardHeaderLbl.text = "Live guided demo"; _cardTxt.text = LiveText(); break;
            default: _cardHeaderLbl.text = "Results: all concepts"; _cardTxt.text = ResultsText(); break;
        }
    }

    private const string ConceptsText =
        "<b><color=#00E5FF>1. CUDA data parallelism</color></b>\n" +
        "One GPU thread per commuter (64 threads per block). Each physics step copies the 80-byte agent records to the GPU, runs " +
        "Predict → Clear/Build hash → Assign LOD → Jacobi solve × iterations → Finalize, then copies them back via P/Invoke.\n" +
        "<i>Architecture point:</i> thousands of lightweight threads hide memory latency; the fixed PCIe copy and launch cost means the " +
        "GPU only pulls ahead once the crowd is large.\n\n" +

        "<b><color=#00E5FF>2. Spatial hashing vs O(N²)</color></b>\n" +
        "The floor is divided into 4 m cells hashed into 8,192 buckets (16 slots + overflow each). A high-detail agent only tests the 3×3 " +
        "cells around it; medium/low agents test their own cell.\n" +
        "<i>Why:</i> neighbour search drops from N² pair tests to roughly N × (agents nearby). <i>Limit:</i> extremely dense clusters can " +
        "overflow a bucket.\n<color=#8FA3BF>See it: Station Ops → Hash-Grid Heat Map.</color>\n\n" +

        "<b><color=#00E5FF>3. Position-Based Dynamics, Jacobi solver</color></b>\n" +
        "Positions are predicted, then projected to satisfy non-overlap (and long-range avoidance) constraints. In Jacobi form every thread " +
        "writes its correction to its own slot and all corrections are applied afterwards, so there are no write races. That is ideal for the GPU, " +
        "unlike sequential Gauss-Seidel.\n" +
        "<i>Trade-off:</i> more iterations means less residual overlap but linearly more kernel time.\n\n" +

        "<b><color=#00E5FF>4. 4-factor importance-based simulation LOD</color></b>\n" +
        "importance = 0.30·distance + 0.25·visibility (camera frustum) + 0.20·contacts + 0.25·local density.\n" +
        "High (≥0.60): all iterations, 9-cell search, 16 contacts.  Med (≥0.30): half the iterations, 6 contacts.  " +
        "Low: one iteration every 4th frame.\n" +
        "<i>Why:</i> spend compute where the viewer looks and where collisions actually happen.\n<color=#8FA3BF>See it: Station Ops → LOD Rings.</color>\n\n" +

        "<b><color=#00E5FF>5. GPU memory layout: AoS vs SoA</color></b>\n" +
        "Array-of-Structures keeps each agent's fields together (80 B stride). Structure-of-Arrays stores every field contiguously, so the 32 " +
        "threads of a warp read neighbouring addresses that coalesce into fewer 128-byte memory transactions.\n\n" +

        "<b><color=#8FA3BF>CGVR side (presentation only)</color></b>\n" +
        "<color=#8FA3BF>A separate visual LOD throttles animators by distance (near: every frame, mid: 10 Hz, far: 3 Hz) and hides off-screen " +
        "commuters. It never changes the physics. The station itself (fare gates, sliding platform doors, train timetable, signage, " +
        "smooth CCTV camera glides) is the digital-twin presentation layer.</color>";

    string LiveText()
    {
        if (_lab == null) return "Performance lab unavailable.";
        var sb = new System.Text.StringBuilder();
        int phase = _lab.PhaseIndex;

        if (phase < 0 && !_lab.isRunning)
        {
            sb.AppendLine("Press <b>▶ RUN GUIDED DEMO</b>. One continuous rush-hour commute plays for 60 seconds while the camera visits " +
                          "five views, one per concept. Each concept is measured live on the GPU, and at the end the Results tab opens with every result.\n");
            sb.AppendLine("While it runs, the station controls are locked so nothing disturbs the measurements. Pause freezes the demo; Stop ends it.\n");
        }

        sb.AppendLine("<b>Timeline</b>");
        for (int i = 0; i < PAGPPerformanceLab.PhaseNames.Length; i++)
        {
            string mark = phase > i ? "<color=#33F273>✓</color>" : phase == i ? "<color=#FFA500>▶</color>" : "<color=#8FA3BF>○</color>";
            sb.AppendLine($"  {mark}  {i * 12:00}-{(i + 1) * 12:00} s   {PAGPPerformanceLab.PhaseNames[i]}");
        }

        if (phase >= 0)
        {
            sb.AppendLine();
            sb.AppendLine($"<b><color=#00E5FF>{_lab.PhaseTitle}</color></b>");
            sb.AppendLine(_lab.PhaseBody);
            sb.AppendLine($"\n<color=#FFA500>{_lab.PhaseStatus}</color>");

            if (_lab.isRunning)
            {
                _mgr.GetFidelityCounts(out int h, out int m, out int l);
                var cuda = _mgr.GetComponent<PBDCudaSolver>();
                string gpu = cuda != null && cuda.IsInitialized ? $"{cuda.LastTimings.totalMs:F2} ms" : "n/a";
                sb.AppendLine($"\n<b>Live:</b> {_mgr.LiveAgentCount} commuters | GPU step {gpu} | LOD H/M/L {h}/{m}/{l} | " +
                              $"{_mgr.pbdIterations} iterations | {1f / Mathf.Max(Time.smoothDeltaTime, 1e-4f):F0} FPS");
            }

            if (phase < PAGPPerformanceLab.ConceptCount && _lab.ConceptResults[phase] != null)
                sb.AppendLine($"\n<b>Result captured:</b>\n{_lab.ConceptResults[phase]}");
        }
        return sb.ToString().TrimEnd();
    }

    string ResultsText()
    {
        if (_lab == null) return "Performance lab unavailable.";
        var sb = new System.Text.StringBuilder();
        if (_lab.isRunning)
            sb.AppendLine($"<color=#FFA500>Running: {_lab.currentBenchmarkName} ({_lab.benchmarkProgress * 100f:F0}%)</color>\n");

        for (int i = 0; i < PAGPPerformanceLab.ConceptCount; i++)
        {
            sb.AppendLine($"<b><color=#00E5FF>{i + 1}. {PAGPPerformanceLab.ConceptNames[i]}</color></b>");
            sb.AppendLine(_lab.ConceptResults[i] ??
                          $"<color=#8FA3BF>Not measured yet. Run the guided demo or experiment {i + 1}.</color>");
            sb.AppendLine();
        }
        sb.Append("<color=#8FA3BF>CSV files: Application.persistentDataPath (and Desktop/pagp/BenchmarkResults if present).</color>");
        return sb.ToString();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Actions
    // ─────────────────────────────────────────────────────────────────────────
    void Toggle(GameObject body, Text label)
    {
        bool show = !body.activeSelf;
        body.SetActive(show);
        label.text = show ? "Hide" : "Show";
    }

    void TogglePause() => Time.timeScale = Time.timeScale > 0f ? 0f : 1f;

    void ResetScene()
    {
        _lab?.StopBenchmark();
        Time.timeScale = 1f;
        var metro = MetroStationManager.Instance;
        if (metro != null && metro.isEvacuationActive) metro.ToggleEmergencyEvacuation();
        _mgr.Configure(SolverMode.PBDCuda, ScenePreset.StationFlow, 300);
        Viz?.SetHashGrid(false);
        Viz?.SetLodRings(false);
        _agentSld.SetValueWithoutNotify(300);
        metro?.ApplyCCTVAngle(MetroStationManager.CCTVAngle.CCTV_Concourse);
    }

    void ToggleVisualLodDiagnostics()
    {
        var vl = _mgr.GetComponent<PBDCudaVisualLodController>();
        if (vl != null) vl.ToggleDiagnostics();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Widget helpers
    // ─────────────────────────────────────────────────────────────────────────
    Transform Panel(Transform root, string name, Vector2 corner, Vector2 offset, float width)
    {
        var go = Box(root, name, ColorBgPanel);
        var rt = go.GetComponent<RectTransform>();
        rt.anchorMin = rt.anchorMax = rt.pivot = corner;
        rt.anchoredPosition = offset;
        rt.sizeDelta = new Vector2(width, 10);
        var vl = go.AddComponent<VerticalLayoutGroup>();
        vl.padding = new RectOffset(10, 10, 10, 10);
        vl.spacing = 5;
        vl.childControlWidth = vl.childControlHeight = true;
        vl.childForceExpandWidth = true;
        vl.childForceExpandHeight = false;
        go.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;
        return go.transform;
    }

    GameObject SecGO(Transform parent, string name)
    {
        var go = new GameObject(name, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var vl = go.AddComponent<VerticalLayoutGroup>();
        vl.spacing = 4;
        vl.childControlWidth = vl.childControlHeight = true;
        vl.childForceExpandWidth = true;
        vl.childForceExpandHeight = false;
        return go;
    }

    GameObject Box(Transform parent, string name, Color col)
    {
        var go = new GameObject(name, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        go.AddComponent<Image>().color = col;
        return go;
    }

    Text Lbl(Transform parent, string txt, int sz, FontStyle fs, Color col, float h)
    {
        var go = new GameObject("L", typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var t = go.AddComponent<Text>();
        t.font = _font; t.text = txt; t.fontSize = sz;
        t.fontStyle = fs; t.color = col; t.supportRichText = true;
        t.raycastTarget = false;
        go.AddComponent<LayoutElement>().preferredHeight = h;
        return t;
    }

    void HSep(Transform parent)
    {
        var le = Box(parent, "Sep", new Color(0.15f, 0.22f, 0.32f)).AddComponent<LayoutElement>();
        le.preferredHeight = le.minHeight = 1;
    }

    Transform HRow(Transform parent, float spacing, float height)
    {
        var go = new GameObject("Row", typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var hl = go.AddComponent<HorizontalLayoutGroup>();
        hl.spacing = spacing;
        hl.childControlWidth = hl.childControlHeight = true;
        hl.childForceExpandWidth = true;
        hl.childForceExpandHeight = false;
        go.AddComponent<LayoutElement>().preferredHeight = height;
        return go.transform;
    }

    /// <summary>Fill the last row of a 3-column grid with invisible spacers so buttons keep equal width.</summary>
    void PadRow(Transform row, int count)
    {
        for (int i = count % 3; i != 0 && i < 3; i++)
            new GameObject("Spacer", typeof(RectTransform), typeof(LayoutElement)).transform.SetParent(row, false);
    }

    Button MkBtn(Transform parent, string label, int sz, UnityEngine.Events.UnityAction cb)
    {
        var go = Box(parent, "Btn", ColorInactiveBtn);
        go.AddComponent<LayoutElement>().preferredHeight = 26;
        var btn = go.AddComponent<Button>();
        btn.onClick.AddListener(cb);
        btn.targetGraphic = go.GetComponent<Image>();
        var colors = btn.colors;
        colors.disabledColor = new Color(0.5f, 0.5f, 0.5f, 0.45f);
        btn.colors = colors;

        var lgo = new GameObject("L", typeof(RectTransform));
        lgo.transform.SetParent(go.transform, false);
        var lrt = lgo.GetComponent<RectTransform>();
        lrt.anchorMin = Vector2.zero; lrt.anchorMax = Vector2.one;
        lrt.offsetMin = lrt.offsetMax = Vector2.zero;
        var t = lgo.AddComponent<Text>();
        t.font = _font; t.text = label; t.fontSize = sz;
        t.fontStyle = FontStyle.Bold;
        t.alignment = TextAnchor.MiddleCenter;
        t.color = ColorHeaderTxt;
        t.supportRichText = true;
        t.raycastTarget = false;
        return btn;
    }

    (Slider, Text) MkSlider(Transform parent, float init, float min, float max, bool whole, UnityEngine.Events.UnityAction<float> cb)
    {
        var lbl = Lbl(parent, "", fontSize - 3, FontStyle.Normal, ColorHeaderTxt, fontSize);

        var sGO = new GameObject("Slider", typeof(RectTransform));
        sGO.transform.SetParent(parent, false);
        sGO.AddComponent<LayoutElement>().preferredHeight = 16;

        var bg = Box(sGO.transform, "Bg", new Color(0.10f, 0.14f, 0.20f));
        var bgRT = bg.GetComponent<RectTransform>();
        bgRT.anchorMin = new Vector2(0, 0.25f); bgRT.anchorMax = new Vector2(1, 0.75f);
        bgRT.offsetMin = bgRT.offsetMax = Vector2.zero;

        var fa = new GameObject("FillArea", typeof(RectTransform));
        fa.transform.SetParent(sGO.transform, false);
        var faRT = fa.GetComponent<RectTransform>();
        faRT.anchorMin = new Vector2(0, 0.25f); faRT.anchorMax = new Vector2(1, 0.75f);
        faRT.offsetMin = new Vector2(4, 0); faRT.offsetMax = new Vector2(-12, 0);
        var fillRT = Box(fa.transform, "Fill", ColorCyanAccent).GetComponent<RectTransform>();
        fillRT.anchorMin = Vector2.zero; fillRT.anchorMax = Vector2.one;
        fillRT.offsetMin = fillRT.offsetMax = Vector2.zero;

        var ha = new GameObject("HandleArea", typeof(RectTransform));
        ha.transform.SetParent(sGO.transform, false);
        var haRT = ha.GetComponent<RectTransform>();
        haRT.anchorMin = Vector2.zero; haRT.anchorMax = Vector2.one;
        haRT.offsetMin = new Vector2(8, 0); haRT.offsetMax = new Vector2(-8, 0);
        var hnd = Box(ha.transform, "Handle", Color.white);
        var hndRT = hnd.GetComponent<RectTransform>();
        hndRT.anchorMin = Vector2.zero; hndRT.anchorMax = new Vector2(0, 1);
        hndRT.offsetMin = new Vector2(-5, 0); hndRT.offsetMax = new Vector2(5, 0);

        var sld = sGO.AddComponent<Slider>();
        sld.fillRect = fillRT; sld.handleRect = hndRT; sld.targetGraphic = hnd.GetComponent<Image>();
        sld.direction = Slider.Direction.LeftToRight;
        sld.minValue = min; sld.maxValue = max;
        sld.wholeNumbers = whole;
        sld.value = Mathf.Clamp(init, min, max);
        sld.onValueChanged.AddListener(cb);
        return (sld, lbl);
    }

    static void SetLbl(Text t, string value)
    {
        if (t != null && t.text != value) t.text = value;
    }
}
