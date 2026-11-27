using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Metro Operations Control Center UI (OCC Digital Twin interface)
/// Visual Theme: Dark navy & charcoal background (#0B1320 / #1C2541), cyan highlights (#00E5FF),
/// clean typography, green/yellow/red status indicators, and PAGP Performance Lab dashboard.
/// </summary>
public class SimulationUI : MonoBehaviour
{
    [Header("Appearance")]
    public float panelWidth = 470f;
    public int   fontSize   = 17;

    // Palette: Metro OCC Digital Twin Theme
    private static readonly Color ColorBgPanel     = new Color(0.043f, 0.075f, 0.125f, 0.94f); // #0B1320
    private static readonly Color ColorCyanAccent  = new Color(0.0f, 0.90f, 1.0f);              // #00E5FF
    private static readonly Color ColorActiveBtn   = new Color(0.0f, 0.70f, 0.85f);
    private static readonly Color ColorInactiveBtn = new Color(0.12f, 0.17f, 0.24f);
    private static readonly Color ColorHeaderTxt   = new Color(0.85f, 0.92f, 1.0f);
    private static readonly Color ColorSubTxt      = new Color(0.55f, 0.65f, 0.78f);
    private static readonly Color ColorGreenStatus = new Color(0.2f, 0.95f, 0.45f);
    private static readonly Color ColorOrangeAlert = new Color(1.0f, 0.65f, 0.15f);
    private static readonly Color ColorRedAlarm    = new Color(1.0f, 0.25f, 0.25f);

    private SimulationManager _mgr;
    private Font _font;
    private GameObject _canvasRoot;
    private GameObject _controlsRoot;
    private Text _toggleControlsLabel;
    private Text _statusLbl;
    private Text _telemetryLbl;
    private Text _perfLabStatusLbl;
    private Text _cctvBtnLabel;
    private Text _evacBtnLabel;
    private Text _pauseBtnLabel;
    private Text _visualLodBtnLabel;

    private static readonly SolverMode[] Modes =
    {
        SolverMode.Force,       SolverMode.ForceJobs,
        SolverMode.PBD,         SolverMode.PBDJobs,
        SolverMode.PBDCLODJobs, SolverMode.PBDCuda,
        SolverMode.PBDCLODCullingJobs
    };
    private static readonly string[] ModeLabels =
        { "Force", "F-Jobs", "PBD", "PBD-Jobs", "CLOD", "CUDA (GPU)", "Cull" };

    private static readonly ScenePreset[] Presets =
    {
        ScenePreset.Random, ScenePreset.Circle,
        ScenePreset.Crossing, ScenePreset.Bidirectional, ScenePreset.Bottleneck,
        ScenePreset.DoubleBottleneck, ScenePreset.MergeLane
    };
    private static readonly string[] PresetLabels =
        { "Concourse", "Circle", "Crossing", "Platform", "Turnstiles", "Double-Gate", "Stair-Merge" };

    private Button[] _solverBtns;
    private Image[]  _solverImgs;
    private Button[] _presetBtns;
    private Image[]  _presetImgs;

    private Slider _agentSld, _speedSld, _radiusSld, _neighSld;
    private Slider _pbdIterSld, _pbdKsiSld;
    private Slider _forceKSld, _forceT0Sld;

    private Text _agentLbl, _speedLbl, _radiusLbl, _neighLbl;
    private Text _pbdIterLbl, _pbdKsiLbl;
    private Text _forceKLbl, _forceT0Lbl;

    private GameObject _pbdSec, _forceSec;

    void Start()
    {
        _mgr = SimulationManager.Instance;
        if (_mgr == null) { enabled = false; return; }

        if (GetComponent<PAGPPerformanceLab>() == null)
            gameObject.AddComponent<PAGPPerformanceLab>();

        _font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        BuildOCCInterface();
    }

    void Update()
    {
        if (_mgr == null) return;

        float fps = 1.0f / Mathf.Max(Time.smoothDeltaTime, 1e-4f);
        _mgr.GetFidelityCounts(out int high, out int med, out int low);
        int total = high + med + low;
        int savedPct = total > 0 ? (int)((float)low / total * 100f) : 0;

        var metro = MetroStationManager.Instance;
        string cctvInfo = metro != null ? metro.GetCCTVLabel() : "CCTV OFF";
        string trainStatus = (metro != null && metro.isTrainAtPlatform) ? "<color=#00E5FF>[PLATFORM: TRAIN DOCKED]</color>" : "<color=#FFA500>[TRAIN IN TRANSIT]</color>";

        // Top Status Header
        _statusLbl.text = $"<b>METRO OCC DIGITAL TWIN</b> | <color=#00E5FF>{_mgr.activeSolver}</color> ({fps:F0} FPS)\n" +
                          $"Live Commuters: <b>{_mgr.LiveAgentCount}</b> | {trainStatus}\n" +
                           $"Simulation LOD: High: {high} | Med: {med} | Low: {low} (<b>{savedPct}% Saved</b>)\n" +
                           $"{cctvInfo}";

        if (metro != null && metro.isEvacuationActive)
        {
            _statusLbl.text += $"\n<color=#FF3B30><b>[EMERGENCY EVACUATION ACTIVE]</b> Time: {metro.evacuationTimer:F1}s | Remaining: {metro.remainingEvacuees}</color>";
        }

        var visualLod = _mgr.GetComponent<PBDCudaVisualLodController>();
        if (visualLod != null)
        {
            _statusLbl.text += $"\nVisual LOD: {visualLod.VisibleCommuterCount} visible | " +
                               $"Near {visualLod.NearCount} | Mid {visualLod.MidCount} | " +
                               $"Far {visualLod.FarCount} | Hidden {visualLod.HiddenCount}";
        }

        // Live GPU / PAGP Telemetry block
// Live GPU / PAGP Telemetry block
        var cuda = _mgr.GetComponent<PBDCudaSolver>();
        if (cuda != null && cuda.IsInitialized && _mgr.activeSolver == SolverMode.PBDCuda)
        {
            var t = cuda.LastTimings;
            _telemetryLbl.text = $"CUDA Total Step: <b>{t.totalMs:F2} ms</b> | Jacobi Kernel: <b>{t.solveMs:F2} ms</b>\n" +
                                 $"Memcpy H2D: {t.h2dMs:F2} ms | D2H: {t.d2hMs:F2} ms | Spatial Grid: {t.hashAgentsMs:F2} ms";
        }
        else
        {
            float fAvg = ProfilerManager.Instance != null ? ProfilerManager.Instance.GetAverageMs("Frame") : 0f;
            _telemetryLbl.text = $"Host Frame Avg: <b>{fAvg:F2} ms</b> | Pipeline: CPU Burst Jobs";
        }

        // Performance Lab live status
        var lab = PAGPPerformanceLab.Instance;
        if (lab != null && _perfLabStatusLbl != null)
        {
            _perfLabStatusLbl.text = lab.isRunning
                ? $"<color=#FFA500>RUNNING: {lab.currentBenchmarkName} ({(lab.benchmarkProgress * 100f):F0}%)</color>"
                : $"<color=#00E5FF>STATUS: {lab.currentBenchmarkName}</color>";
        }

        SetLbl(_agentLbl,   "Commuters (Reset to apply): ", (int)_agentSld.value);
        SetLbl(_speedLbl,   "Walking Speed: ",    _speedSld.value,  "F2");
        SetLbl(_radiusLbl,  "Body Radius: ",      _radiusSld.value, "F2");
        SetLbl(_neighLbl,   "Perception Dist: ",  _neighSld.value,  "F1");
        SetLbl(_pbdIterLbl, "PBD Iterations: ",   (int)_pbdIterSld.value);
        SetLbl(_pbdKsiLbl,  "Relaxation (Ksi): ", _pbdKsiSld.value, "F3");
        SetLbl(_forceKLbl,  "Stiffness (K): ",    _forceKSld.value, "F2");
        SetLbl(_forceT0Lbl, "Lookahead (t0): ",   _forceT0Sld.value,"F1");

        for (int i = 0; i < Modes.Length; i++)
        {
            bool on = _mgr.activeSolver == Modes[i];
            _solverImgs[i].color = on ? ColorActiveBtn : ColorInactiveBtn;
            _solverBtns[i].GetComponentInChildren<Text>().color = on ? Color.black : ColorHeaderTxt;
        }

        for (int i = 0; i < Presets.Length; i++)
        {
            bool on = _mgr.activePreset == Presets[i];
            _presetImgs[i].color = on ? new Color(0.1f, 0.75f, 0.6f) : ColorInactiveBtn;
            _presetBtns[i].GetComponentInChildren<Text>().color = on ? Color.black : ColorHeaderTxt;
        }

        bool isPBD = _mgr.activeSolver != SolverMode.Force && _mgr.activeSolver != SolverMode.ForceJobs;
        _pbdSec.SetActive(isPBD);
        _forceSec.SetActive(!isPBD);
    }

    void BuildOCCInterface()
    {
        var cGO = new GameObject("OCC_Canvas");
        _canvasRoot = cGO;
        cGO.transform.SetParent(transform);
        var cv = cGO.AddComponent<Canvas>();
        cv.renderMode = RenderMode.ScreenSpaceOverlay;
        cv.sortingOrder = 100;
        var sc = cGO.AddComponent<CanvasScaler>();
        sc.uiScaleMode         = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        sc.referenceResolution = new Vector2(1920, 1080);
        sc.matchWidthOrHeight  = 0.5f;
        cGO.AddComponent<GraphicRaycaster>();

        var pGO = Box(cGO.transform, "OCC_MainPanel", ColorBgPanel);
        var pRT = pGO.GetComponent<RectTransform>();
        pRT.anchorMin = pRT.anchorMax = pRT.pivot = new Vector2(0, 1);
        pRT.anchoredPosition = new Vector2(12, -12);
        pRT.sizeDelta = new Vector2(panelWidth, 10);

        var vl = pGO.AddComponent<VerticalLayoutGroup>();
        vl.padding    = new RectOffset(10, 10, 10, 10);
        vl.spacing    = 4;
        vl.childControlWidth = vl.childControlHeight = true;
        vl.childForceExpandWidth = true;
        vl.childForceExpandHeight = false;
        pGO.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;

        var p = pGO.transform;

        // Header Strip
        var header = HRow(p, 4, 32);
        var title = Lbl(header, "OPERATIONS CONTROL CENTER", fontSize, FontStyle.Bold, ColorCyanAccent, 28);
        title.GetComponent<LayoutElement>().flexibleWidth = 1;
        var toggleBtn = MkBtn(header, "Collapse", fontSize - 2, ToggleControls);
        toggleBtn.GetComponent<LayoutElement>().preferredWidth = 70;
        _toggleControlsLabel = toggleBtn.GetComponentInChildren<Text>();

        _statusLbl = Lbl(p, "", fontSize - 2, FontStyle.Normal, ColorHeaderTxt, fontSize * 5.0f);

        // Telemetry Box
        var telBox = Box(p, "TelemetryBox", new Color(0.08f, 0.12f, 0.18f, 0.9f));
        telBox.AddComponent<LayoutElement>().preferredHeight = 36;
        var telL = telBox.AddComponent<VerticalLayoutGroup>();
        telL.padding = new RectOffset(6, 6, 4, 4);
        _telemetryLbl = Lbl(telBox.transform, "", fontSize - 3, FontStyle.Normal, ColorCyanAccent, 30);

        _controlsRoot = SecGO(p, "ControlsContent");
        var controls = _controlsRoot.transform;

        HSep(controls);

        // ── Solver Switcher ──
        Lbl(controls, "PARALLEL ARCHITECTURE SOLVER:", fontSize - 1, FontStyle.Bold, ColorCyanAccent, 20);
        var r1 = HRow(controls, 4, 26);
        var r2 = HRow(controls, 4, 26);
        var r3 = HRow(controls, 4, 26);
        r3.GetComponent<HorizontalLayoutGroup>().childForceExpandWidth = false;

        _solverBtns = new Button[Modes.Length];
        _solverImgs = new Image[Modes.Length];
        Transform[] btnParents = { r1, r1, r1, r2, r2, r2, r3 };

        for (int i = 0; i < Modes.Length; i++)
        {
            int idx = i;
            var b = MkBtn(btnParents[i], ModeLabels[i], fontSize - 2, () => _mgr.SetSolverMode(Modes[idx]));
            if (i >= 6)
            {
                var le = b.gameObject.AddComponent<LayoutElement>();
                le.preferredWidth = (panelWidth - 20f) / 3f;
            }
            _solverBtns[i] = b;
            _solverImgs[i] = b.GetComponent<Image>();
        }

        HSep(controls);

        // ── Station Scenarios ──
        Lbl(controls, "METRO SCENARIO PRESETS:", fontSize - 1, FontStyle.Bold, ColorCyanAccent, 20);
        var pr1 = HRow(controls, 4, 26);
        var pr2 = HRow(controls, 4, 26);
        var pr3 = HRow(controls, 4, 26);
        pr2.GetComponent<HorizontalLayoutGroup>().childForceExpandWidth = false;
        pr3.GetComponent<HorizontalLayoutGroup>().childForceExpandWidth = false;

        _presetBtns = new Button[Presets.Length];
        _presetImgs = new Image[Presets.Length];
        Transform[] presetParents = { pr1, pr1, pr1, pr2, pr2, pr2, pr3 };

        for (int i = 0; i < Presets.Length; i++)
        {
            int idx = i;
            var b = MkBtn(presetParents[i], PresetLabels[i], fontSize - 3, () => _mgr.SetPreset(Presets[idx]));
            if (i >= 3)
            {
                var le = b.gameObject.AddComponent<LayoutElement>();
                le.preferredWidth = (panelWidth - 20f) / 3f;
            }
            _presetBtns[i] = b;
            _presetImgs[i] = b.GetComponent<Image>();
        }

        HSep(controls);

        // ── Sliders ──
        Lbl(controls, "─── SIMULATION TUNING ───", fontSize - 2, FontStyle.Normal, ColorSubTxt, 18);
        (_agentSld,  _agentLbl)  = MkSlider(controls, "Commuters (Reset to apply): 50", _mgr.agentCount, 5, 5000, true,  v => _mgr.agentCount  = (int)v);
        (_speedSld,  _speedLbl)  = MkSlider(controls, "Walking Speed: 5.00",   _mgr.agentSpeed, 0.5f, 15f, false, v => _mgr.agentSpeed  = v);
        (_radiusSld, _radiusLbl) = MkSlider(controls, "Body Radius: 0.50",     _mgr.agentRadius, 0.1f, 2f,  false, v => _mgr.agentRadius = v);
        (_neighSld,  _neighLbl)  = MkSlider(controls, "Perception Dist: 4.0",  _mgr.neighborDist, 1f, 20f, false, v => _mgr.neighborDist = v);

        _pbdSec = SecGO(controls, "PBDSec");
        Lbl(_pbdSec.transform, "─── PBD PARAMETERS ───", fontSize - 2, FontStyle.Normal, ColorCyanAccent, 18);
        (_pbdIterSld, _pbdIterLbl) = MkSlider(_pbdSec.transform, "PBD Iterations: 3", _mgr.pbdIterations, 1, 20, true,  v => _mgr.pbdIterations = (int)v);
        (_pbdKsiSld,  _pbdKsiLbl)  = MkSlider(_pbdSec.transform, "Relaxation: 0.050",   _mgr.pbdKsi, 0.01f, 1f, false, v => _mgr.pbdKsi = v);

        _forceSec = SecGO(controls, "ForceSec");
        Lbl(_forceSec.transform, "─── FORCE PARAMETERS ───", fontSize - 2, FontStyle.Normal, ColorOrangeAlert, 18);
        (_forceKSld,  _forceKLbl)  = MkSlider(_forceSec.transform, "Stiffness: 1.00", _mgr.forceK, 0.1f, 5f, false, v => _mgr.forceK = v);
        (_forceT0Sld, _forceT0Lbl) = MkSlider(_forceSec.transform, "Lookahead: 3.0",  _mgr.forceT0, 0.5f, 10f, false, v => _mgr.forceT0 = v);

        HSep(controls);

        // ── Showcase Runtime Controls ──
        HSep(controls);
        Lbl(controls, "SHOWCASE CONTROLS:", fontSize - 1, FontStyle.Bold, ColorCyanAccent, 20);
        var showcaseRow = HRow(controls, 4, 28);

        var pauseBtn = MkBtn(showcaseRow, "Pause", fontSize - 2, TogglePause);
        _pauseBtnLabel = pauseBtn.GetComponentInChildren<Text>();

        var resetBtn = MkBtn(showcaseRow, "Reset Showcase", fontSize - 2, ResetShowcase);
        resetBtn.GetComponent<Image>().color = new Color(0.16f, 0.38f, 0.30f);

        var visualLodBtn = MkBtn(showcaseRow, "Show Visual LOD", fontSize - 3, ToggleVisualLodDiagnostics);
        _visualLodBtnLabel = visualLodBtn.GetComponentInChildren<Text>();

        // ── Station Operations Controls ──
// ── Station Operations Controls ──
        Lbl(controls, "TRANSIT OPERATIONS:", fontSize - 1, FontStyle.Bold, ColorCyanAccent, 20);
        var opRow = HRow(controls, 4, 28);
        var cctvBtn = MkBtn(opRow, "CCTV Angle", fontSize - 2, () => MetroStationManager.Instance?.CycleCCTVCamera());
        _cctvBtnLabel = cctvBtn.GetComponentInChildren<Text>();

        var trainBtn = MkBtn(opRow, "Cycle Train", fontSize - 2, () => MetroStationManager.Instance?.TriggerTrainCycle());
        trainBtn.GetComponent<Image>().color = new Color(0.12f, 0.38f, 0.32f);

        var evacBtn = MkBtn(controls, "Emergency Evacuation Alarm: OFF", fontSize - 1, () => {
            var m = MetroStationManager.Instance;
            if (m != null)
            {
                m.ToggleEmergencyEvacuation();
                _evacBtnLabel.text = m.isEvacuationActive ? "ALARM: EVACUATION ACTIVE" : "Emergency Evacuation Alarm: OFF";
            }
        });
        evacBtn.GetComponent<Image>().color = new Color(0.55f, 0.15f, 0.15f);
        _evacBtnLabel = evacBtn.GetComponentInChildren<Text>();

        // ── PAGP Performance Lab Scientific Suite ──
        HSep(controls);
        Lbl(controls, "PAGP PERFORMANCE LAB SCIENTIFIC SUITE:", fontSize - 1, FontStyle.Bold, ColorCyanAccent, 20);
        _perfLabStatusLbl = Lbl(controls, "STATUS: Idle", fontSize - 3, FontStyle.Normal, ColorSubTxt, 18);

        var labRow1 = HRow(controls, 4, 26);
        var bSuite = MkBtn(labRow1, "1: CPU vs CUDA", fontSize - 3, () => PAGPPerformanceLab.Instance?.RunComprehensiveSuite());
        bSuite.GetComponent<Image>().color = new Color(0.08f, 0.30f, 0.45f);

        var bHash = MkBtn(labRow1, "2: Spatial Hash vs O(N²)", fontSize - 3, () => PAGPPerformanceLab.Instance?.RunSpatialHashAnalysis());
        bHash.GetComponent<Image>().color = new Color(0.18f, 0.28f, 0.42f);

        var labRow2 = HRow(controls, 4, 26);
        var bAbl = MkBtn(labRow2, "3: LOD Ablation", fontSize - 3, () => PAGPPerformanceLab.Instance?.RunLODAblationStudy());
        bAbl.GetComponent<Image>().color = new Color(0.12f, 0.35f, 0.28f);

        var bConv = MkBtn(labRow2, "4: PBD Convergence", fontSize - 3, () => PAGPPerformanceLab.Instance?.RunPBDConvergenceTest());
        bConv.GetComponent<Image>().color = new Color(0.35f, 0.22f, 0.40f);

        var labRow3 = HRow(controls, 4, 26);
        var bMem = MkBtn(labRow3, "5: AoS vs SoA Memory", fontSize - 3, () => PAGPPerformanceLab.Instance?.RunMemoryLayoutBenchmark());
        bMem.GetComponent<Image>().color = new Color(0.20f, 0.35f, 0.45f);

        var bStop = MkBtn(labRow3, "Stop Test", fontSize - 3, () => PAGPPerformanceLab.Instance?.StopBenchmark());
        bStop.GetComponent<Image>().color = new Color(0.50f, 0.15f, 0.15f);

        var respawnBtn = MkBtn(controls, "Respawn Digital Twin Crowd", fontSize, () => _mgr.Respawn());
        respawnBtn.GetComponent<Image>().color = new Color(0.12f, 0.45f, 0.25f);
    }

    void ToggleControls()
    {
        bool show = !_controlsRoot.activeSelf;
        _controlsRoot.SetActive(show);
        _toggleControlsLabel.text = show ? "Collapse" : "Expand";
    }

void TogglePause()
    {
        bool paused = Time.timeScale > 0f;
        Time.timeScale = paused ? 0f : 1f;
        if (_pauseBtnLabel != null)
            _pauseBtnLabel.text = paused ? "Resume" : "Pause";
    }

    void ResetShowcase()
    {
        Time.timeScale = 1f;
        if (_pauseBtnLabel != null) _pauseBtnLabel.text = "Pause";

        var metro = MetroStationManager.Instance;
        if (metro != null && metro.isEvacuationActive)
        {
            // The existing normal-operation path already respawns the active preset.
            metro.ToggleEmergencyEvacuation();
            return;
        }

        _mgr?.Respawn();
    }

    void ToggleVisualLodDiagnostics()
    {
        var visualLod = _mgr != null ? _mgr.GetComponent<PBDCudaVisualLodController>() : null;
        if (visualLod == null) return;

        visualLod.ToggleDiagnostics();
        if (_visualLodBtnLabel != null)
            _visualLodBtnLabel.text = visualLod.showDiagnostics ? "Hide Visual LOD" : "Show Visual LOD";
    }


    GameObject SecGO(Transform parent, string name)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        var vl = go.AddComponent<VerticalLayoutGroup>();
        vl.spacing = 3;
        vl.childControlWidth = vl.childControlHeight = true;
        vl.childForceExpandWidth = true;
        vl.childForceExpandHeight = false;
        go.AddComponent<ContentSizeFitter>().verticalFit =
            ContentSizeFitter.FitMode.PreferredSize;
        return go;
    }

    // ── Helper Constructors ──
    GameObject Box(Transform parent, string name, Color col)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        go.AddComponent<Image>().color = col;
        return go;
    }

    Text Lbl(Transform parent, string txt, int sz, FontStyle fs, Color col, float h)
    {
        var go = new GameObject("L");
        go.transform.SetParent(parent, false);
        var t = go.AddComponent<Text>();
        t.font = _font; t.text = txt; t.fontSize = sz;
        t.fontStyle = fs; t.color = col; t.supportRichText = true;
        go.AddComponent<LayoutElement>().preferredHeight = h;
        return t;
    }

    void HSep(Transform parent)
    {
        var go = Box(parent, "Sep", new Color(0.15f, 0.22f, 0.32f));
        var le = go.AddComponent<LayoutElement>();
        le.preferredHeight = le.minHeight = 1;
    }

    Transform HRow(Transform parent, float spacing, float height)
    {
        var go = new GameObject("HRow");
        go.transform.SetParent(parent, false);
        var hl = go.AddComponent<HorizontalLayoutGroup>();
        hl.spacing = spacing;
        hl.childControlWidth = hl.childControlHeight = true;
        hl.childForceExpandWidth = true;
        hl.childForceExpandHeight = false;
        go.AddComponent<LayoutElement>().preferredHeight = height;
        return go.transform;
    }

    Button MkBtn(Transform parent, string label, int sz, UnityEngine.Events.UnityAction cb)
    {
        var go = Box(parent, "Btn", ColorInactiveBtn);
        var le = go.AddComponent<LayoutElement>();
        le.preferredHeight = 28;
        var btn = go.AddComponent<Button>();
        btn.onClick.AddListener(cb);
        btn.targetGraphic = go.GetComponent<Image>();

        var lgo = new GameObject("L");
        lgo.transform.SetParent(go.transform, false);
        var lrt = lgo.AddComponent<RectTransform>();
        lrt.anchorMin = Vector2.zero; lrt.anchorMax = Vector2.one;
        lrt.offsetMin = lrt.offsetMax = Vector2.zero;
        var t = lgo.AddComponent<Text>();
        t.font = _font; t.text = label; t.fontSize = sz;
        t.fontStyle = FontStyle.Bold;
        t.alignment = TextAnchor.MiddleCenter;
        t.color = ColorHeaderTxt;
        t.supportRichText = true;
        return btn;
    }

    (Slider, Text) MkSlider(Transform parent, string labelTxt, float init, float min, float max, bool whole, UnityEngine.Events.UnityAction<float> cb)
    {
        var lbl = Lbl(parent, labelTxt, fontSize - 3, FontStyle.Normal, ColorHeaderTxt, fontSize);

        var sGO = new GameObject("SLD");
        sGO.transform.SetParent(parent, false);
        sGO.AddComponent<LayoutElement>().preferredHeight = 16;

        var bg = Box(sGO.transform, "Bg", new Color(0.10f, 0.14f, 0.20f));
        var bgRT = bg.GetComponent<RectTransform>();
        bgRT.anchorMin = new Vector2(0, 0.25f); bgRT.anchorMax = new Vector2(1, 0.75f);
        bgRT.offsetMin = bgRT.offsetMax = Vector2.zero;

        var fa = new GameObject("FA");
        fa.transform.SetParent(sGO.transform, false);
        var faRT = fa.AddComponent<RectTransform>();
        faRT.anchorMin = new Vector2(0, 0.25f); faRT.anchorMax = new Vector2(1, 0.75f);
        faRT.offsetMin = new Vector2(4, 0);   faRT.offsetMax = new Vector2(-12, 0);
        var fill = Box(fa.transform, "F", ColorCyanAccent);
        var fillRT = fill.GetComponent<RectTransform>();
        fillRT.anchorMin = Vector2.zero; fillRT.anchorMax = Vector2.one;
        fillRT.offsetMin = fillRT.offsetMax = Vector2.zero;

        var ha = new GameObject("HA");
        ha.transform.SetParent(sGO.transform, false);
        var haRT = ha.AddComponent<RectTransform>();
        haRT.anchorMin = Vector2.zero; haRT.anchorMax = Vector2.one;
        haRT.offsetMin = new Vector2(8, 0); haRT.offsetMax = new Vector2(-8, 0);
        var hnd = Box(ha.transform, "H", Color.white);
        var hndRT = hnd.GetComponent<RectTransform>();
        hndRT.anchorMin = Vector2.zero; hndRT.anchorMax = new Vector2(0, 1);
        hndRT.sizeDelta = new Vector2(10, 0);
        hndRT.offsetMin = new Vector2(-5, 0); hndRT.offsetMax = new Vector2(5, 0);

        var sld = sGO.AddComponent<Slider>();
        sld.fillRect = fillRT; sld.handleRect = hndRT;
        sld.direction = Slider.Direction.LeftToRight;
        sld.minValue = min; sld.maxValue = max;
        sld.wholeNumbers = whole;
        sld.value = Mathf.Clamp(init, min, max);
        sld.onValueChanged.AddListener(cb);

        return (sld, lbl);
    }

    static void SetLbl(Text t, string prefix, int v) => t.text = prefix + v;
    static void SetLbl(Text t, string prefix, float v, string fmt) => t.text = prefix + v.ToString(fmt);
}
