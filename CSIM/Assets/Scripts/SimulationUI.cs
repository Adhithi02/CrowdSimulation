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
/// Compact Canvas UI — matches the style of photo 2.
/// Attach to the same GameObject as SimulationManager.
/// No TextMeshPro required.
/// </summary>
public class SimulationUI : MonoBehaviour
{
    [Header("Appearance")]
    public float panelWidth = 440f;
    public int   fontSize   = 18;

    // ── runtime refs ──────────────────────────────────────────────────────
    private SimulationManager _mgr;
    private Font _font;
    private GameObject _canvasRoot;
    private GameObject _controlsRoot;
    private Text _toggleControlsLabel;
    private Text _comparisonStatusLbl;
    private Text _compareButtonLabel;
    private Text _cpuGpuCompareButtonLabel;
    private Button _compareButton;
    private CanvasGroup _controlsCanvasGroup;
    private Coroutine _comparisonRoutine;

    private static readonly SolverMode[] Modes =
    {
        SolverMode.Force,    SolverMode.ForceJobs,
        SolverMode.PBD,      SolverMode.PBDJobs,
        SolverMode.PBDCLODJobs, SolverMode.PBDCompute,
        SolverMode.PBDCLODCullingJobs
    };
    private static readonly string[] ModeLabels =
        { "Force", "F-Jobs", "PBD", "PBD-Jobs", "CLOD", "PBD-Cmp", "Cull" };
    private static readonly SolverMode[] PbdBackendModes =
        { SolverMode.PBDJobs, SolverMode.PBDCompute };

    private static readonly ScenePreset[] Presets =
    {
        ScenePreset.Random, ScenePreset.Circle,
        ScenePreset.Crossing, ScenePreset.Bidirectional, ScenePreset.Bottleneck,
        ScenePreset.DoubleBottleneck, ScenePreset.MergeLane
    };
    private static readonly string[] PresetLabels =
        { "Random", "Circle", "Crossing", "Bi-Dir", "Bottleneck", "Double-BN", "MergeLane" };

    private Button[] _solverBtns;
    private Image[]  _solverImgs;
    private Button[] _presetBtns;
    private Image[]  _presetImgs;

    private Slider _agentSld, _speedSld, _radiusSld, _neighSld;
    private Slider _pbdIterSld, _pbdKsiSld;
    private Slider _forceKSld, _forceT0Sld;

    private Text _statusLbl;
    private Text _agentLbl, _speedLbl, _radiusLbl, _neighLbl;
    private Text _pbdIterLbl, _pbdKsiLbl;
    private Text _forceKLbl, _forceT0Lbl;

    private GameObject _pbdSec, _forceSec;

    private sealed class SolverComparisonResult
    {
        public SolverMode Mode;
        public string ImageFile;
        public float Fps;
        public float FrameAverage;
        public float Frame95;
        public float NeighborAverage;
        public float Neighbor95;
        public float SolverAverage;
        public float Solver95;
        public float GpuComputeAverage;
        public float GpuCompute95;
    }

    // ─────────────────────────────────────────────────────────────────────
    void Start()
    {
        _mgr = SimulationManager.Instance;
        if (_mgr == null) { enabled = false; return; }

        _font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

        Build();
    }

    void Update()
    {
        if (_mgr == null) return;
        string status = $"{_mgr.activeSolver} | {_mgr.LiveAgentCount} agents";
        if (_mgr.activeSolver == SolverMode.PBDCLODJobs ||
            _mgr.activeSolver == SolverMode.PBDCompute ||
            _mgr.activeSolver == SolverMode.PBDCLODCullingJobs)
        {
            _mgr.GetFidelityCounts(out int high, out int med, out int low);
            status += $" | H:{high} M:{med} L:{low}";
        }
        _statusLbl.text = status;

        SetLbl(_agentLbl,   "Agent Count: ",  (int)_agentSld.value);
        SetLbl(_speedLbl,   "Speed: ",         _speedSld.value,  "F2");
        SetLbl(_radiusLbl,  "Agent Radius: ",  _radiusSld.value, "F2");
        SetLbl(_neighLbl,   "Neighbour Dist: ",_neighSld.value,  "F1");
        SetLbl(_pbdIterLbl, "Iterations: ",   (int)_pbdIterSld.value);
        SetLbl(_pbdKsiLbl,  "Ksi: ",           _pbdKsiSld.value, "F3");
        SetLbl(_forceKLbl,  "K: ",             _forceKSld.value, "F2");
        SetLbl(_forceT0Lbl, "t0: ",            _forceT0Sld.value,"F1");

        for (int i = 0; i < Modes.Length; i++)
        {
            bool on = _mgr.activeSolver == Modes[i];
            _solverImgs[i].color = on
                ? new Color(0.85f, 0.65f, 0f)
                : new Color(0.28f, 0.28f, 0.28f);
            _solverBtns[i].GetComponentInChildren<Text>().color =
                on ? Color.black : new Color(0.8f, 0.8f, 0.8f);
        }

        for (int i = 0; i < Presets.Length; i++)
        {
            bool on = _mgr.activePreset == Presets[i];
            _presetImgs[i].color = on
                ? new Color(0.2f, 0.7f, 0.55f)   // teal = active preset
                : new Color(0.28f, 0.28f, 0.28f);
            _presetBtns[i].GetComponentInChildren<Text>().color =
                on ? Color.black : new Color(0.8f, 0.8f, 0.8f);
        }

        bool isPBD = _mgr.activeSolver != SolverMode.Force &&
                     _mgr.activeSolver != SolverMode.ForceJobs;
        _pbdSec.SetActive(isPBD);
        _forceSec.SetActive(!isPBD);
    }

    // ── construction ──────────────────────────────────────────────────────

    void Build()
    {
        // Canvas
        var cGO = new GameObject("SimUI_Canvas");
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

        // Panel
        var pGO = Box(cGO.transform, "Panel", new Color(0.12f, 0.12f, 0.12f, 0.93f));
        var pRT = pGO.GetComponent<RectTransform>();
        pRT.anchorMin = pRT.anchorMax = pRT.pivot = new Vector2(0, 1);
        pRT.anchoredPosition = new Vector2(10, -10);
        pRT.sizeDelta = new Vector2(panelWidth, 10);

        var vl = pGO.AddComponent<VerticalLayoutGroup>();
        vl.padding    = new RectOffset(8, 8, 8, 10);
        vl.spacing    = 4;
        vl.childControlWidth = vl.childControlHeight = true;
        vl.childForceExpandWidth = true;
        vl.childForceExpandHeight = false;
        pGO.AddComponent<ContentSizeFitter>().verticalFit =
            ContentSizeFitter.FitMode.PreferredSize;

        var p = pGO.transform;

        // Keep a compact title/status strip visible while the controls are collapsed.
        var header = HRow(p, 4, 30);
        var headerLayout = header.GetComponent<HorizontalLayoutGroup>();
        headerLayout.childForceExpandWidth = false;
        var title = Lbl(header, "Crowd Manager", fontSize + 1, FontStyle.Bold,
            Color.white, fontSize + 7);
        title.GetComponent<LayoutElement>().flexibleWidth = 1;
        var toggleButton = MkBtn(header, "Show", fontSize - 1, ToggleControls);
        var toggleButtonLayout = toggleButton.GetComponent<LayoutElement>();
        toggleButtonLayout.preferredWidth = 54;
        toggleButtonLayout.flexibleWidth = 0;
        _toggleControlsLabel = toggleButton.GetComponentInChildren<Text>();

        _statusLbl = Lbl(p, "", fontSize - 2, FontStyle.Normal,
                         new Color(0.7f, 0.9f, 0.7f), fontSize + 2);

        _controlsRoot = SecGO(p, "Controls");
        _controlsCanvasGroup = _controlsRoot.AddComponent<CanvasGroup>();
        var controls = _controlsRoot.transform;

        HSep(controls);

        // ── Switch Solver ───────────────────────────────────────────────
        Lbl(controls, "Switch Solver:", fontSize, FontStyle.Bold,
            Color.white, fontSize + 4);

        // Row 1: Force | F-Jobs | PBD
        var r1 = HRow(controls, 4, 28);
        // Row 2: PBD-Jobs | CLOD | PBD-Cmp
        var r2 = HRow(controls, 4, 28);
        // Row 3: Cull (left-aligned, fixed width so it doesn't stretch full panel)
        var r3 = HRow(controls, 4, 28);
        r3.GetComponent<HorizontalLayoutGroup>().childForceExpandWidth = false;

        _solverBtns = new Button[Modes.Length];
        _solverImgs = new Image[Modes.Length];

        Transform[] btnParents = { r1, r1, r1, r2, r2, r2, r3 };
        for (int i = 0; i < Modes.Length; i++)
        {
            int idx = i;
            var b = MkBtn(btnParents[i], ModeLabels[i], fontSize - 1,
                          () => _mgr.SetSolverMode(Modes[idx]));
            if (i == Modes.Length - 1)
            {
                // Row 3 has non-expanding children; give Cull a fixed width so it stays visible.
                var le = b.gameObject.AddComponent<LayoutElement>();
                le.preferredWidth = (panelWidth - 16f - 4f) / 3f;
                le.flexibleWidth = 0;
            }
            _solverBtns[i] = b;
            _solverImgs[i] = b.GetComponent<Image>();
        }

        HSep(controls);

        // ── Scene presets ────────────────────────────────────────────────
        Lbl(controls, "Scene Preset:", fontSize, FontStyle.Bold,
            Color.white, fontSize + 4);

        // Row 1: Random | Circle | Crossing
        var pr1 = HRow(controls, 4, 28);
        // Row 2: Bi-Dir | Bottleneck | Double-BN
        var pr2 = HRow(controls, 4, 28);
        // Row 3: MergeLane (left-aligned)
        var pr3 = HRow(controls, 4, 28);
        pr2.GetComponent<HorizontalLayoutGroup>().childForceExpandWidth = false;
        pr3.GetComponent<HorizontalLayoutGroup>().childForceExpandWidth = false;

        _presetBtns = new Button[Presets.Length];
        _presetImgs = new Image[Presets.Length];

        Transform[] presetParents = { pr1, pr1, pr1, pr2, pr2, pr2, pr3 };
        for (int i = 0; i < Presets.Length; i++)
        {
            int idx = i;
            var b = MkBtn(presetParents[i], PresetLabels[i], fontSize - 2,
                          () => _mgr.SetPreset(Presets[idx]));
            if (i >= 3)
            {
                var le = b.gameObject.AddComponent<LayoutElement>();
                le.preferredWidth = (panelWidth - 16f - 4f) / 3f;
                le.flexibleWidth  = 0;
            }
            _presetBtns[i] = b;
            _presetImgs[i] = b.GetComponent<Image>();
        }

        HSep(controls);

        // ── Common sliders ──────────────────────────────────────────────
        Lbl(controls, "─── Common ───", fontSize - 1, FontStyle.Normal,
            new Color(0.6f, 0.6f, 0.6f), fontSize + 2);

        (_agentSld,  _agentLbl)  = MkSlider(controls, "Agent Count: 50",
            _mgr.agentCount, 5, 500, true,  v => _mgr.agentCount  = (int)v);
        (_speedSld,  _speedLbl)  = MkSlider(controls, "Speed: 5.00",
            _mgr.agentSpeed,  0.5f, 15f, false, v => _mgr.agentSpeed  = v);
        (_radiusSld, _radiusLbl) = MkSlider(controls, "Agent Radius: 0.50",
            _mgr.agentRadius, 0.1f, 2f,  false, v => _mgr.agentRadius = v);
        (_neighSld,  _neighLbl)  = MkSlider(controls, "Neighbour Dist: 4.0",
            _mgr.neighborDist, 1f, 20f, false, v => _mgr.neighborDist = v);

        // ── PBD section ─────────────────────────────────────────────────
        _pbdSec = SecGO(controls, "PBDSec");
        Lbl(_pbdSec.transform, "─── PBD ───", fontSize - 1, FontStyle.Normal,
            new Color(0.4f, 0.75f, 1f), fontSize + 2);
        (_pbdIterSld, _pbdIterLbl) = MkSlider(_pbdSec.transform, "Iterations: 3",
            _mgr.pbdIterations, 1, 20, true,  v => _mgr.pbdIterations = (int)v);
        (_pbdKsiSld,  _pbdKsiLbl)  = MkSlider(_pbdSec.transform, "Ksi: 0.050",
            _mgr.pbdKsi, 0.01f, 1f, false, v => _mgr.pbdKsi = v);

        // ── Force section ────────────────────────────────────────────────
        _forceSec = SecGO(controls, "ForceSec");
        Lbl(_forceSec.transform, "─── Force ───", fontSize - 1, FontStyle.Normal,
            new Color(1f, 0.6f, 0.2f), fontSize + 2);
        (_forceKSld,  _forceKLbl)  = MkSlider(_forceSec.transform, "K: 1.00",
            _mgr.forceK,  0.1f, 5f,  false, v => _mgr.forceK  = v);
        (_forceT0Sld, _forceT0Lbl) = MkSlider(_forceSec.transform, "t0: 3.0",
            _mgr.forceT0, 0.5f, 10f, false, v => _mgr.forceT0 = v);

        // ── Respawn ──────────────────────────────────────────────────────
        HSep(controls);
        var rb = MkBtn(controls, "Respawn Crowd", fontSize, () => _mgr.Respawn());
        rb.GetComponent<Image>().color = new Color(0.15f, 0.45f, 0.15f);
        rb.GetComponent<LayoutElement>().preferredHeight = 30;
        var rbc = rb.colors;
        rbc.highlightedColor = new Color(0.22f, 0.6f, 0.22f);
        rbc.pressedColor     = new Color(0.1f,  0.3f, 0.1f);
        rb.colors = rbc;

        HSep(controls);
        Lbl(controls, "Solver Comparisons", fontSize - 1, FontStyle.Bold,
            new Color(0.75f, 0.85f, 1f), fontSize + 4);
        _compareButton = MkBtn(controls, "Compare All Solvers", fontSize,
            StartAllSolversComparison);
        _compareButton.GetComponent<Image>().color = new Color(0.12f, 0.32f, 0.5f);
        _compareButtonLabel = _compareButton.GetComponentInChildren<Text>();
        var cpuGpuCompareButton = MkBtn(controls, "Compare CPU vs GPU PBD", fontSize,
            StartCpuGpuPbdComparison);
        cpuGpuCompareButton.GetComponent<Image>().color = new Color(0.12f, 0.38f, 0.34f);
        _cpuGpuCompareButtonLabel = cpuGpuCompareButton.GetComponentInChildren<Text>();
        _comparisonStatusLbl = Lbl(controls, "", fontSize - 2, FontStyle.Normal,
            new Color(0.75f, 0.9f, 0.75f), fontSize * 2);

        _controlsRoot.SetActive(true);
        _toggleControlsLabel.text = "Hide";
    }

    void StartAllSolversComparison()
    {
        StartSolverComparison(
            Modes, "All solver comparison", "AllSolvers",
            _compareButtonLabel, "Compare All Solvers");
    }

    void StartCpuGpuPbdComparison()
    {
        StartSolverComparison(
            PbdBackendModes, "CPU vs GPU PBD comparison", "CpuGpuPbd",
            _cpuGpuCompareButtonLabel, "Compare CPU vs GPU PBD");
    }

    void StartSolverComparison(
        SolverMode[] modes,
        string reportTitle,
        string directoryName,
        Text buttonLabel,
        string idleButtonText)
    {
        if (_comparisonRoutine != null) return;
        if (ProfilerManager.Instance == null)
        {
            _comparisonStatusLbl.text = "ProfilerManager is required for comparison.";
            Debug.LogError("[SimulationUI] Cannot compare solvers without a ProfilerManager.");
            return;
        }

        _comparisonRoutine = StartCoroutine(
            CompareSolvers(modes, reportTitle, directoryName, buttonLabel, idleButtonText));
    }

    IEnumerator CompareSolvers(
        SolverMode[] modes,
        string reportTitle,
        string directoryName,
        Text buttonLabel,
        string idleButtonText)
    {
        SolverMode originalSolver = _mgr.activeSolver;
        ScenePreset originalPreset = _mgr.activePreset;
        UnityEngine.Random.State randomState = UnityEngine.Random.state;
        var profiler = ProfilerManager.Instance;
        var results = new List<SolverComparisonResult>(modes.Length);
        string outputDirectory = Path.Combine(
            Application.persistentDataPath,
            "SolverComparisons",
            directoryName + "-" +
            DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture));

        Directory.CreateDirectory(outputDirectory);
        _controlsCanvasGroup.interactable = false;
        _controlsCanvasGroup.blocksRaycasts = false;
        buttonLabel.text = "Comparing...";
        _comparisonStatusLbl.text =
            $"Runs {modes.Length} solvers; each gets 1s warm-up and 5s measurement.";

        try
        {
            UnityEngine.Random.InitState(4142026);
            if (_mgr.activePreset != ScenePreset.Circle)
                _mgr.SetPreset(ScenePreset.Circle);
            else
                _mgr.Respawn();

            for (int i = 0; i < modes.Length; i++)
            {
                SolverMode mode = modes[i];
                _comparisonStatusLbl.text = $"Running {mode} ({i + 1}/{modes.Length})...";

                UnityEngine.Random.InitState(4142026);
                if (_mgr.activeSolver != mode)
                    _mgr.SetSolverMode(mode);
                else
                    _mgr.Respawn();

                yield return null;
                yield return new WaitForSecondsRealtime(1f);
                profiler.ResetMeasurements();

                ProfilerRecorder gpuRecorder = default;
                try
                {
                    if (mode == SolverMode.PBDCompute)
                    {
                        gpuRecorder = ProfilerRecorder.StartNew(
                            ProfilerCategory.Render,
                            "PBDCompute.GPU",
                            512,
                            ProfilerRecorderOptions.GpuRecorder |
                            ProfilerRecorderOptions.SumAllSamplesInFrame |
                            ProfilerRecorderOptions.WrapAroundWhenCapacityReached);
                    }

                    float elapsed = 0f;
                    int renderedFrames = 0;
                    while (elapsed < 5f)
                    {
                        yield return null;
                        float frameTime = Time.unscaledDeltaTime;
                        if (frameTime > 0f) renderedFrames++;
                        elapsed += frameTime;
                    }

                    var result = new SolverComparisonResult
                    {
                        Mode = mode,
                        ImageFile = $"{mode}.png",
                        Fps = elapsed > 0f ? renderedFrames / elapsed : float.NaN,
                        FrameAverage = profiler.GetAverageMs("Frame"),
                        Frame95 = profiler.GetPercentileMs("Frame", 95f),
                        NeighborAverage = profiler.GetAverageMs("NeighborQuery"),
                        Neighbor95 = profiler.GetPercentileMs("NeighborQuery", 95f),
                        SolverAverage = profiler.GetAverageMs("Solver"),
                        Solver95 = profiler.GetPercentileMs("Solver", 95f)
                    };
                    if (mode == SolverMode.PBDCompute && gpuRecorder.Valid)
                        ReadGpuComputeMetrics(gpuRecorder, result);
                    results.Add(result);

                    _comparisonStatusLbl.text = $"Capturing {mode} ({i + 1}/{modes.Length})...";
                    yield return CaptureComparisonImage(
                        Path.Combine(outputDirectory, result.ImageFile));
                }
                finally
                {
                    if (gpuRecorder.Valid)
                        gpuRecorder.Dispose();
                }
            }

            string reportPath = Path.Combine(outputDirectory, "index.html");
            File.WriteAllText(
                Path.Combine(outputDirectory, "metrics.csv"),
                BuildComparisonCsv(results),
                Encoding.UTF8);
            File.WriteAllText(
                reportPath,
                BuildComparisonHtml(
                    reportTitle,
                    results,
                    _mgr.agentCount,
                    _mgr.agentSpeed,
                    _mgr.agentRadius,
                    _mgr.neighborDist,
                    _mgr.pbdIterations),
                Encoding.UTF8);

            _comparisonStatusLbl.text = "Comparison complete. Opening report...";
            Debug.Log($"[SimulationUI] Solver comparison saved to {reportPath}");
            Application.OpenURL(new Uri(reportPath).AbsoluteUri);
        }
        finally
        {
            UnityEngine.Random.state = randomState;
            if (_mgr != null)
            {
                if (_mgr.activePreset != originalPreset)
                    _mgr.SetPreset(originalPreset);
                if (_mgr.activeSolver != originalSolver)
                    _mgr.SetSolverMode(originalSolver);
            }

            _controlsCanvasGroup.interactable = true;
            _controlsCanvasGroup.blocksRaycasts = true;
            buttonLabel.text = idleButtonText;
            _comparisonRoutine = null;
        }
    }

    static void ReadGpuComputeMetrics(
        ProfilerRecorder recorder,
        SolverComparisonResult result)
    {
        var samples = new List<ProfilerRecorderSample>();
        recorder.CopyTo(samples, false);
        if (samples.Count == 0) return;

        var values = new List<float>(samples.Count);
        double total = 0d;
        foreach (var sample in samples)
        {
            if (sample.Value <= 0) continue;
            float milliseconds = sample.Value / 1000000f;
            values.Add(milliseconds);
            total += milliseconds;
        }

        if (values.Count == 0) return;
        values.Sort();
        result.GpuComputeAverage = (float)(total / values.Count);
        result.GpuCompute95 = Percentile(values, 0.95f);
    }

    static float Percentile(List<float> sortedValues, float percentile)
    {
        float rank = percentile * (sortedValues.Count - 1);
        int lower = (int)Math.Floor(rank);
        int upper = Math.Min(lower + 1, sortedValues.Count - 1);
        return Mathf.Lerp(sortedValues[lower], sortedValues[upper], rank - lower);
    }

    IEnumerator CaptureComparisonImage(string path)
    {
        bool canvasWasActive = _canvasRoot.activeSelf;
        Texture2D image = null;

        try
        {
            _canvasRoot.SetActive(false);

            yield return new WaitForEndOfFrame();
            image = ScreenCapture.CaptureScreenshotAsTexture();
            if (image == null)
                throw new InvalidOperationException("Unity did not return a comparison screenshot.");

            File.WriteAllBytes(path, image.EncodeToPNG());
        }
        finally
        {
            if (image != null) Destroy(image);
            _canvasRoot.SetActive(canvasWasActive);
        }
    }

    static string BuildComparisonCsv(List<SolverComparisonResult> results)
    {
        var csv = new StringBuilder();
        csv.AppendLine("solver,fps,script_frame_avg_ms,script_frame_p95_ms,neighbor_avg_ms,neighbor_p95_ms,cpu_solver_avg_ms,cpu_solver_p95_ms,gpu_compute_avg_ms,gpu_compute_p95_ms");
        foreach (var result in results)
        {
            csv.Append(result.Mode).Append(',')
                .Append(FormatMetric(result.Fps)).Append(',')
                .Append(FormatMetric(result.FrameAverage)).Append(',')
                .Append(FormatMetric(result.Frame95)).Append(',')
                .Append(FormatMetric(result.NeighborAverage)).Append(',')
                .Append(FormatMetric(result.Neighbor95)).Append(',')
                .Append(FormatMetric(result.SolverAverage)).Append(',')
                .Append(FormatMetric(result.Solver95)).Append(',')
                .Append(FormatMetric(result.GpuComputeAverage)).Append(',')
                .AppendLine(FormatMetric(result.GpuCompute95));
        }
        return csv.ToString();
    }

    static string BuildComparisonHtml(
        string reportTitle,
        List<SolverComparisonResult> results,
        int agentCount,
        float speed,
        float radius,
        float neighborDistance,
        int pbdIterations)
    {
        var html = new StringBuilder();
        html.AppendLine("<!doctype html><html><head><meta charset=\"utf-8\">");
        html.AppendLine("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">");
        html.Append("<title>").Append(reportTitle).AppendLine("</title><style>");
        html.AppendLine("body{font:16px Arial,sans-serif;background:#111827;color:#e5e7eb;margin:24px}");
        html.AppendLine("h1{margin-bottom:6px}p{color:#b8c2d1}");
        html.AppendLine(".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}");
        html.AppendLine("article{background:#1f2937;border:1px solid #374151;border-radius:10px;padding:14px}");
        html.AppendLine("img{width:100%;height:auto;display:block;background:#000;border-radius:6px}");
        html.AppendLine("table{width:100%;border-collapse:collapse;margin-top:10px}");
        html.AppendLine("td{padding:4px;border-bottom:1px solid #374151}td:last-child{text-align:right;font-variant-numeric:tabular-nums}");
        html.AppendLine("</style></head><body>");
        html.Append("<h1>").Append(reportTitle).AppendLine("</h1>");
        html.Append("<p>Circle preset; ").Append(agentCount)
            .Append(" agents; speed ").Append(FormatMetric(speed))
            .Append(" m/s; radius ").Append(FormatMetric(radius))
            .Append(" m; neighbor distance ").Append(FormatMetric(neighborDistance))
            .Append(" m; ").Append(pbdIterations)
            .AppendLine(" PBD iterations. Screenshots omit the runtime UI. CPU solver time is CPU-side; GPU compute time is a GPU profiler sample and may be n/a if unsupported.</p>");
        html.AppendLine("<section class=\"grid\">");

        foreach (var result in results)
        {
            html.Append("<article><h2>").Append(result.Mode).Append("</h2><img src=\"")
                .Append(result.ImageFile).Append("\" alt=\"").Append(result.Mode)
                .Append(" simulation screenshot\"><table>");
            AppendMetricRow(html, "Average FPS", result.Fps);
            AppendMetricRow(html, "Script-frame avg / p95 (ms)", result.FrameAverage, result.Frame95);
            AppendMetricRow(html, "Neighbour avg / p95 (ms)", result.NeighborAverage, result.Neighbor95);
            AppendMetricRow(html, "CPU solver avg / p95 (ms)", result.SolverAverage, result.Solver95);
            AppendMetricRow(html, "GPU compute avg / p95 (ms)", result.GpuComputeAverage, result.GpuCompute95);
            html.AppendLine("</table></article>");
        }

        html.AppendLine("</section></body></html>");
        return html.ToString();
    }

    static void AppendMetricRow(StringBuilder html, string label, float value)
    {
        html.Append("<tr><td>").Append(label).Append("</td><td>")
            .Append(FormatMetric(value)).AppendLine("</td></tr>");
    }

    static void AppendMetricRow(StringBuilder html, string label, float average, float p95)
    {
        html.Append("<tr><td>").Append(label).Append("</td><td>")
            .Append(FormatMetric(average)).Append(" / ")
            .Append(FormatMetric(p95)).AppendLine("</td></tr>");
    }

    static string FormatMetric(float value) =>
        float.IsNaN(value) || float.IsInfinity(value)
            ? "n/a"
            : value.ToString("F2", CultureInfo.InvariantCulture);

    void ToggleControls()
    {
        bool showControls = !_controlsRoot.activeSelf;
        _controlsRoot.SetActive(showControls);
        _toggleControlsLabel.text = showControls ? "Hide" : "Show";
    }

    // ── helpers ───────────────────────────────────────────────────────────

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
        t.fontStyle = fs; t.color = col; t.supportRichText = false;
        go.AddComponent<LayoutElement>().preferredHeight = h;
        return t;
    }

    void HSep(Transform parent)
    {
        var go = Box(parent, "Sep", new Color(0.35f, 0.35f, 0.35f));
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

    Button MkBtn(Transform parent, string label, int sz,
                 UnityEngine.Events.UnityAction cb)
    {
        var go = Box(parent, "Btn", new Color(0.28f, 0.28f, 0.28f));
        var le = go.AddComponent<LayoutElement>();
        le.preferredHeight = 28;
        var btn = go.AddComponent<Button>();
        btn.onClick.AddListener(cb);
        btn.targetGraphic = go.GetComponent<Image>();
        var bc = btn.colors;
        bc.normalColor      = new Color(0.28f, 0.28f, 0.28f);
        bc.highlightedColor = new Color(0.38f, 0.38f, 0.38f);
        bc.pressedColor     = new Color(0.15f, 0.15f, 0.15f);
        btn.colors = bc;

        var lgo = new GameObject("L");
        lgo.transform.SetParent(go.transform, false);
        var lrt = lgo.AddComponent<RectTransform>();
        lrt.anchorMin = Vector2.zero; lrt.anchorMax = Vector2.one;
        lrt.offsetMin = lrt.offsetMax = Vector2.zero;
        var t = lgo.AddComponent<Text>();
        t.font = _font; t.text = label; t.fontSize = sz;
        t.fontStyle = FontStyle.Bold;
        t.alignment = TextAnchor.MiddleCenter;
        t.color = Color.white;
        t.supportRichText = false;
        return btn;
    }

    // A vertical-layout container (for PBD/Force sections)
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

    // One slider row: label on top, slider below
    (Slider, Text) MkSlider(Transform parent, string labelTxt,
                             float init, float min, float max, bool whole,
                             UnityEngine.Events.UnityAction<float> cb)
    {
        // Value label
        var lbl = Lbl(parent, labelTxt, fontSize - 2, FontStyle.Normal,
                      Color.white, fontSize);

        // Slider sits in a thin container
        var sGO = new GameObject("SLD");
        sGO.transform.SetParent(parent, false);
        sGO.AddComponent<LayoutElement>().preferredHeight = 18;

        // track
        var bg = Box(sGO.transform, "Bg", new Color(0.18f, 0.18f, 0.18f));
        var bgRT = bg.GetComponent<RectTransform>();
        bgRT.anchorMin = new Vector2(0, 0.2f); bgRT.anchorMax = new Vector2(1, 0.8f);
        bgRT.offsetMin = bgRT.offsetMax = Vector2.zero;

        // fill area
        var fa = new GameObject("FA");
        fa.transform.SetParent(sGO.transform, false);
        var faRT = fa.AddComponent<RectTransform>();
        faRT.anchorMin = new Vector2(0, 0.2f); faRT.anchorMax = new Vector2(1, 0.8f);
        faRT.offsetMin = new Vector2(4, 0);   faRT.offsetMax = new Vector2(-12, 0);
        var fill = Box(fa.transform, "F", new Color(0.2f, 0.55f, 1f));
        var fillRT = fill.GetComponent<RectTransform>();
        fillRT.anchorMin = Vector2.zero; fillRT.anchorMax = Vector2.one;
        fillRT.offsetMin = fillRT.offsetMax = Vector2.zero;

        // handle area
        var ha = new GameObject("HA");
        ha.transform.SetParent(sGO.transform, false);
        var haRT = ha.AddComponent<RectTransform>();
        haRT.anchorMin = Vector2.zero; haRT.anchorMax = Vector2.one;
        haRT.offsetMin = new Vector2(8, 0); haRT.offsetMax = new Vector2(-8, 0);
        var hnd = Box(ha.transform, "H", Color.white);
        var hndRT = hnd.GetComponent<RectTransform>();
        hndRT.anchorMin = Vector2.zero; hndRT.anchorMax = new Vector2(0, 1);
        hndRT.sizeDelta = new Vector2(12, 0);
        hndRT.offsetMin = new Vector2(-6, 1); hndRT.offsetMax = new Vector2(6, -1);

        // Slider component on sGO (auto has RectTransform from Canvas parenting)
        var sld = sGO.AddComponent<Slider>();
        sld.fillRect = fillRT; sld.handleRect = hndRT;
        sld.direction = Slider.Direction.LeftToRight;
        sld.minValue = min; sld.maxValue = max;
        sld.wholeNumbers = whole;
        sld.value = Mathf.Clamp(init, min, max);
        sld.onValueChanged.AddListener(cb);

        return (sld, lbl);
    }

    static void SetLbl(Text t, string prefix, int v)    => t.text = prefix + v;
    static void SetLbl(Text t, string prefix, float v, string fmt) =>
        t.text = prefix + v.ToString(fmt);
}
