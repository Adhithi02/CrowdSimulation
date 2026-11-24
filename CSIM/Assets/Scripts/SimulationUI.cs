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
    public float panelWidth = 220f;
    public int   fontSize   = 14;

    // ── runtime refs ──────────────────────────────────────────────────────
    private SimulationManager _mgr;
    private Font _font;

    private static readonly SolverMode[] Modes =
    {
        SolverMode.Force,    SolverMode.ForceJobs,
        SolverMode.PBD,      SolverMode.PBDJobs,
        SolverMode.PBDCLODJobs, SolverMode.PBDCompute,
        SolverMode.PBDCLODCullingJobs
    };
    private static readonly string[] ModeLabels =
        { "Force", "F-Jobs", "PBD", "PBD-Jobs", "CLOD", "PBD-Cmp", "Cull" };

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

    // ─────────────────────────────────────────────────────────────────────
    void Start()
    {
        _mgr = SimulationManager.Instance;
        if (_mgr == null) { enabled = false; return; }

        _font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf")
             ?? Resources.GetBuiltinResource<Font>("Arial.ttf");

        Build();
    }

    void Update()
    {
        if (_mgr == null) return;
        if (_mgr.activeSolver == SolverMode.PBDCLODJobs ||
            _mgr.activeSolver == SolverMode.PBDCompute ||
            _mgr.activeSolver == SolverMode.PBDCLODCullingJobs)
        {
            _mgr.GetFidelityCounts(out int high, out int med, out int low);
            _statusLbl.text =
                $"Solver: {_mgr.activeSolver}   Agents: {_mgr.LiveAgentCount} (H:{high} M:{med} L:{low})";
        }
        else
        {
            _statusLbl.text =
                $"Solver: {_mgr.activeSolver}   Agents: {_mgr.LiveAgentCount}";
        }

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

        // ── Title ───────────────────────────────────────────────────────
        Lbl(p, "Crowd Manager", fontSize + 1, FontStyle.Bold,
            Color.white, fontSize + 7);

        // ── Status line ─────────────────────────────────────────────────
        _statusLbl = Lbl(p, "", fontSize - 1, FontStyle.Normal,
                         new Color(0.7f, 0.9f, 0.7f), fontSize + 3);

        HSep(p);

        // ── Switch Solver ───────────────────────────────────────────────
        Lbl(p, "Switch Solver:", fontSize, FontStyle.Bold,
            Color.white, fontSize + 4);

        // Row 1: Force | F-Jobs | PBD
        var r1 = HRow(p, 4, 28);
        // Row 2: PBD-Jobs | CLOD | PBD-Cmp
        var r2 = HRow(p, 4, 28);
        // Row 3: Cull (left-aligned, fixed width so it doesn't stretch full panel)
        var r3 = HRow(p, 4, 28);
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

        HSep(p);

        // ── Scene presets ────────────────────────────────────────────────
        Lbl(p, "Scene Preset:", fontSize, FontStyle.Bold,
            Color.white, fontSize + 4);

        // Row 1: Random | Circle | Crossing
        var pr1 = HRow(p, 4, 28);
        // Row 2: Bi-Dir | Bottleneck | Double-BN
        var pr2 = HRow(p, 4, 28);
        // Row 3: MergeLane (left-aligned)
        var pr3 = HRow(p, 4, 28);
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

        HSep(p);

        // ── Common sliders ──────────────────────────────────────────────
        Lbl(p, "─── Common ───", fontSize - 1, FontStyle.Normal,
            new Color(0.6f, 0.6f, 0.6f), fontSize + 2);

        (_agentSld,  _agentLbl)  = MkSlider(p, "Agent Count: 50",
            _mgr.agentCount, 5, 500, true,  v => _mgr.agentCount  = (int)v);
        (_speedSld,  _speedLbl)  = MkSlider(p, "Speed: 5.00",
            _mgr.agentSpeed,  0.5f, 15f, false, v => _mgr.agentSpeed  = v);
        (_radiusSld, _radiusLbl) = MkSlider(p, "Agent Radius: 0.50",
            _mgr.agentRadius, 0.1f, 2f,  false, v => _mgr.agentRadius = v);
        (_neighSld,  _neighLbl)  = MkSlider(p, "Neighbour Dist: 4.0",
            _mgr.neighborDist, 1f, 20f, false, v => _mgr.neighborDist = v);

        // ── PBD section ─────────────────────────────────────────────────
        _pbdSec = SecGO(p, "PBDSec");
        Lbl(_pbdSec.transform, "─── PBD ───", fontSize - 1, FontStyle.Normal,
            new Color(0.4f, 0.75f, 1f), fontSize + 2);
        (_pbdIterSld, _pbdIterLbl) = MkSlider(_pbdSec.transform, "Iterations: 3",
            _mgr.pbdIterations, 1, 20, true,  v => _mgr.pbdIterations = (int)v);
        (_pbdKsiSld,  _pbdKsiLbl)  = MkSlider(_pbdSec.transform, "Ksi: 0.050",
            _mgr.pbdKsi, 0.01f, 1f, false, v => _mgr.pbdKsi = v);

        // ── Force section ────────────────────────────────────────────────
        _forceSec = SecGO(p, "ForceSec");
        Lbl(_forceSec.transform, "─── Force ───", fontSize - 1, FontStyle.Normal,
            new Color(1f, 0.6f, 0.2f), fontSize + 2);
        (_forceKSld,  _forceKLbl)  = MkSlider(_forceSec.transform, "K: 1.00",
            _mgr.forceK,  0.1f, 5f,  false, v => _mgr.forceK  = v);
        (_forceT0Sld, _forceT0Lbl) = MkSlider(_forceSec.transform, "t0: 3.0",
            _mgr.forceT0, 0.5f, 10f, false, v => _mgr.forceT0 = v);

        // ── Respawn ──────────────────────────────────────────────────────
        HSep(p);
        var rb = MkBtn(p, "Respawn Crowd", fontSize, () => _mgr.Respawn());
        rb.GetComponent<Image>().color = new Color(0.15f, 0.45f, 0.15f);
        rb.GetComponent<LayoutElement>().preferredHeight = 30;
        var rbc = rb.colors;
        rbc.highlightedColor = new Color(0.22f, 0.6f, 0.22f);
        rbc.pressedColor     = new Color(0.1f,  0.3f, 0.1f);
        rb.colors = rbc;
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
        t.color = new Color(0.85f, 0.85f, 0.85f);
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
