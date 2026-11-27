using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Presentation-only LOD for the PBDCuda crowd. It never disables an agent GameObject,
/// NavMeshAgent, PBDAgent, or PBDCudaSolver; it only controls renderers and animators.
/// </summary>
[DefaultExecutionOrder(100)]
public sealed class PBDCudaVisualLodController : MonoBehaviour
{
    [Header("Visual LOD")]
    public bool enableVisualLod = true;
    [Min(1f)] public float nearDistance = 12f;
    [Min(1f)] public float midDistance = 26f;
    [Tooltip("Far agents keep only their primary body mesh; simulation is unaffected.")]
    public bool simplifyFarMeshes = true;
    [Range(1f, 20f)] public float midAnimationFps = 10f;
    [Range(0.5f, 10f)] public float farAnimationFps = 3f;
    [Range(0.05f, 0.5f)] public float refreshInterval = 0.12f;
    [Tooltip("Disable commuter shadows outside the near band to protect GPU fill rate.")]
    public bool reduceDistantShadows = true;

    [Header("Presentation")]
    [Tooltip("F8 toggles this optional overlay at runtime.")]
    public bool showDiagnostics = false;
    public bool cleanShowcaseMode = true;
    public KeyCode diagnosticsToggleKey = KeyCode.F8;

    public int VisibleCommuterCount { get; private set; }
    public int NearCount { get; private set; }
    public int MidCount { get; private set; }
    public int FarCount { get; private set; }
    public int HiddenCount { get; private set; }
    public int ActiveAnimatorCount { get; private set; }
    public float FrameTimeMs { get; private set; }
    public float FramesPerSecond { get; private set; }

    private SimulationManager _simulation;
    private PBDCudaSolver _cuda;
    private Camera _presentationCamera;
    private readonly List<AgentPresentation> _agents = new();
    private readonly List<CameraState> _extraCameras = new();
    private float _nextRefreshTime;
    private float _nextCacheCheckTime;
    private Rect _diagnosticsRect = new Rect(16f, 16f, 340f, 170f);

    private enum VisualBand { Near, Mid, Far, Hidden }

    private sealed class AgentPresentation
    {
        public AgentBase agent;
        public Renderer[] renderers;
        public Animator[] animators;
        public bool[] originalRendererEnabled;
        public bool[] diagnosticRenderer;
        public bool[] farSimplifiedRenderer;
        public ShadowCastingMode[] originalShadowCasting;
        public bool[] originalReceiveShadows;
        public bool[] originalAnimatorEnabled;
        public AnimatorCullingMode[] originalCullingMode;
        public bool[] originalFoveatedEnabled;
        public FoveatedAnimationTarget[] foveatedTargets;
        public float nextAnimationUpdate;
        public VisualBand band;
    }

    private struct CameraState
    {
        public Camera camera;
        public bool cameraEnabled;
        public AudioListener listener;
        public bool listenerEnabled;
    }

    private void Awake()
    {
        _simulation = GetComponent<SimulationManager>();
        _cuda = GetComponent<PBDCudaSolver>();

        if (cleanShowcaseMode && _simulation != null)
        {
            // Fidelity colours instantiate/tint presentation materials and are useful
            // for solver inspection, not for the final station view.
            _simulation.showFidelityColors = false;
            _simulation.drawGrid = false;
        }

        ConfigurePresentationCameras();
    }

    private void OnEnable()
    {
        _nextRefreshTime = 0f;
        _nextCacheCheckTime = 0f;
    }

    private void OnDisable()
    {
        RestorePresentation();
        RestoreExtraCameras();
    }

    private void Update()
    {
        if (_simulation == null) _simulation = GetComponent<SimulationManager>();
        if (_cuda == null) _cuda = GetComponent<PBDCudaSolver>();

        if (Input.GetKeyDown(diagnosticsToggleKey))
            showDiagnostics = !showDiagnostics;

        float dt = Mathf.Max(Time.unscaledDeltaTime, 0.0001f);
        FrameTimeMs = dt * 1000f;
        FramesPerSecond = 1f / dt;

        if (!enableVisualLod || _simulation == null || _simulation.activeSolver != SolverMode.PBDCuda)
        {
            RestorePresentation();
            return;
        }

        if (Time.unscaledTime >= _nextCacheCheckTime)
        {
            _nextCacheCheckTime = Time.unscaledTime + 0.5f;
            if (NeedsRecache()) CacheAgents();
        }

        if (Time.unscaledTime < _nextRefreshTime) return;
        _nextRefreshTime = Time.unscaledTime + refreshInterval;

        _presentationCamera = ResolvePresentationCamera();
        if (_presentationCamera == null) return;

        UpdatePresentation();
    }

    /// <summary>Called after SimulationManager respawns the crowd.</summary>
    public void RefreshAgents()
    {
        CacheAgents();
    }

    public void ToggleDiagnostics()
    {
        showDiagnostics = !showDiagnostics;
    }

    private bool NeedsRecache()
    {
        if (_simulation == null || _agents.Count != _simulation.LiveAgentCount) return true;
        for (int i = 0; i < _agents.Count; i++)
            if (_agents[i].agent != _simulation.GetAgent(i)) return true;
        return false;
    }

    private Camera ResolvePresentationCamera()
    {
        if (_simulation != null && _simulation.benchmarkCamera != null &&
            _simulation.benchmarkCamera.isActiveAndEnabled)
            return _simulation.benchmarkCamera;

        return Camera.main;
    }

    private void ConfigurePresentationCameras()
    {
        if (!cleanShowcaseMode) return;

        Camera primary = Camera.main;
        if (primary == null) return;

        foreach (Camera camera in FindObjectsOfType<Camera>(true))
        {
            if (camera == null || camera == primary || !camera.isActiveAndEnabled) continue;

            var listener = camera.GetComponent<AudioListener>();
            _extraCameras.Add(new CameraState
            {
                camera = camera,
                cameraEnabled = camera.enabled,
                listener = listener,
                listenerEnabled = listener != null && listener.enabled
            });

            // MetroStationManager controls the tagged Main Camera for CCTV and showcase
            // views. The untagged duplicate camera provides no independent output.
            camera.enabled = false;
            if (listener != null) listener.enabled = false;
        }
    }

    private void RestoreExtraCameras()
    {
        foreach (CameraState state in _extraCameras)
        {
            if (state.camera != null) state.camera.enabled = state.cameraEnabled;
            if (state.listener != null) state.listener.enabled = state.listenerEnabled;
        }
        _extraCameras.Clear();
    }

    private void CacheAgents()
    {
        RestorePresentation();
        _agents.Clear();

        if (_simulation == null) return;

        for (int i = 0; i < _simulation.LiveAgentCount; i++)
        {
            AgentBase agent = _simulation.GetAgent(i);
            if (agent == null) continue;

            Renderer[] renderers = agent.GetComponentsInChildren<Renderer>(true);
            Animator[] animators = agent.GetComponentsInChildren<Animator>(true);
            FoveatedAnimationTarget[] foveated =
                agent.GetComponentsInChildren<FoveatedAnimationTarget>(true);

            var state = new AgentPresentation
            {
                agent = agent,
                renderers = renderers,
                animators = animators,
                originalRendererEnabled = new bool[renderers.Length],
                diagnosticRenderer = new bool[renderers.Length],
                farSimplifiedRenderer = new bool[renderers.Length],
                originalShadowCasting = new ShadowCastingMode[renderers.Length],
                originalReceiveShadows = new bool[renderers.Length],
                originalAnimatorEnabled = new bool[animators.Length],
                originalCullingMode = new AnimatorCullingMode[animators.Length],
                originalFoveatedEnabled = new bool[foveated.Length],
                foveatedTargets = foveated,
                band = VisualBand.Near
            };

            for (int r = 0; r < renderers.Length; r++)
            {
                if (renderers[r] == null) continue;
                state.originalRendererEnabled[r] = renderers[r].enabled;
                state.diagnosticRenderer[r] = cleanShowcaseMode &&
                                               renderers[r].gameObject.name == "Sphere";
                state.farSimplifiedRenderer[r] =
                    renderers[r] is SkinnedMeshRenderer &&
                    renderers[r].gameObject.name != "Ch01_Body";
                state.originalShadowCasting[r] = renderers[r].shadowCastingMode;
                state.originalReceiveShadows[r] = renderers[r].receiveShadows;
            }

            for (int a = 0; a < animators.Length; a++)
            {
                if (animators[a] == null) continue;
                state.originalAnimatorEnabled[a] = animators[a].enabled;
                state.originalCullingMode[a] = animators[a].cullingMode;
            }

            // The legacy foveated component also controls Animator playable graphs.
            // Visual LOD takes exclusive ownership of those visual components so the
            // two presentation systems cannot fight; PBDCuda does not depend on it.
            for (int f = 0; f < foveated.Length; f++)
            {
                if (foveated[f] == null) continue;
                state.originalFoveatedEnabled[f] = foveated[f].enabled;
                foveated[f].enabled = false;
            }

            _agents.Add(state);
        }
    }

    private void UpdatePresentation()
    {
        VisibleCommuterCount = NearCount = MidCount = FarCount = HiddenCount = ActiveAnimatorCount = 0;
        Plane[] planes = GeometryUtility.CalculateFrustumPlanes(_presentationCamera);
        Vector3 cameraPosition = _presentationCamera.transform.position;
        float nearSq = nearDistance * nearDistance;
        float midSq = midDistance * midDistance;
        float now = Time.unscaledTime;

        foreach (AgentPresentation state in _agents)
        {
            if (state.agent == null) continue;

            bool visible = IsVisible(state, planes);
            VisualBand nextBand;
            if (!visible) nextBand = VisualBand.Hidden;
            else
            {
                Vector3 delta = state.agent.transform.position - cameraPosition;
                float distanceSq = delta.sqrMagnitude;
                nextBand = distanceSq <= nearSq ? VisualBand.Near :
                           distanceSq <= midSq ? VisualBand.Mid : VisualBand.Far;
            }

            ApplyBand(state, nextBand, now);

            switch (nextBand)
            {
                case VisualBand.Near: NearCount++; VisibleCommuterCount++; break;
                case VisualBand.Mid: MidCount++; VisibleCommuterCount++; break;
                case VisualBand.Far: FarCount++; VisibleCommuterCount++; break;
                default: HiddenCount++; break;
            }
        }
    }

    private static bool IsVisible(AgentPresentation state, Plane[] planes)
    {
        for (int i = 0; i < state.renderers.Length; i++)
        {
            Renderer renderer = state.renderers[i];
            if (renderer != null && GeometryUtility.TestPlanesAABB(planes, renderer.bounds))
                return true;
        }
        return false;
    }

    private void ApplyBand(AgentPresentation state, VisualBand nextBand, float now)
    {
        bool visible = nextBand != VisualBand.Hidden;
        bool near = nextBand == VisualBand.Near;

        for (int r = 0; r < state.renderers.Length; r++)
        {
            Renderer renderer = state.renderers[r];
            if (renderer == null) continue;

            renderer.enabled = visible && state.originalRendererEnabled[r] &&
                               !state.diagnosticRenderer[r] &&
                               !(simplifyFarMeshes && nextBand == VisualBand.Far &&
                                 state.farSimplifiedRenderer[r]);
            if (reduceDistantShadows && !near)
            {
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
            }
            else
            {
                renderer.shadowCastingMode = state.originalShadowCasting[r];
                renderer.receiveShadows = state.originalReceiveShadows[r];
            }
        }

        if (nextBand == VisualBand.Near)
        {
            SetAnimators(state, true, false);
        }
        else if (nextBand == VisualBand.Mid)
        {
            // Keep the current pose between sparse manual evaluations. Positions still
            // follow the CUDA result every simulation step.
            if (now >= state.nextAnimationUpdate)
            {
                EvaluateAnimators(state, 1f / midAnimationFps);
                state.nextAnimationUpdate = now + 1f / midAnimationFps;
            }
            SetAnimators(state, false, true);
        }
        else if (nextBand == VisualBand.Far)
        {
            if (now >= state.nextAnimationUpdate)
            {
                EvaluateAnimators(state, 1f / farAnimationFps);
                state.nextAnimationUpdate = now + 1f / farAnimationFps;
            }
            SetAnimators(state, false, true);
        }
        else
        {
            SetAnimators(state, false, true);
        }

        state.band = nextBand;
    }

    private void SetAnimators(AgentPresentation state, bool enabled, bool culled)
    {
        for (int i = 0; i < state.animators.Length; i++)
        {
            Animator animator = state.animators[i];
            if (animator == null) continue;
            animator.enabled = enabled;
            animator.cullingMode = culled
                ? AnimatorCullingMode.CullCompletely
                : state.originalCullingMode[i];
            if (enabled) ActiveAnimatorCount++;
        }
    }

    private static void EvaluateAnimators(AgentPresentation state, float step)
    {
        for (int i = 0; i < state.animators.Length; i++)
        {
            Animator animator = state.animators[i];
            if (animator == null) continue;
            animator.enabled = true;
            animator.Update(step);
            animator.enabled = false;
        }
    }

    private void RestorePresentation()
    {
        foreach (AgentPresentation state in _agents)
        {
            for (int r = 0; r < state.renderers.Length; r++)
            {
                Renderer renderer = state.renderers[r];
                if (renderer == null) continue;
                renderer.enabled = state.originalRendererEnabled[r];
                renderer.shadowCastingMode = state.originalShadowCasting[r];
                renderer.receiveShadows = state.originalReceiveShadows[r];
            }

            for (int a = 0; a < state.animators.Length; a++)
            {
                Animator animator = state.animators[a];
                if (animator == null) continue;
                animator.enabled = state.originalAnimatorEnabled[a];
                animator.cullingMode = state.originalCullingMode[a];
            }

            for (int f = 0; f < state.foveatedTargets.Length; f++)
            {
                if (state.foveatedTargets[f] != null)
                    state.foveatedTargets[f].enabled = state.originalFoveatedEnabled[f];
            }
        }
    }

    private void OnGUI()
    {
        if (!showDiagnostics) return;

        // Bottom-centre: the OCC control panel owns the left edge and the PAGP lab
        // panel owns the right edge, so this box must never sit in either corner.
        _diagnosticsRect.x = (Screen.width - _diagnosticsRect.width) * 0.5f;
        _diagnosticsRect.y = Screen.height - _diagnosticsRect.height - 16f;

        GUI.Box(_diagnosticsRect, "PBDCuda Presentation LOD");
        GUILayout.BeginArea(new Rect(_diagnosticsRect.x + 12f, _diagnosticsRect.y + 26f,
            _diagnosticsRect.width - 24f, _diagnosticsRect.height - 36f));

        float cudaMs = _cuda != null && _cuda.IsInitialized ? _cuda.LastTimings.totalMs : 0f;
        GUILayout.Label("Live agents: " + (_simulation != null ? _simulation.LiveAgentCount : 0));
        GUILayout.Label("Visible: " + VisibleCommuterCount + "  Near: " + NearCount +
                        "  Mid: " + MidCount + "  Far: " + FarCount);
        GUILayout.Label("Offscreen: " + HiddenCount + "  Active Animators: " + ActiveAnimatorCount);
        GUILayout.Label("Frame: " + FrameTimeMs.ToString("F1") + " ms  |  FPS: " +
                        FramesPerSecond.ToString("F1"));
        GUILayout.Label("CUDA step: " + cudaMs.ToString("F2") + " ms  |  F8 hides diagnostics");
        GUILayout.EndArea();
    }
}