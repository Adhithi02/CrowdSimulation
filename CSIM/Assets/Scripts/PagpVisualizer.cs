using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Makes two PAGP concepts visible in the 3D scene (presentation only; it never
/// changes the simulation):
///
///  • Hash-grid heat map: the station floor is tiled with the solver's spatial-hash
///    cells (PBDCudaSolver.cellSize, 4 m). Each tile is coloured by how many
///    commuters are in it. A commuter only tests its own tile and the 8 around it.
///  • LOD rings: a disc under each commuter shows the simulation fidelity tier the
///    GPU assigned this step (green = High, amber = Medium, blue = Low).
/// </summary>
public sealed class PagpVisualizer : MonoBehaviour
{
    public static readonly Color HighColor = new Color(0.20f, 0.95f, 0.40f, 0.85f);
    public static readonly Color MedColor  = new Color(1.00f, 0.75f, 0.10f, 0.85f);
    public static readonly Color LowColor  = new Color(0.30f, 0.60f, 1.00f, 0.85f);

    public bool ShowHashGrid { get; private set; }
    public bool ShowLodRings { get; private set; }
    public int OccupiedCells { get; private set; }
    public int MaxPerCell { get; private set; }
    public float CellSize { get; private set; } = 4f;

    private SimulationManager _sim;
    private Mesh _quad;
    private Material[] _cellMats;
    private Material[] _ringMats;
    private Texture2D _ringTex;
    private readonly List<GameObject> _cells = new();
    private readonly List<MeshRenderer> _cellRenderers = new();
    private int _ix0, _iz0, _nx, _nz;
    private int[] _counts;
    private readonly List<(AgentBase agent, MeshRenderer ring)> _rings = new();
    private float _nextTick;

    private void Awake()
    {
        _sim = GetComponent<SimulationManager>();
        var shader = Shader.Find("Sprites/Default");
        _quad = BuildQuad();

        // Occupancy buckets: empty, 1-2, 3-5, 6-9, 10+
        Color[] cellCols =
        {
            new Color(0.0f, 0.9f, 1.0f, 0.07f),
            new Color(0.0f, 0.9f, 1.0f, 0.28f),
            new Color(0.2f, 1.0f, 0.4f, 0.38f),
            new Color(1.0f, 0.8f, 0.1f, 0.48f),
            new Color(1.0f, 0.3f, 0.2f, 0.58f)
        };
        _cellMats = new Material[cellCols.Length];
        for (int i = 0; i < cellCols.Length; i++) _cellMats[i] = new Material(shader) { color = cellCols[i] };

        _ringTex = BuildRingTexture();
        Color[] ringCols = { HighColor, MedColor, LowColor };
        _ringMats = new Material[3];
        for (int i = 0; i < 3; i++) _ringMats[i] = new Material(shader) { color = ringCols[i], mainTexture = _ringTex };
    }

    public void SetHashGrid(bool on)
    {
        if (ShowHashGrid == on) return;
        ShowHashGrid = on;
        if (on) BuildCells(); else DestroyCells();
    }

    public void SetLodRings(bool on)
    {
        if (ShowLodRings == on) return;
        ShowLodRings = on;
        if (on) BuildRings(); else DestroyRings();
    }

    /// <summary>Called by SimulationManager after a respawn (old agents and their rings are gone).</summary>
    public void OnCrowdSpawned()
    {
        _rings.Clear();
        if (ShowLodRings) BuildRings();
        if (ShowHashGrid) { DestroyCells(); BuildCells(); }
    }

    private void Update()
    {
        if ((!ShowHashGrid && !ShowLodRings) || _sim == null || Time.unscaledTime < _nextTick) return;
        _nextTick = Time.unscaledTime + 0.2f;
        if (ShowHashGrid) UpdateCells();
        if (ShowLodRings) UpdateRings();
    }

    // ── Hash-grid heat map ───────────────────────────────────────────────────
    private void BuildCells()
    {
        var cuda = GetComponent<PBDCudaSolver>();
        CellSize = cuda != null ? Mathf.Max(1f, cuda.cellSize) : 4f;
        Vector3 c = _sim.Center();
        float L = StationLayout.HalfLength;

        // World-aligned cells, exactly like WorldToCell() in the CUDA kernels.
        _ix0 = Mathf.FloorToInt((c.x - L) / CellSize);
        _iz0 = Mathf.FloorToInt((c.z + StationLayout.FloorMinZ) / CellSize);
        _nx = Mathf.FloorToInt((c.x + L) / CellSize) - _ix0 + 1;
        _nz = Mathf.FloorToInt((c.z + StationLayout.PsdZ) / CellSize) - _iz0 + 1;
        _counts = new int[_nx * _nz];

        float tile = CellSize - 0.25f;
        for (int z = 0; z < _nz; z++)
        for (int x = 0; x < _nx; x++)
        {
            var go = new GameObject("HashCell");
            go.transform.SetParent(transform, false);
            go.transform.position = new Vector3((_ix0 + x + 0.5f) * CellSize, c.y + 0.04f, (_iz0 + z + 0.5f) * CellSize);
            go.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
            go.transform.localScale = new Vector3(tile, tile, 1f);
            go.AddComponent<MeshFilter>().sharedMesh = _quad;
            var mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterial = _cellMats[0];
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            mr.receiveShadows = false;
            _cells.Add(go);
            _cellRenderers.Add(mr);
        }
        UpdateCells();
    }

    private void UpdateCells()
    {
        if (_counts == null) return;
        if (_cellRenderers.Count != _counts.Length)
        {
            // State mismatch (e.g. a script hot-reload during Play restores the array but
            // not the readonly lists): rebuild the tiles instead of indexing past the end.
            DestroyCells();
            BuildCells();
            return;
        }
        System.Array.Clear(_counts, 0, _counts.Length);
        float floorY = _sim.Center().y;
        for (int i = 0; i < _sim.LiveAgentCount; i++)
        {
            var a = _sim.GetAgent(i);
            if (a == null) continue;
            Vector3 p = a.transform.position;
            if (p.y < floorY - 3f) continue;    // riding the train (parked below the track)
            int x = Mathf.FloorToInt(p.x / CellSize) - _ix0;
            int z = Mathf.FloorToInt(p.z / CellSize) - _iz0;
            if (x >= 0 && x < _nx && z >= 0 && z < _nz) _counts[z * _nx + x]++;
        }
        int occupied = 0, max = 0;
        for (int i = 0; i < _counts.Length; i++)
        {
            int n = _counts[i];
            if (n > 0) occupied++;
            if (n > max) max = n;
            int bucket = n == 0 ? 0 : n <= 2 ? 1 : n <= 5 ? 2 : n <= 9 ? 3 : 4;
            var mat = _cellMats[bucket];
            if (_cellRenderers[i].sharedMaterial != mat) _cellRenderers[i].sharedMaterial = mat;
        }
        OccupiedCells = occupied;
        MaxPerCell = max;
    }

    private void DestroyCells()
    {
        foreach (var go in _cells) if (go != null) Destroy(go);
        _cells.Clear();
        _cellRenderers.Clear();
        _counts = null;
    }

    // ── LOD rings ────────────────────────────────────────────────────────────
    private void BuildRings()
    {
        DestroyRings();
        for (int i = 0; i < _sim.LiveAgentCount; i++)
        {
            var a = _sim.GetAgent(i);
            if (a == null) continue;
            var go = new GameObject("LodRing");
            go.transform.SetParent(a.transform, false);
            float s = 1.25f / Mathf.Max(0.01f, a.transform.lossyScale.x);
            go.transform.localPosition = new Vector3(0f, 0.06f / Mathf.Max(0.01f, a.transform.lossyScale.y), 0f);
            go.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
            go.transform.localScale = new Vector3(s, s, 1f);
            go.AddComponent<MeshFilter>().sharedMesh = _quad;
            var mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterial = _ringMats[Mathf.Clamp(a.fidelityLevel, 0, 2)];
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            mr.receiveShadows = false;
            _rings.Add((a, mr));
        }
    }

    private void UpdateRings()
    {
        foreach (var (agent, ring) in _rings)
        {
            if (agent == null || ring == null) continue;
            var mat = _ringMats[Mathf.Clamp(agent.fidelityLevel, 0, 2)];
            if (ring.sharedMaterial != mat) ring.sharedMaterial = mat;
        }
    }

    private void DestroyRings()
    {
        foreach (var (_, ring) in _rings) if (ring != null) Destroy(ring.gameObject);
        _rings.Clear();
    }

    // ── Procedural assets ────────────────────────────────────────────────────
    private static Mesh BuildQuad()
    {
        var m = new Mesh { name = "PagpQuad" };
        m.vertices = new[] { new Vector3(-0.5f, -0.5f), new Vector3(0.5f, -0.5f), new Vector3(-0.5f, 0.5f), new Vector3(0.5f, 0.5f) };
        m.uv = new[] { new Vector2(0, 0), new Vector2(1, 0), new Vector2(0, 1), new Vector2(1, 1) };
        m.colors = new[] { Color.white, Color.white, Color.white, Color.white };   // Sprites/Default multiplies by vertex colour
        m.triangles = new[] { 0, 2, 1, 2, 3, 1 };
        m.RecalculateBounds();
        return m;
    }

    private static Texture2D BuildRingTexture()
    {
        const int N = 64;
        var tex = new Texture2D(N, N, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
        var px = new Color[N * N];
        for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
        {
            float dx = (x + 0.5f) / N * 2f - 1f, dy = (y + 0.5f) / N * 2f - 1f;
            float r = Mathf.Sqrt(dx * dx + dy * dy);
            float a = r > 1f ? 0f : r > 0.7f ? 1f : 0.22f;   // solid rim, faint fill
            px[y * N + x] = new Color(1f, 1f, 1f, a);
        }
        tex.SetPixels(px);
        tex.Apply();
        return tex;
    }
}
