using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Drives the "Metro Commute" scenario: a continuous, realistic passenger cycle
/// through the station, with a constant population (so PAGP measurements are
/// taken on a fixed N):
///
///   street stairs → entry fare gate → platform spot → wait for the train
///   → walk to a door while the doors are open → board (rides away on the train)
///   → alight at a later stop → exit fare gate → street stairs → arrives again
///
/// Passengers "on the train" are parked out of sight below the track. They have
/// no navigation goal, so they stay still and never enter the visible station.
/// This component only sets navigation goals; collision handling remains the
/// job of the active PBD/CUDA solver.
/// </summary>
public sealed class StationFlowController : MonoBehaviour
{
    private enum Stage { ToEntryGate, ThroughEntryGate, ToPlatformSpot, Waiting, ToDoor, OnTrain, ToExitGate, ThroughExitGate, ToStreet }

    private sealed class Commuter
    {
        public AgentBase agent;
        public Transform target;
        public Stage stage;
        public float laneX;
    }

    public bool Active { get; private set; }
    public int WaitingCount { get; private set; }
    public int OnTrainCount { get; private set; }
    public int InConcourseCount { get; private set; }
    public int TotalBoarded { get; private set; }
    public int TotalAlighted { get; private set; }

    private readonly List<Commuter> _commuters = new();
    private Transform _targetRoot;
    private SimulationManager _sim;
    private MetroStationManager _metro;
    private Vector3 _c;
    private float _nextTick;
    private bool _doorsWereOpen;

    // ── Spawn layout (called by SimulationManager before agents exist) ───────
    // Role mix: 40% waiting on platform, 30% walking in from the street side,
    // 20% leaving towards the exit, 10% already arriving at the stairs.
    private static int RoleOf(int i)
    {
        int r = i % 10;
        return r < 4 ? 0 : r < 7 ? 1 : r < 9 ? 2 : 3;
    }

    public Vector3 InitialSpawnPoint(int i, Vector3 centre)
    {
        switch (RoleOf(i))
        {
            case 0: return centre + RandomPlatformSpot();
            case 1: return centre + new Vector3(Random.Range(-30f, 30f), 0f, Random.Range(-21f, -12f));
            case 2: return centre + new Vector3(Random.Range(-20f, 20f), 0f, Random.Range(-5f, -0.5f));
            default:
                float sx = StationLayout.StreetX[i % 2];
                return centre + new Vector3(sx + Random.Range(-2f, 2f), 0f, StationLayout.StreetZ + Random.Range(0f, 2.5f));
        }
    }

    /// <summary>Called by SimulationManager after every respawn.</summary>
    public void OnCrowdSpawned(bool active)
    {
        Clear();
        Active = active;
        if (!active) return;

        _sim = SimulationManager.Instance;
        _metro = MetroStationManager.Instance;
        _c = _sim.Center();
        _targetRoot = new GameObject("CommuterTargets").transform;
        _targetRoot.SetParent(transform, false);

        for (int i = 0; i < _sim.LiveAgentCount; i++)
        {
            AgentBase a = _sim.GetAgent(i);
            if (a == null) continue;
            var t = new GameObject("Target").transform;
            t.SetParent(_targetRoot, false);
            var cm = new Commuter { agent = a, target = t };
            a.Goals = new[] { t };   // a single goal disables the agent's own random goal switching
            a.goal = t;
            _commuters.Add(cm);

            switch (RoleOf(i))
            {
                case 0: cm.stage = Stage.Waiting; t.position = a.transform.position; break;
                case 1: StartEntry(cm); break;
                case 2: StartExit(cm); break;
                default: StartEntry(cm); break;
            }
        }
        _doorsWereOpen = false;
    }

    private void Clear()
    {
        _commuters.Clear();
        if (_targetRoot != null) Destroy(_targetRoot.gameObject);
        _targetRoot = null;
        WaitingCount = OnTrainCount = InConcourseCount = 0;
        TotalBoarded = TotalAlighted = 0;
    }

    // ── Main loop: cheap state checks at ~7 Hz ───────────────────────────────
    private void Update()
    {
        if (!Active || Time.time < _nextTick) return;
        _nextTick = Time.time + 0.15f;

        if (_metro == null) _metro = MetroStationManager.Instance;
        if (_metro != null && _metro.isEvacuationActive) return;   // the evacuation owns the goals

        bool doorsOpen = _metro != null && _metro.DoorsOpen;
        if (doorsOpen && !_doorsWereOpen) AlightPassengers();
        _doorsWereOpen = doorsOpen;

        int waiting = 0, onTrain = 0, concourse = 0;
        foreach (var cm in _commuters)
        {
            if (cm.agent == null) continue;
            Vector3 p = cm.agent.transform.position;
            float d = FlatDistance(p, cm.target.position);

            switch (cm.stage)
            {
                case Stage.ToEntryGate:
                    concourse++;
                    if (d < 1.3f) { cm.stage = Stage.ThroughEntryGate; SetTarget(cm, new Vector3(cm.laneX, 0f, StationLayout.GateZ + 2.5f)); }
                    break;
                case Stage.ThroughEntryGate:
                    if (d < 1.3f) { cm.stage = Stage.ToPlatformSpot; SetTarget(cm, RandomPlatformSpot()); }
                    break;
                case Stage.ToPlatformSpot:
                    if (d < 0.9f) cm.stage = Stage.Waiting;
                    else if (doorsOpen && p.z - _c.z > -1f && Random.value < 0.25f) GoToDoor(cm, p);
                    break;
                case Stage.Waiting:
                    waiting++;
                    // Staggered so the platform does not empty in a single frame.
                    if (doorsOpen && Random.value < 0.3f) GoToDoor(cm, p);
                    break;
                case Stage.ToDoor:
                    waiting++;
                    if (!doorsOpen) { cm.stage = Stage.ToPlatformSpot; SetTarget(cm, RandomPlatformSpot()); }   // missed it
                    else if (d < 1.0f) Board(cm);
                    break;
                case Stage.OnTrain:
                    onTrain++;
                    break;
                case Stage.ToExitGate:
                    if (d < 1.3f) { cm.stage = Stage.ThroughExitGate; SetTarget(cm, new Vector3(cm.laneX, 0f, StationLayout.GateZ - 2.5f)); }
                    break;
                case Stage.ThroughExitGate:
                    concourse++;
                    if (d < 1.3f)
                    {
                        cm.stage = Stage.ToStreet;
                        SetTarget(cm, new Vector3(StationLayout.NearestStreetX(p.x - _c.x) + Random.Range(-1.5f, 1.5f), 0f, StationLayout.StreetZ));
                    }
                    break;
                case Stage.ToStreet:
                    concourse++;
                    if (d < 1.4f) ArriveFromStreet(cm);   // one passenger leaves, another walks in
                    break;
            }
        }
        WaitingCount = waiting;
        OnTrainCount = onTrain;
        InConcourseCount = concourse;
    }

    // ── Transitions ──────────────────────────────────────────────────────────
    private void StartEntry(Commuter cm)
    {
        cm.stage = Stage.ToEntryGate;
        cm.laneX = StationLayout.EntryLaneX[Random.Range(0, StationLayout.EntryLaneX.Length)];
        SetTarget(cm, new Vector3(cm.laneX, 0f, StationLayout.GateZ - 2.5f));
    }

    private void StartExit(Commuter cm)
    {
        cm.stage = Stage.ToExitGate;
        cm.laneX = StationLayout.ExitLaneX[Random.Range(0, StationLayout.ExitLaneX.Length)];
        SetTarget(cm, new Vector3(cm.laneX, 0f, StationLayout.GateZ + 2.5f));
    }

    private void GoToDoor(Commuter cm, Vector3 p)
    {
        cm.stage = Stage.ToDoor;
        float door = StationLayout.NearestDoorX(p.x - _c.x);
        SetTarget(cm, new Vector3(door + Random.Range(-0.7f, 0.7f), 0f, StationLayout.DoorWaitZ));
    }

    private void Board(Commuter cm)
    {
        cm.stage = Stage.OnTrain;
        TotalBoarded++;
        var a = cm.agent;
        a.goal = null;                         // no goal → no preferred velocity → stays parked
        if (a._agent != null && a._agent.isOnNavMesh) a._agent.ResetPath();
        // Park out of sight below the track, spread out so parked riders don't collide.
        Vector3 park = _c + new Vector3(Random.Range(-38f, 38f), -6f, StationLayout.TrackZ + Random.Range(-1.5f, 1.5f));
        Place(a, park, false);
    }

    private void AlightPassengers()
    {
        foreach (var cm in _commuters)
        {
            if (cm.stage != Stage.OnTrain || cm.agent == null) continue;
            float door = StationLayout.DoorX[Random.Range(0, StationLayout.DoorX.Length)];
            Place(cm.agent, _c + new Vector3(door + Random.Range(-0.6f, 0.6f), 0f, StationLayout.DoorWaitZ - 0.3f), true);
            cm.agent.goal = cm.target;
            StartExit(cm);
            TotalAlighted++;
        }
    }

    private void ArriveFromStreet(Commuter cm)
    {
        float sx = StationLayout.StreetX[Random.Range(0, StationLayout.StreetX.Length)];
        Place(cm.agent, _c + new Vector3(sx + Random.Range(-1.8f, 1.8f), 0f, StationLayout.StreetZ + 0.4f), true);
        StartEntry(cm);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────
    private void SetTarget(Commuter cm, Vector3 offset)
    {
        Vector3 world = _c + offset;
        cm.target.position = world;
        cm.agent.goal = cm.target;
        _sim.TrySetAgentDestination(cm.agent, world);
    }

    private Vector3 RandomPlatformSpot()
    {
        for (int k = 0; k < 12; k++)
        {
            float x = Random.Range(-36f, 36f);
            float z = Random.Range(StationLayout.PlatformMinZ, StationLayout.PlatformMaxZ);
            if (StationLayout.PlatformSpotIsFree(x, z)) return new Vector3(x, 0f, z);
        }
        return new Vector3(Random.Range(-36f, 36f), 0f, 4.5f);
    }

    /// <summary>Teleport an agent (used only for boarding/alighting/street arrivals).</summary>
    private static void Place(AgentBase a, Vector3 pos, bool onNavMesh)
    {
        if (onNavMesh && UnityEngine.AI.NavMesh.SamplePosition(pos, out var hit, 2f, UnityEngine.AI.NavMesh.AllAreas))
            pos = hit.position;
        a.transform.position = pos;
        a.predictedPosition = pos;
        a.velocity = Vector3.zero;
        if (onNavMesh && a._agent != null && a._agent.enabled) a._agent.Warp(pos);
    }

    private static float FlatDistance(Vector3 a, Vector3 b)
    {
        float dx = a.x - b.x, dz = a.z - b.z;
        return Mathf.Sqrt(dx * dx + dz * dz);
    }

    /// <summary>True for passengers currently riding the train (parked below the track).</summary>
    public bool IsOnTrain(AgentBase a)
    {
        if (!Active || a == null) return false;
        return a.transform.position.y < _c.y - 3f;
    }
}
