using UnityEngine;

/// <summary>
/// Single source of truth for the metro station geometry. All values are offsets
/// from SimulationManager.Center() (the station centre). The visual station
/// (MetroStationManager), the walkable obstacles (SimulationManager) and the
/// commuter routes (StationFlowController) all read from here, so they line up.
///
///   z = -25  south wall with two street stairways (x = -20, +20)
///   z = -8   fare-gate line: 6 gate cabinets, 5 lanes (3 entry, 2 exit)
///   z = 0..6 platform (pillars + benches at z = 2)
///   z = 7    platform screen doors (PSD), 6 door openings
///   z = 10   track with a 3-car train
/// </summary>
public static class StationLayout
{
    public const float HalfLength = 44f;
    public const float FloorMinZ = -25f;
    public const float FloorMaxZ = 9f;
    public const float WallHeight = 6f;

    public const float GateZ = -8f;
    public static readonly float[] GateCabinetX = { -7.5f, -4.5f, -1.5f, 1.5f, 4.5f, 7.5f };
    public static readonly float[] EntryLaneX = { -6f, -3f, 0f };
    public static readonly float[] ExitLaneX = { 3f, 6f };

    public const float PlatformMinZ = 0.6f;
    public const float PlatformMaxZ = 5.6f;
    public const float PsdZ = 7f;
    public const float DoorWaitZ = 6.2f;
    public const float TrackZ = 10f;
    public static readonly float[] CarX = { -25f, 0f, 25f };
    public static readonly float[] DoorX = { -31f, -19f, -6f, 6f, 19f, 31f };

    public const float PillarZ = 2f;
    public static readonly float[] PillarX = { -28f, -14f, 0f, 14f, 28f };
    public static readonly float[] BenchX = { -23f, -9f, 19f, 33f };

    public static readonly float[] StreetX = { -20f, 20f };
    public const float StreetZ = -23.3f;

    /// <summary>True if an (x,z) offset is clear of pillars and benches on the platform.</summary>
    public static bool PlatformSpotIsFree(float x, float z)
    {
        foreach (float px in PillarX)
            if (Mathf.Abs(x - px) < 1.4f && Mathf.Abs(z - PillarZ) < 1.4f) return false;
        foreach (float bx in BenchX)
            if (Mathf.Abs(x - bx) < 1.8f && Mathf.Abs(z - PillarZ) < 1.0f) return false;
        return true;
    }

    public static float NearestDoorX(float x)
    {
        float best = DoorX[0];
        foreach (float d in DoorX)
            if (Mathf.Abs(d - x) < Mathf.Abs(best - x)) best = d;
        return best;
    }

    public static float NearestStreetX(float x) => x < 0f ? StreetX[0] : StreetX[1];
}
