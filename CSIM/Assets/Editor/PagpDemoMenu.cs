using UnityEditor;
using UnityEngine;

/// <summary>Editor shortcuts for the PAGP suite (Play mode only).</summary>
public static class PagpDemoMenu
{
    [MenuItem("Tools/PAGP/Run Guided Demo")]
    private static void RunDemo()
    {
        var lab = Object.FindObjectOfType<PAGPPerformanceLab>();
        if (lab != null && !lab.isRunning) lab.Run60SecondShowcaseSequence();
    }

    [MenuItem("Tools/PAGP/Stop Run")]
    private static void StopRun() => Object.FindObjectOfType<PAGPPerformanceLab>()?.StopBenchmark();

    [MenuItem("Tools/PAGP/Run Guided Demo", true)]
    [MenuItem("Tools/PAGP/Stop Run", true)]
    private static bool InPlayMode() => Application.isPlaying;
}
