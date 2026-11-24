using System.Collections;
using TMPro;
using UnityEngine;

public class ProfilerUI : MonoBehaviour
{
    public TMP_Text sign;
    public FocusPointSphere focusPoint;

    void Start()
    {
        StartCoroutine(UpdateLoop());
    }

    private IEnumerator UpdateLoop()
    {
        yield return new WaitForSeconds(0.5f);
        while (true)
        {
            UpdateText();
            yield return new WaitForSeconds(0.1f);
        }
    }

    private void UpdateText()
    {
        string fpsStr = focusPoint != null ? $"{focusPoint.fpsEstimate():F1}" : $"{(1.0f / Time.deltaTime):F1}";
        string leftRightEye = focusPoint != null && focusPoint.UsingRighteye() ? "Right Eye" : "Left Eye";

        var pm = ProfilerManager.Instance;
        string frameAvg = pm != null ? $"{pm.GetAverageMs("Frame"):F2} ms" : "n/a";
        string frame95 = pm != null ? $"{pm.GetPercentileMs("Frame", 95):F2} ms" : "n/a";
        string neighborAvg = pm != null ? $"{pm.GetAverageMs("NeighborQuery"):F2} ms" : "n/a";
        string neighbor95 = pm != null ? $"{pm.GetPercentileMs("NeighborQuery", 95):F2} ms" : "n/a";
        string solverAvg = pm != null ? $"{pm.GetAverageMs("Solver"):F2} ms" : "n/a";
        string solver95 = pm != null ? $"{pm.GetPercentileMs("Solver", 95):F2} ms" : "n/a";

        sign.text = $"FPS ≈ {fpsStr} | Frame avg/95: {frameAvg} / {frame95}\n" +
                    $"Neighbor avg/95: {neighborAvg} / {neighbor95}\n" +
                    $"Solver avg/95: {solverAvg} / {solver95}\n" +
                    (focusPoint != null ? $"Fov = {focusPoint.foveationFactor:F3} | Using {leftRightEye}" : "");
    }
}