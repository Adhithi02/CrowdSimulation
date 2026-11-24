using UnityEngine;

public class GlobalFrameProfiler : MonoBehaviour
{
    void Update()
    {
        if (ProfilerManager.Instance != null) ProfilerManager.Instance.BeginSample("Frame");
    }

    void LateUpdate()
    {
        if (ProfilerManager.Instance != null) ProfilerManager.Instance.EndSample("Frame");
        // optional: write CSV once per frame
        if (ProfilerManager.Instance != null)
        {
            ProfilerManager.Instance.LogFrameData(Time.frameCount);
        }
    }
}