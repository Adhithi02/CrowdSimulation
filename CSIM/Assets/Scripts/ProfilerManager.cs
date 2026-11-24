using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using UnityEngine.Profiling;

/// Simple profiler to measure named code sections with a sliding window of recent samples.
/// *Not thread-safe (main-thread usage) *. Uses Stopwatch timestamps for low-overhead timing.
/// Also emits Unity Profiler samples for editor visibility.
/// #TODO: make safe for multi-threaded usage (e.g. neighbor queries in parallel jobs)
public class ProfilerManager : MonoBehaviour
{
    public static ProfilerManager Instance { get; private set; }

    [Tooltip("Number of recent samples kept per metric for averages/percentiles")]
    public int historySize = 300; // keep last 300 frames ~5s at 60fps

    [Tooltip("Enable logging to CSV (to Application.persistentDataPath)")]
    public bool enableCsvLogging = false;

    private Dictionary<string, long> _starts = new Dictionary<string, long>();
    private Dictionary<string, RollingBuffer> _buffers = new Dictionary<string, RollingBuffer>();
    private long _freq;

    private StreamWriter _csvWriter;

    void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(this); return; }
        Instance = this;
        _freq = Stopwatch.Frequency;
        if (enableCsvLogging)
        {
            var path = Path.Combine(Application.persistentDataPath, "profiler_log.csv");
            _csvWriter = new StreamWriter(path, append: true);
            _csvWriter.WriteLine("time_unix_ms,frame,frame_ms,neighbor_ms,solver_ms,animation_ms");
            UnityEngine.Debug.Log($"Profiler CSV: {path}");
        }
    }

    void OnDestroy()
    {
        if (_csvWriter != null)
        {
            _csvWriter.Flush();
            _csvWriter.Close();
            _csvWriter = null;
        }
    }

    /// Start a named sample (records timestamp)
    public void BeginSample(string name)
    {
        long t = Stopwatch.GetTimestamp();
        _starts[name] = t;
        // mark for Unity profiler (vis in Editor)
        Profiler.BeginSample(name);
    }

    /// End a previously begun named sample and record duration in ms
    public void EndSample(string name)
    {
        Profiler.EndSample();
        if (!_starts.TryGetValue(name, out long start))
        {
            UnityEngine.Debug.LogWarning($"ProfilerManager.EndSample called without BeginSample for '{name}'");
            return;
        }

        long end = Stopwatch.GetTimestamp();
        double ms = (end - start) * 1000.0 / _freq;

        if (!_buffers.TryGetValue(name, out var buf))
        {
            buf = new RollingBuffer(historySize);
            _buffers[name] = buf;
        }
        buf.Add((float)ms);

        _starts.Remove(name);
    }

    /// Get average duration in ms for the named sample
    public float GetAverageMs(string name)
    {
        if (_buffers.TryGetValue(name, out var buf)) return buf.Average;
        return float.NaN;
    }

    /// Get max duration in ms for the named sample
    public float GetMaxMs(string name)
    {
        if (_buffers.TryGetValue(name, out var buf)) return buf.Max;
        return float.NaN;
    }

    /// Get percentile (0-100) for the named sample
    public float GetPercentileMs(string name, float p)
    {
        if (_buffers.TryGetValue(name, out var buf)) return buf.Percentile(p);
        return float.NaN;
    }

    /// Call once per frame to optionally log CSV row. call at end of Update()
    public void LogFrameData(int frameNumber)
    {
        if (!enableCsvLogging || _csvWriter == null) return;
        float frameMs = GetAverageMs("Frame"); 
        float neighborMs = GetAverageMs("NeighborQuery");
        float solverMs = GetAverageMs("Solver");
        float animMs = GetAverageMs("Animation");
        long unixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        _csvWriter.WriteLine($"{unixMs},{frameNumber},{frameMs},{neighborMs},{solverMs},{animMs}");
        _csvWriter.Flush();
    }

    /// Forcibly add a sample (ms) to a named metric
    public void AddSampleMs(string name, float ms)
    {
        if (!_buffers.TryGetValue(name, out var buf))
        {
            buf = new RollingBuffer(historySize);
            _buffers[name] = buf;
        }
        buf.Add(ms);
    }

    // Small circular buffer for rolling stats
    private class RollingBuffer
    {
        readonly float[] _data;
        int _idx = 0;
        int _count = 0;
        float _sum = 0.0f;
        float _max = float.MinValue;

        public RollingBuffer(int capacity)
        {
            _data = new float[Math.Max(8, capacity)];
        }

        public void Add(float v)
        {
            if (_count < _data.Length)
            {
                _data[_idx++] = v;
                _count++;
                _sum += v;
            }
            else
            {
                // overwrite oldest
                _idx = _idx % _data.Length;
                _sum -= _data[_idx];
                _data[_idx] = v;
                _sum += v;
                _idx++;
            }
            if (v > _max) _max = v;
            if (_count == _data.Length)
            {
                // recompute max occasionally to avoid stale max after wrap
                _max = float.MinValue;
                for (int i = 0; i < _data.Length; ++i) if (_data[i] > _max) _max = _data[i];
            }
        }

        public float Average { get { return _count == 0 ? float.NaN : _sum / _count; } }
        public float Max { get { return _count == 0 ? float.NaN : _max; } }

        // percentile: copying+sorting (small arrays --> cheap)
        public float Percentile(float p)
        {
            if (_count == 0) return float.NaN;
            float[] copy = new float[_count];
            Array.Copy(_data, 0, copy, 0, _count);
            Array.Sort(copy);
            float rank = Math.Clamp(p / 100f * (_count - 1), 0f, _count - 1f);
            int i = (int)Math.Floor(rank);
            int j = Math.Min(i + 1, _count - 1);
            float t = rank - i;
            return Mathf.Lerp(copy[i], copy[j], t);
        }
    }
}