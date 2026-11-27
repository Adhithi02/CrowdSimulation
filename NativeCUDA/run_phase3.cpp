// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 3: Final Performance Evaluation & Experimental Benchmarking
// Metro Station Digital Twin — PAGP Project
// RTX 2050 Mobile (sm_86) | CUDA 13.5
//
// Experiments A–E with strict repeatability protocol:
//   2 warmup trials + 20 measured trials per configuration
//   Individual trial timings recorded
//   Full statistics: mean, median, stddev, min, max, p95
// ═══════════════════════════════════════════════════════════════════════════════

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include "CrowdCommon.h"

// ─────────────────────────────────────────────────────────────────────────────
// DLL Interface
// ─────────────────────────────────────────────────────────────────────────────
extern "C" {
    bool InitCUDA(int maxAgents);
    bool StepCUDA(AgentGPU* hostAgents, int agentCount, SimParams* params, int iterations);
    void ShutdownCUDA();
    bool GetLastTimings(CUDATimingData* out);
    bool GetGPUMemoryInfo(GPUMemoryInfo* out);
    bool RunBruteForceNeighborBenchmark(AgentGPU* hostAgents, int agentCount, float interactionRadius, long long* outTotalCandidates, long long* outTotalNeighbors);
    bool GetHashGridStats(HashGridStats* out);
    bool GetDetailedHashStats(DetailedHashGridStats* out);
    bool CollectCUDACollisionPairs(AgentGPU* hostAgents, int agentCount, float cellSize, float threshold, int* outPairA, int* outPairB, int maxPairs, int* outPairCount, AuditedNeighborStats* outStats);
    bool MeasureConstraintViolations(AgentGPU* hostAgents, int agentCount, SimParams* params, ConstraintStats* out);
    bool RunAoSvsSoABenchmark(AgentGPU* hostAgents, int agentCount, SimParams* params, int iterations, SoABenchmarkResult* out);
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
static const int WARMUP_TRIALS   = 2;
static const int MEASURED_TRIALS = 20;
static const int AGENT_COUNTS[]  = { 500, 1000, 2500, 5000 };
static const int NUM_COUNTS      = 4;
static const float CPU_BF_TIMEOUT_MS = 10000.0f; // 10 second timeout

// Output directory
static const char* OUTPUT_DIR = "Phase3_Final";

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
float RandomFloat(float minVal, float maxVal) {
    return minVal + ((float)rand() / (float)RAND_MAX) * (maxVal - minVal);
}

inline int CPUHash(int cx, int cz) {
    return ((cx * 73856093) ^ (cz * 19349663)) & HASH_BUCKET_MASK;
}

struct Stats {
    float mean, median, stddev, minVal, maxVal, p95;
};

Stats ComputeStats(const std::vector<float>& data) {
    Stats s = {};
    int n = (int)data.size();
    if (n == 0) return s;

    std::vector<float> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    float sum = 0.0f;
    for (float v : sorted) sum += v;
    s.mean   = sum / n;
    s.minVal = sorted[0];
    s.maxVal = sorted[n - 1];
    s.median = (n % 2 == 0) ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0f : sorted[n/2];

    int p95idx = (int)(0.95f * n);
    if (p95idx >= n) p95idx = n - 1;
    s.p95 = sorted[p95idx];

    float varSum = 0.0f;
    for (float v : sorted) varSum += (v - s.mean) * (v - s.mean);
    s.stddev = sqrtf(varSum / n);

    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent Setup (Uniform station distribution, deterministic seed)
// ─────────────────────────────────────────────────────────────────────────────
void SetupAgentsUniform(AgentGPU* agents, int count, float radius) {
    for (int i = 0; i < count; i++) {
        agents[i].position = make_float3(RandomFloat(-40.0f, 40.0f), 0.0f, RandomFloat(-15.0f, 15.0f));
        agents[i].predictedPosition = agents[i].position;
        agents[i].velocity = make_float3(RandomFloat(-2.0f, 2.0f), 0.0f, RandomFloat(-2.0f, 2.0f));
        agents[i].preferredVelocity = make_float3(RandomFloat(-3.0f, 3.0f), 0.0f, RandomFloat(-3.0f, 3.0f));
        agents[i].radius = radius;
        agents[i].mass = 1.0f;
        agents[i].ksi = 0.05f;
        agents[i].lodLevel = (i % 3);
        agents[i].importanceScore = RandomFloat(0.1f, 0.9f);
        agents[i].lastContactCount = rand() % 5;
    }
}

void SetupParams(SimParams& p, int count) {
    memset(&p, 0, sizeof(SimParams));
    p.agentCount = count;
    p.deltaTime = 0.02f;
    p.cellSize = 4.0f;
    p.highDensityCutoff = 12;
    p.medDensityCutoff = 4;
    p.focusPoint = make_float3(0.0f, 0.0f, 0.0f);
    p.hasFrustumPlanes = 0;
    p.weightDistance = 0.30f;
    p.weightVisibility = 0.25f;
    p.weightInteraction = 0.20f;
    p.weightDensity = 0.25f;
    p.highImportanceThresh = 0.60f;
    p.medImportanceThresh = 0.30f;
    p.maxImportanceDist = 60.0f;
    p.maxInteractionContacts = 6;
    p.iterationCount = 3;
    p.solveLowThisFrame = 1;
    p.kShortRange = 1.0f;
    p.kLongRange = 1.0f;
    p.muStatic = 0.21f;
    p.muKinematic = 0.10f;
    p.avgCoefficient = 0.25f;
    p.lrcLookAhead = 2.0f;
    p.lodAblationMode = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU Brute-Force Solver (O(N^2) reference)
// ─────────────────────────────────────────────────────────────────────────────
void CPUSolveBruteForce(AgentGPU* agents, int count, SimParams p, int iterations) {
    for (int i = 0; i < count; i++) {
        agents[i].velocity = lerp(agents[i].velocity, agents[i].preferredVelocity, agents[i].ksi);
        agents[i].predictedPosition = agents[i].position + agents[i].velocity * p.deltaTime;
    }
    for (int it = 0; it < iterations; it++) {
        for (int i = 0; i < count; i++) {
            float3 delta = make_float3(0, 0, 0);
            int c = 0;
            for (int j = 0; j < count; j++) {
                if (i == j) continue;
                float3 diff = agents[i].predictedPosition - agents[j].predictedPosition;
                diff.y = 0;
                float d = length(diff);
                float margin = agents[i].radius + agents[j].radius;
                if (d < margin && d > 1e-4f) {
                    float3 n = diff / d;
                    float pen = margin - d;
                    delta += 0.5f * p.kShortRange * pen * n;
                    c++;
                }
            }
            if (c > 0) agents[i].predictedPosition += delta / (float)c;
        }
    }
    for (int i = 0; i < count; i++) {
        agents[i].velocity = (agents[i].predictedPosition - agents[i].position) / p.deltaTime;
        agents[i].position = agents[i].predictedPosition;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU Spatial Hash Solver (fair comparison to CUDA)
// ─────────────────────────────────────────────────────────────────────────────
void CPUSolveSpatialHash(AgentGPU* agents, int count, SimParams p, int iterations) {
    for (int i = 0; i < count; i++) {
        agents[i].velocity = lerp(agents[i].velocity, agents[i].preferredVelocity, agents[i].ksi);
        agents[i].predictedPosition = agents[i].position + agents[i].velocity * p.deltaTime;
    }

    std::vector<std::vector<int>> grid(HASH_BUCKET_COUNT);
    for (int i = 0; i < count; i++) {
        int cx = (int)floorf(agents[i].predictedPosition.x / p.cellSize);
        int cz = (int)floorf(agents[i].predictedPosition.z / p.cellSize);
        int h  = CPUHash(cx, cz);
        grid[h].push_back(i);
    }

    std::vector<float3> corrections(count, make_float3(0, 0, 0));
    std::vector<int> counts(count, 0);

    for (int it = 0; it < iterations; it++) {
        for (int i = 0; i < count; i++) {
            corrections[i] = make_float3(0, 0, 0);
            counts[i] = 0;

            int cx = (int)floorf(agents[i].predictedPosition.x / p.cellSize);
            int cz = (int)floorf(agents[i].predictedPosition.z / p.cellSize);
            float3 totalDelta = make_float3(0, 0, 0);
            int c = 0;

            int visitedBuckets[9];
            int visitedCount = 0;

            for (int dx = -1; dx <= 1; dx++) {
                for (int dz = -1; dz <= 1; dz++) {
                    int h = CPUHash(cx + dx, cz + dz);
                    bool dup = false;
                    for (int v = 0; v < visitedCount; v++) {
                        if (visitedBuckets[v] == h) { dup = true; break; }
                    }
                    if (dup) continue;
                    visitedBuckets[visitedCount++] = h;

                    const auto& bucket = grid[h];
                    for (int nb : bucket) {
                        if (nb == i) continue;
                        float3 diff = agents[i].predictedPosition - agents[nb].predictedPosition;
                        diff.y = 0;
                        float d = length(diff);
                        float margin = agents[i].radius + agents[nb].radius;
                        if (d < margin && d > 1e-4f) {
                            float3 n = diff / d;
                            float pen = margin - d;
                            totalDelta += 0.5f * p.kShortRange * pen * n;
                            c++;
                        }
                    }
                }
            }
            corrections[i] = totalDelta;
            counts[i] = c;
        }

        for (int i = 0; i < count; i++) {
            if (counts[i] > 0)
                agents[i].predictedPosition += (p.avgCoefficient * corrections[i]) / (float)counts[i];
        }
    }

    for (int i = 0; i < count; i++) {
        agents[i].velocity = (agents[i].predictedPosition - agents[i].position) / p.deltaTime;
        agents[i].position = agents[i].predictedPosition;
    }
}

// Helper: Save original agent positions for deterministic re-run
void CopyAgents(const AgentGPU* src, AgentGPU* dst, int count) {
    memcpy(dst, src, sizeof(AgentGPU) * count);
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT A: CPU BF vs CPU SH vs CUDA SH Scalability
// ═══════════════════════════════════════════════════════════════════════════════
void ExperimentA(FILE* fTrials, FILE* fStats) {
    printf("\n==================================================================\n");
    printf("  EXPERIMENT A: CPU vs CUDA Scalability (2 warmup + 20 measured)\n");
    printf("==================================================================\n");

    fprintf(fTrials, "Experiment,AgentCount,Solver,Trial,StepMs,SolveMs,Overflow,Dropped\n");

    for (int ci = 0; ci < NUM_COUNTS; ci++) {
        int n = AGENT_COUNTS[ci];
        AgentGPU* agents = (AgentGPU*)malloc(sizeof(AgentGPU) * n);
        AgentGPU* backup = (AgentGPU*)malloc(sizeof(AgentGPU) * n);
        SimParams p;

        // ── CPU Brute Force ──
        {
            std::vector<float> trials;
            bool timedOut = false;

            for (int t = 0; t < WARMUP_TRIALS + MEASURED_TRIALS; t++) {
                srand(4142026);
                SetupAgentsUniform(agents, n, 0.5f);
                SetupParams(p, n);

                auto start = std::chrono::high_resolution_clock::now();
                CPUSolveBruteForce(agents, n, p, 3);
                auto end = std::chrono::high_resolution_clock::now();
                float ms = std::chrono::duration<float, std::milli>(end - start).count();

                if (ms > CPU_BF_TIMEOUT_MS) {
                    timedOut = true;
                    if (t >= WARMUP_TRIALS) {
                        fprintf(fTrials, "A,%d,CPU_BruteForce,%d,TIMEOUT,TIMEOUT,0,0\n", n, t - WARMUP_TRIALS + 1);
                    }
                    // Record the measurement anyway if it finished, even if slow
                    if (t >= WARMUP_TRIALS) trials.push_back(ms);
                    break; // stop further trials for this config
                }
                if (t >= WARMUP_TRIALS) {
                    trials.push_back(ms);
                    fprintf(fTrials, "A,%d,CPU_BruteForce,%d,%.4f,%.4f,0,0\n", n, t - WARMUP_TRIALS + 1, ms, ms);
                }
            }

            if (!timedOut && trials.size() > 0) {
                Stats s = ComputeStats(trials);
                fprintf(fStats, "A,%d,CPU_BruteForce,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                    n, s.mean, s.median, s.stddev, s.minVal, s.maxVal, s.p95, (int)trials.size());
                printf("  [CPU BF ] N=%5d | mean=%.2f ms | median=%.2f | stddev=%.2f | min=%.2f | max=%.2f | p95=%.2f (%d trials)\n",
                    n, s.mean, s.median, s.stddev, s.minVal, s.maxVal, s.p95, (int)trials.size());
            } else if (timedOut && trials.size() > 0) {
                Stats s = ComputeStats(trials);
                fprintf(fStats, "A,%d,CPU_BruteForce,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d_TIMEOUT\n",
                    n, s.mean, s.median, s.stddev, s.minVal, s.maxVal, s.p95, (int)trials.size());
                printf("  [CPU BF ] N=%5d | TIMEOUT (>%.0fs) partial: mean=%.2f ms (%d trials)\n",
                    n, CPU_BF_TIMEOUT_MS/1000.0f, s.mean, (int)trials.size());
            } else {
                fprintf(fStats, "A,%d,CPU_BruteForce,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,0\n", n);
                printf("  [CPU BF ] N=%5d | TIMEOUT (>%.0fs)\n", n, CPU_BF_TIMEOUT_MS/1000.0f);
            }
        }

        // ── CPU Spatial Hash ──
        {
            std::vector<float> trials;
            for (int t = 0; t < WARMUP_TRIALS + MEASURED_TRIALS; t++) {
                srand(4142026);
                SetupAgentsUniform(agents, n, 0.5f);
                SetupParams(p, n);

                auto start = std::chrono::high_resolution_clock::now();
                CPUSolveSpatialHash(agents, n, p, 3);
                auto end = std::chrono::high_resolution_clock::now();
                float ms = std::chrono::duration<float, std::milli>(end - start).count();

                if (t >= WARMUP_TRIALS) {
                    trials.push_back(ms);
                    fprintf(fTrials, "A,%d,CPU_SpatialHash,%d,%.4f,%.4f,0,0\n", n, t - WARMUP_TRIALS + 1, ms, ms);
                }
            }
            Stats s = ComputeStats(trials);
            fprintf(fStats, "A,%d,CPU_SpatialHash,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                n, s.mean, s.median, s.stddev, s.minVal, s.maxVal, s.p95, (int)trials.size());
            printf("  [CPU SH ] N=%5d | mean=%.2f ms | median=%.2f | stddev=%.2f | min=%.2f | max=%.2f | p95=%.2f\n",
                n, s.mean, s.median, s.stddev, s.minVal, s.maxVal, s.p95);
        }

        // ── CUDA Spatial Hash (Full Pipeline) ──
        {
            std::vector<float> totalTrials;
            std::vector<float> solveTrials;

            // Reset to initial state for all CUDA trials
            for (int t = 0; t < WARMUP_TRIALS + MEASURED_TRIALS; t++) {
                srand(4142026);
                SetupAgentsUniform(agents, n, 0.5f);
                SetupParams(p, n);

                StepCUDA(agents, n, &p, 3);
                CUDATimingData timing;
                GetLastTimings(&timing);

                if (t >= WARMUP_TRIALS) {
                    totalTrials.push_back(timing.totalMs);
                    solveTrials.push_back(timing.solveMs);

                    // Get overflow stats for this trial
                    DetailedHashGridStats dhs;
                    GetDetailedHashStats(&dhs);

                    fprintf(fTrials, "A,%d,CUDA_SpatialHash,%d,%.4f,%.4f,%d,%d\n",
                        n, t - WARMUP_TRIALS + 1, timing.totalMs, timing.solveMs,
                        dhs.totalOverflowInsertions, dhs.totalDroppedInsertions);
                }
            }
            Stats sTotal = ComputeStats(totalTrials);
            Stats sSolve = ComputeStats(solveTrials);
            fprintf(fStats, "A,%d,CUDA_Total,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                n, sTotal.mean, sTotal.median, sTotal.stddev, sTotal.minVal, sTotal.maxVal, sTotal.p95, (int)totalTrials.size());
            fprintf(fStats, "A,%d,CUDA_SolveOnly,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                n, sSolve.mean, sSolve.median, sSolve.stddev, sSolve.minVal, sSolve.maxVal, sSolve.p95, (int)solveTrials.size());
            printf("  [CUDA   ] N=%5d | total: mean=%.3f ms median=%.3f stddev=%.3f | solve: mean=%.3f ms median=%.3f\n",
                n, sTotal.mean, sTotal.median, sTotal.stddev, sSolve.mean, sSolve.median);
        }

        free(agents);
        free(backup);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT B: CUDA Pipeline Stage Breakdown
// ═══════════════════════════════════════════════════════════════════════════════
void ExperimentB(FILE* fTrials, FILE* fStats) {
    printf("\n==================================================================\n");
    printf("  EXPERIMENT B: CUDA Pipeline Stage Breakdown\n");
    printf("==================================================================\n");

    fprintf(fTrials, "Experiment,AgentCount,Trial,H2D_ms,Predict_ms,ClearHash_ms,HashAgents_ms,LOD_ms,Solve_ms,Finalize_ms,D2H_ms,Total_ms,SumOfStages_ms,Discrepancy_ms\n");

    for (int ci = 0; ci < NUM_COUNTS; ci++) {
        int n = AGENT_COUNTS[ci];
        AgentGPU* agents = (AgentGPU*)malloc(sizeof(AgentGPU) * n);
        SimParams p;

        std::vector<float> h2d, predict, clearH, hashA, lod, solve, finalize, d2h, total;

        for (int t = 0; t < WARMUP_TRIALS + MEASURED_TRIALS; t++) {
            srand(4142026);
            SetupAgentsUniform(agents, n, 0.5f);
            SetupParams(p, n);

            StepCUDA(agents, n, &p, 3);
            CUDATimingData td;
            GetLastTimings(&td);

            if (t >= WARMUP_TRIALS) {
                h2d.push_back(td.h2dMs);
                predict.push_back(td.predictMs);
                clearH.push_back(td.clearHashMs);
                hashA.push_back(td.hashAgentsMs);
                lod.push_back(td.lodMs);
                solve.push_back(td.solveMs);
                finalize.push_back(td.finalizeMs);
                d2h.push_back(td.d2hMs);
                total.push_back(td.totalMs);

                float sumStages = td.h2dMs + td.predictMs + td.clearHashMs + td.hashAgentsMs
                                + td.lodMs + td.solveMs + td.finalizeMs + td.d2hMs;
                float discrepancy = td.totalMs - sumStages;

                fprintf(fTrials, "B,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    n, t - WARMUP_TRIALS + 1,
                    td.h2dMs, td.predictMs, td.clearHashMs, td.hashAgentsMs,
                    td.lodMs, td.solveMs, td.finalizeMs, td.d2hMs,
                    td.totalMs, sumStages, discrepancy);
            }
        }

        Stats sH2D    = ComputeStats(h2d);
        Stats sPred   = ComputeStats(predict);
        Stats sClear  = ComputeStats(clearH);
        Stats sHash   = ComputeStats(hashA);
        Stats sLOD    = ComputeStats(lod);
        Stats sSolve  = ComputeStats(solve);
        Stats sFin    = ComputeStats(finalize);
        Stats sD2H    = ComputeStats(d2h);
        Stats sTotal  = ComputeStats(total);

        float sumMeans = sH2D.mean + sPred.mean + sClear.mean + sHash.mean
                       + sLOD.mean + sSolve.mean + sFin.mean + sD2H.mean;

        fprintf(fStats, "B,%d,H2D,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", n, sH2D.mean, sH2D.median, sH2D.stddev, sH2D.minVal, sH2D.maxVal, sH2D.p95, MEASURED_TRIALS);
        fprintf(fStats, "B,%d,Predict,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", n, sPred.mean, sPred.median, sPred.stddev, sPred.minVal, sPred.maxVal, sPred.p95, MEASURED_TRIALS);
        fprintf(fStats, "B,%d,ClearHash,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", n, sClear.mean, sClear.median, sClear.stddev, sClear.minVal, sClear.maxVal, sClear.p95, MEASURED_TRIALS);
        fprintf(fStats, "B,%d,HashAgents,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", n, sHash.mean, sHash.median, sHash.stddev, sHash.minVal, sHash.maxVal, sHash.p95, MEASURED_TRIALS);
        fprintf(fStats, "B,%d,LOD,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", n, sLOD.mean, sLOD.median, sLOD.stddev, sLOD.minVal, sLOD.maxVal, sLOD.p95, MEASURED_TRIALS);
        fprintf(fStats, "B,%d,Solve,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", n, sSolve.mean, sSolve.median, sSolve.stddev, sSolve.minVal, sSolve.maxVal, sSolve.p95, MEASURED_TRIALS);
        fprintf(fStats, "B,%d,Finalize,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", n, sFin.mean, sFin.median, sFin.stddev, sFin.minVal, sFin.maxVal, sFin.p95, MEASURED_TRIALS);
        fprintf(fStats, "B,%d,D2H,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", n, sD2H.mean, sD2H.median, sD2H.stddev, sD2H.minVal, sD2H.maxVal, sD2H.p95, MEASURED_TRIALS);
        fprintf(fStats, "B,%d,Total,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", n, sTotal.mean, sTotal.median, sTotal.stddev, sTotal.minVal, sTotal.maxVal, sTotal.p95, MEASURED_TRIALS);
        fprintf(fStats, "B,%d,SumOfMeans,%.4f,,,,,,%d\n", n, sumMeans, MEASURED_TRIALS);

        printf("  N=%5d | H2D=%.3f Pred=%.3f Clear=%.3f Hash=%.3f LOD=%.3f Solve=%.3f Fin=%.3f D2H=%.3f | Total=%.3f (sum=%.3f disc=%.3f)\n",
            n, sH2D.mean, sPred.mean, sClear.mean, sHash.mean, sLOD.mean,
            sSolve.mean, sFin.mean, sD2H.mean, sTotal.mean, sumMeans, sTotal.mean - sumMeans);

        free(agents);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT C: AoS vs SoA Solver-Kernel Comparison
// ═══════════════════════════════════════════════════════════════════════════════
void ExperimentC(FILE* fTrials, FILE* fStats) {
    printf("\n==================================================================\n");
    printf("  EXPERIMENT C: AoS vs SoA Solver-Kernel Comparison\n");
    printf("==================================================================\n");

    fprintf(fTrials, "Experiment,AgentCount,Trial,AoS_SolveMs,SoA_SolveMs,Speedup\n");

    for (int ci = 0; ci < NUM_COUNTS; ci++) {
        int n = AGENT_COUNTS[ci];
        AgentGPU* agents = (AgentGPU*)malloc(sizeof(AgentGPU) * n);
        SimParams p;

        std::vector<float> aosTrials, soaTrials, speedups;

        for (int t = 0; t < WARMUP_TRIALS + MEASURED_TRIALS; t++) {
            srand(4142026);
            SetupAgentsUniform(agents, n, 0.5f);
            SetupParams(p, n);

            // Need to hash agents first for the SoA benchmark to work
            StepCUDA(agents, n, &p, 3);

            // Reset agents to initial state for the AoS vs SoA benchmark
            srand(4142026);
            SetupAgentsUniform(agents, n, 0.5f);

            SoABenchmarkResult res;
            RunAoSvsSoABenchmark(agents, n, &p, 3, &res);

            if (t >= WARMUP_TRIALS) {
                aosTrials.push_back(res.aosSolveMs);
                soaTrials.push_back(res.soaSolveMs);
                speedups.push_back(res.speedup);

                fprintf(fTrials, "C,%d,%d,%.4f,%.4f,%.4f\n",
                    n, t - WARMUP_TRIALS + 1, res.aosSolveMs, res.soaSolveMs, res.speedup);
            }
        }

        Stats sAos = ComputeStats(aosTrials);
        Stats sSoa = ComputeStats(soaTrials);
        Stats sSpd = ComputeStats(speedups);

        fprintf(fStats, "C,%d,AoS_Solve,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
            n, sAos.mean, sAos.median, sAos.stddev, sAos.minVal, sAos.maxVal, sAos.p95, MEASURED_TRIALS);
        fprintf(fStats, "C,%d,SoA_Solve,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
            n, sSoa.mean, sSoa.median, sSoa.stddev, sSoa.minVal, sSoa.maxVal, sSoa.p95, MEASURED_TRIALS);
        fprintf(fStats, "C,%d,Speedup,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
            n, sSpd.mean, sSpd.median, sSpd.stddev, sSpd.minVal, sSpd.maxVal, sSpd.p95, MEASURED_TRIALS);

        printf("  N=%5d | AoS: mean=%.3f ms med=%.3f | SoA: mean=%.3f ms med=%.3f | Speedup: mean=%.2fx med=%.2fx\n",
            n, sAos.mean, sAos.median, sSoa.mean, sSoa.median, sSpd.mean, sSpd.median);

        free(agents);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT D: PBD Convergence at Varying Iteration Counts
// ═══════════════════════════════════════════════════════════════════════════════
void ExperimentD(FILE* fTrials, FILE* fStats) {
    printf("\n==================================================================\n");
    printf("  EXPERIMENT D: PBD Convergence (2500 agents, iterations 1-16)\n");
    printf("==================================================================\n");

    int iterList[] = { 1, 2, 4, 8, 12, 16 };
    int nIters = 6;
    int n = 2500;

    fprintf(fTrials, "Experiment,AgentCount,Iterations,Trial,SolveMs,TotalMs,AvgPenetration,MaxPenetration,ActivePairs\n");

    AgentGPU* agents = (AgentGPU*)malloc(sizeof(AgentGPU) * n);
    SimParams p;

    for (int ii = 0; ii < nIters; ii++) {
        int iters = iterList[ii];

        std::vector<float> solveTrials, totalTrials, avgPenTrials, maxPenTrials;
        std::vector<float> pairTrials;

        for (int t = 0; t < WARMUP_TRIALS + MEASURED_TRIALS; t++) {
            srand(4142026);
            SetupAgentsUniform(agents, n, 0.5f);
            SetupParams(p, n);
            p.iterationCount = iters;

            StepCUDA(agents, n, &p, iters);
            CUDATimingData td;
            GetLastTimings(&td);

            ConstraintStats cs;
            MeasureConstraintViolations(agents, n, &p, &cs);

            if (t >= WARMUP_TRIALS) {
                solveTrials.push_back(td.solveMs);
                totalTrials.push_back(td.totalMs);
                avgPenTrials.push_back(cs.avgViolationBefore);
                maxPenTrials.push_back(cs.maxViolationBefore);
                pairTrials.push_back((float)cs.totalConstraints);

                fprintf(fTrials, "D,%d,%d,%d,%.4f,%.4f,%.6f,%.6f,%d\n",
                    n, iters, t - WARMUP_TRIALS + 1,
                    td.solveMs, td.totalMs,
                    cs.avgViolationBefore, cs.maxViolationBefore, cs.totalConstraints);
            }
        }

        Stats sSolve  = ComputeStats(solveTrials);
        Stats sTotal  = ComputeStats(totalTrials);
        Stats sAvgPen = ComputeStats(avgPenTrials);
        Stats sMaxPen = ComputeStats(maxPenTrials);
        Stats sPairs  = ComputeStats(pairTrials);

        fprintf(fStats, "D,%d,Iter%d_Solve,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
            n, iters, sSolve.mean, sSolve.median, sSolve.stddev, sSolve.minVal, sSolve.maxVal, sSolve.p95, MEASURED_TRIALS);
        fprintf(fStats, "D,%d,Iter%d_AvgPen,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
            n, iters, sAvgPen.mean, sAvgPen.median, sAvgPen.stddev, sAvgPen.minVal, sAvgPen.maxVal, sAvgPen.p95, MEASURED_TRIALS);
        fprintf(fStats, "D,%d,Iter%d_MaxPen,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
            n, iters, sMaxPen.mean, sMaxPen.median, sMaxPen.stddev, sMaxPen.minVal, sMaxPen.maxVal, sMaxPen.p95, MEASURED_TRIALS);
        fprintf(fStats, "D,%d,Iter%d_Pairs,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d\n",
            n, iters, sPairs.mean, sPairs.median, sPairs.stddev, sPairs.minVal, sPairs.maxVal, sPairs.p95, MEASURED_TRIALS);

        printf("  Iters=%2d | solve: mean=%.3f ms | avgPen: %.4f m | maxPen: %.4f m | pairs: %.0f\n",
            iters, sSolve.mean, sAvgPen.mean, sMaxPen.mean, sPairs.mean);
    }

    free(agents);
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT E: LOD Ablation Study
// ═══════════════════════════════════════════════════════════════════════════════
void ExperimentE(FILE* fTrials, FILE* fStats) {
    printf("\n==================================================================\n");
    printf("  EXPERIMENT E: LOD Ablation (2500 agents, no-LOD baseline + 4 modes)\n");
    printf("==================================================================\n");

    // Mode 1: Distance Only (baseline)
    // Mode 2: Distance + Visibility
    // Mode 3: Distance + Visibility + Interaction
    // Mode 0: Full 4-Factor (Distance + Visibility + Interaction + Density)
    // Mode -1: No-LOD baseline. Every agent is forced to HIGH (thresholds below 0),
    //          i.e. the cost of solving everyone at full detail.
    int modes[] = { -1, 1, 2, 3, 0 };
    const char* modeLabels[] = { "NoLOD_AllHigh", "DistOnly", "Dist+Vis", "Dist+Vis+Inter", "Full_4Factor" };
    int nModes = 5;
    int n = 2500;

    fprintf(fTrials, "Experiment,AgentCount,AblationMode,ModeLabel,Trial,SolveMs,TotalMs,LOD0_Count,LOD1_Count,LOD2_Count,AvgPenetration,MaxPenetration\n");

    AgentGPU* agents = (AgentGPU*)malloc(sizeof(AgentGPU) * n);
    SimParams p;

    for (int mi = 0; mi < nModes; mi++) {
        int mode = modes[mi];

        std::vector<float> solveTrials, totalTrials, avgPenTrials, maxPenTrials;
        std::vector<float> lod0Trials, lod1Trials, lod2Trials;

        for (int t = 0; t < WARMUP_TRIALS + MEASURED_TRIALS; t++) {
            srand(4142026);
            SetupAgentsUniform(agents, n, 0.5f);
            SetupParams(p, n);
            if (mode < 0) {
                p.lodAblationMode = 0;
                p.highImportanceThresh = -1.0f;   // importance >= 0 always -> HIGH
                p.medImportanceThresh  = -1.0f;
            } else {
                p.lodAblationMode = mode;
            }

            StepCUDA(agents, n, &p, 3);
            CUDATimingData td;
            GetLastTimings(&td);

            // Count LOD levels from agent readback
            int lod0 = 0, lod1 = 0, lod2 = 0;
            for (int i = 0; i < n; i++) {
                if (agents[i].lodLevel == 0) lod0++;
                else if (agents[i].lodLevel == 1) lod1++;
                else lod2++;
            }

            // Measure penetration
            ConstraintStats cs;
            MeasureConstraintViolations(agents, n, &p, &cs);

            if (t >= WARMUP_TRIALS) {
                solveTrials.push_back(td.solveMs);
                totalTrials.push_back(td.totalMs);
                lod0Trials.push_back((float)lod0);
                lod1Trials.push_back((float)lod1);
                lod2Trials.push_back((float)lod2);
                avgPenTrials.push_back(cs.avgViolationBefore);
                maxPenTrials.push_back(cs.maxViolationBefore);

                fprintf(fTrials, "E,%d,%d,%s,%d,%.4f,%.4f,%d,%d,%d,%.6f,%.6f\n",
                    n, mode, modeLabels[mi], t - WARMUP_TRIALS + 1,
                    td.solveMs, td.totalMs, lod0, lod1, lod2,
                    cs.avgViolationBefore, cs.maxViolationBefore);
            }
        }

        Stats sSolve  = ComputeStats(solveTrials);
        Stats sTotal  = ComputeStats(totalTrials);
        Stats sLOD0   = ComputeStats(lod0Trials);
        Stats sLOD1   = ComputeStats(lod1Trials);
        Stats sLOD2   = ComputeStats(lod2Trials);
        Stats sAvgPen = ComputeStats(avgPenTrials);
        Stats sMaxPen = ComputeStats(maxPenTrials);

        fprintf(fStats, "E,%d,%s_Solve,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
            n, modeLabels[mi], sSolve.mean, sSolve.median, sSolve.stddev, sSolve.minVal, sSolve.maxVal, sSolve.p95, MEASURED_TRIALS);
        fprintf(fStats, "E,%d,%s_Total,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
            n, modeLabels[mi], sTotal.mean, sTotal.median, sTotal.stddev, sTotal.minVal, sTotal.maxVal, sTotal.p95, MEASURED_TRIALS);
        fprintf(fStats, "E,%d,%s_LOD0,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d\n",
            n, modeLabels[mi], sLOD0.mean, sLOD0.median, sLOD0.stddev, sLOD0.minVal, sLOD0.maxVal, sLOD0.p95, MEASURED_TRIALS);
        fprintf(fStats, "E,%d,%s_LOD1,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d\n",
            n, modeLabels[mi], sLOD1.mean, sLOD1.median, sLOD1.stddev, sLOD1.minVal, sLOD1.maxVal, sLOD1.p95, MEASURED_TRIALS);
        fprintf(fStats, "E,%d,%s_LOD2,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d\n",
            n, modeLabels[mi], sLOD2.mean, sLOD2.median, sLOD2.stddev, sLOD2.minVal, sLOD2.maxVal, sLOD2.p95, MEASURED_TRIALS);
        fprintf(fStats, "E,%d,%s_AvgPen,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
            n, modeLabels[mi], sAvgPen.mean, sAvgPen.median, sAvgPen.stddev, sAvgPen.minVal, sAvgPen.maxVal, sAvgPen.p95, MEASURED_TRIALS);

        printf("  [%s] solve: mean=%.3f ms | total: mean=%.3f ms | LOD: %d/%d/%d | avgPen: %.4f m\n",
            modeLabels[mi], sSolve.mean, sTotal.mean,
            (int)sLOD0.mean, (int)sLOD1.mean, (int)sLOD2.mean, sAvgPen.mean);
    }

    free(agents);
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPERIMENT E2: LOD Ablation with a real camera (matches the Unity runtime)
//
// Experiment E has no camera (every agent "visible"), a focus point in the middle
// of the crowd and Low agents solved every step, so LOD cannot save work there.
// E2 reproduces the runtime: a CCTV-style camera at the west end of an 80 m
// platform looking east (60 deg vertical FOV, 16:9, 40 m far plane), focus point
// at the camera, and Low agents solved on every 4th step only.
// Work = agent-iteration solves actually performed by the Jacobi kernel.
// ═══════════════════════════════════════════════════════════════════════════════
static void SetupCameraParams(SimParams& p, int frameIndex) {
    const float camX = -45.0f, farDist = 40.0f;
    const float halfH = 46.8f * 3.14159265f / 180.0f;   // horizontal half-FOV for 60 deg vertical @ 16:9
    const float sh = sinf(halfH), ch = cosf(halfH);
    p.focusPoint = make_float3(camX, 0.0f, 0.0f);
    p.hasFrustumPlanes = 1;
    p.frustumPlanes[0] = make_float4( 1.0f, 0.0f, 0.0f, -(camX + 0.3f));   // near
    p.frustumPlanes[1] = make_float4(-1.0f, 0.0f, 0.0f,  camX + farDist);  // far
    p.frustumPlanes[2] = make_float4(sh, 0.0f, -ch, -sh * camX);           // right
    p.frustumPlanes[3] = make_float4(sh, 0.0f,  ch, -sh * camX);           // left
    p.frustumPlanes[4] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);              // top (open)
    p.frustumPlanes[5] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);              // bottom (open)
    p.solveLowThisFrame = (frameIndex % 4 == 0) ? 1 : 0;                   // runtime cadence
}

void ExperimentE2(FILE* fTrials, FILE* fStats) {
    printf("\n==================================================================\n");
    printf("  EXPERIMENT E2: LOD Ablation with camera (2500 agents, runtime cadence)\n");
    printf("==================================================================\n");

    int modes[] = { -1, 1, 2, 3, 0 };
    const char* modeLabels[] = { "Cam_NoLOD_AllHigh", "Cam_DistOnly", "Cam_Dist+Vis", "Cam_Dist+Vis+Inter", "Cam_Full_4Factor" };
    const int nModes = 5, n = 2500, iters = 3, medCap = (iters + 1) / 2;

    fprintf(fTrials, "Experiment,AgentCount,AblationMode,ModeLabel,Trial,SolveMs,TotalMs,LOD0_Count,LOD1_Count,LOD2_Count,AgentSolves\n");
    AgentGPU* agents = (AgentGPU*)malloc(sizeof(AgentGPU) * n);
    SimParams p;
    double baseWork = 0.0, baseSolve = 0.0;

    for (int mi = 0; mi < nModes; mi++) {
        int mode = modes[mi];
        std::vector<float> solveTrials, totalTrials, workTrials, l0, l1, l2;

        for (int t = 0; t < WARMUP_TRIALS + MEASURED_TRIALS; t++) {
            srand(4142026);
            SetupAgentsUniform(agents, n, 0.5f);
            SetupParams(p, n);
            SetupCameraParams(p, t);
            if (mode < 0) { p.lodAblationMode = 0; p.highImportanceThresh = -1.0f; p.medImportanceThresh = -1.0f; }
            else            p.lodAblationMode = mode;

            StepCUDA(agents, n, &p, iters);
            CUDATimingData td;
            GetLastTimings(&td);

            int c0 = 0, c1 = 0, c2 = 0;
            for (int i = 0; i < n; i++) {
                if (agents[i].lodLevel == 0) c0++; else if (agents[i].lodLevel == 1) c1++; else c2++;
            }
            float work = (float)(c0 * iters + c1 * medCap + (p.solveLowThisFrame ? c2 : 0));

            if (t >= WARMUP_TRIALS) {
                solveTrials.push_back(td.solveMs); totalTrials.push_back(td.totalMs); workTrials.push_back(work);
                l0.push_back((float)c0); l1.push_back((float)c1); l2.push_back((float)c2);
                fprintf(fTrials, "E2,%d,%d,%s,%d,%.4f,%.4f,%d,%d,%d,%.0f\n",
                    n, mode, modeLabels[mi], t - WARMUP_TRIALS + 1, td.solveMs, td.totalMs, c0, c1, c2, work);
            }
        }

        Stats sSolve = ComputeStats(solveTrials), sTotal = ComputeStats(totalTrials), sWork = ComputeStats(workTrials);
        Stats s0 = ComputeStats(l0), s1 = ComputeStats(l1), s2 = ComputeStats(l2);
        if (mi == 0) { baseWork = sWork.mean; baseSolve = sSolve.mean; }
        fprintf(fStats, "E2,%d,%s_Solve,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", n, modeLabels[mi], sSolve.mean, sSolve.median, sSolve.stddev, sSolve.minVal, sSolve.maxVal, sSolve.p95, MEASURED_TRIALS);
        fprintf(fStats, "E2,%d,%s_Total,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", n, modeLabels[mi], sTotal.mean, sTotal.median, sTotal.stddev, sTotal.minVal, sTotal.maxVal, sTotal.p95, MEASURED_TRIALS);
        fprintf(fStats, "E2,%d,%s_AgentSolves,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d\n", n, modeLabels[mi], sWork.mean, sWork.median, sWork.stddev, sWork.minVal, sWork.maxVal, sWork.p95, MEASURED_TRIALS);
        fprintf(fStats, "E2,%d,%s_LOD0,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d\n", n, modeLabels[mi], s0.mean, s0.median, s0.stddev, s0.minVal, s0.maxVal, s0.p95, MEASURED_TRIALS);
        fprintf(fStats, "E2,%d,%s_LOD1,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d\n", n, modeLabels[mi], s1.mean, s1.median, s1.stddev, s1.minVal, s1.maxVal, s1.p95, MEASURED_TRIALS);
        fprintf(fStats, "E2,%d,%s_LOD2,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d\n", n, modeLabels[mi], s2.mean, s2.median, s2.stddev, s2.minVal, s2.maxVal, s2.p95, MEASURED_TRIALS);

        printf("  [%-19s] solve=%.3f ms (%+.0f%%) | work=%.0f agent-solves (%+.0f%%) | LOD %d/%d/%d\n",
            modeLabels[mi], sSolve.mean, 100.0 * (sSolve.mean / baseSolve - 1.0),
            sWork.mean, 100.0 * (sWork.mean / baseWork - 1.0), (int)s0.mean, (int)s1.mean, (int)s2.mean);
    }
    free(agents);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════
int main() {
    printf("=================================================================\n");
    printf("   PHASE 3: FINAL PERFORMANCE EVALUATION & BENCHMARKING\n");
    printf("   NVIDIA GeForce RTX 2050 Mobile (sm_86) | CUDA 13.5\n");
    printf("   Seed: 4142026 | dt=0.02s | PBD=3iter | Radius=0.5m\n");
    printf("   Cell=4.0m | 8192 buckets | Warmup=%d | Trials=%d\n", WARMUP_TRIALS, MEASURED_TRIALS);
    printf("=================================================================\n");

    const int maxCap = 10000;
    if (!InitCUDA(maxCap)) {
        printf("Error: Failed to initialize CUDA device!\n");
        return 1;
    }

    GPUMemoryInfo mem;
    GetGPUMemoryInfo(&mem);
    printf("VRAM Allocated: %lld KB (Capacity: %d agents)\n", mem.totalAllocatedBytes / 1024, mem.allocatedAgentCapacity);
    printf("VRAM Free: %lld MB / %lld MB\n", mem.deviceFreeBytes / (1024*1024), mem.deviceTotalBytes / (1024*1024));
    printf("AgentGPU: %d bytes, alignment: %d bytes\n\n", mem.agentStructSize, mem.agentStructAlignment);

    // Open output files
    FILE* fTrials = fopen("Phase3_Trials.csv", "w");
    FILE* fStats  = fopen("Phase3_Statistics.csv", "w");

    if (!fTrials || !fStats) {
        printf("Error: Failed to create output CSV files!\n");
        ShutdownCUDA();
        return 1;
    }

    // Stats header
    fprintf(fStats, "Experiment,AgentCount,Metric,Mean,Median,StdDev,Min,Max,P95,Trials\n");

    // Run all experiments
    ExperimentA(fTrials, fStats);
    ExperimentB(fTrials, fStats);
    ExperimentC(fTrials, fStats);
    ExperimentD(fTrials, fStats);
    ExperimentE(fTrials, fStats);
    ExperimentE2(fTrials, fStats);

    fclose(fTrials);
    fclose(fStats);

    ShutdownCUDA();

    printf("\n=================================================================\n");
    printf("   PHASE 3 COMPLETE: All experiments (A-E, E2) finished.\n");
    printf("   Output: Phase3_Trials.csv, Phase3_Statistics.csv\n");
    printf("   Note: Experiment F (Unity runtime) requires Unity Editor.\n");
    printf("=================================================================\n");

    return 0;
}
