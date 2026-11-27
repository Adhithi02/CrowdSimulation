#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "CrowdCommon.h"

extern "C" {
    bool InitCUDA(int maxAgents);
    bool StepCUDA(AgentGPU* hostAgents, int agentCount, SimParams* params, int iterations);
    void ShutdownCUDA();
    bool GetLastTimings(CUDATimingData* out);
    bool GetDetailedHashStats(DetailedHashGridStats* out);
}

float RandomFloat(float minVal, float maxVal) {
    return minVal + ((float)rand() / (float)RAND_MAX) * (maxVal - minVal);
}

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
    p = {};
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

int main() {
    InitCUDA(10000);
    int counts[] = {500, 1000, 2500, 5000};
    for (int c = 0; c < 4; c++) {
        int n = counts[c];
        AgentGPU* agents = new AgentGPU[n];
        SimParams p;
        SetupParams(p, n);
        
        printf("\n=== Testing exact Exp A vs Exp B setup for N=%d ===\n", n);
        // Mode A: exact loop as in Exp A
        printf("--- Running 5 trials with Exp A loop (with GetDetailedHashStats) ---\n");
        for (int t = 0; t < 5; t++) {
            srand(4142026);
            SetupAgentsUniform(agents, n, 0.5f);
            StepCUDA(agents, n, &p, 3);
            CUDATimingData td;
            GetLastTimings(&td);
            DetailedHashGridStats dhs;
            GetDetailedHashStats(&dhs);
            
            int lod0 = 0, lod1 = 0, lod2 = 0;
            for (int i = 0; i < n; i++) {
                if (agents[i].lodLevel == 0) lod0++;
                else if (agents[i].lodLevel == 1) lod1++;
                else lod2++;
            }
            printf("ExpA Trial %d: total=%.3f, solve=%.3f, clear=%.3f, d2h=%.3f, lod=[%d,%d,%d]\n",
                t, td.totalMs, td.solveMs, td.clearHashMs, td.d2hMs, lod0, lod1, lod2);
        }
        
        printf("--- Running 5 trials with Exp B loop (no GetDetailedHashStats) ---\n");
        for (int t = 0; t < 5; t++) {
            srand(4142026);
            SetupAgentsUniform(agents, n, 0.5f);
            StepCUDA(agents, n, &p, 3);
            CUDATimingData td;
            GetLastTimings(&td);
            
            int lod0 = 0, lod1 = 0, lod2 = 0;
            for (int i = 0; i < n; i++) {
                if (agents[i].lodLevel == 0) lod0++;
                else if (agents[i].lodLevel == 1) lod1++;
                else lod2++;
            }
            printf("ExpB Trial %d: total=%.3f, solve=%.3f, clear=%.3f, d2h=%.3f, lod=[%d,%d,%d]\n",
                t, td.totalMs, td.solveMs, td.clearHashMs, td.d2hMs, lod0, lod1, lod2);
        }
        
        delete[] agents;
    }
    ShutdownCUDA();
    return 0;
}
