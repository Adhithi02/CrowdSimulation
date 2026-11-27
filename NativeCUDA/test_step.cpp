#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "CrowdCommon.h"

extern "C" {
    bool InitCUDA(int maxAgents);
    bool StepCUDA(AgentGPU* hostAgents, int agentCount, SimParams* params, int iterations);
    void ShutdownCUDA();
    bool GetLastTimings(CUDATimingData* out);
}

void SetupAgentsUniform(AgentGPU* agents, int count, float radius) {
    for (int i = 0; i < count; i++) {
        agents[i].position = make_float3((float)(rand() % 80 - 40), 0.0f, (float)(rand() % 30 - 15));
        agents[i].predictedPosition = agents[i].position;
        agents[i].velocity = make_float3(0, 0, 0);
        agents[i].preferredVelocity = make_float3(1, 0, 0);
        agents[i].radius = radius;
        agents[i].mass = 1.0f;
        agents[i].ksi = 0.05f;
        agents[i].lodLevel = (i % 3);
        agents[i].importanceScore = 0.5f;
        agents[i].lastContactCount = 0;
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
        
        printf("=== Testing N=%d ===\n", n);
        for (int t = 0; t < 5; t++) {
            srand(4142026);
            SetupAgentsUniform(agents, n, 0.5f);
            StepCUDA(agents, n, &p, 3);
            CUDATimingData td;
            GetLastTimings(&td);
            printf("Trial %d: total=%.3f, solve=%.3f, clear=%.3f, d2h=%.3f, h2d=%.3f, pred=%.3f, lod=%.3f, fin=%.3f\n",
                t, td.totalMs, td.solveMs, td.clearHashMs, td.d2hMs, td.h2dMs, td.predictMs, td.lodMs, td.finalizeMs);
        }
        delete[] agents;
    }
    ShutdownCUDA();
    return 0;
}
