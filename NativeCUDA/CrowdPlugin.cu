#include "CrowdCommon.h"
#include <stdio.h>

// Forward declarations of kernels
extern __global__ void KernelPredict(AgentGPU* agents, SimParams p);
extern __global__ void KernelClearHashGrid(int* hashGrid, int* hashGridCount, int* hashGridOverflow, int* hashOverflowCount);
extern __global__ void KernelHashAgents(AgentGPU* agents, int* hashGrid, int* hashGridCount, int* hashGridOverflow, int* hashOverflowCount, int* outTotalOverflows, int* outTotalDropped, SimParams p);
extern __global__ void KernelAssignLOD(AgentGPU* agents, const int* hashGridCount, SimParams p);
extern __global__ void KernelSolveConstraints(AgentGPU* agents, CorrectionData* corrections, const int* hashGrid, const int* hashGridCount, const int* hashGridOverflow, const int* hashOverflowCount, SimParams p);
extern __global__ void KernelApplyCorrections(AgentGPU* agents, const CorrectionData* corrections, SimParams p);
extern __global__ void KernelFinalize(AgentGPU* agents, const CorrectionData* corrections, SimParams p);
extern __global__ void KernelSolveConstraintsSoA(
    const float* predX, const float* predY, const float* predZ,
    const float* posX, const float* posY, const float* posZ,
    const float* radius, const float* mass, const int* lodLevel,
    CorrectionData* corrections,
    const int* hashGrid, const int* hashGridCount,
    const int* hashGridOverflow, const int* hashOverflowCount,
    SimParams p);
extern __global__ void KernelBruteForceNeighbors(const AgentGPU* agents, int* outCandidates, int* outNeighbors, int agentCount, float interactionRadius);
extern __global__ void KernelMeasureViolations(const AgentGPU* agents, const int* hashGrid, const int* hashGridCount, const int* hashGridOverflow, const int* hashOverflowCount, float* outViolationSum, int* outViolationCount, float* outMaxViolation, SimParams p);
extern __global__ void KernelHashGridStats(const int* hashGridCount, int* outOccupiedCells, int* outMaxOccupancy);
extern __global__ void KernelDetailedHashStats(
    const int* hashGridCount,
    const int* hashOverflowCount,
    int* outOccupiedBuckets,
    int* outMaxPrimaryOccupancy,
    int* outMaxSecondaryOccupancy,
    int* outMaxTotalOccupancy,
    int* outBucketsUsingOverflow,
    int* outBucketsExceedingSecondary);
extern __global__ void KernelCollectSpatialHashPairs(
    const AgentGPU* agents,
    int agentCount,
    float cellSize,
    float threshold,
    const int* hashGrid,
    const int* hashGridCount,
    const int* hashGridOverflow,
    const int* hashOverflowCount,
    int* outPairA,
    int* outPairB,
    int* outPairCount,
    int maxPairs,
    long long* outEntriesTraversed,
    long long* outDistanceChecks);

// Internal device state
static AgentGPU*       d_agents = nullptr;
static CorrectionData* d_corrections = nullptr;
static int*            d_hashGrid = nullptr;
static int*            d_hashGridCount = nullptr;
static int*            d_hashGridOverflow = nullptr;
static int*            d_hashOverflowCount = nullptr;
static int*            d_totalOverflowCounter = nullptr;
static int*            d_totalDroppedCounter = nullptr;
static cudaStream_t    stream = nullptr;
static int             currentAllocatedAgents = 0;

// SoA Device Buffers for Layout Comparison
static float* d_soa_predX = nullptr;
static float* d_soa_predY = nullptr;
static float* d_soa_predZ = nullptr;
static float* d_soa_posX  = nullptr;
static float* d_soa_posY  = nullptr;
static float* d_soa_posZ  = nullptr;
static float* d_soa_radius = nullptr;
static float* d_soa_mass   = nullptr;
static int*   d_soa_lodLevel = nullptr;
static int    currentSoAAllocated = 0;

// PAGP Instrumentation: CUDA Event Timing
static cudaEvent_t ev_start, ev_h2d_done;
static cudaEvent_t ev_predict_done, ev_clearHash_done, ev_hashAgents_done;
static cudaEvent_t ev_lod_done, ev_solve_done, ev_apply_done;
static cudaEvent_t ev_finalize_done, ev_d2h_done;
static cudaEvent_t ev_total_start, ev_total_end;
static bool        timingEventsCreated = false;
static CUDATimingData lastTimings = {};

// Instrumentation scratch buffers (device)
static int*   d_bruteForce_candidates = nullptr;
static int*   d_bruteForce_neighbors  = nullptr;
static float* d_violationSum = nullptr;
static int*   d_violationCount = nullptr;
static float* d_maxViolation = nullptr;
static int*   d_occupiedCells = nullptr;
static int*   d_maxOccupancy = nullptr;
static int    instrumentationAllocated = 0;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
    } while (0)

static void CreateTimingEvents()
{
    if (timingEventsCreated) return;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_h2d_done);
    cudaEventCreate(&ev_predict_done);
    cudaEventCreate(&ev_clearHash_done);
    cudaEventCreate(&ev_hashAgents_done);
    cudaEventCreate(&ev_lod_done);
    cudaEventCreate(&ev_solve_done);
    cudaEventCreate(&ev_apply_done);
    cudaEventCreate(&ev_finalize_done);
    cudaEventCreate(&ev_d2h_done);
    cudaEventCreate(&ev_total_start);
    cudaEventCreate(&ev_total_end);
    timingEventsCreated = true;
}

static void DestroyTimingEvents()
{
    if (!timingEventsCreated) return;
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_h2d_done);
    cudaEventDestroy(ev_predict_done);
    cudaEventDestroy(ev_clearHash_done);
    cudaEventDestroy(ev_hashAgents_done);
    cudaEventDestroy(ev_lod_done);
    cudaEventDestroy(ev_solve_done);
    cudaEventDestroy(ev_apply_done);
    cudaEventDestroy(ev_finalize_done);
    cudaEventDestroy(ev_d2h_done);
    cudaEventDestroy(ev_total_start);
    cudaEventDestroy(ev_total_end);
    timingEventsCreated = false;
}

static bool EnsureInstrumentationBuffers(int agentCount)
{
    if (instrumentationAllocated >= agentCount) return true;
    if (d_bruteForce_candidates) cudaFree(d_bruteForce_candidates);
    if (d_bruteForce_neighbors)  cudaFree(d_bruteForce_neighbors);
    if (d_violationSum)          cudaFree(d_violationSum);
    if (d_violationCount)        cudaFree(d_violationCount);
    if (d_maxViolation)          cudaFree(d_maxViolation);
    if (d_occupiedCells)         cudaFree(d_occupiedCells);
    if (d_maxOccupancy)          cudaFree(d_maxOccupancy);

    CUDA_CHECK(cudaMalloc((void**)&d_bruteForce_candidates, sizeof(int) * agentCount));
    CUDA_CHECK(cudaMalloc((void**)&d_bruteForce_neighbors,  sizeof(int) * agentCount));
    CUDA_CHECK(cudaMalloc((void**)&d_violationSum,   sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_violationCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_maxViolation,   sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_occupiedCells,  sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_maxOccupancy,   sizeof(int)));

    instrumentationAllocated = agentCount;
    return true;
}

static bool EnsureSoABuffers(int maxAgents)
{
    if (currentSoAAllocated >= maxAgents) return true;
    if (d_soa_predX) cudaFree(d_soa_predX);
    if (d_soa_predY) cudaFree(d_soa_predY);
    if (d_soa_predZ) cudaFree(d_soa_predZ);
    if (d_soa_posX)  cudaFree(d_soa_posX);
    if (d_soa_posY)  cudaFree(d_soa_posY);
    if (d_soa_posZ)  cudaFree(d_soa_posZ);
    if (d_soa_radius) cudaFree(d_soa_radius);
    if (d_soa_mass)   cudaFree(d_soa_mass);
    if (d_soa_lodLevel) cudaFree(d_soa_lodLevel);

    CUDA_CHECK(cudaMalloc((void**)&d_soa_predX, sizeof(float) * maxAgents));
    CUDA_CHECK(cudaMalloc((void**)&d_soa_predY, sizeof(float) * maxAgents));
    CUDA_CHECK(cudaMalloc((void**)&d_soa_predZ, sizeof(float) * maxAgents));
    CUDA_CHECK(cudaMalloc((void**)&d_soa_posX,  sizeof(float) * maxAgents));
    CUDA_CHECK(cudaMalloc((void**)&d_soa_posY,  sizeof(float) * maxAgents));
    CUDA_CHECK(cudaMalloc((void**)&d_soa_posZ,  sizeof(float) * maxAgents));
    CUDA_CHECK(cudaMalloc((void**)&d_soa_radius, sizeof(float) * maxAgents));
    CUDA_CHECK(cudaMalloc((void**)&d_soa_mass,   sizeof(float) * maxAgents));
    CUDA_CHECK(cudaMalloc((void**)&d_soa_lodLevel, sizeof(int) * maxAgents));

    currentSoAAllocated = maxAgents;
    return true;
}

static void FreeInstrumentationBuffers()
{
    if (d_bruteForce_candidates) { cudaFree(d_bruteForce_candidates); d_bruteForce_candidates = nullptr; }
    if (d_bruteForce_neighbors)  { cudaFree(d_bruteForce_neighbors);  d_bruteForce_neighbors  = nullptr; }
    if (d_violationSum)          { cudaFree(d_violationSum);          d_violationSum          = nullptr; }
    if (d_violationCount)        { cudaFree(d_violationCount);        d_violationCount        = nullptr; }
    if (d_maxViolation)          { cudaFree(d_maxViolation);          d_maxViolation          = nullptr; }
    if (d_occupiedCells)         { cudaFree(d_occupiedCells);         d_occupiedCells         = nullptr; }
    if (d_maxOccupancy)          { cudaFree(d_maxOccupancy);          d_maxOccupancy          = nullptr; }
    instrumentationAllocated = 0;
}

static void FreeSoABuffers()
{
    if (d_soa_predX) { cudaFree(d_soa_predX); d_soa_predX = nullptr; }
    if (d_soa_predY) { cudaFree(d_soa_predY); d_soa_predY = nullptr; }
    if (d_soa_predZ) { cudaFree(d_soa_predZ); d_soa_predZ = nullptr; }
    if (d_soa_posX)  { cudaFree(d_soa_posX);  d_soa_posX  = nullptr; }
    if (d_soa_posY)  { cudaFree(d_soa_posY);  d_soa_posY  = nullptr; }
    if (d_soa_posZ)  { cudaFree(d_soa_posZ);  d_soa_posZ  = nullptr; }
    if (d_soa_radius) { cudaFree(d_soa_radius); d_soa_radius = nullptr; }
    if (d_soa_mass)   { cudaFree(d_soa_mass);   d_soa_mass   = nullptr; }
    if (d_soa_lodLevel) { cudaFree(d_soa_lodLevel); d_soa_lodLevel = nullptr; }
    currentSoAAllocated = 0;
}

extern "C" {

__declspec(dllexport) bool InitCUDA(int maxAgents)
{
    if (d_agents != nullptr) {
        cudaFree(d_agents);
        cudaFree(d_corrections);
        cudaFree(d_hashGrid);
        cudaFree(d_hashGridCount);
        if (d_hashGridOverflow) cudaFree(d_hashGridOverflow);
        if (d_hashOverflowCount) cudaFree(d_hashOverflowCount);
        if (d_totalOverflowCounter) cudaFree(d_totalOverflowCounter);
        if (d_totalDroppedCounter) cudaFree(d_totalDroppedCounter);
        if (stream) cudaStreamDestroy(stream);
        d_agents = nullptr;
    }

    currentAllocatedAgents = maxAgents;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc((void**)&d_agents, sizeof(AgentGPU) * maxAgents));
    CUDA_CHECK(cudaMalloc((void**)&d_corrections, sizeof(CorrectionData) * maxAgents));
    CUDA_CHECK(cudaMalloc((void**)&d_hashGrid, sizeof(int) * HASH_BUCKET_COUNT * MAX_PER_BUCKET));
    CUDA_CHECK(cudaMalloc((void**)&d_hashGridCount, sizeof(int) * HASH_BUCKET_COUNT));
    CUDA_CHECK(cudaMalloc((void**)&d_hashGridOverflow, sizeof(int) * HASH_BUCKET_COUNT * MAX_OVERFLOW_PER_BUCKET));
    CUDA_CHECK(cudaMalloc((void**)&d_hashOverflowCount, sizeof(int) * HASH_BUCKET_COUNT));
    CUDA_CHECK(cudaMalloc((void**)&d_totalOverflowCounter, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_totalDroppedCounter, sizeof(int)));

    CreateTimingEvents();

    return true;
}

__declspec(dllexport) bool StepCUDA(AgentGPU* hostAgents, int agentCount, SimParams* params, int iterations)
{
    if (!d_agents || agentCount <= 0 || agentCount > currentAllocatedAgents)
        return false;

    int agentThreads = THREAD_GROUP_SIZE;
    int agentBlocks  = (agentCount + agentThreads - 1) / agentThreads;
    int clearBlocks  = ((HASH_BUCKET_COUNT * MAX_OVERFLOW_PER_BUCKET) + agentThreads - 1) / agentThreads;

    cudaEventRecord(ev_total_start, stream);

    // 1. Host -> Device copy
    cudaEventRecord(ev_start, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_agents, hostAgents, sizeof(AgentGPU) * agentCount, cudaMemcpyHostToDevice, stream));
    cudaEventRecord(ev_h2d_done, stream);

    // 2. KernelPredict
    KernelPredict<<<agentBlocks, agentThreads, 0, stream>>>(d_agents, *params);
    cudaEventRecord(ev_predict_done, stream);

    // 3. Clear & Hash Agents (Density Map + Overflow Handling)
    KernelClearHashGrid<<<clearBlocks, agentThreads, 0, stream>>>(
        d_hashGrid, d_hashGridCount, d_hashGridOverflow, d_hashOverflowCount);
    cudaEventRecord(ev_clearHash_done, stream);

    // Explicitly reset overflow and dropped counters EVERY simulation step
    CUDA_CHECK(cudaMemsetAsync(d_totalOverflowCounter, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_totalDroppedCounter, 0, sizeof(int), stream));

    KernelHashAgents<<<agentBlocks, agentThreads, 0, stream>>>(
        d_agents, d_hashGrid, d_hashGridCount, d_hashGridOverflow, d_hashOverflowCount,
        d_totalOverflowCounter, d_totalDroppedCounter, *params);
    cudaEventRecord(ev_hashAgents_done, stream);

    // 4. KernelAssignLOD (Importance Metric)
    KernelAssignLOD<<<agentBlocks, agentThreads, 0, stream>>>(d_agents, d_hashGridCount, *params);
    cudaEventRecord(ev_lod_done, stream);

    // 5. Jacobi constraint iterations
    for (int iter = 0; iter < iterations; iter++)
    {
        params->currentIteration = iter;
        KernelSolveConstraints<<<agentBlocks, agentThreads, 0, stream>>>(
            d_agents, d_corrections, d_hashGrid, d_hashGridCount, d_hashGridOverflow, d_hashOverflowCount, *params);
        KernelApplyCorrections<<<agentBlocks, agentThreads, 0, stream>>>(d_agents, d_corrections, *params);
    }
    cudaEventRecord(ev_solve_done, stream);

    // 6. KernelFinalize
    KernelFinalize<<<agentBlocks, agentThreads, 0, stream>>>(d_agents, d_corrections, *params);
    cudaEventRecord(ev_finalize_done, stream);

    // 7. Device -> Host copy
    CUDA_CHECK(cudaMemcpyAsync(hostAgents, d_agents, sizeof(AgentGPU) * agentCount, cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(ev_d2h_done, stream);

    cudaEventRecord(ev_total_end, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Extract timing data ──
    cudaEventElapsedTime(&lastTimings.h2dMs,       ev_start,          ev_h2d_done);
    cudaEventElapsedTime(&lastTimings.predictMs,   ev_h2d_done,       ev_predict_done);
    cudaEventElapsedTime(&lastTimings.clearHashMs, ev_predict_done,   ev_clearHash_done);
    cudaEventElapsedTime(&lastTimings.hashAgentsMs,ev_clearHash_done, ev_hashAgents_done);
    cudaEventElapsedTime(&lastTimings.lodMs,       ev_hashAgents_done,ev_lod_done);
    cudaEventElapsedTime(&lastTimings.solveMs,     ev_lod_done,       ev_solve_done);
    cudaEventElapsedTime(&lastTimings.finalizeMs,  ev_solve_done,     ev_finalize_done);
    cudaEventElapsedTime(&lastTimings.d2hMs,       ev_finalize_done,  ev_d2h_done);
    cudaEventElapsedTime(&lastTimings.totalMs,     ev_total_start,    ev_total_end);
    lastTimings.applyMs         = 0.0f;
    lastTimings.streamSyncMs    = 0.0f;
    lastTimings.iterationCount  = iterations;

    return true;
}

__declspec(dllexport) void ShutdownCUDA()
{
    if (d_agents != nullptr) {
        cudaFree(d_agents);
        cudaFree(d_corrections);
        cudaFree(d_hashGrid);
        cudaFree(d_hashGridCount);
        if (d_hashGridOverflow) cudaFree(d_hashGridOverflow);
        if (d_hashOverflowCount) cudaFree(d_hashOverflowCount);
        if (d_totalOverflowCounter) cudaFree(d_totalOverflowCounter);
        if (d_totalDroppedCounter) cudaFree(d_totalDroppedCounter);
        if (stream) cudaStreamDestroy(stream);
        d_agents = nullptr;
        d_corrections = nullptr;
        d_hashGrid = nullptr;
        d_hashGridCount = nullptr;
        d_hashGridOverflow = nullptr;
        d_hashOverflowCount = nullptr;
        d_totalOverflowCounter = nullptr;
        d_totalDroppedCounter = nullptr;
        stream = nullptr;
        currentAllocatedAgents = 0;
    }
    FreeInstrumentationBuffers();
    FreeSoABuffers();
    DestroyTimingEvents();
}

__declspec(dllexport) bool GetLastTimings(CUDATimingData* out)
{
    if (out == nullptr) return false;
    *out = lastTimings;
    return true;
}

__declspec(dllexport) bool GetGPUMemoryInfo(GPUMemoryInfo* out)
{
    if (out == nullptr) return false;
    out->agentBufferBytes      = (long long)sizeof(AgentGPU) * currentAllocatedAgents;
    out->correctionBufferBytes = (long long)sizeof(CorrectionData) * currentAllocatedAgents;
    out->hashGridBytes         = (long long)sizeof(int) * HASH_BUCKET_COUNT * (MAX_PER_BUCKET + MAX_OVERFLOW_PER_BUCKET);
    out->hashGridCountBytes    = (long long)sizeof(int) * HASH_BUCKET_COUNT * 2;
    out->totalAllocatedBytes   = out->agentBufferBytes + out->correctionBufferBytes +
                                  out->hashGridBytes + out->hashGridCountBytes;
    out->agentStructSize       = sizeof(AgentGPU);
    out->agentStructAlignment  = __alignof(AgentGPU);
    out->allocatedAgentCapacity = currentAllocatedAgents;

    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    out->deviceFreeBytes  = (long long)freeMem;
    out->deviceTotalBytes = (long long)totalMem;

    return true;
}

__declspec(dllexport) bool RunBruteForceNeighborBenchmark(
    AgentGPU* hostAgents, int agentCount, float interactionRadius,
    long long* outTotalCandidates, long long* outTotalNeighbors)
{
    if (!d_agents || agentCount <= 0 || agentCount > currentAllocatedAgents)
        return false;
    if (!EnsureInstrumentationBuffers(agentCount))
        return false;

    int agentThreads = THREAD_GROUP_SIZE;
    int agentBlocks  = (agentCount + agentThreads - 1) / agentThreads;

    CUDA_CHECK(cudaMemcpy(d_agents, hostAgents, sizeof(AgentGPU) * agentCount, cudaMemcpyHostToDevice));

    cudaEvent_t bfStart, bfEnd;
    cudaEventCreate(&bfStart);
    cudaEventCreate(&bfEnd);

    cudaEventRecord(bfStart);
    KernelBruteForceNeighbors<<<agentBlocks, agentThreads>>>(
        d_agents, d_bruteForce_candidates, d_bruteForce_neighbors,
        agentCount, interactionRadius);
    cudaEventRecord(bfEnd);
    cudaEventSynchronize(bfEnd);

    int* h_candidates = new int[agentCount];
    int* h_neighbors  = new int[agentCount];
    cudaMemcpy(h_candidates, d_bruteForce_candidates, sizeof(int) * agentCount, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_neighbors,  d_bruteForce_neighbors,  sizeof(int) * agentCount, cudaMemcpyDeviceToHost);

    long long totalC = 0, totalN = 0;
    for (int i = 0; i < agentCount; i++) {
        totalC += h_candidates[i];
        totalN += h_neighbors[i];
    }
    *outTotalCandidates = totalC;
    *outTotalNeighbors  = totalN;

    delete[] h_candidates;
    delete[] h_neighbors;
    cudaEventDestroy(bfStart);
    cudaEventDestroy(bfEnd);

    return true;
}

__declspec(dllexport) bool GetHashGridStats(HashGridStats* out)
{
    if (!d_hashGridCount || out == nullptr) return false;
    if (!EnsureInstrumentationBuffers(1)) return false;

    int zero = 0;
    cudaMemcpy(d_occupiedCells, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxOccupancy,  &zero, sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (HASH_BUCKET_COUNT + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
    KernelHashGridStats<<<blocks, THREAD_GROUP_SIZE>>>(d_hashGridCount, d_occupiedCells, d_maxOccupancy);
    cudaDeviceSynchronize();

    cudaMemcpy(&out->occupiedCells, d_occupiedCells, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out->maxOccupancy,  d_maxOccupancy,  sizeof(int), cudaMemcpyDeviceToHost);

    if (d_totalOverflowCounter != nullptr) {
        int of = 0;
        cudaMemcpy(&of, d_totalOverflowCounter, sizeof(int), cudaMemcpyDeviceToHost);
        out->overflowCount = of;
    } else {
        out->overflowCount = 0;
    }

    out->totalInserted = 0;
    out->totalCandidates = 0;
    out->actualNeighbors = 0;
    out->distanceChecks  = 0;

    return true;
}

__declspec(dllexport) bool GetDetailedHashStats(DetailedHashGridStats* out)
{
    if (!d_hashGridCount || out == nullptr) return false;

    int* d_occ = nullptr;
    int* d_maxPrim = nullptr;
    int* d_maxSec = nullptr;
    int* d_maxTot = nullptr;
    int* d_useOf = nullptr;
    int* d_excSec = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_occ, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_maxPrim, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_maxSec, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_maxTot, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_useOf, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_excSec, sizeof(int)));

    cudaMemset(d_occ, 0, sizeof(int));
    cudaMemset(d_maxPrim, 0, sizeof(int));
    cudaMemset(d_maxSec, 0, sizeof(int));
    cudaMemset(d_maxTot, 0, sizeof(int));
    cudaMemset(d_useOf, 0, sizeof(int));
    cudaMemset(d_excSec, 0, sizeof(int));

    int blocks = (HASH_BUCKET_COUNT + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
    KernelDetailedHashStats<<<blocks, THREAD_GROUP_SIZE>>>(
        d_hashGridCount, d_hashOverflowCount,
        d_occ, d_maxPrim, d_maxSec, d_maxTot, d_useOf, d_excSec);
    cudaDeviceSynchronize();

    cudaMemcpy(&out->occupiedBuckets, d_occ, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out->maxPrimaryOccupancy, d_maxPrim, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out->maxSecondaryOccupancy, d_maxSec, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out->maxTotalOccupancy, d_maxTot, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out->bucketsUsingOverflow, d_useOf, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out->bucketsExceedingSecondary, d_excSec, sizeof(int), cudaMemcpyDeviceToHost);

    if (d_totalOverflowCounter != nullptr)
        cudaMemcpy(&out->totalOverflowInsertions, d_totalOverflowCounter, sizeof(int), cudaMemcpyDeviceToHost);
    else
        out->totalOverflowInsertions = 0;

    if (d_totalDroppedCounter != nullptr)
        cudaMemcpy(&out->totalDroppedInsertions, d_totalDroppedCounter, sizeof(int), cudaMemcpyDeviceToHost);
    else
        out->totalDroppedInsertions = 0;

    cudaFree(d_occ);
    cudaFree(d_maxPrim);
    cudaFree(d_maxSec);
    cudaFree(d_maxTot);
    cudaFree(d_useOf);
    cudaFree(d_excSec);

    return true;
}

__declspec(dllexport) bool CollectCUDACollisionPairs(
    AgentGPU* hostAgents, int agentCount, float cellSize, float threshold,
    int* outPairA, int* outPairB, int maxPairs, int* outPairCount,
    AuditedNeighborStats* outStats)
{
    if (!d_agents || agentCount <= 0 || outPairCount == nullptr) return false;

    int* d_pA = nullptr;
    int* d_pB = nullptr;
    int* d_pCount = nullptr;
    long long* d_trav = nullptr;
    long long* d_chk = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_pA, sizeof(int) * maxPairs));
    CUDA_CHECK(cudaMalloc((void**)&d_pB, sizeof(int) * maxPairs));
    CUDA_CHECK(cudaMalloc((void**)&d_pCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_trav, sizeof(long long)));
    CUDA_CHECK(cudaMalloc((void**)&d_chk, sizeof(long long)));

    cudaMemset(d_pCount, 0, sizeof(int));
    cudaMemset(d_trav, 0, sizeof(long long));
    cudaMemset(d_chk, 0, sizeof(long long));

    int agentThreads = THREAD_GROUP_SIZE;
    int agentBlocks  = (agentCount + agentThreads - 1) / agentThreads;
    int clearBlocks  = ((HASH_BUCKET_COUNT * MAX_OVERFLOW_PER_BUCKET) + agentThreads - 1) / agentThreads;

    CUDA_CHECK(cudaMemcpy(d_agents, hostAgents, sizeof(AgentGPU) * agentCount, cudaMemcpyHostToDevice));

    KernelClearHashGrid<<<clearBlocks, agentThreads>>>(d_hashGrid, d_hashGridCount, d_hashGridOverflow, d_hashOverflowCount);
    cudaMemset(d_totalOverflowCounter, 0, sizeof(int));
    cudaMemset(d_totalDroppedCounter, 0, sizeof(int));

    SimParams p = {};
    p.agentCount = agentCount;
    p.cellSize = cellSize;
    KernelHashAgents<<<agentBlocks, agentThreads>>>(
        d_agents, d_hashGrid, d_hashGridCount, d_hashGridOverflow, d_hashOverflowCount,
        d_totalOverflowCounter, d_totalDroppedCounter, p);
    cudaDeviceSynchronize();

    KernelCollectSpatialHashPairs<<<agentBlocks, agentThreads>>>(
        d_agents, agentCount, cellSize, threshold,
        d_hashGrid, d_hashGridCount, d_hashGridOverflow, d_hashOverflowCount,
        d_pA, d_pB, d_pCount, maxPairs, d_trav, d_chk);
    cudaDeviceSynchronize();

    int foundCount = 0;
    cudaMemcpy(&foundCount, d_pCount, sizeof(int), cudaMemcpyDeviceToHost);
    int copyCount = min(foundCount, maxPairs);

    if (outPairA && outPairB && copyCount > 0)
    {
        cudaMemcpy(outPairA, d_pA, sizeof(int) * copyCount, cudaMemcpyDeviceToHost);
        cudaMemcpy(outPairB, d_pB, sizeof(int) * copyCount, cudaMemcpyDeviceToHost);
    }
    *outPairCount = foundCount;

    if (outStats != nullptr)
    {
        long long trav = 0, chk = 0;
        cudaMemcpy(&trav, d_trav, sizeof(long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&chk, d_chk, sizeof(long long), cudaMemcpyDeviceToHost);

        long long theoretical = (long long)agentCount * (long long)(agentCount - 1) / 2;
        outStats->theoreticalCandidateBudget = theoretical;
        outStats->entriesTraversed = trav;
        outStats->distanceChecks = chk;
        outStats->collisionPairsDetected = foundCount;
        outStats->reductionPercentage = (theoretical > 0) ? (1.0f - (float)chk / (float)theoretical) * 100.0f : 0.0f;
    }

    cudaFree(d_pA);
    cudaFree(d_pB);
    cudaFree(d_pCount);
    cudaFree(d_trav);
    cudaFree(d_chk);

    return true;
}

__declspec(dllexport) bool MeasureConstraintViolations(
    AgentGPU* hostAgents, int agentCount, SimParams* params,
    ConstraintStats* out)
{
    if (!d_agents || agentCount <= 0 || out == nullptr) return false;
    if (!EnsureInstrumentationBuffers(agentCount)) return false;

    int agentThreads = THREAD_GROUP_SIZE;
    int agentBlocks  = (agentCount + agentThreads - 1) / agentThreads;

    cudaMemcpy(d_agents, hostAgents, sizeof(AgentGPU) * agentCount, cudaMemcpyHostToDevice);

    float zeroF = 0.0f;
    int   zeroI = 0;
    cudaMemcpy(d_violationSum,   &zeroF, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_violationCount, &zeroI, sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxViolation,   &zeroF, sizeof(float), cudaMemcpyHostToDevice);

    KernelMeasureViolations<<<agentBlocks, agentThreads>>>(
        d_agents, d_hashGrid, d_hashGridCount, d_hashGridOverflow, d_hashOverflowCount,
        d_violationSum, d_violationCount, d_maxViolation, *params);
    cudaDeviceSynchronize();

    float violSum = 0.0f;
    int   violCount = 0;
    float violMax = 0.0f;
    cudaMemcpy(&violSum,   d_violationSum,   sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&violCount, d_violationCount, sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(&violMax,   d_maxViolation,   sizeof(float), cudaMemcpyDeviceToHost);

    out->avgViolationBefore = (violCount > 0) ? violSum / violCount : 0.0f;
    out->maxViolationBefore = violMax;
    out->avgViolationAfter  = 0.0f;
    out->maxViolationAfter  = 0.0f;
    out->avgCorrectionMag   = 0.0f;
    out->maxCorrectionMag   = 0.0f;
    out->totalConstraints   = violCount;
    out->activeAgents       = agentCount;

    return true;
}

__declspec(dllexport) bool RunAoSvsSoABenchmark(
    AgentGPU* hostAgents, int agentCount, SimParams* params, int iterations,
    SoABenchmarkResult* out)
{
    if (!d_agents || agentCount <= 0 || out == nullptr) return false;
    if (!EnsureSoABuffers(agentCount)) return false;

    int agentThreads = THREAD_GROUP_SIZE;
    int agentBlocks  = (agentCount + agentThreads - 1) / agentThreads;
    int clearBlocks  = ((HASH_BUCKET_COUNT * MAX_OVERFLOW_PER_BUCKET) + agentThreads - 1) / agentThreads;

    // 1. Prepare SoA host data
    float* h_predX = new float[agentCount];
    float* h_predY = new float[agentCount];
    float* h_predZ = new float[agentCount];
    float* h_posX  = new float[agentCount];
    float* h_posY  = new float[agentCount];
    float* h_posZ  = new float[agentCount];
    float* h_rad   = new float[agentCount];
    float* h_mass  = new float[agentCount];
    int*   h_lod   = new int[agentCount];

    for (int i = 0; i < agentCount; i++) {
        h_predX[i] = hostAgents[i].predictedPosition.x;
        h_predY[i] = hostAgents[i].predictedPosition.y;
        h_predZ[i] = hostAgents[i].predictedPosition.z;
        h_posX[i]  = hostAgents[i].position.x;
        h_posY[i]  = hostAgents[i].position.y;
        h_posZ[i]  = hostAgents[i].position.z;
        h_rad[i]   = hostAgents[i].radius;
        h_mass[i]  = hostAgents[i].mass;
        h_lod[i]   = hostAgents[i].lodLevel;
    }

    CUDA_CHECK(cudaMemcpy(d_soa_predX, h_predX, sizeof(float) * agentCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_soa_predY, h_predY, sizeof(float) * agentCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_soa_predZ, h_predZ, sizeof(float) * agentCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_soa_posX,  h_posX,  sizeof(float) * agentCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_soa_posY,  h_posY,  sizeof(float) * agentCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_soa_posZ,  h_posZ,  sizeof(float) * agentCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_soa_radius, h_rad,  sizeof(float) * agentCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_soa_mass,   h_mass, sizeof(float) * agentCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_soa_lodLevel, h_lod, sizeof(int) * agentCount, cudaMemcpyHostToDevice));

    delete[] h_predX; delete[] h_predY; delete[] h_predZ;
    delete[] h_posX;  delete[] h_posY;  delete[] h_posZ;
    delete[] h_rad;   delete[] h_mass;  delete[] h_lod;

    // Hash grid preparation
    CUDA_CHECK(cudaMemcpy(d_agents, hostAgents, sizeof(AgentGPU) * agentCount, cudaMemcpyHostToDevice));
    KernelClearHashGrid<<<clearBlocks, agentThreads>>>(d_hashGrid, d_hashGridCount, d_hashGridOverflow, d_hashOverflowCount);
    cudaMemset(d_totalOverflowCounter, 0, sizeof(int));
    cudaMemset(d_totalDroppedCounter, 0, sizeof(int));
    KernelHashAgents<<<agentBlocks, agentThreads>>>(d_agents, d_hashGrid, d_hashGridCount, d_hashGridOverflow, d_hashOverflowCount, d_totalOverflowCounter, d_totalDroppedCounter, *params);
    cudaDeviceSynchronize();

    cudaEvent_t evAosStart, evAosEnd, evSoaStart, evSoaEnd;
    cudaEventCreate(&evAosStart);
    cudaEventCreate(&evAosEnd);
    cudaEventCreate(&evSoaStart);
    cudaEventCreate(&evSoaEnd);

    // Run AoS
    cudaEventRecord(evAosStart);
    for (int it = 0; it < iterations; it++) {
        params->currentIteration = it;
        KernelSolveConstraints<<<agentBlocks, agentThreads>>>(
            d_agents, d_corrections, d_hashGrid, d_hashGridCount, d_hashGridOverflow, d_hashOverflowCount, *params);
    }
    cudaEventRecord(evAosEnd);

    // Run SoA
    cudaEventRecord(evSoaStart);
    for (int it = 0; it < iterations; it++) {
        params->currentIteration = it;
        KernelSolveConstraintsSoA<<<agentBlocks, agentThreads>>>(
            d_soa_predX, d_soa_predY, d_soa_predZ,
            d_soa_posX, d_soa_posY, d_soa_posZ,
            d_soa_radius, d_soa_mass, d_soa_lodLevel,
            d_corrections, d_hashGrid, d_hashGridCount,
            d_hashGridOverflow, d_hashOverflowCount, *params);
    }
    cudaEventRecord(evSoaEnd);
    cudaEventSynchronize(evSoaEnd);

    float aosTime = 0.0f, soaTime = 0.0f;
    cudaEventElapsedTime(&aosTime, evAosStart, evAosEnd);
    cudaEventElapsedTime(&soaTime, evSoaStart, evSoaEnd);

    out->aosSolveMs = aosTime;
    out->soaSolveMs = soaTime;
    out->speedup    = (soaTime > 1e-4f) ? (aosTime / soaTime) : 1.0f;

    cudaEventDestroy(evAosStart);
    cudaEventDestroy(evAosEnd);
    cudaEventDestroy(evSoaStart);
    cudaEventDestroy(evSoaEnd);

    return true;
}

} // extern "C"
