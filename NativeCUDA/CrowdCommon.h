#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define THREAD_GROUP_SIZE       64
#define HASH_BUCKET_COUNT       8192    // power of 2
#define HASH_BUCKET_MASK        8191    // HASH_BUCKET_COUNT - 1
#define MAX_PER_BUCKET          16      // fast-path primary bucket storage
#define MAX_OVERFLOW_PER_BUCKET 64      // secondary overflow storage per bucket

// Byte-for-byte identical to C# AgentGPU (80 bytes total, 8-byte aligned)
struct AgentGPU
{
    float3 position;            // 12 bytes [0..12]
    float  ksi;                 //  4 bytes [12..16]
    float3 predictedPosition;   // 12 bytes [16..28]
    float  radius;              //  4 bytes [28..32]
    float3 velocity;            // 12 bytes [32..44]
    float  mass;                //  4 bytes [44..48]
    float3 preferredVelocity;   // 12 bytes [48..60]
    int    lodLevel;            //  4 bytes [60..64] (0=HIGH, 1=MED, 2=LOW)
    float  importanceScore;     //  4 bytes [64..68]
    int    lastContactCount;    //  4 bytes [68..72]
    float2 _padding;            //  8 bytes [72..80]
};

// Structure-of-Arrays (SoA) layout for comparative memory layout benchmarking
struct AgentSoA
{
    float* posX;
    float* posY;
    float* posZ;
    float* predX;
    float* predY;
    float* predZ;
    float* velX;
    float* velY;
    float* velZ;
    float* prefVelX;
    float* prefVelY;
    float* prefVelZ;
    float* radius;
    float* mass;
    float* ksi;
    int*   lodLevel;
    float* importanceScore;
    int*   lastContactCount;
};

struct CorrectionData
{
    float3 delta;   // 12 bytes
    int    count;   //  4 bytes -> 16
};

struct SimParams
{
    int   agentCount;
    float deltaTime;
    float cellSize;

    // Density thresholds
    int   highDensityCutoff;
    int   medDensityCutoff;

    // Importance parameters
    float3 focusPoint;
    float4 frustumPlanes[6];
    int    hasFrustumPlanes;
    float  weightDistance;
    float  weightVisibility;
    float  weightInteraction;
    float  weightDensity;
    float  highImportanceThresh;
    float  medImportanceThresh;
    float  maxImportanceDist;
    int    maxInteractionContacts;
    int    currentIteration;
    int    iterationCount;
    int    solveLowThisFrame;

    // Collision physics
    float kShortRange;
    float kLongRange;
    float muStatic;
    float muKinematic;
    float avgCoefficient;
    float lrcLookAhead;

    // Ablation configuration (0=all, 1=dist, 2=dist+vis, 3=dist+vis+inter, 4=all4)
    int   lodAblationMode;
};

// ═══════════════════════════════════════════════════════════════════════════
// PAGP Instrumentation Structs — exported via DLL for C# P/Invoke
// ═══════════════════════════════════════════════════════════════════════════

// Per-step CUDA event timing (ms)
struct CUDATimingData
{
    float h2dMs;              // Host → Device memcpy
    float predictMs;          // KernelPredict
    float clearHashMs;        // KernelClearHashGrid
    float hashAgentsMs;       // KernelHashAgents
    float lodMs;              // KernelAssignLOD
    float solveMs;            // All solver iterations combined
    float applyMs;            // All apply iterations combined
    float finalizeMs;         // KernelFinalize
    float d2hMs;              // Device → Host memcpy
    float totalMs;            // Full StepCUDA wall-clock
    float streamSyncMs;       // cudaStreamSynchronize
    int   iterationCount;     // Number of Jacobi iterations this step
};

// Spatial hash grid occupancy statistics
struct HashGridStats
{
    int occupiedCells;        // Cells with >= 1 agent
    int maxOccupancy;         // Max agents in any single cell
    int totalInserted;        // Total agent insertions
    int overflowCount;        // Insertions that entered secondary overflow storage
    long long totalCandidates;// Total neighbor candidates examined
    long long actualNeighbors;// Neighbors within interaction radius
    long long distanceChecks; // Total distance calculations performed
};

// Detailed Phase 2.1 spatial hash occupancy and overflow telemetry
struct DetailedHashGridStats
{
    int occupiedBuckets;           // Buckets containing >= 1 agent
    int maxPrimaryOccupancy;       // Max agents in primary tier of any bucket (<= 16)
    int maxSecondaryOccupancy;     // Max agents in secondary tier of any bucket (<= 64)
    int maxTotalOccupancy;         // Max total agents mapped to any bucket
    int bucketsUsingOverflow;      // Number of buckets where count > MAX_PER_BUCKET (16)
    int bucketsExceedingSecondary; // Number of buckets exceeding MAX_OVERFLOW_PER_BUCKET (64)
    int totalOverflowInsertions;   // Insertions entering secondary tier this step
    int totalDroppedInsertions;    // Insertions dropped due to exceeding nominal 80 capacity
};

// Audited candidate-reduction metrics
struct AuditedNeighborStats
{
    long long theoreticalCandidateBudget; // N * (N - 1) / 2
    long long entriesTraversed;           // Total candidate entries pulled from grid
    long long distanceChecks;             // Total distance calculations performed (j > i)
    long long collisionPairsDetected;     // Total pairs with distance < threshold
    float reductionPercentage;            // (1 - distanceChecks / theoreticalBudget) * 100
};

// PBD constraint violation metrics
struct ConstraintStats
{
    float avgViolationBefore; // Average penetration depth before solving
    float maxViolationBefore; // Max penetration depth before solving
    float avgViolationAfter;  // Average penetration depth after solving
    float maxViolationAfter;  // Max penetration depth after solving
    float avgCorrectionMag;   // Average correction vector magnitude
    float maxCorrectionMag;   // Max correction vector magnitude
    int   totalConstraints;   // Total pairwise constraints detected
    int   activeAgents;       // Agents that received corrections
};

// GPU memory allocation summary
struct GPUMemoryInfo
{
    long long agentBufferBytes;       // d_agents
    long long correctionBufferBytes;  // d_corrections
    long long hashGridBytes;          // d_hashGrid + d_hashGridOverflow
    long long hashGridCountBytes;     // d_hashGridCount + d_hashOverflowCount
    long long totalAllocatedBytes;    // Sum of all device allocations
    long long deviceFreeBytes;        // cudaMemGetInfo free
    long long deviceTotalBytes;       // cudaMemGetInfo total
    int   agentStructSize;            // sizeof(AgentGPU)
    int   agentStructAlignment;       // __alignof(AgentGPU)
    int   allocatedAgentCapacity;     // Max agents allocated for
};

// SoA benchmark timing result
struct SoABenchmarkResult
{
    float aosSolveMs;
    float soaSolveMs;
    float speedup;
};

// Math operators for CUDA vector types
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline float3 operator/(const float3& a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}
__host__ __device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z; return a;
}
__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float length(const float3& a) {
    return sqrtf(dot(a, a));
}
__host__ __device__ inline float clampVal(float v, float minVal, float maxVal) {
    return fminf(fmaxf(v, minVal), maxVal);
}
__host__ __device__ inline float saturate(float v) {
    return clampVal(v, 0.0f, 1.0f);
}
__host__ __device__ inline float3 lerp(const float3& a, const float3& b, float t) {
    return a + (b - a) * t;
}
