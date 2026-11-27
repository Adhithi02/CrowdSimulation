#include "CrowdCommon.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
__device__ inline int Hash(int cx, int cz)
{
    return ((cx * 73856093) ^ (cz * 19349663)) & HASH_BUCKET_MASK;
}

__device__ inline int2 WorldToCell(const float3& p, float cellSize)
{
    int2 cell;
    cell.x = (int)floorf(p.x / cellSize);
    cell.y = (int)floorf(p.z / cellSize);
    return cell;
}

__device__ float3 ShortRangeCorrection(const AgentGPU& a, const AgentGPU& b, float kShortRange, float muStatic, float muKinematic, int& count)
{
    float3 n = a.predictedPosition - b.predictedPosition;
    n.y = 0.0f;
    float d = length(n);
    float margin = fmaxf(a.radius, 0.25f) + fmaxf(b.radius, 0.25f);

    float f = d - margin;
    if (f >= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

    n = (d > 1e-6f) ? (n / d) : make_float3(1.0f, 0.0f, 0.0f);

    float wA = 1.0f / fmaxf(a.mass, 1e-4f);
    float wB = 1.0f / fmaxf(b.mass, 1e-4f);
    float w  = wA / (wA + wB);

    float3 dx = -w * kShortRange * f * n;

    // Tangential friction
    float3 xi    = a.predictedPosition + dx;
    float3 xj    = b.predictedPosition - dx;
    float3 d_rel = (xi - a.position) - (xj - b.position);
    d_rel.y      = 0.0f;
    float3 d_tan = d_rel - (dot(d_rel, n) * n);
    float  d_len = length(d_tan);

    if (d_len >= muStatic * d)
        d_tan = fminf(muKinematic * d / fmaxf(d_len, 1e-6f), 1.0f) * d_tan;

    count++;
    return dx + (w * d_tan);
}

__device__ float3 LongRangeCorrection(const AgentGPU& a, const AgentGPU& b, float dt, float lrcLookAhead, float kLongRange, int& count)
{
    float r = fmaxf(a.radius, 0.3f) + fmaxf(b.radius, 0.3f);
    float r_sq = r * r;

    float3 x_ij = a.position - b.position;
    x_ij.y = 0.0f;
    float dist = length(x_ij);

    if (dist < r) r_sq = (r - dist) * (r - dist);

    float3 v_ij = ((a.predictedPosition - a.position) - (b.predictedPosition - b.position)) / fmaxf(dt, 1e-6f);
    v_ij.y = 0.0f;

    float aq = dot(v_ij, v_ij);
    float bq = -dot(x_ij, v_ij);
    float cq = dot(x_ij, x_ij) - r_sq;
    float discr = bq * bq - aq * cq;

    if (discr < 0.0f || fabsf(aq) < 1e-6f) return make_float3(0.0f, 0.0f, 0.0f);

    discr = sqrtf(discr);
    float t = (bq - discr) / aq;
    float t0 = fmaxf(lrcLookAhead, 0.1f);

    if (t < 1e-6f || t > t0) return make_float3(0.0f, 0.0f, 0.0f);

    float t_free = dt * floorf(t / dt);
    float t_hit  = dt + t_free;

    float3 xi_free = a.position + t_free * a.velocity;
    float3 xi_hit  = a.position + t_hit  * a.velocity;
    float3 xj_free = b.position + t_free * b.velocity;
    float3 xj_hit  = b.position + t_hit  * b.velocity;

    float3 n_hit = xi_hit - xj_hit;
    n_hit.y = 0.0f;
    float d_hit = length(n_hit);
    if (d_hit < 1e-6f) return make_float3(0.0f, 0.0f, 0.0f);
    n_hit = n_hit / d_hit;

    float wA = 1.0f / fmaxf(a.mass, 1e-4f);
    float wB = 1.0f / fmaxf(b.mass, 1e-4f);
    float w  = wA / (wA + wB);
    float k  = kLongRange * (dt / fmaxf(t, dt));

    float3 dx        = -w * (d_hit - r) * n_hit;
    float3 d_vec     = (xi_hit - xi_free) - (xj_hit - xj_free);
    d_vec.y          = 0.0f;
    float3 d_tangent = d_vec - (dot(d_vec, n_hit) * n_hit);
    dx               = dx + (w * d_tangent);

    count++;
    return k * dx;
}

// ---------------------------------------------------------------------------
// 1. KernelPredict: Velocity blend and candidate position prediction
// ---------------------------------------------------------------------------
__global__ void KernelPredict(AgentGPU* agents, SimParams p)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= p.agentCount) return;

    AgentGPU a = agents[index];
    a.velocity            = lerp(a.velocity, a.preferredVelocity, a.ksi);
    a.predictedPosition   = a.position + a.velocity * p.deltaTime;
    a.predictedPosition.y = a.position.y; // keep grounded
    agents[index]         = a;
}

// ---------------------------------------------------------------------------
// 2. KernelClearHashGrid: Zero count, primary, and secondary overflow slots
// ---------------------------------------------------------------------------
__global__ void KernelClearHashGrid(
    int* hashGrid,
    int* hashGridCount,
    int* hashGridOverflow,
    int* hashOverflowCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HASH_BUCKET_COUNT)
    {
        hashGridCount[i] = 0;
        if (hashOverflowCount != nullptr)
            hashOverflowCount[i] = 0;
    }
    if (i < (HASH_BUCKET_COUNT * MAX_PER_BUCKET))
        hashGrid[i] = -1;
    if (hashGridOverflow != nullptr && i < (HASH_BUCKET_COUNT * MAX_OVERFLOW_PER_BUCKET))
        hashGridOverflow[i] = -1;
}

// ---------------------------------------------------------------------------
// 3. KernelHashAgents: Two-tier atomic insertion preserving all agents
// ---------------------------------------------------------------------------
__global__ void KernelHashAgents(
    AgentGPU* agents,
    int* hashGrid,
    int* hashGridCount,
    int* hashGridOverflow,
    int* hashOverflowCount,
    int* outTotalOverflows,
    int* outTotalDropped,
    SimParams p)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= p.agentCount) return;

    AgentGPU a = agents[index];
    int2 cell  = WorldToCell(a.predictedPosition, p.cellSize);
    int  h     = Hash(cell.x, cell.y);

    int slot = atomicAdd(&hashGridCount[h], 1);
    if (slot < MAX_PER_BUCKET)
    {
        // Fast-path: primary bucket slot
        hashGrid[h * MAX_PER_BUCKET + slot] = index;
    }
    else
    {
        // Secondary overflow path: ensures NO agent silently disappears
        if (outTotalOverflows != nullptr)
            atomicAdd(outTotalOverflows, 1);

        if (hashGridOverflow != nullptr && hashOverflowCount != nullptr)
        {
            int overSlot = atomicAdd(&hashOverflowCount[h], 1);
            if (overSlot < MAX_OVERFLOW_PER_BUCKET)
            {
                hashGridOverflow[h * MAX_OVERFLOW_PER_BUCKET + overSlot] = index;
            }
            else
            {
                // Explicitly report dropped insertions if bucket exceeds 16 + 64 = 80!
                if (outTotalDropped != nullptr)
                    atomicAdd(outTotalDropped, 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. KernelAssignLOD: 4-factor importance calculation with Ablation support
// ---------------------------------------------------------------------------
__global__ void KernelAssignLOD(AgentGPU* agents, const int* hashGridCount, SimParams p)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= p.agentCount) return;

    AgentGPU a = agents[index];

    // 1. Distance score
    float dist = length(a.position - p.focusPoint);
    float distScore = saturate(1.0f - dist / fmaxf(p.maxImportanceDist, 1.0f));

    // 2. Visibility score (6 camera frustum planes)
    float visScore = 1.0f;
    if (p.hasFrustumPlanes == 1 && (p.lodAblationMode == 0 || p.lodAblationMode >= 2))
    {
        for (int i = 0; i < 6; i++)
        {
            float4 plane = p.frustumPlanes[i];
            float dotP = plane.x * a.position.x + plane.y * a.position.y + plane.z * a.position.z + plane.w;
            if (dotP < 0.0f)
            {
                visScore = 0.0f;
                break;
            }
        }
    }

    // 3. Interaction score
    float interScore = 0.0f;
    if (p.lodAblationMode == 0 || p.lodAblationMode >= 3)
    {
        interScore = saturate((float)a.lastContactCount / fmaxf((float)p.maxInteractionContacts, 1.0f));
    }

    // 4. Local crowd density score
    float densityScore = 0.0f;
    if (p.lodAblationMode == 0 || p.lodAblationMode >= 4)
    {
        int2 cell = WorldToCell(a.predictedPosition, p.cellSize);
        int  h    = Hash(cell.x, cell.y);
        int  dens = hashGridCount[h];
        densityScore = saturate((float)dens / fmaxf((float)p.highDensityCutoff, 1.0f));
    }

    // Weight combination according to ablation mode:
    float wDist  = p.weightDistance;
    float wVis   = (p.lodAblationMode == 1) ? 0.0f : p.weightVisibility;
    float wInter = (p.lodAblationMode == 1 || p.lodAblationMode == 2) ? 0.0f : p.weightInteraction;
    float wDens  = (p.lodAblationMode != 0 && p.lodAblationMode < 4) ? 0.0f : p.weightDensity;

    float totalWeight = fmaxf(wDist + wVis + wInter + wDens, 1e-5f);
    float importance = saturate((wDist * distScore +
                                wVis * visScore +
                                wInter * interScore +
                                wDens * densityScore) / totalWeight);

    a.importanceScore = importance;
    if      (importance >= p.highImportanceThresh) a.lodLevel = 0; // HIGH
    else if (importance >= p.medImportanceThresh)  a.lodLevel = 1; // MED
    else                                           a.lodLevel = 2; // LOW

    agents[index] = a;
}

// ---------------------------------------------------------------------------
// 5. KernelSolveConstraints: Jacobi PBD collision resolution checking primary + overflow
// ---------------------------------------------------------------------------
__global__ void KernelSolveConstraints(
    AgentGPU* agents,
    CorrectionData* corrections,
    const int* hashGrid,
    const int* hashGridCount,
    const int* hashGridOverflow,
    const int* hashOverflowCount,
    SimParams p)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= p.agentCount) return;

    AgentGPU a = agents[index];
    int lod = a.lodLevel;

    // Computational effort budgeting
    int medCap = fmaxf(1, (p.iterationCount + 1) / 2);
    if (lod == 1 && p.currentIteration >= medCap) return;
    if (lod == 2 && (p.currentIteration > 0 || p.solveLowThisFrame == 0)) return;

    float3 totalDelta = make_float3(0.0f, 0.0f, 0.0f);
    int count = 0;

    int2 center = WorldToCell(a.predictedPosition, p.cellSize);
    int searchRad = (lod == 0) ? 1 : 0;
    int maxContact = (lod == 0) ? 16 : (lod == 1) ? 6 : 2;

    int visitedBuckets[9];
    int visitedCount = 0;

    for (int i = -searchRad; i <= searchRad; i++)
    {
        for (int j = -searchRad; j <= searchRad; j++)
        {
            if (count >= maxContact) break;

            int h = Hash(center.x + i, center.y + j);

            // Deduplicate visited buckets in case of hash collisions in the 3x3 window
            bool dup = false;
            for (int v = 0; v < visitedCount; v++) {
                if (visitedBuckets[v] == h) { dup = true; break; }
            }
            if (dup) continue;
            visitedBuckets[visitedCount++] = h;

            int primaryCount = min(hashGridCount[h], MAX_PER_BUCKET);

            // Tier 1: Check primary fast-path bucket entries
            for (int s = 0; s < primaryCount; s++)
            {
                if (count >= maxContact) break;

                int nb = hashGrid[h * MAX_PER_BUCKET + s];
                if (nb < 0 || nb == index) continue;

                AgentGPU b = agents[nb];
                totalDelta += ShortRangeCorrection(a, b, p.kShortRange, p.muStatic, p.muKinematic, count);

                if (lod == 0 && p.lrcLookAhead > 0.0f)
                    totalDelta += LongRangeCorrection(a, b, p.deltaTime, p.lrcLookAhead, p.kLongRange, count);
            }

            // Tier 2: Check secondary overflow bucket entries if cell overflowed
            if (hashGridOverflow != nullptr && hashOverflowCount != nullptr && count < maxContact)
            {
                int overCount = min(hashOverflowCount[h], MAX_OVERFLOW_PER_BUCKET);
                for (int s = 0; s < overCount; s++)
                {
                    if (count >= maxContact) break;

                    int nb = hashGridOverflow[h * MAX_OVERFLOW_PER_BUCKET + s];
                    if (nb < 0 || nb == index) continue;

                    AgentGPU b = agents[nb];
                    totalDelta += ShortRangeCorrection(a, b, p.kShortRange, p.muStatic, p.muKinematic, count);

                    if (lod == 0 && p.lrcLookAhead > 0.0f)
                        totalDelta += LongRangeCorrection(a, b, p.deltaTime, p.lrcLookAhead, p.kLongRange, count);
                }
            }
        }
    }

    CorrectionData cd;
    cd.delta = totalDelta;
    cd.count = count;
    corrections[index] = cd;
}

// ---------------------------------------------------------------------------
// 6. KernelApplyCorrections: Averaged Jacobi displacement applied
// ---------------------------------------------------------------------------
__global__ void KernelApplyCorrections(AgentGPU* agents, const CorrectionData* corrections, SimParams p)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= p.agentCount) return;

    AgentGPU a = agents[index];
    int lod = a.lodLevel;
    int medCap = fmaxf(1, (p.iterationCount + 1) / 2);
    if (lod == 1 && p.currentIteration >= medCap) return;
    if (lod == 2 && (p.currentIteration > 0 || p.solveLowThisFrame == 0)) return;

    CorrectionData cd = corrections[index];
    if (cd.count <= 0) return;

    a.predictedPosition += (p.avgCoefficient * cd.delta) / fmaxf((float)cd.count, 1.0f);
    a.predictedPosition.y = a.position.y;
    agents[index] = a;
}

// ---------------------------------------------------------------------------
// 7. KernelFinalize: Commit velocity and position
// ---------------------------------------------------------------------------
__global__ void KernelFinalize(AgentGPU* agents, const CorrectionData* corrections, SimParams p)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= p.agentCount) return;

    AgentGPU a = agents[index];
    a.velocity = (a.predictedPosition - a.position) / fmaxf(p.deltaTime, 1e-5f);
    a.position = a.predictedPosition;
    a.lastContactCount = corrections[index].count;
    agents[index] = a;
}

// ===========================================================================
// Structure-of-Arrays (SoA) Comparative Kernels
// ===========================================================================

__global__ void KernelSolveConstraintsSoA(
    const float* predX, const float* predY, const float* predZ,
    const float* posX, const float* posY, const float* posZ,
    const float* radius, const float* mass, const int* lodLevel,
    CorrectionData* corrections,
    const int* hashGrid, const int* hashGridCount,
    const int* hashGridOverflow, const int* hashOverflowCount,
    SimParams p)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= p.agentCount) return;

    int lod = lodLevel[index];
    int medCap = fmaxf(1, (p.iterationCount + 1) / 2);
    if (lod == 1 && p.currentIteration >= medCap) return;
    if (lod == 2 && (p.currentIteration > 0 || p.solveLowThisFrame == 0)) return;

    float3 aPred = make_float3(predX[index], predY[index], predZ[index]);
    float  aRad  = radius[index];
    float  aMass = mass[index];

    float3 totalDelta = make_float3(0.0f, 0.0f, 0.0f);
    int count = 0;

    int2 center = WorldToCell(aPred, p.cellSize);
    int searchRad = (lod == 0) ? 1 : 0;
    int maxContact = (lod == 0) ? 16 : (lod == 1) ? 6 : 2;

    int visitedBuckets[9];
    int visitedCount = 0;

    for (int i = -searchRad; i <= searchRad; i++)
    {
        for (int j = -searchRad; j <= searchRad; j++)
        {
            if (count >= maxContact) break;

            int h = Hash(center.x + i, center.y + j);

            // Deduplicate visited buckets in case of hash collisions in the 3x3 window
            bool dup = false;
            for (int v = 0; v < visitedCount; v++) {
                if (visitedBuckets[v] == h) { dup = true; break; }
            }
            if (dup) continue;
            visitedBuckets[visitedCount++] = h;

            int primaryCount = min(hashGridCount[h], MAX_PER_BUCKET);

            for (int s = 0; s < primaryCount; s++)
            {
                if (count >= maxContact) break;

                int nb = hashGrid[h * MAX_PER_BUCKET + s];
                if (nb < 0 || nb == index) continue;

                float3 bPred = make_float3(predX[nb], predY[nb], predZ[nb]);
                float  bRad  = radius[nb];
                float  bMass = mass[nb];

                float3 n = aPred - bPred;
                n.y = 0.0f;
                float d = length(n);
                float margin = fmaxf(aRad, 0.25f) + fmaxf(bRad, 0.25f);
                float f = d - margin;

                if (f < 0.0f)
                {
                    n = (d > 1e-6f) ? (n / d) : make_float3(1.0f, 0.0f, 0.0f);
                    float wA = 1.0f / fmaxf(aMass, 1e-4f);
                    float wB = 1.0f / fmaxf(bMass, 1e-4f);
                    float w  = wA / (wA + wB);
                    totalDelta += -w * p.kShortRange * f * n;
                    count++;
                }
            }

            if (hashGridOverflow != nullptr && hashOverflowCount != nullptr && count < maxContact)
            {
                int overCount = min(hashOverflowCount[h], MAX_OVERFLOW_PER_BUCKET);
                for (int s = 0; s < overCount; s++)
                {
                    if (count >= maxContact) break;

                    int nb = hashGridOverflow[h * MAX_OVERFLOW_PER_BUCKET + s];
                    if (nb < 0 || nb == index) continue;

                    float3 bPred = make_float3(predX[nb], predY[nb], predZ[nb]);
                    float  bRad  = radius[nb];
                    float  bMass = mass[nb];

                    float3 n = aPred - bPred;
                    n.y = 0.0f;
                    float d = length(n);
                    float margin = fmaxf(aRad, 0.25f) + fmaxf(bRad, 0.25f);
                    float f = d - margin;

                    if (f < 0.0f)
                    {
                        n = (d > 1e-6f) ? (n / d) : make_float3(1.0f, 0.0f, 0.0f);
                        float wA = 1.0f / fmaxf(aMass, 1e-4f);
                        float wB = 1.0f / fmaxf(bMass, 1e-4f);
                        float w  = wA / (wA + wB);
                        totalDelta += -w * p.kShortRange * f * n;
                        count++;
                    }
                }
            }
        }
    }

    CorrectionData cd;
    cd.delta = totalDelta;
    cd.count = count;
    corrections[index] = cd;
}

// ===========================================================================
// PAGP Instrumentation Kernels
// ===========================================================================

__global__ void KernelBruteForceNeighbors(
    const AgentGPU* agents,
    int* outCandidates,
    int* outNeighbors,
    int agentCount,
    float interactionRadius)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= agentCount) return;

    AgentGPU a = agents[index];
    int candidates = 0;
    int neighbors = 0;
    float rSq = interactionRadius * interactionRadius;

    for (int j = 0; j < agentCount; j++)
    {
        if (j == index) continue;
        candidates++;

        float3 diff = a.predictedPosition - agents[j].predictedPosition;
        diff.y = 0.0f;
        float dSq = dot(diff, diff);
        if (dSq < rSq)
            neighbors++;
    }

    outCandidates[index] = candidates;
    outNeighbors[index]  = neighbors;
}

__global__ void KernelMeasureViolations(
    const AgentGPU* agents,
    const int* hashGrid,
    const int* hashGridCount,
    const int* hashGridOverflow,
    const int* hashOverflowCount,
    float* outViolationSum,
    int*   outViolationCount,
    float* outMaxViolation,
    SimParams p)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= p.agentCount) return;

    AgentGPU a = agents[index];
    int2 center = WorldToCell(a.predictedPosition, p.cellSize);

    int visitedBuckets[9];
    int visitedCount = 0;

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            int h = Hash(center.x + i, center.y + j);

            // Deduplicate visited buckets in case of hash collisions in the 3x3 window
            bool dup = false;
            for (int v = 0; v < visitedCount; v++) {
                if (visitedBuckets[v] == h) { dup = true; break; }
            }
            if (dup) continue;
            visitedBuckets[visitedCount++] = h;

            int primaryCount = min(hashGridCount[h], MAX_PER_BUCKET);

            for (int s = 0; s < primaryCount; s++)
            {
                int nb = hashGrid[h * MAX_PER_BUCKET + s];
                if (nb <= index) continue;

                AgentGPU b = agents[nb];
                float3 diff = a.predictedPosition - b.predictedPosition;
                diff.y = 0.0f;
                float d = length(diff);
                float margin = fmaxf(a.radius, 0.25f) + fmaxf(b.radius, 0.25f);
                float penetration = margin - d;

                if (penetration > 0.0f)
                {
                    atomicAdd(outViolationSum, penetration);
                    atomicAdd(outViolationCount, 1);
                    int* addr = (int*)outMaxViolation;
                    int expected = *addr;
                    while (penetration > __int_as_float(expected))
                    {
                        int old = atomicCAS(addr, expected, __float_as_int(penetration));
                        if (old == expected) break;
                        expected = old;
                    }
                }
            }

            if (hashGridOverflow != nullptr && hashOverflowCount != nullptr)
            {
                int overCount = min(hashOverflowCount[h], MAX_OVERFLOW_PER_BUCKET);
                for (int s = 0; s < overCount; s++)
                {
                    int nb = hashGridOverflow[h * MAX_OVERFLOW_PER_BUCKET + s];
                    if (nb <= index) continue;

                    AgentGPU b = agents[nb];
                    float3 diff = a.predictedPosition - b.predictedPosition;
                    diff.y = 0.0f;
                    float d = length(diff);
                    float margin = fmaxf(a.radius, 0.25f) + fmaxf(b.radius, 0.25f);
                    float penetration = margin - d;

                    if (penetration > 0.0f)
                    {
                        atomicAdd(outViolationSum, penetration);
                        atomicAdd(outViolationCount, 1);
                        int* addr = (int*)outMaxViolation;
                        int expected = *addr;
                        while (penetration > __int_as_float(expected))
                        {
                            int old = atomicCAS(addr, expected, __float_as_int(penetration));
                            if (old == expected) break;
                            expected = old;
                        }
                    }
                }
            }
        }
    }
}

__global__ void KernelHashGridStats(
    const int* hashGridCount,
    int* outOccupiedCells,
    int* outMaxOccupancy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HASH_BUCKET_COUNT) return;

    int count = hashGridCount[i];
    if (count > 0)
    {
        atomicAdd(outOccupiedCells, 1);
        atomicMax(outMaxOccupancy, count);
    }
}

__global__ void KernelDetailedHashStats(
    const int* hashGridCount,
    const int* hashOverflowCount,
    int* outOccupiedBuckets,
    int* outMaxPrimaryOccupancy,
    int* outMaxSecondaryOccupancy,
    int* outMaxTotalOccupancy,
    int* outBucketsUsingOverflow,
    int* outBucketsExceedingSecondary)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HASH_BUCKET_COUNT) return;

    int count = hashGridCount[i];
    int overCount = (hashOverflowCount != nullptr) ? hashOverflowCount[i] : 0;

    if (count > 0)
    {
        atomicAdd(outOccupiedBuckets, 1);
        atomicMax(outMaxPrimaryOccupancy, min(count, MAX_PER_BUCKET));
        atomicMax(outMaxTotalOccupancy, count);
    }
    if (overCount > 0)
    {
        atomicAdd(outBucketsUsingOverflow, 1);
        atomicMax(outMaxSecondaryOccupancy, min(overCount, MAX_OVERFLOW_PER_BUCKET));
        if (overCount > MAX_OVERFLOW_PER_BUCKET)
        {
            atomicAdd(outBucketsExceedingSecondary, 1);
        }
    }
}

__global__ void KernelCollectSpatialHashPairs(
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
    long long* outDistanceChecks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= agentCount) return;

    AgentGPU a = agents[i];
    int2 center = WorldToCell(a.predictedPosition, cellSize);
    float threshSq = threshold * threshold;

    int visitedBuckets[9];
    int visitedCount = 0;
    long long localTraversed = 0;
    long long localChecks = 0;

    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dz = -1; dz <= 1; dz++)
        {
            int h = Hash(center.x + dx, center.y + dz);
            bool dup = false;
            for (int v = 0; v < visitedCount; v++) {
                if (visitedBuckets[v] == h) { dup = true; break; }
            }
            if (dup) continue;
            visitedBuckets[visitedCount++] = h;

            int primaryCount = min(hashGridCount[h], MAX_PER_BUCKET);
            localTraversed += primaryCount;

            for (int s = 0; s < primaryCount; s++)
            {
                int nb = hashGrid[h * MAX_PER_BUCKET + s];
                if (nb <= i) continue;

                localChecks++;
                AgentGPU b = agents[nb];
                float3 diff = a.predictedPosition - b.predictedPosition;
                diff.y = 0.0f;
                float dSq = dot(diff, diff);
                if (dSq < threshSq)
                {
                    int slot = atomicAdd(outPairCount, 1);
                    if (slot < maxPairs && outPairA != nullptr && outPairB != nullptr)
                    {
                        outPairA[slot] = i;
                        outPairB[slot] = nb;
                    }
                }
            }

            if (hashGridOverflow != nullptr && hashOverflowCount != nullptr)
            {
                int overCount = min(hashOverflowCount[h], MAX_OVERFLOW_PER_BUCKET);
                localTraversed += overCount;

                for (int s = 0; s < overCount; s++)
                {
                    int nb = hashGridOverflow[h * MAX_OVERFLOW_PER_BUCKET + s];
                    if (nb <= i) continue;

                    localChecks++;
                    AgentGPU b = agents[nb];
                    float3 diff = a.predictedPosition - b.predictedPosition;
                    diff.y = 0.0f;
                    float dSq = dot(diff, diff);
                    if (dSq < threshSq)
                    {
                        int slot = atomicAdd(outPairCount, 1);
                        if (slot < maxPairs && outPairA != nullptr && outPairB != nullptr)
                        {
                            outPairA[slot] = i;
                            outPairB[slot] = nb;
                        }
                    }
                }
            }
        }
    }

    if (outEntriesTraversed != nullptr) atomicAdd((unsigned long long int*)outEntriesTraversed, (unsigned long long int)localTraversed);
    if (outDistanceChecks != nullptr) atomicAdd((unsigned long long int*)outDistanceChecks, (unsigned long long int)localChecks);
}
