#include <stdio.h>
#include <stddef.h>
#include "CrowdCommon.h"

int main() {
    printf("Sizeof AgentGPU: %zu bytes\n", sizeof(AgentGPU));
    printf("Alignof AgentGPU: %zu bytes\n", __alignof(AgentGPU));
    printf("Offset position: %zu\n", offsetof(AgentGPU, position));
    printf("Offset ksi: %zu\n", offsetof(AgentGPU, ksi));
    printf("Offset predictedPosition: %zu\n", offsetof(AgentGPU, predictedPosition));
    printf("Offset radius: %zu\n", offsetof(AgentGPU, radius));
    printf("Offset velocity: %zu\n", offsetof(AgentGPU, velocity));
    printf("Offset mass: %zu\n", offsetof(AgentGPU, mass));
    printf("Offset preferredVelocity: %zu\n", offsetof(AgentGPU, preferredVelocity));
    printf("Offset lodLevel: %zu\n", offsetof(AgentGPU, lodLevel));
    printf("Offset importanceScore: %zu\n", offsetof(AgentGPU, importanceScore));
    printf("Offset lastContactCount: %zu\n", offsetof(AgentGPU, lastContactCount));
    printf("Offset _padding: %zu\n", offsetof(AgentGPU, _padding));
    return 0;
}
