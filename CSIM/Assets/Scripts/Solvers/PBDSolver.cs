using System.Collections.Generic;
using UnityEngine;

public class PBDSolver : MonoBehaviour
{
    /// <summary>
    /// Executes the Position-Based Dynamics loop for a given list of PBD Agents.
    /// </summary>
    public void Solve(List<AgentBase> agents, float dt, int iterations)
    {
        if (dt <= 0f || agents.Count == 0) return;

        // ---------------------------------------------------------------------
        // Step 1: Predict Positions & Plan Velocity
        // ---------------------------------------------------------------------
        foreach (var agent in agents)
        {
            Vector3 vPref = Vector3.zero;

            if (agent._agent != null && agent._agent.hasPath)
            {
                // Calculate preferred velocity directed at the NavMesh path corner
                Vector3 desiredDirection = (agent._agent.steeringTarget - agent.transform.position).normalized;
                vPref = desiredDirection * agent.prefSpeed;
            }

            // Interpolate current velocity towards preferred velocity using the agent's ksi parameter
            agent.velocity = Vector3.Lerp(agent.velocity, vPref, agent.ksi);

            // Predict the position for the next frame
            agent.predictedPosition = agent.transform.position + agent.velocity * dt;
            agent.predictedPosition.y = agent.transform.position.y; // Keep vertical movement fixed
        }

        // ---------------------------------------------------------------------
        // Step 2: Project Constraints (Collision Avoidance)
        // ---------------------------------------------------------------------
        for (int iter = 0; iter < iterations; iter++)
        {
            foreach (var agent in agents)
            {
                foreach (var neighborBase in agent.neighbours)
                {
                    PBDAgent neighbor = neighborBase as PBDAgent;
                    if (neighbor == null || neighbor == agent) continue;

                    Vector3 diff = agent.predictedPosition - neighbor.predictedPosition;
                    diff.y = 0;
                    float distance = diff.magnitude;

                    // Margin is the sum of radii (minimum allowed distance between the two)
                    float margin = (agent.radius > 0 ? agent.radius : 0.5f) +
                                   (neighbor.radius > 0 ? neighbor.radius : 0.5f);

                    float penetration = margin - distance;

                    // If penetration > 0, they are predicted to overlap. Resolve it!
                    if (penetration > 0 && distance > 0.001f)
                    {
                        Vector3 contactNormal = diff / distance;

                        // Inverse mass weighting
                        float wI = 1f / (agent.mass > 0 ? agent.mass : 1f);
                        float wJ = 1f / (neighbor.mass > 0 ? neighbor.mass : 1f);
                        float wSum = wI + wJ;

                        // Calculate positional constraint correction
                        Vector3 correction = (penetration / wSum) * contactNormal;

                        // Apply the correction directly to the predicted position
                        agent.predictedPosition += correction * wI;
                    }
                }
            }
        }

        // ---------------------------------------------------------------------
        // Step 3: Update Final Positions and Velocities
        // ---------------------------------------------------------------------
        foreach (var agent in agents)
        {
            // Derive velocity from the constrained displacement and suppress elastic rebound.
            Vector3 rawVelocity = (agent.predictedPosition - agent.transform.position) / dt;
            Vector3 vPref = Vector3.zero;
            if (agent._agent != null && agent._agent.hasPath)
            {
                Vector3 desiredDirection = (agent._agent.steeringTarget - agent.transform.position).normalized;
                vPref = desiredDirection * agent.prefSpeed;
            }

            float preferredSpeed = vPref.magnitude;
            float maxSpeed = Mathf.Max(preferredSpeed * 1.15f, 0.75f);
            if (rawVelocity.magnitude > maxSpeed)
                rawVelocity = rawVelocity.normalized * maxSpeed;

            if (preferredSpeed > 0.01f)
            {
                Vector3 prefDir = vPref / preferredSpeed;
                float backward = Vector3.Dot(rawVelocity, -prefDir);
                if (backward > 0f)
                    rawVelocity += prefDir * (backward * 0.85f);
            }
            agent.velocity = rawVelocity;

            // Apply the actual physical movement
            agent.transform.position = agent.predictedPosition;

            // Smoothly rotate the agent to face its velocity direction
            if (agent.velocity.sqrMagnitude > 0.01f)
            {
                Quaternion targetRot = Quaternion.LookRotation(agent.velocity.normalized);
                agent.transform.rotation = Quaternion.Slerp(agent.transform.rotation, targetRot, dt * 10f);
            }

            // Sync NavMeshAgent to the newly constrained position so Unity pathfinding doesn't break
            if (agent._agent != null)
            {
                agent._agent.nextPosition = agent.transform.position;
            }

            // Note: If you have Unity Rigidbodies on the agents and they jitter, 
            // you can ensure they don't fight the PBD math by adding:
            // if (agent.TryGetComponent(out Rigidbody rb)) rb.isKinematic = true;
        }
    }
}