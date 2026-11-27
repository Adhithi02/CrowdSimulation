using UnityEngine;

public class PBDAgent : AgentBase
{
    private int pbdIterations = 0;

    public int PbdIterations
    {
        get => pbdIterations;
        set => pbdIterations = value;
    }

    protected override void Start()
    {
        base.Start();
        if (goal != null)
            SimulationManager.Instance?.TrySetAgentDestination(this, goal.position);

        _agent.speed = speed;
        // PBD INTEGRATION: Stop NavMeshAgent from directly driving the Transform
        _agent.updatePosition = false;
        _agent.updateRotation = false;
    }

    private void FixedUpdate()
    {
        if (ProfilerManager.Instance != null)
            ProfilerManager.Instance.BeginSample("Animation");

        CheckDestinationReached();
        UpdateAnimation();

        if (ProfilerManager.Instance != null)
            ProfilerManager.Instance.EndSample("Animation");
    }

    public void SetGoal(Transform newGoal)
    {
        goal = newGoal;
        if (goal == null)
        {
            Debug.LogWarning("Goal is missing. Cannot set destination.");
            return;
        }

        SimulationManager.Instance?.TrySetAgentDestination(this, goal.position);
    }

    public void CheckDestinationReached()
    {
        // We need at least 2 goals to be able to pick a different random one.
        if (goals == null || goals.Length <= 1 || _agent == null ||
            !_agent.enabled || !gameObject.activeInHierarchy)
            return;

        if (!SimulationManager.Instance.TryRecoverAgentOnNavMesh(this))
            return;

        if (!_agent.pathPending && _agent.remainingDistance < stopThreshold)
        {
            currentGoal = GetNextGoalIndex();
            SetGoal(goals[currentGoal]);
        }
    }

    public void UpdateAnimation()
    {
        // In PBD, the true velocity is driven by the constraint solver.
        // We evaluate the actual current speed to drive the animator.
        float currentSpeed = _agent.velocity.magnitude;
        _anim.SetBool(_fastID, currentSpeed >= slowFastThreshold);
    }

    // -------------------------------------------------------------------------
    // PBD Solver (Weiss 2019 Implementation)
    // -------------------------------------------------------------------------
    public void SolveSeparation()
    {
        float dt = Time.deltaTime;
        if (dt <= 0f) return;

        // Step 1: Predict Positions and Plan Velocity
        // Calculate preferred velocity directed at the NavMesh path corner
        Vector3 desiredDirection = (_agent.steeringTarget - this.transform.position).normalized;
        Vector3 vPref = desiredDirection * _agent.speed;

        // Interpolate velocity using Ksi
        velocity = Vector3.Lerp(velocity, vPref, ksi);

        // Derive predicted position
        predictedPosition = this.transform.position + velocity * dt;
        predictedPosition.y = this.transform.position.y;

        // Step 2: Project Constraints (Collision Avoidance)
        for (int iter = 0; iter < pbdIterations; ++iter)
        {
            foreach (var neighbor in neighbours)
            {
                Vector3 diff = predictedPosition - neighbor.predictedPosition;
                diff.y = 0;
                float distance = diff.magnitude;

                float margin = (radius > 0 ? radius : 0.5f) +
                               (neighbor.radius > 0 ? neighbor.radius : 0.5f);

                float penetration = margin - distance;

                if (penetration > 0 && distance > 0.001f)
                {
                    Vector3 contactNormal = diff / distance;
                    float wI = 1f / (mass > 0 ? mass : 1f);
                    float wJ = 1f / (neighbor.mass > 0 ? neighbor.mass : 1f);
                    float wSum = wI + wJ;
                    predictedPosition += (penetration / wSum) * contactNormal * wI;
                }
            }

            // Wall constraints — walls have infinite mass so only the agent moves.
            // Applied every iteration so crowd pressure can't accumulate into the wall.
            var walls = SimulationManager.Instance?.WallBounds;
            if (walls != null)
            {
                foreach (var wall in walls)
                {
                    Vector3 closest = wall.ClosestPoint(predictedPosition);
                    Vector3 diff    = predictedPosition - closest;
                    diff.y = 0;
                    float dist = diff.magnitude;
                    if (dist < 0.001f)
                    {
                        // Inside wall: eject through thinnest horizontal face
                        float penX = Mathf.Min(Mathf.Abs(predictedPosition.x - wall.min.x),
                                               Mathf.Abs(predictedPosition.x - wall.max.x));
                        float penZ = Mathf.Min(Mathf.Abs(predictedPosition.z - wall.min.z),
                                               Mathf.Abs(predictedPosition.z - wall.max.z));
                        if (penZ <= penX)
                            predictedPosition.z = predictedPosition.z < wall.center.z
                                                ? wall.min.z - radius
                                                : wall.max.z + radius;
                        else
                            predictedPosition.x = predictedPosition.x < wall.center.x
                                                ? wall.min.x - radius
                                                : wall.max.x + radius;
                    }
                    else if (dist < radius)
                    {
                        predictedPosition += diff.normalized * (radius - dist);
                    }
                }
            }
        }

        // Step 3: Update Final Positions and Velocities
        // Clamp post-constraint velocity so dense contacts do not look like elastic rebounds.
        Vector3 rawVelocity = (predictedPosition - this.transform.position) / dt;
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
        velocity = rawVelocity;

        // Apply the actual movement
        this.transform.position = predictedPosition;

        // Handle rotation
        if (velocity.sqrMagnitude > 0.01f)
        {
            Quaternion targetRot = Quaternion.LookRotation(velocity.normalized);
            this.transform.rotation = Quaternion.Slerp(this.transform.rotation, targetRot, dt * 10f);
        }

        // Sync NavMeshAgent to the newly constrained position so pathing doesn't break
        if (_agent != null)
        {
            _agent.nextPosition = this.transform.position;
        }
    }
}
