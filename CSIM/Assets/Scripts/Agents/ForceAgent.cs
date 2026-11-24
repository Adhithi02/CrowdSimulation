using UnityEngine;

public class ForceAgent : AgentBase
{
    private Rigidbody _rigidbody;
    private const float EPSILON = 0.0001f;
    protected override void Awake()
    {
        base.Awake();
    }

    protected override void Start()
    {
        base.Start();

        if (_agent != null)
        {
            // Prevent NavMeshAgent from directly driving the Transform
            // We want ForceAgent to handle its own movement logic.
            _agent.updatePosition = false;
            _agent.updateRotation = false;
        }

        _rigidbody = this.gameObject.GetComponent<Rigidbody>();
        if (_rigidbody == null)
        {
            _rigidbody = this.gameObject.AddComponent<Rigidbody>();
        }

        // Disable gravity and freeze constraints so it doesn't fall through the floor!
        _rigidbody.useGravity = false;
        _rigidbody.constraints = RigidbodyConstraints.FreezeRotation | RigidbodyConstraints.FreezePositionY;
    }

    public void ComputeForces()
    {
        Vector3 position = transform.position;
        Vector3 vPref = Vector3.zero;

        if (goal != null)
        {
            Vector3 diff = goal.position - position;
            diff.y = 0;

            float distSqToGoal = diff.sqrMagnitude;
            if (distSqToGoal < goalRadius * goalRadius)
            {
                velocity = Vector3.zero;
                currentGoal = GetNextGoalIndex();
                goal = goals[currentGoal];
                return;
            }
            vPref = diff.normalized * prefSpeed;
        }

        // Driving force
        F = (vPref - velocity) / ksi;

        // Anticipatory force from each neighbor
        foreach (var other in neighbours)
        {
            if (other == this) continue;

            Vector3 otherPos = other.transform.position;
            Vector3 w = otherPos - position;
            w.y = 0;
            Vector3 v = velocity - other.velocity;
            v.y = 0;

            float distanceSq = w.sqrMagnitude;
            float rSq = (other.radius + radius) * (other.radius + radius);

            if (distanceSq < rSq)
            {
                rSq = Mathf.Pow(other.radius + radius - Mathf.Sqrt(distanceSq), 2);
            }

            float a = Vector3.Dot(v, v);
            float b = Vector3.Dot(w, v);
            float c = Vector3.Dot(w, w) - rSq;
            float discr = b * b - a * c;

            if (discr > 0.0f && (a < -EPSILON || a > EPSILON))
            {
                discr = Mathf.Sqrt(discr);
                float t = (b - discr) / a;
                if (t > 0)
                {
                    Vector3 forcePart = (v - (b * v - a * w) / discr) / (a * Mathf.Pow(t, m)) * (m / t + 1 / t0);
                    F += -k * Mathf.Exp(-t / t0) * forcePart;
                }
            }
        }

        // Wall / static obstacle repulsion — treated as stationary agents with zero velocity.
        // This runs inside the force loop so the result is integrated like any other force,
        // preventing the social-force push from driving boundary agents through corridor walls.
        var wallBounds = SimulationManager.Instance?.WallBounds;
        if (wallBounds != null)
        {
            foreach (var wall in wallBounds)
            {
                Vector3 closest = wall.ClosestPoint(position);
                Vector3 away    = position - closest;
                away.y = 0;
                float dist = away.magnitude;
                if (dist < 0.001f || dist > t0 * 4f) continue;

                // Same exponential decay as agent repulsion; wall is heavier so scale by 2
                float mag = k * 2f * Mathf.Exp(-dist / t0);
                F += away.normalized * mag;
            }
        }

        F.y = 0;
    }

    public void UpdateAgent()
    {
        // update the agents
        Vector3 acceleration = F;
        if (acceleration.sqrMagnitude > maxAccel * maxAccel)
        {
            acceleration = acceleration.normalized * maxAccel;
        }

        float dt = Time.deltaTime;

        velocity += acceleration * dt;

        // Clamp velocity to the preferred speed (plus a tiny bit of leeway if desired)
        if (velocity.sqrMagnitude > prefSpeed * prefSpeed)
        {
            velocity = velocity.normalized * prefSpeed;
        }

        if (_rigidbody != null)
        {
            //face to movement direction
            if (velocity.sqrMagnitude > 0.01f)
            {
                Quaternion targetRotation = Quaternion.LookRotation(velocity.normalized, Vector3.up);
                _rigidbody.MoveRotation(Quaternion.Slerp(transform.rotation, targetRotation, dt * 5f));
            }

            _rigidbody.velocity = new Vector3(velocity.x, 0, velocity.z); // Explicitly zero out vertical velocity just in case

        }
        else
        {
            Debug.LogWarning("No Rigidbody found on ForceAgent. Movement will not work properly.");
        }
    }

    private Vector3 ClosestPointLineSegment(Vector3 p1, Vector3 p2, Vector3 p0)
    {
        Vector3 v = p2 - p1;
        Vector3 w = p0 - p1;

        float c1 = Vector3.Dot(w, v);
        if (c1 <= 0) return p1;

        float c2 = Vector3.Dot(v, v);
        if (c2 <= c1) return p2;

        float b = c1 / c2;
        return p1 + b * v;
    }

    private float Det2D(Vector3 a, Vector3 b)
    {
        // Equivalent to 2D cross product in xz plane: a.x * b.y - a.y * b.x
        return a.x * b.z - a.z * b.x;
    }
}
