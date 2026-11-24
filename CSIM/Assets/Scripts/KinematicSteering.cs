using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

[RequireComponent(typeof(NavMeshAgent))]
public class KinematicSteering : MonoBehaviour
{
    [Header("PBD Physical Properties")]
    [Tooltip("Heavier agents push lighter agents out of the way.")]
    public float mass = 1f;
    [Tooltip("Collision boundary for the PBD constraint solver.")]
    public float radius = 0.5f;

    [Header("Spatial Query")]
    public float perceptionRadius = 4f;
    // public float separationRadius = 1.5f; //No longer needed

    [HideInInspector] public List<KinematicSteering> neighbours = new();
    [HideInInspector] public Vector3 velocity;
    [HideInInspector] public Vector3 predictedPosition;

    public NavMeshAgent Agent { get; private set; }

    void Awake()
    {
        Agent = GetComponent<NavMeshAgent>();
    }
    /* LATE UPDATE NO LONGER NEEDED
    void LateUpdate()
    {
        velocity = _agent.velocity;

        if (accumulatedDisplacement.sqrMagnitude > 0.0001f)
        {
            _agent.Move(accumulatedDisplacement);
            accumulatedDisplacement = Vector3.zero;
        }
    }
    */
}
