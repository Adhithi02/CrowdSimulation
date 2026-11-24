using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using Random = UnityEngine.Random;

public class goal_move : MonoBehaviour
{
    //Private components. Fetched during Start.
    private Animator _anim;
    public NavMeshAgent _agent { get; private set; }

    //Serialized Private Field

    [SerializeField] private float speed;
    [SerializeField] private float slowFastThreshold;
    [SerializeField] private int stopThreshold;
    [SerializeField] private int currentGoal = 0;
    [SerializeField] private GameObject[] goals;

    public GameObject[] Goals
    {
        get => goals;
        set => goals = value;
    }

    //Animator speed thresholds
    [SerializeField] private float MinimumAnimationSpeed = 0.8f;
    [SerializeField] private float MaximumAnimationSpeed = 1.2f;

    //Animator variables
    private int _runningID;
    private int _fastID;
    private const string RunningString = "Running";
    private const string FastString = "Fast";

    private int nrOfGoals;

    void Start()
    {
        _anim = GetComponent<Animator>();
        _agent = GetComponent<NavMeshAgent>();

        // PBD INTEGRATION: Stop NavMeshAgent from directly driving the Transform
        _agent.updatePosition = false;
        _agent.updateRotation = false;

        // If FlockManager successfully passed the goals array...
        if (goals != null && goals.Length > 0)
        {
            nrOfGoals = goals.Length;
            // Pick a random starting goal from the array!
            currentGoal = Random.Range(0, nrOfGoals);
            _agent.SetDestination(goals[currentGoal].transform.position);
        }

        _agent.speed = speed;

        _anim.speed = Random.Range(MinimumAnimationSpeed, MaximumAnimationSpeed);
        _runningID = Animator.StringToHash(RunningString);
        _fastID = Animator.StringToHash(FastString);
        StartRunningAnimation();
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

    private void StartRunningAnimation()
    {
        _anim.SetBool(_fastID, _agent.speed >= slowFastThreshold);
        _anim.SetBool(_runningID, true);
    }


    private void CheckDestinationReached()
    {
        // We need at least 2 goals to be able to pick a different random one
        if (goals == null || goals.Length <= 1) return;

        if (!_agent.pathPending && _agent.remainingDistance < stopThreshold)
        {
            int nextGoal = currentGoal;

            // Keep picking a random goal until it's a different one
            while (nextGoal == currentGoal)
            {
                nextGoal = Random.Range(0, nrOfGoals);
            }

            currentGoal = nextGoal;
            _agent.SetDestination(goals[currentGoal].transform.position);
        }
    }

    private void UpdateAnimation()
    {
        // In PBD, the true velocity is driven by the constraint solver.
        // We evaluate the actual current speed to drive the animator.
        float currentSpeed = _agent.velocity.magnitude;
        _anim.SetBool(_fastID, currentSpeed >= slowFastThreshold);
    }
}