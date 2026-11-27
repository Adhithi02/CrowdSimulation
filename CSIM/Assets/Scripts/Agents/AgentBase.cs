using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

[RequireComponent(typeof(NavMeshAgent))]
public class AgentBase : MonoBehaviour
{
    [Header("Spatial Query")]
    public float separationRadius = 1.5f;

    [Header("Agent Parameters")]
    public float radius = 0.5f;
    public float prefSpeed = 1.6f;
    public float maxAccel = 5f;
    public float neighborDist = 4.0f;
    public float k = 1.0f;
    public float ksi = 0.05f;
    public float m = 2.0f;
    public float t0 = 3.0f;
    public float goalRadius = 2.0f;

    [Tooltip("Per-commuter walking-speed multiplier (natural variation).")]
    public float speedFactor = 1f;
    [Tooltip("Temporary multiplier, e.g. slowing down to tap a card at the fare gate.")]
    public float speedScale = 1f;

    [Header("PBD Physical Properties")]
    [Tooltip("Heavier agents push lighter agents out of the way.")]
    public float mass = 1f;

    public int currentGoal = 0;
    public Transform[] goals;

    public Transform[] Goals
    {
        get => goals;
        set => goals = value;
    }

    public int nrOfGoals
    {
        get => goals.Length;
    }

    //Animator variables
    protected int _runningID;
    protected int _fastID;
    private const string RunningString = "Running";
    private const string FastString = "Fast";

    [HideInInspector] public Vector3 velocity;
    [HideInInspector] public Vector3 predictedPosition;
    [HideInInspector] public Vector3 accumulatedDisplacement;
    [HideInInspector] public Vector3 F;
    [HideInInspector] public Transform goal;

    [Header("Importance & Fidelity (Runtime)")]
    [Tooltip("Normalized importance score in [0, 1] computed from distance, visibility, interaction, and local density.")]
    [Range(0f, 1f)] public float importanceScore = 0.5f;
    [Tooltip("Assigned simulation fidelity tier: 0 = HIGH, 1 = MED, 2 = LOW.")]
    public int fidelityLevel = 0;

    [HideInInspector] public List<AgentBase> neighbours = new();

    public float perceptionRadius => neighborDist;

    //Animator speed thresholds
    [SerializeField] protected float MinimumAnimationSpeed = 0.8f;
    [SerializeField] protected float MaximumAnimationSpeed = 1.2f;
    [SerializeField] protected float slowFastThreshold = 10.0f;
    [SerializeField] protected int stopThreshold = 6;
    [SerializeField] protected float speed = 5.0f;

    public Animator _anim { get; private set; }
    public NavMeshAgent _agent { get; private set; }


    protected virtual void Awake()
    {
        _agent = GetComponent<NavMeshAgent>();
        _anim = GetComponent<Animator>();
    }
    protected virtual void Start()
    {
        if (goals != null && goals.Length > 0)
        {
            currentGoal = Random.Range(0, goals.Length);
            goal = goals[currentGoal];
        }

        // _agent.speed = speed;
        _anim.speed = Random.Range(MinimumAnimationSpeed, MaximumAnimationSpeed);
        _runningID = Animator.StringToHash(RunningString);
        _fastID = Animator.StringToHash(FastString);
        StartRunningAnimation();
    }

    private void StartRunningAnimation()
    {
        _anim.SetBool(_fastID, _agent.speed >= slowFastThreshold);
        _anim.SetBool(_runningID, true);
    }

    public int GetNextGoalIndex()
    {
        if (goals == null || goals.Length == 0) return -1;

        if (goals.Length == 1) return 0;

        int nextGoal = currentGoal;
        while (nextGoal == currentGoal)
        {
            nextGoal = Random.Range(0, nrOfGoals);
        }
        return nextGoal;
    }

    protected virtual void LateUpdate()
    {
    }
}
