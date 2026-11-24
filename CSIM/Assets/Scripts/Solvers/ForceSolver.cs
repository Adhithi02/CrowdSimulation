using System.Collections.Generic;
using UnityEngine;

public class ForceSolver : MonoBehaviour
{
    public void Solve(List<AgentBase> agents)
    {
        foreach (var agent in agents)
        {
            if (agent is ForceAgent forceAgent)
            {
                forceAgent.ComputeForces();
                forceAgent.UpdateAgent();
            }
        }
    }
}