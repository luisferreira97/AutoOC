from autooc.agent.agent import Agent
from autooc.algorithm.parameters import params
# from autooc.fitness.evaluation import evaluate_fitness
from autooc.stats.stats import stats

# from autooc.utilities.stats import trackers
# from autooc.operators.initialisation import initialisation
# from autooc.utilities.algorithm.initialise_run import pool_init


def create_agents(n, p):
    """
    Create a list of agent specified by n parameter
    """
    return [Agent(p) for a in range(n)]


def search_loop():
    """
    This loop is used when the multi-agent parameter is passed
    """
    # Create a list of agents based on the parameter passed
    agents = create_agents(params["AGENT_SIZE"],
                           params["INTERACTION_PROBABILITY"])

    # Multi-Agent based GE
    for generation in range(1, (params["GENERATIONS"] + 1)):
        stats["gen"] = generation

        # New generation
        agents = params["STEP"](agents)

    return [agent.individual[0] for agent in agents]
