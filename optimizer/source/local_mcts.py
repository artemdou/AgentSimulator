import math
import random
from copy import deepcopy
from typing import Any, List

from mcts_node import MCTSNode  # assumes MCTSNode class in mcts_node.py
from state import Case

Proposal = Any  # replace with your actual Proposal type


# def evaluate_cost(case: Case) -> float:
#     """
#     Compute the trace duration for a single case.
#     """
#     start = case.start_time
#     end = case.current_time
#     return (end - start).total_seconds()

def evaluate_cost(case: Case):
    base = (case.current_time - case.start_time).total_seconds()
    repeats = sum(case.performed.count(a) - 1 for a in set(case.performed))
    return base + 300 * repeats



def mcts_select(simulation: Any,
                proposals: List[Proposal],
                budget: int = 1000,
                exploration_constant: float = math.sqrt(2)
               ) -> Proposal:
    """
    Perform Local MCTS to choose the best next proposal from the given simulation state.
    :param simulation: the real simulation state (will not be mutated)
    :param proposals: list of valid proposals at this decision point
    :param budget: number of rollouts to perform
    :param exploration_constant: UCT exploration parameter
    :return: the Proposal with highest visit count at root
    """
    root = MCTSNode(state=deepcopy(simulation), parent=None, proposal=None)
    root.untried_actions = proposals.copy()

    for rollout_idx in range(budget):
        print(f"\n\U0001f39f️ Rollout {rollout_idx + 1}/{budget}")
        node = root

        while node.fully_expanded() and not node.is_terminal():
            next_node = node.best_uct_child(exploration_constant)
            if next_node is None:
                print("⚠️ No children to expand from node.")
                break
            print(f"🔹 Descending to child via proposal: {next_node.proposal.activity} by agent {next_node.proposal.agent.agent_id}")
            node = next_node

        if not node.is_terminal() and node.untried_actions:
            prop = node.untried_proposal()
            print(f"🔄 Expanding new child with proposal: {prop.activity} by agent {prop.agent.agent_id}")
            child_state = deepcopy(node.state)

            # Rewire proposal for cloned simulation state
            case = child_state.cases[0]
            agent_map = {a.agent_id: a for a in child_state.agents}
            agent = agent_map[prop.agent.agent_id]
            available = case.get_available_activities(child_state.rules)
            calendars = getattr(child_state, 'calendars', {})

            reproposals = child_state.make_proposals(agent, case, child_state.rules, child_state.durations, calendars, available)
            matching = [p for p in reproposals if p.activity == prop.activity]

            if matching:
                child_state.perform_proposal(matching[0], child_state.rules)
            else:
                print("⚠️ Could not rebind proposal to cloned simulation.")

            print(case.performed)
            node = node.add_child(prop, child_state)
            rollout_state = deepcopy(child_state)
        else:
            rollout_state = deepcopy(node.state)

        case = rollout_state.cases[0]  # assuming one case per MCTS run
        calendars = rollout_state.calendars if hasattr(rollout_state, 'calendars') else {}
        steps = 0

        while True:
            available_activities = case.get_available_activities(rollout_state.rules)
            proposals = []
            for agent in rollout_state.agents:
                props = rollout_state.make_proposals(
                    agent, case, rollout_state.rules,
                    rollout_state.durations, calendars, available_activities
                )
                proposals.extend(props)
            if not proposals:
                print(f"🛑 Rollout ended prematurelly after {steps} steps (no more proposals).")
                break

            # selected = rollout_state.select_proposal(proposals)
            selected = rollout_state.select_proposal_greedy(proposals)
            # trying earliest and shortest selection
            # selected = min(proposals, key=lambda p: p.duration + (p.start_time - case.current_time).total_seconds())
            # print(f"➡️ Rollout step {steps + 1}: selected {selected.activity} by agent {selected.agent.agent_id}")
            rollout_state.perform_proposal(selected, rollout_state.rules)
            steps += 1

            if rollout_state.rules.is_case_end(selected.activity, case.outstanding_obligations):
                print(f"✅ Case ended at activity {selected.activity} after {steps} steps.")
                print(f"🟢 Performed activities: {case.performed}")
                break

        cost = evaluate_cost(case)
        print(f"💸 Rollout cost: {cost}")
        node.backpropagate(-cost)

    best_child = root.child_with_max_visits()
    print(f"🔝 Selected proposal: {best_child.proposal.activity} by agent {best_child.proposal.agent.agent_id} with {best_child.n_visits} visits")
    return best_child.proposal


def local_mcts_tick(simulation: Any, calendars: dict, budget: int = 1000) -> None:
    """
    Perform one scheduling tick on all active cases using Local MCTS.
    :param simulation: your Simulation instance
    :param calendars: mapping of agent_id to RCalendar
    :param budget: number of rollouts per decision
    Modifies simulation in-place.
    """
    active_cases = [c for c in simulation.cases if not c.done and c.get_available_activities(simulation.rules)]

    for case in active_cases:
        available_activities = case.get_available_activities(simulation.rules)
        print(f"🏋🏻 Available activities: {available_activities}")
        all_proposals = []
        for agent in simulation.agents:
            props = simulation.make_proposals(agent, case, simulation.rules, simulation.durations, calendars, available_activities)
            all_proposals.extend(props)
        if not all_proposals:
            continue

        simulation_clone = deepcopy(simulation)
        simulation_clone.calendars = calendars  # ensure clone has access to calendars
        best_prop = mcts_select(simulation_clone, all_proposals, budget)

        print(f"👆🏼 Best proposal: {best_prop.agent.agent_id} - {best_prop.activity}")
        simulation.perform_proposal(best_prop, simulation.rules)
