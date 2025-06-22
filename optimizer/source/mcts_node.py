import math
from typing import Any, Dict, Optional

class MCTSNode:
    """
    A node in the Monte Carlo Tree Search (MCTS) representing a simulation state.
    Each node tracks visit counts, total reward, child nodes, and untried actions.
    """
    def __init__(self, state: Any, parent: Optional['MCTSNode'] = None, proposal: Any = None):
        """
        :param state: A clone of the Simulation at this node.
        :param parent: Reference to the parent MCTSNode (None for root).
        :param proposal: The Proposal that led from the parent to this node.
        """
        self.state = state
        self.parent = parent
        self.proposal = proposal
        # Children keyed by proposal; values are MCTSNode
        self.children: Dict[Any, MCTSNode] = {}
        # Actions not yet tried from this node
        self.untried_actions = []
        # Statistics
        self.n_visits: int = 0
        self.w_value: float = 0.0  # cumulative (negative) cost or reward

    def is_terminal(self) -> bool:
        """
        Check if the simulation state has no further proposals for the current case.
        """
        case = self.state.cases[0]  # assuming single-case MCTS
        available = case.get_available_activities(self.state.rules)
        calendars = getattr(self.state, 'calendars', {})

        for agent in self.state.agents:
            proposals = self.state.make_proposals(
                agent, case, self.state.rules,
                self.state.durations, calendars, available
            )
            if proposals:
                return False  # at least one move is available

        return True  # no agents can propose anything


    def fully_expanded(self) -> bool:
        """
        A node is fully expanded if all actions have been tried (no remaining untried_actions).
        """
        return len(self.untried_actions) == 0

    def uct_score(self, child: 'MCTSNode', exploration_constant: float = math.sqrt(2)) -> float:
        """
        Compute the UCT score for a child node.
        UCT = (w / n) + C * sqrt(ln(N) / n)
        where:
          w = child.w_value, n = child.n_visits, N = self.n_visits
        """
        if child.n_visits == 0:
            return float('inf')
        exploitation = child.w_value / child.n_visits
        exploration = exploration_constant * math.sqrt(math.log(self.n_visits) / child.n_visits)
        return exploitation + exploration

    def best_uct_child(self, exploration_constant: float = math.sqrt(2)) -> Optional['MCTSNode']:
        """
        Select the child with the highest UCT score.
        """
        if not self.children:
            return None
        return max(
            self.children.values(),
            key=lambda c: self.uct_score(c, exploration_constant)
        )


    def untried_proposal(self) -> Any:
        """
        Pop and return one untried proposal (action) to expand.
        """
        return self.untried_actions.pop()

    def add_child(self, proposal: Any, new_state: Any) -> 'MCTSNode':
        """
        Create a new child node for the given proposal and simulation state.
        :param proposal: The action leading to the new state.
        :param new_state: The cloned Simulation after applying the proposal.
        :return: The newly created child node.
        """
        child = MCTSNode(new_state, parent=self, proposal=proposal)
        self.children[proposal] = child
        return child

    def backpropagate(self, value: float) -> None:
        """
        Update this node's statistics with the given value and recurse to parent.
        :param value: The (negative) cost or reward from a rollout.
        """
        self.n_visits += 1
        self.w_value += value
        if self.parent:
            self.parent.backpropagate(value)

    def child_with_max_visits(self) -> 'MCTSNode':
        """
        Return the child node with the highest visit count (used to select the best action).
        """
        return max(self.children.values(), key=lambda c: c.n_visits)
