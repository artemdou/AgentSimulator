import random
import pandas as pd
import numpy as np
from copy import deepcopy
from proposal import Proposal, LogEntry
from activity_rules import ActivityRules
from utils import get_activity_duration, sample_from_distribution, shift_activity_start_to_next_valid_window, compute_transition_weight

from local_mcts import mcts_select

# Add project root to sys.path
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join("../../..")))
# from source.utils import sample_from_distribution

class Simulation:
    def __init__(self, agents, cases, rules, durations, case_arrivals, is_orchestrated, transition_probabilities, agent_transition_probabilities, activity_durations):
        self.agents = agents
        self.cases = cases
        self.rules = rules
        self.durations = durations
        self.case_arrivals = case_arrivals
        self.log = []
        self.is_orchestrated = is_orchestrated
        self.transition_probabilities = transition_probabilities
        self.agent_transition_probabilities = agent_transition_probabilities
        self.activity_durations = activity_durations

    def make_proposals(self, agent, case, rules: ActivityRules, durations: dict, calendars: dict, available_activities: list):
        """
        Generate a list of proposals for a given agent to perform eligible activities
        for a given case at this point in the simulation, considering:
        - prerequisites,
        - post-condition obligations,
        - agent availability,
        - activity durations,
        - and the agent's working calendar.

        Parameters:
        - agent: Agent object making the proposal.
        - case: Case object requesting work.
        - rules: ActivityRules object containing prerequisites and post-conditions.
        - durations: Dict[agent_id][activity] → DurationDistribution object.
        - calendars: Dict[agent_id] → RCalendar object for validating working hours.

        Returns:
        - List of Proposal objects, one for each valid agent-case-activity match.

        Notes:
        - Activities with missing or empty duration distributions are skipped.
        - Proposals are only created if the agent is available *and* the full duration
        fits within their working hours.
        - Duration is sampled upfront (may be deferred in the future).
        - No cost or optimization logic is applied yet.
        
        Limitations / TODOs:
        - Doesn’t yet consider multitasking or interruptions.
        """
    
        proposals = []

        for act in available_activities:
            if act not in agent.capable_activities:
                continue
            if not rules.is_activity_allowed(act, case):
                continue

            try:
                duration = get_activity_duration(agent.agent_id, act, self.activity_durations)
            except (KeyError, AttributeError) as e:
                print(e)
                continue

            # Start checking from current case time
            candidate_start = case.current_time
            calendar = calendars.get(agent.agent_id)
            calendar_json = calendar.intervals_to_json() if calendar else None

            while True:
                # Step 1: Check for overlap with busy windows
                overlaps = any(
                    start < candidate_start + pd.Timedelta(seconds=duration) and
                    candidate_start < end
                    for start, end in agent.busy_windows
                )
                if overlaps:
                    for start, end in sorted(agent.busy_windows):
                        if candidate_start < end and candidate_start + pd.Timedelta(seconds=duration) > start:
                            candidate_start = end
                            break
                    continue  # Loop back with updated time

                # Step 2: Check calendar compliance
                if calendar_json:
                    if not self.is_within_calendar(candidate_start, duration, calendar_json):
                        shifted = shift_activity_start_to_next_valid_window(candidate_start, duration, calendar_json)
                        if shifted is None:
                            candidate_start = None
                            break
                        candidate_start = shifted
                        continue  # Need to recheck for busy conflict

                break  # If we pass both checks

            if candidate_start is not None:
                proposals.append(Proposal(case, agent, act, candidate_start, duration))

        return proposals


    def is_within_calendar(self, start_time: pd.Timestamp, duration: float, calendar_json: list) -> bool:
        """
        Checks if the given start_time and duration fall entirely within
        the agent's working hours based on calendar_json (from intervals_to_json()).

        Parameters:
        - start_time: When the activity would begin.
        - duration: Duration in seconds.
        - calendar_json: List of availability windows from RCalendar.intervals_to_json().

        Returns:
        - True if the entire activity fits within any working window of that day.
        - False otherwise.

        Limitations:
        - Does not check cross-day durations.
        """

        from datetime import datetime

        end_time = start_time + pd.Timedelta(seconds=duration)
        day = start_time.strftime('%A').upper()

        for entry in calendar_json:
            if entry['from'] == day:
                try:
                    begin = datetime.strptime(entry['beginTime'], '%H:%M:%S').time()
                    end = datetime.strptime(entry['endTime'], '%H:%M:%S').time()
                except ValueError:
                    continue  # Skip malformed entries

                activity_start = start_time.time()
                activity_end = end_time.time()

                if begin <= activity_start and activity_end <= end:
                    return True

        return False


    def handle_post_conditions(self, activity, case):
        for post_act in self.rules.post_conditions.get(activity, []):
            case.add_obligation(post_act)

    
    # place holder
    # def select_proposal(self, proposals: list) -> Proposal:
    #     """
    #     Select one proposal with weighted random logic:
    #     - Boosts proposals that fulfill obligations
    #     - Penalizes proposals that repeat the case's last activity

    #     Returns:
    #     - A single Proposal object
    #     """

    #     if not proposals:
    #         return None

    #     weights = []
    #     for p in proposals:
    #         weight = 1.0

    #         # 📌 Boost if it's an obligation
    #         if p.activity in p.case.outstanding_obligations:
    #             weight *= 10  # Total = 10

    #         # 🚫 Penalize if it's an immediate repeat
    #         elif len(p.case.performed) > 0 and p.activity == p.case.performed[-1]:
    #             weight *= 0.01  # 90% penalty, Total = 0.1
            
    #         else:
    #             weight *= 0.05

    #         weights.append(weight)

            

    #     # for p, w in zip(proposals, weights):
    #         # print(f"Weighted option: {p.activity} by {p.agent.agent_id} → weight {w}")

        
    #     return random.choices(proposals, weights=weights, k=1)[0]


    # def select_proposal(self, proposals: list) -> Proposal:
    #     if not proposals:
    #         return None

    #     weights = []
    #     for p in proposals:
    #         performed_activities = p.case.performed
    #         if p.case.agent_pairing:
    #             last_agent = p.case.agent_pairing[-1].agent_id
    #             proposed_activity = p.activity
    #             proposal_agent = p.agent.agent_id
    #             weight = compute_transition_weight(
    #                 performed_activities, 
    #                 last_agent, 
    #                 self.transition_probabilities, 
    #                 self.agent_transition_probabilities, 
    #                 self.is_orchestrated,
    #                 proposal_agent,
    #                 proposed_activity
    #             )
    #         else:
    #             weight = 0.0

    #         if weight == 1:
    #             weight = weight * 100
    
    #         weights.append(weight)

    #     # fallback logic if none of the proposals could get a weight
    #     if np.array(weights).sum() == 0:
    #         weights = np.array(weights) + 0.001
    #         weights = weights.tolist()


    #     # for p, w in zip(proposals, weights):
    #     #     print(f"Weighted option: {p.activity} by {p.agent.agent_id} → weight {w}")

        
    #     return random.choices(proposals, weights=weights, k=1)[0]

    """
    agent transition probabilities and penalization for delay and duration
    """
    def select_proposal(self, proposals: list) -> Proposal:
        if not proposals:
            return None

        weights = []
        for p in proposals:
            performed_activities = p.case.performed
            case = p.case
            last_agent = p.case.agent_pairing[-1].agent_id if p.case.agent_pairing else None

            proposed_activity = p.activity
            proposal_agent = p.agent.agent_id

            # 1. Base weight from transition probabilities
            weight = compute_transition_weight(
                performed_activities,
                last_agent,
                self.transition_probabilities,
                self.agent_transition_probabilities,
                self.is_orchestrated,
                proposal_agent,
                proposed_activity
            )

            # 2. Delay penalty (later start = worse)
            delay = max(0.1, (p.start_time - case.current_time).total_seconds())

            # 3. Duration penalty
            duration = max(0.1, p.duration)

            # 4. Hybrid score: prefer early and short with high transition probability
            penalty_factor = 1
            adjusted_weight = weight / ((delay + duration) ** penalty_factor)


            weights.append(adjusted_weight)

        # fallback in case all weights are zero
        if sum(weights) == 0:
            weights = [1.0 for _ in proposals]

        return random.choices(proposals, weights=weights, k=1)[0]
    


    def select_proposal_greedy(self, proposals: list) -> Proposal:

        from datetime import datetime, timezone


        if not proposals:
            return None

        weights = []
        max_weight = 0
        selected_activity = None

        for p in proposals:
            performed_activites = p.case.performed
            if performed_activites:
                prefix = tuple(performed_activites)
                proposed_activity = p.activity
                try:
                    subdict = self.transition_probabilities[prefix]
                    probs = [v.get(proposed_activity, 0.0) for v in subdict.values()]
                    weight = sum(probs) / len(probs)
                except:
                    weight = 0
                

                if weight == 1:
                    weight = weight * 100

                if weight >= max_weight:
                    max_weight = weight
                    selected_activity = proposed_activity
            else:
                weight = 0

            weights.append(weight)
                

        # fallback logic if none of the proposals could get a weight
        if np.array(weights).sum() == 0:
            weights = np.array(weights) + 0.001
            weights = weights.tolist()

        selected_activity = random.choices(proposals, weights=weights, k=1)[0].activity
        print("Pre-selection:", selected_activity)
        
        min_end_time = datetime.max.replace(tzinfo=timezone.utc)
        i = -1
        for p in proposals:
            if selected_activity:
                if p.activity != selected_activity:
                    continue
            if p.end_time <= min_end_time:
                min_end_time = p.end_time
                i = proposals.index(p) 

        return proposals[i]

       
        


    def perform_proposal(self, proposal: Proposal, rules: ActivityRules):
        """
        Apply the selected proposal by updating simulation state:
        - Advance case time
        - Update performed activities
        - Update agent availability
        - Handle post-condition obligations
        - Log the event

        Parameters:
        - proposal: The selected Proposal to execute
        - rules: ActivityRules object to handle post-conditions

        Side Effects:
        - Updates case and agent state in-place
        - Appends a LogEntry to self.log
        """
        case = proposal.case
        agent = proposal.agent
        activity = proposal.activity

        # 1. Advance simulation time for the case
        case.current_time = proposal.end_time

        # 2. Mark activity as done
        case.performed.append(activity)
        case.agent_pairing.append(agent)
        case.allowed_activity_number[activity] = case.allowed_activity_number[activity] - 1

        # 3. Register the agent as busy during this interval
        agent.busy_windows.append((proposal.start_time, proposal.end_time))

        # 4. Add post-condition obligations
        for post_act in rules.post_conditions.get(activity, []):
            case.add_obligation(post_act)

        # 5. Remove executed action from outstanding_obligations
        if activity in case.outstanding_obligations:
            case.remove_obligation(activity)

        # 6. Log the action
        entry = LogEntry(case_id=case.case_id, agent_id=agent.agent_id,
                        activity=activity, start=proposal.start_time, end=proposal.end_time)
        self.log.append(entry)

        # 7. Optionally check if case is now finished
        if rules.is_case_end(activity, case.outstanding_obligations):
            case.done = True

        
        # 8. Lock XOR decision if needed
        # for anchor, groups in rules.xor_rules.items():
        #     for group_index, group in enumerate(groups):
        #         if activity in group:
        #             case.xor_windows.setdefault(anchor, []).append(group_index)
        #             # print(f"📌 XOR decision made: {anchor} → group {group_index}")
        #             break


    
    def tick(self, calendars: dict):
        """
        Perform one tick of the simulation; push the simulation forward by one step:
        - Collects all valid proposals across agents and active cases.
        - Selects one proposal to execute (currently random).
        - Applies the proposal, updating case and agent state.
        
        Parameters:
        - calendars: Dict[agent_id] → RCalendar

        Returns:
        - True if a proposal was executed
        - False if no valid proposals (simulation may need to advance time)

        Notes:
        - No concurrent activities available
        """

        all_proposals = []

        for case in self.cases:
            if case.done:
                continue
            available_activities = case.get_available_activities(self.rules)
            print(f"Available activities: {available_activities}")
            print(f"Current time {case.current_time}")
            for agent in self.agents:
                proposals = self.make_proposals(agent, case, self.rules, self.durations, calendars, available_activities)
                all_proposals.extend(proposals)

            if not all_proposals:
                return False
            
            selected = self.select_proposal(all_proposals)
            print(f"Selected activity: {selected.activity} by agent {selected.agent.agent_id}")
        
            self.perform_proposal(selected, self.rules)
       
        return True
    


    def local_mcts_tick(self, calendars, budget: int = 100) -> bool:
        """
        Perform one scheduling tick on all active cases using Local MCTS.
        :param calendars: mapping of agent_id to RCalendar
        :param budget: number of rollouts per decision
        :return: True if any proposal was executed; False otherwise
        """
        did_something = False

        active_cases = [c for c in self.cases if not c.done and c.get_available_activities(self.rules)]

        for case in active_cases:
            available_activities = case.get_available_activities(self.rules)
            all_proposals = []

            for agent in self.agents:
                proposals = self.make_proposals(agent, case, self.rules, self.durations, calendars, available_activities)
                all_proposals.extend(proposals)

            if not all_proposals:
                continue

            # simulation = deepcopy(self)
            # simulation.calendars = calendars  # pass calendars into the clone

            exploration_constant = 1
            budget = 300

            best_prop = mcts_select(self, all_proposals, budget, exploration_constant)
            print(f"👆🏼 Best proposal: {best_prop.agent.agent_id} - {best_prop.activity} - {best_prop.end_time}")


            # Map agent and case from real simulation
            real_case = case  # already in context
            agent_map = {a.agent_id: a for a in self.agents}
            real_agent = agent_map[best_prop.agent.agent_id]

            # Create equivalent proposal for the real sim context
            real_proposal = Proposal(
                case=real_case,
                agent=real_agent,
                activity=best_prop.activity,
                start_time=best_prop.start_time,
                duration=best_prop.duration
            )

            self.perform_proposal(real_proposal, self.rules)

            # self.perform_proposal(best_prop, self.rules)
            did_something = True

        return did_something


