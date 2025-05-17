import random
import pandas as pd
from proposal import Proposal, LogEntry
from activity_rules import ActivityRules
from utils import sample_from_distribution, shift_activity_start_to_next_valid_window

# Add project root to sys.path
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join("../../..")))
# from source.utils import sample_from_distribution

class Simulation:
    def __init__(self, agents, cases, rules, durations, case_arrivals):
        self.agents = agents
        self.cases = cases
        self.rules = rules
        self.durations = durations
        self.case_arrivals = case_arrivals
        self.log = []

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
        - Doesn’t handle fallback logic if activity doesn't fit in calendar (e.g., shifting start time).
        - Doesn’t yet consider multitasking or interruptions.
        """

        proposals = []  
        # if 'Check credit history' in available_activities:
        #     available_activities.remove('Check credit history')  
        # if 'Appraise property' in available_activities:
        #     available_activities.remove('Appraise property')  
        # if 'AML check' in available_activities:
        #     available_activities.remove('AML check')  
        for act in available_activities:
            if act in agent.capable_activities:
                if not rules.is_activity_allowed(act, case):
                    continue
                try:
                    dist = durations[agent.agent_id][act]
                    if not dist:  # Skip empty distributions
                        continue
                    duration = sample_from_distribution(dist)
                except (KeyError, AttributeError):
                    continue  # Duration missing or malformed

                start_time = max(agent.available_at, case.current_time)

                # ✅ Check calendar constraints before proposing
                calendar = calendars.get(agent.agent_id, None)
                if calendar is not None:
                    calendar_json = calendar.intervals_to_json()
                    if not self.is_within_calendar(start_time, duration, calendar_json):
                        print(f"Violated calendar rules of agent {agent.agent_id} for time {start_time} to {start_time + pd.Timedelta(seconds=duration)}, agent seems available at {agent.available_at}")
                        # ⏩ Try to shift the start time to next valid calendar slot
                        new_start_time = shift_activity_start_to_next_valid_window(start_time, duration, calendar_json)
                        if new_start_time is None:
                            print(f"⚠️ Could not find a future shift for agent {agent.agent_id}. Skipping proposal.")
                            continue
                        print(f"⏩  Shifted activity start time to {new_start_time}")
                        start_time = new_start_time
                else:
                    print(f"Warning: No calendar found for agent {agent.agent_id}, assuming always available.")
                proposals.append(Proposal(case, agent, act, start_time, duration))

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
    def select_proposal(self, proposals: list) -> Proposal:
        """
        Select one proposal with weighted random logic:
        - Boosts proposals that fulfill obligations
        - Penalizes proposals that repeat the case's last activity

        Returns:
        - A single Proposal object
        """

        if not proposals:
            return None

        weights = []
        for p in proposals:
            weight = 1.0

            # 📌 Boost if it's an obligation
            if p.activity in p.case.outstanding_obligations:
                weight *= 10  # Total = 10

            # 🚫 Penalize if it's an immediate repeat
            elif len(p.case.performed) > 0 and p.activity == p.case.performed[-1]:
                weight *= 0.01  # 90% penalty, Total = 0.1
            
            else:
                weight *= 0.05

            weights.append(weight)

            

        # for p, w in zip(proposals, weights):
            # print(f"Weighted option: {p.activity} by {p.agent.agent_id} → weight {w}")

        
        return random.choices(proposals, weights=weights, k=1)[0]



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

        # 3. Update agent's availability
        agent.available_at = proposal.end_time

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
        for anchor, groups in rules.xor_rules.items():
            for group_index, group in enumerate(groups):
                if activity in group:
                    case.xor_windows.setdefault(anchor, []).append(group_index)
                    print(f"📌 XOR decision made: {anchor} → group {group_index}")
                    break


    
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


