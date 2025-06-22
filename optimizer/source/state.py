import pandas as pd
import pytz
from activity_rules import ActivityRules  # only needed if used inside methods like get_available_activities
from collections import defaultdict


# State Classes

class Case:
    """
    A Case object represents a case of a simulated or raw log. 
    It knows its:
        *  erformed activities,
        * current_timestamp,
        * post_condition_queue,
        * outstanding_obligations.
    """
    
    def __init__(self, case_id, arrival_time, allowed_activity_mapping, xor_decisions):
        self.case_id = case_id
        self.start_time = arrival_time
        self.current_time = arrival_time
        self.performed = []
        self.agent_pairing = []
        self.outstanding_obligations = []
        self.allowed_activity_number = allowed_activity_mapping
        self.done = False
        self.xor_decisions = xor_decisions

        self.xor_windows = defaultdict(list)  # in Case.__init__()


    def get_available_activities(self, rules: ActivityRules):
        """
        Gets available activities based on pre-conditions and prioritizes them based on outstanding 
        obligations list.
        Logic found in ActivityRules class
        """

        all_available = rules.get_available_activities(self.performed, self.xor_windows, self.allowed_activity_number)

        # Sort so obligations come first, preserving order
        prioritized = [a for a in self.outstanding_obligations if a in all_available]
        rest = [a for a in all_available if a not in prioritized]

        return prioritized + rest
    

    # post-condition related functions
    
    def add_obligation(self, activity):
        """
        Adds a post-condition activity to the obligations list
        """
        if activity not in self.outstanding_obligations:
            self.outstanding_obligations.append(activity)

    def remove_obligation(self, activity):
        """
        Removes a post-condition activity from the obligations list
        """
        if activity in self.outstanding_obligations:
            self.outstanding_obligations.remove(activity)


class Agent:
    """
    An Agent object represents an agent mined from a simulated or raw log. 
    It knows:
        * What activities it can perform,
        * When it's available,
        * Proposes actions it could do for a case.
    """
    def __init__(self, agent_id, capable_activities, calendar):
        self.agent_id = agent_id
        self.capable_activities = set(capable_activities)
        self.busy_windows = []
        # self.available_at = pd.Timestamp.min.replace(tzinfo=pytz.UTC)  # updated as tasks are performed
        self.calendar = calendar

    def is_available_at(self, time):
        """
        Checks whether the agent is available at a specific timestamp.
        
        Parameters:
        - time: pd.Timestamp to check

        Returns:
        - True if the agent is free at that time
        - False if the agent is currently busy
        """

        for start, end in self.busy_windows:
            if start <= time < end:
                return False
        return True
    
    def get_next_availability(self, after_time):
        """
        Returns the next timestamp after `after_time` when the agent is available,
        respecting both busy windows and calendar constraints.

        Parameters:
        - after_time: pd.Timestamp to start searching from

        Returns:
        - pd.Timestamp of next available moment
        """
        # Sort the busy windows and calendar for consistency
        self.busy_windows.sort()
        self.calendar.sort()

        for work_start, work_end in self.calendar:
            # Skip calendar windows before the requested time
            if work_end <= after_time:
                continue

            # Determine the actual search start time
            current_time = max(after_time, work_start)

            while current_time < work_end:
                if self.is_available_at(current_time):
                    return current_time
                # Advance time by smallest practical resolution (e.g., 1 minute)
                current_time += pd.Timedelta(minutes=1)

        # If no availability is found
        return None