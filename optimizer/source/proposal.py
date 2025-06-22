import pandas as pd

class Proposal:
    """
    Agents make a Proposal object for each activity they offer to do
    """
    def __init__(self, case, agent, activity, start_time, duration):
        self.case = case
        self.agent = agent
        self.activity = activity
        self.start_time = start_time
        self.end_time = start_time + pd.Timedelta(seconds=duration)
        self.duration = duration

class LogEntry:
    def __init__(self, case_id, agent_id, activity, start, end):
        self.case_id = case_id
        self.agent_id = agent_id
        self.activity = activity
        self.start = start
        self.end = end

    def to_dict(self):
        return {
            "case_id": self.case_id,
            "agent": self.agent_id,
            "activity": self.activity,
            "start": self.start,
            "end": self.end
        }

