import random
import math
from datetime import datetime
from mesa import Agent
import scipy.stats as st
import time
import pandas as pd

class ContractorAgent(Agent):
    """
    One contractor agent to assign tasks using the contraction net protocol
    """
    def __init__(self, unique_id, model, activities, transition_probabilities, agent_activity_mapping):
        super().__init__(unique_id, model)
        self.activities = activities
        self.transition_probabilities = transition_probabilities
        self.agent_activity_mapping = agent_activity_mapping
        self.model = model
        self.current_activity_index = None
        self.activity_performed = False


    def step(self, scheduler, agent_keys, cases):
        method = "step"
        # If the agent list contains just the contractor and one resource, no need for complex sorting
        if len(agent_keys) <= 2:
            sorted_agent_keys = agent_keys[1:]
        else:
            agent_keys = agent_keys[1:] # exclude contractor agent
            # 1) sort by specialism
            def get_key_length(key):
                return len(self.agent_activity_mapping[key])

            if isinstance(agent_keys[0], list):
                sorted_agent_keys = []
                for agent_list in agent_keys:
                    sorted_agent_keys.append(sorted(agent_list, key=get_key_length))
            else:
                sorted_agent_keys = sorted(agent_keys, key=get_key_length)
            
            # 2) sort by next availability
            sorted_agent_keys = self.sort_agents_by_availability(sorted_agent_keys)
                
            if self.model.central_orchestration == False:
                # 3) sort by transition probs
                current_agent = self.case.previous_agent
                if current_agent != -1:
                    current_activity = self.case.activities_performed[-1]
                    next_activity = self.activities[self.new_activity_index]
                    
                    if current_agent in self.model.agent_transition_probabilities:
                        if current_activity in self.model.agent_transition_probabilities[current_agent]:
                            current_probabilities = {}
                            for agent in sorted_agent_keys:
                                if agent in self.model.agent_transition_probabilities[current_agent][current_activity]:
                                    prob = self.model.agent_transition_probabilities[current_agent][current_activity][agent].get(next_activity, 0)
                                    current_probabilities[agent] = prob
                                else:
                                    current_probabilities[agent] = 0
                            
                            sorted_agent_keys_ = [agent for agent in sorted_agent_keys if current_probabilities.get(agent, 0) > 0]
                            if len(sorted_agent_keys_) > 0:
                                probabilities = [current_probabilities[agent] for agent in sorted_agent_keys_]
                                sorted_agent_keys = random.choices(sorted_agent_keys_, weights=probabilities, k=len(sorted_agent_keys_))
                            else:
                                sorted_agent_keys = sorted_agent_keys
        
        last_possible_agent = False

        if sorted_agent_keys and isinstance(sorted_agent_keys[0], list):
            for agent_key in sorted_agent_keys:
                for inner_key in agent_key:
                    if inner_key == agent_key[-1]:
                        last_possible_agent = True
                    if inner_key in scheduler._agents:
                        if self.activity_performed:
                            break
                        else:
                            current_timestamp = self.get_current_timestamp(inner_key, parallel_activity=True)
                            getattr(scheduler._agents[inner_key], method)(last_possible_agent, parallel_activity=True, current_timestamp=current_timestamp, perform_multitask=False)
        else:
            for agent_key in sorted_agent_keys:
                if agent_key == sorted_agent_keys[-1]:
                    last_possible_agent = True
                if agent_key in scheduler._agents:
                    if self.activity_performed:
                        break
                    else:
                        current_timestamp = self.get_current_timestamp(agent_key)
                        getattr(scheduler._agents[agent_key], method)(last_possible_agent, parallel_activity=False, current_timestamp=current_timestamp, perform_multitask=False)
        self.activity_performed = False

    def sort_agents_by_availability(self, sorted_agent_keys):
        if not sorted_agent_keys:
            return []
        if isinstance(sorted_agent_keys[0], list):
            sorted_agent_keys_new = []
            for agent_list in sorted_agent_keys:
                sorted_agent_keys_new.append(sorted(agent_list, key=lambda x: self.model.agents_busy_until[x]))
        else:
            sorted_agent_keys_new = sorted(sorted_agent_keys, key=lambda x: self.model.agents_busy_until[x])
        return sorted_agent_keys_new
    
    def get_current_timestamp(self, agent_id, parallel_activity=False):
        if parallel_activity == False:
            return self.case.current_timestamp
        else:
            return self.case.timestamp_before_and_gateway

    def get_activity_duration(self, agent, activity):
        activity_distribution = self.model.activity_durations_dict[agent][activity]
        if activity_distribution.type.value == "expon":
            scale = activity_distribution.mean - activity_distribution.min
            if scale < 0.0:
                print("Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value.")
                scale = activity_distribution.mean
            return st.expon.rvs(loc=activity_distribution.min, scale=scale, size=1)[0]
        elif activity_distribution.type.value == "gamma":
            return st.gamma.rvs(pow(activity_distribution.mean, 2) / activity_distribution.var, loc=0, scale=activity_distribution.var / activity_distribution.mean, size=1)[0]
        elif activity_distribution.type.value == "norm":
            return st.norm.rvs(loc=activity_distribution.mean, scale=activity_distribution.std, size=1)[0]
        elif activity_distribution.type.value == "uniform":
            return st.uniform.rvs(loc=activity_distribution.min, scale=activity_distribution.max - activity_distribution.min, size=1)[0]
        elif activity_distribution.type.value == "lognorm":
            pow_mean = pow(activity_distribution.mean, 2)
            phi = math.sqrt(activity_distribution.var + pow_mean)
            mu = math.log(pow_mean / phi)
            sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
            return st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=1)[0]
        elif activity_distribution.type.value == "fix":
            return activity_distribution.mean
        return 0
    
    def sample_starting_activity(self):
        if not hasattr(self, '_start_activities_dist'):
            df = self.model.data
            first_activities = (df.sort_values(['case_id', 'start_timestamp', 'end_timestamp']).groupby('case_id')['activity_name'].first())
            if "Start" in first_activities.values or "start" in first_activities.values:
                self._start_activities_dist = ("Start" if "Start" in first_activities.values else "start", None)
            else:
                total_cases = len(df['case_id'].unique())
                start_count = first_activities.value_counts() / total_cases
                self._start_activities_dist = (list(start_count.index), list(start_count.values))
        activities, weights = self._start_activities_dist
        if isinstance(activities, str):
            return activities
        else:
            return random.choices(activities, weights=weights, k=1)[0]
    
    def check_for_other_possible_next_activity(self, next_activity):
        possible_other_next_activities = []
        for key, value in self.model.prerequisites.items():
            for item in value:
                if not isinstance(item, list):
                    if item == next_activity: possible_other_next_activities.append(key)
                else:
                    if any(next_activity in sublist for sublist in item):
                        if all(val in self.case.activities_performed for val in item):
                            possible_other_next_activities.append(key)
        return possible_other_next_activities

    def check_if_all_preceding_activities_performed(self, activity):
        for key, value in self.model.prerequisites.items():
            if activity == key:
                return all(val in self.case.activities_performed for val in value)
        return False

    # def get_potential_agents(self, case):
    #     self.case = case
    #     case_ended = False

    #     if case.get_last_activity() is None:
    #         next_activity = self.sample_starting_activity()
    #         potential_agents = [key for key, value in self.agent_activity_mapping.items() if next_activity in value]
    #         potential_agents.insert(0, 9999)
    #         self.new_activity_index = self.activities.index(next_activity)
    #         return potential_agents, False

    #     prefix = self.case.activities_performed
    #     activity_list, probabilities = None, None
        
    #     if self.model.central_orchestration:
    #         while tuple(prefix) not in self.transition_probabilities:
    #             prefix = prefix[1:] if len(prefix) > 1 else []
    #             if not prefix: break
    #         if tuple(prefix) in self.transition_probabilities:
    #             activity_list = list(self.transition_probabilities[tuple(prefix)].keys())
    #             probabilities = list(self.transition_probabilities[tuple(prefix)].values())
    #     else:
    #         while tuple(prefix) not in self.transition_probabilities or self.case.previous_agent not in self.transition_probabilities.get(tuple(prefix), {}):
    #             prefix = prefix[1:] if len(prefix) > 1 else []
    #             if not prefix: break
    #         if tuple(prefix) in self.transition_probabilities and self.case.previous_agent in self.transition_probabilities[tuple(prefix)]:
    #             activity_list = list(self.transition_probabilities[tuple(prefix)][self.case.previous_agent].keys())
    #             probabilities = list(self.transition_probabilities[tuple(prefix)][self.case.previous_agent].values())

    #     if not activity_list:
    #         return None, True

    #     if self.model.params.get('execution_type') == 'greedy_optimize':
    #         progress_scores = self.model.params.get('activity_progress_scores', {})
    #         max_time = self.model.params.get('max_time_per_step', 1.0)
    #         agent_costs = self.model.params.get('agent_costs', {})
    #         max_cost = self.model.params.get('max_task_cost', 1.0)
    #         weights = self.model.params.get('optimizer_weights', {})

    #         best_choice = {'activity': None, 'agent_id': None, 'score': float('inf')}

    #         for possible_activity in activity_list:
    #             if possible_activity == 'zzz_end': continue
    #             capable_agents = [key for key, value in self.agent_activity_mapping.items() if possible_activity in value]
                
    #             for agent_id in capable_agents:
    #                 agent_availability = self.model.agents_busy_until.get(agent_id, self.model.params['start_timestamp'])
    #                 start_time = max(case.current_timestamp, agent_availability)
                    
    #                 activity_duration_dist = self.model.activity_durations_dict[agent_id][possible_activity]
    #                 work_duration_seconds = activity_duration_dist.mean

    #                 agent_calendar = self.model.calendars.get(agent_id)
    #                 if agent_calendar and work_duration_seconds > 0:
    #                     wall_clock_duration_seconds = agent_calendar.find_idle_time(start_time, work_duration_seconds)
    #                     end_time = start_time + pd.Timedelta(seconds=wall_clock_duration_seconds)
    #                 else:
    #                     end_time = start_time + pd.Timedelta(seconds=work_duration_seconds)
                    
    #                 resource_name = self.model.resources[agent_id]
    #                 cost_per_hour = agent_costs.get(resource_name, 0)
    #                 task_cost = (work_duration_seconds / 3600) * cost_per_hour
                    
    #                 duration_for_score = (end_time - case.current_timestamp).total_seconds()
    #                 normalized_time = min(1.0, duration_for_score / max_time)
    #                 normalized_cost = min(1.0, task_cost / max_cost)
    #                 progress_score = progress_scores.get(possible_activity, 0)
                    
    #                 total_score = (weights.get('time', 0.0) * normalized_time) + \
    #                               (weights.get('progress', 0.0) * (1 - progress_score)) + \
    #                               (weights.get('cost', 0.0) * normalized_cost)

    #                 if total_score < best_choice['score']:
    #                     best_choice = {'activity': possible_activity, 'agent_id': agent_id, 'score': total_score}
            
    #         if best_choice['activity'] is None:
    #             next_activity = 'zzz_end'
    #         else:
    #             next_activity = best_choice['activity']
    #             chosen_agent_id = best_choice['agent_id']

    #         if next_activity == 'zzz_end': return None, True
    #         self.new_activity_index = self.activities.index(next_activity)
    #         return [9999, chosen_agent_id], False

    #     else: # Default simulation behavior
    #         next_activity = random.choices(activity_list, weights=probabilities, k=1)[0]
    #         if next_activity == 'zzz_end': return None, True
    #         self.new_activity_index = self.activities.index(next_activity)
    #         potential_agents = [key for key, value in self.agent_activity_mapping.items() if next_activity in value]
    #         potential_agents.insert(0, 9999)
    #         return potential_agents, False


    def get_potential_agents(self, case):
        """
        Finds the single best (activity, agent) pair to execute next by
        holistically evaluating every possible combination based on a weighted score
        of progress, task cost, and waiting cost.
        """
        self.case = case
        case_ended = False

        # Handle the very first activity in a case
        if case.get_last_activity() is None:
            next_activity = self.sample_starting_activity()
            # For the first activity, we don't optimize the agent choice,
            # we let the original mechanism handle it.
            potential_agents = [key for key, value in self.agent_activity_mapping.items() if next_activity in value]
            potential_agents.insert(0, 9999)
            self.new_activity_index = self.activities.index(next_activity)
            return potential_agents, False

        # --- Get the list of possible next activities based on the process model ---
        prefix = self.case.activities_performed
        activity_list, probabilities = None, None
        
        if self.model.central_orchestration:
            while tuple(prefix) not in self.transition_probabilities:
                prefix = prefix[1:] if len(prefix) > 1 else []
                if not prefix: break
            if tuple(prefix) in self.transition_probabilities:
                activity_list = list(self.transition_probabilities[tuple(prefix)].keys())
                probabilities = list(self.transition_probabilities[tuple(prefix)].values())
        else:
            while tuple(prefix) not in self.transition_probabilities or self.case.previous_agent not in self.transition_probabilities.get(tuple(prefix), {}):
                prefix = prefix[1:] if len(prefix) > 1 else []
                if not prefix: break
            if tuple(prefix) in self.transition_probabilities and self.case.previous_agent in self.transition_probabilities[tuple(prefix)]:
                activity_list = list(self.transition_probabilities[tuple(prefix)][self.case.previous_agent].keys())
                probabilities = list(self.transition_probabilities[tuple(prefix)][self.case.previous_agent].values())

        # If no path forward is found, end the case
        if not activity_list:
            return None, True

        # ===== DECISION LOGIC: HOLISTIC OPTIMIZATION OR DEFAULT SIMULATION =====
        if self.model.params.get('execution_type') == 'greedy_optimize':
            
            # --- Retrieve all necessary optimizer parameters from the model ---
            progress_scores = self.model.params.get('activity_progress_scores', {})
            agent_costs = self.model.params.get('agent_costs', {})
            max_task_cost = self.model.params.get('max_task_cost', 1.0)
            cost_of_delay_per_hour = self.model.params.get('cost_of_delay_per_hour', 0)
            max_wait_cost = self.model.params.get('max_wait_cost', 1.0)
            weights = self.model.params.get('optimizer_weights', {})

            best_choice = {'activity': None, 'agent_id': None, 'score': float('inf')}

            # === HOLISTIC EVALUATION LOOP: Iterate through every possible (activity, agent) pair ===
            for possible_activity in activity_list:
                if possible_activity == 'zzz_end':
                    continue
                
                capable_agents = [key for key, value in self.agent_activity_mapping.items() if possible_activity in value]
                
                for agent_id in capable_agents:
                    # --- Explicitly calculate each component of the score for this specific pair ---
                    
                    # 1. Calculate Wait Time and Wait Cost
                    agent_availability = self.model.agents_busy_until.get(agent_id, self.model.params['start_timestamp'])
                    wait_duration_seconds = max(0, (agent_availability - case.current_timestamp).total_seconds())
                    wait_cost = (wait_duration_seconds / 3600) * cost_of_delay_per_hour
                    normalized_wait_cost = min(1.0, wait_cost / max_wait_cost) if max_wait_cost > 0 else 0

                    # 2. Calculate Task Cost (cost of the work itself)
                    activity_duration_dist = self.model.activity_durations_dict[agent_id][possible_activity]
                    work_duration_seconds = activity_duration_dist.mean
                    resource_name = self.model.resources[agent_id]
                    cost_per_hour = agent_costs.get(resource_name, 0)
                    task_cost = (work_duration_seconds / 3600) * cost_per_hour
                    normalized_task_cost = min(1.0, task_cost / max_task_cost) if max_task_cost > 0 else 0

                    # 3. Calculate Progress Score
                    progress_score = progress_scores.get(possible_activity, 0)
                    
                    # --- Combine into the final score using flexible weights ---
                    total_score = (weights.get('progress', 0.0) * (1 - progress_score)) + \
                                  (weights.get('task_cost', 0.0) * normalized_task_cost) + \
                                  (weights.get('wait_cost', 0.0) * normalized_wait_cost)

                    # If this pair has the best score so far, record it
                    if total_score < best_choice['score']:
                        best_choice = {'activity': possible_activity, 'agent_id': agent_id, 'score': total_score}
            
            # --- After checking all pairs, commit to the best one found ---
            if best_choice['activity'] is None:
                next_activity = 'zzz_end'
            else:
                next_activity = best_choice['activity']
                chosen_agent_id = best_choice['agent_id']
                
            if next_activity == 'zzz_end': 
                return None, True

            self.new_activity_index = self.activities.index(next_activity)
            # Return a list with ONLY the chosen agent for the scheduler to use
            return [9999, chosen_agent_id], False

        else: # Default simulation behavior (remains unchanged)
            next_activity = random.choices(activity_list, weights=probabilities, k=1)[0]
            if next_activity == 'zzz_end': 
                return None, True
            
            self.new_activity_index = self.activities.index(next_activity)
            potential_agents = [key for key, value in self.agent_activity_mapping.items() if next_activity in value]
            potential_agents.insert(0, 9999)
            return potential_agents, False