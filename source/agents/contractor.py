import random
import math
from datetime import datetime
from mesa import Agent
import scipy.stats as st
import time
import pandas as pd # <-- ADD THIS IMPORT



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
        agent_keys = agent_keys[1:] # exclude contractor agent here as we only want to perform resource agent steps
        # 1) sort by specialism
        # bring the agents in an order to first ask the most specialized agents to not waste agent capacity for future cases -> principle of specialization
        def get_key_length(key):
            return len(self.agent_activity_mapping[key])

        # Sort the keys using the custom key function
        if isinstance(agent_keys[0], list):
            sorted_agent_keys = []
            for agent_list in agent_keys:
                sorted_agent_keys.append(sorted(agent_list, key=get_key_length))
        else:
            sorted_agent_keys = sorted(agent_keys, key=get_key_length)
        # print(f"Agents sorted by specialism: {sorted_agent_keys}")
        
        # # 2) sort by next availability
        sorted_agent_keys = self.sort_agents_by_availability(sorted_agent_keys)
            
        if self.model.central_orchestration == False:
            # 3) sort by transition probs
            current_agent = self.case.previous_agent
            if current_agent != -1:
                current_activity = self.case.activities_performed[-1]
                next_activity = self.activities[self.new_activity_index]  # Get the next activity
                
                # Navigate through the nested dictionary structure
                if current_agent in self.model.agent_transition_probabilities:
                    if current_activity in self.model.agent_transition_probabilities[current_agent]:
                        # Create a dictionary to store probabilities for each potential next agent
                        current_probabilities = {}
                        for agent in sorted_agent_keys:
                            # Sum up probabilities for the specific next activity across all agents
                            if agent in self.model.agent_transition_probabilities[current_agent][current_activity]:
                                prob = self.model.agent_transition_probabilities[current_agent][current_activity][agent].get(next_activity, 0)
                                current_probabilities[agent] = prob
                            else:
                                current_probabilities[agent] = 0
                        # print(f"current_activity: {current_activity}")
                        # print(f"current_agent: {current_agent}")
                        # print(f"current_probabilities: {self.model.agent_transition_probabilities[current_agent][current_activity]}")
                        # print(f"sorted_agent_keys before: {sorted_agent_keys}")
                        # Filter out agents with zero probability and sort remaining agents
                        sorted_agent_keys_ = [
                            agent for agent in sorted_agent_keys 
                            if current_probabilities.get(agent, 0) > 0
                        ]
                        if len(sorted_agent_keys_) > 0:
                            # sorted_agent_keys = sorted_agent_keys_
                            # sorted_agent_keys = sorted(sorted_agent_keys, 
                            #                      key=lambda x: current_probabilities.get(x, 0), 
                            #                      reverse=True)
                            probabilities = [current_probabilities[agent] for agent in sorted_agent_keys_]
                            sorted_agent_keys = random.choices(
                                sorted_agent_keys_,
                                weights=probabilities,
                                k=len(sorted_agent_keys_)
                            )
                        else:
                            sorted_agent_keys = sorted_agent_keys
                        # print(f"sorted_agent_keys after: {sorted_agent_keys}")
                        end_time = time.time()
                        
        # print(f"sorted_agent_keys: {sorted_agent_keys}")

        last_possible_agent = False

        if isinstance(sorted_agent_keys[0], list):
            for agent_key in sorted_agent_keys:
                for inner_key in agent_key:
                    if inner_key == agent_key[-1]:
                        last_possible_agent = True
                    if inner_key in scheduler._agents:
                        if self.activity_performed:
                            break
                        else:
                            current_timestamp = self.get_current_timestamp(inner_key, parallel_activity=True)
                            perform_multitask = False
                            getattr(scheduler._agents[inner_key], method)(last_possible_agent, 
                                                                          parallel_activity=True, current_timestamp=current_timestamp, perform_multitask=perform_multitask)
        else:
            for agent_key in sorted_agent_keys:
                if agent_key == sorted_agent_keys[-1]:
                    last_possible_agent = True
                if agent_key in scheduler._agents:
                    if self.activity_performed:
                        break
                    else:
                        current_timestamp = self.get_current_timestamp(agent_key)
                        perform_multitask = False
                        getattr(scheduler._agents[agent_key], method)(last_possible_agent, parallel_activity=False, current_timestamp=current_timestamp, perform_multitask=perform_multitask)
        self.activity_performed = False

    def sort_agents_by_availability(self, sorted_agent_keys):
        # print(f"agents before sorting: {sorted_agent_keys}")
        # for agent in sorted_agent_keys:
            # print(f"agent: {agent}, busy until: {self.model.agents_busy_until[agent]}")
        if isinstance(sorted_agent_keys[0], list):
            sorted_agent_keys_new = []
            for agent_list in sorted_agent_keys:
                sorted_agent_keys_new.append(sorted(agent_list, key=lambda x: self.model.agents_busy_until[x]))
        else:
            sorted_agent_keys_new = sorted(sorted_agent_keys, key=lambda x: self.model.agents_busy_until[x])


        # print(f"agents after sorting: {sorted_agent_keys_new}")
        return sorted_agent_keys_new
    
    
    def get_current_timestamp(self, agent_id, parallel_activity=False):
        if parallel_activity == False:
            current_timestamp = self.case.current_timestamp
        else:
            current_timestamp = self.case.timestamp_before_and_gateway

        return current_timestamp


    def get_activity_duration(self, agent, activity):
        activity_distribution = self.model.activity_durations_dict[agent][activity]
        if activity_distribution.type.value == "expon":
            scale = activity_distribution.mean - activity_distribution.min
            if scale < 0.0:
                print("Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value.")
                scale = activity_distribution.mean
            activity_duration = st.expon.rvs(loc=activity_distribution.min, scale=scale, size=1)[0]
        elif activity_distribution.type.value == "gamma":
            activity_duration = st.gamma.rvs(
                pow(activity_distribution.mean, 2) / activity_distribution.var,
                loc=0,
                scale=activity_distribution.var / activity_distribution.mean,
                size=1,
            )[0]
        elif activity_distribution.type.value == "norm":
            activity_duration = st.norm.rvs(loc=activity_distribution.mean, scale=activity_distribution.std, size=1)[0]
        elif activity_distribution.type.value == "uniform":
            activity_duration = st.uniform.rvs(loc=activity_distribution.min, scale=activity_distribution.max - activity_distribution.min, size=1)[0]
        elif activity_distribution.type.value == "lognorm":
            pow_mean = pow(activity_distribution.mean, 2)
            phi = math.sqrt(activity_distribution.var + pow_mean)
            mu = math.log(pow_mean / phi)
            sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
            activity_duration = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=1)[0]
        elif activity_distribution.type.value == "fix":
            activity_duration = activity_distribution.mean

        return activity_duration
    
    def sample_starting_activity(self):
        """
        Sample the activity that starts the case based on the frequency of starting activities in the train log
        """
        
        # Cache the start activities if not already cached
        if not hasattr(self, '_start_activities_dist'):
            # Get first activity for each case more efficiently
            df = self.model.data
            # Sort once and get first activity for each case
            first_activities = (df.sort_values(['case_id', 'start_timestamp', 'end_timestamp'])
                            .groupby('case_id')['activity_name']
                            .first())
            
            # Handle Start/start cases
            if "Start" in first_activities.values or "start" in first_activities.values:
                self._start_activities_dist = ("Start" if "Start" in first_activities.values else "start", None)
            else:
                # Calculate frequencies
                total_cases = len(df['case_id'].unique())
                start_count = first_activities.value_counts() / total_cases
                self._start_activities_dist = (list(start_count.index), list(start_count.values))

        # Use cached distribution
        activities, weights = self._start_activities_dist
        if isinstance(activities, str):  # Handle Start/start case
            sampled_activity = activities
        else:
            sampled_activity = random.choices(activities, weights=weights, k=1)[0]
        
        return sampled_activity
    
    def check_for_other_possible_next_activity(self, next_activity):
        possible_other_next_activities = []
        for key, value in self.model.prerequisites.items():
            for i in range(len(value)):
                # if values is a single list, then only ONE of the entries must have been performed already (XOR gateway)
                if not isinstance(value[i], list):
                    if value[i] == next_activity:
                        possible_other_next_activities.append(key)
                # if value contains sublists, then all of the values in the sublist must have been performed (AND gateway)
                else:
                    # if current next_activity contained in prerequisites
                    if any(next_activity in sublist for sublist in value[i]):
                        # if all prerequisites are fulfilled
                        if all(value_ in self.case.activities_performed for value_ in value[i]):
                            possible_other_next_activities.append(key)

        return possible_other_next_activities
    

    def check_if_all_preceding_activities_performed(self, activity):
        print(f"activities_performed: {self.case.activities_performed}")
        print(f"prerequisites: {self.model.prerequisites}")
        print(f"activity: {activity}")
        for key, value in self.model.prerequisites.items():
            if activity == key:
                if all(value_ in self.case.activities_performed for value_ in value):
                    return True
        return False
    
    # def get_potential_agents(self, case):
    #     """
    #     check if there already happened activities in the current case
    #         if no: current activity is usual start activity
    #         if yes: current activity is the last activity of the current case
    #     """
    #     self.case = case
    #     case_ended = False

    #     if case.get_last_activity() is None: # if first activity in case
    #         sampled_start_act = self.sample_starting_activity()
    #         next_activity = sampled_start_act
    #     else:
    #         prefix = self.case.activities_performed
    #         activity_list = None

    #         # Logic to get the list of possible next activities
    #         if self.model.central_orchestration:
    #             while tuple(prefix) not in self.transition_probabilities:
    #                 prefix = prefix[1:]
    #             activity_list = list(self.transition_probabilities[tuple(prefix)].keys())
    #             probabilities = list(self.transition_probabilities[tuple(prefix)].values())
    #         else:
    #             while tuple(prefix) not in self.transition_probabilities or self.case.previous_agent not in self.transition_probabilities[tuple(prefix)]:
    #                 prefix = prefix[1:]
    #             activity_list = list(self.transition_probabilities[tuple(prefix)][self.case.previous_agent].keys())
    #             probabilities = list(self.transition_probabilities[tuple(prefix)][self.case.previous_agent].values())

    #         # ===== DECISION LOGIC: GREEDY OPTIMIZATION OR DEFAULT SIMULATION =====
    #         if self.model.params.get('execution_type') == 'greedy_optimize':
    #             progress_scores = self.model.params.get('activity_progress_scores', {})
    #             max_time = self.model.params.get('max_time_per_step', 1.0)
    #             weights = self.model.params.get('optimizer_weights', {'time': 0.5, 'progress': 0.5})

    #             best_choice = {
    #                 'activity': None,
    #                 'score': float('inf'),
    #             }

    #             # The lookahead now considers each agent-activity pair individually
    #             for possible_activity in activity_list:
    #                 if possible_activity == 'zzz_end':
    #                     continue

    #                 progress_score = progress_scores.get(possible_activity, 0)
                    
    #                 capable_agents = [key for key, value in self.agent_activity_mapping.items() if possible_activity in value]
                    
    #                 for agent_id in capable_agents:
    #                     # --- Calculate the score for this specific (agent, activity) pair ---
    #                     agent_availability = self.model.agents_busy_until.get(agent_id, self.model.params['start_timestamp'])
                        
    #                     # This is the ACTUAL duration of the work itself
    #                     activity_duration_dist = self.model.activity_durations_dict[agent_id][possible_activity]
    #                     work_duration_seconds = activity_duration_dist.mean

    #                     # This is the ACTUAL time the case has to wait for this specific agent
    #                     wait_duration_seconds = max(0, (agent_availability - case.current_timestamp).total_seconds())

    #                     # Now we use the correct, decoupled values for scoring
    #                     # NOTE: We can use the same max_time for normalization, or create separate ones.
    #                     # For an MVP, using one is fine.
    #                     normalized_time = min(1.0, (wait_duration_seconds + work_duration_seconds) / max_time)

    #                     total_score = (weights['time'] * normalized_time) + \
    #                                 (weights['progress'] * (1 - progress_score))

    #                     if total_score < best_choice['score']:
    #                         best_choice['activity'] = possible_activity
    #                         best_choice['score'] = total_score
                
    #             next_activity = best_choice['activity']
    #             if next_activity is None:
    #                 next_activity = 'zzz_end'
    #         else: # Default simulation behavior
    #             next_activity = random.choices(activity_list, weights=probabilities, k=1)[0]

    #     # ===== COMMON LOGIC FOR ANY CHOICE =====
    #     if next_activity == 'zzz_end':
    #         return None, True

    #     self.new_activity_index = self.activities.index(next_activity)
        
    #     # ... (rest of the function, including activity_allowed checks, remains the same) ...
    #     activity_allowed = True
    #     number_occurence_of_next_activity = self.case.activities_performed.count(next_activity) + 1
    #     if number_occurence_of_next_activity > self.model.max_activity_count_per_case.get(next_activity, float('inf')):
    #         activity_allowed = False
    #         possible_other_next_activities = self.check_for_other_possible_next_activity(next_activity)
    #         if len(possible_other_next_activities) > 0:
    #             next_activity = random.choice(possible_other_next_activities)
    #             self.new_activity_index = self.activities.index(next_activity)
    #             activity_allowed = True
    #             if next_activity == 'zzz_end':
    #                 return None, True
    #         else:
    #             activity_allowed = True

    #     if not activity_allowed:
    #         return None, False

    #     potential_agents = [key for key, value in self.agent_activity_mapping.items() if next_activity in value]
    #     potential_agents.insert(0, 9999)

    #     return potential_agents, case_ended

    def get_potential_agents(self, case):
        """
        Finds the single best (activity, agent) pair to execute next by
        holistically evaluating every possible combination.
        """
        self.case = case
        case_ended = False

        if case.get_last_activity() is None:
            # Logic for starting activity remains the same
            sampled_start_act = self.sample_starting_activity()
            next_activity = sampled_start_act
            # For the first activity, we still need to find potential agents
            potential_agents = [key for key, value in self.agent_activity_mapping.items() if next_activity in value]
            potential_agents.insert(0, 9999)
            self.new_activity_index = self.activities.index(next_activity)
            return potential_agents, False

        # --- Get the list of possible next activities ---
        prefix = self.case.activities_performed
        activity_list = None
        probabilities = None # Keep probabilities for the default mode

        if self.model.central_orchestration:
            while tuple(prefix) not in self.transition_probabilities:
                prefix = prefix[1:]
            activity_list = list(self.transition_probabilities[tuple(prefix)].keys())
            probabilities = list(self.transition_probabilities[tuple(prefix)].values())
        else:
            while tuple(prefix) not in self.transition_probabilities or self.case.previous_agent not in self.transition_probabilities[tuple(prefix)]:
                prefix = prefix[1:]
            activity_list = list(self.transition_probabilities[tuple(prefix)][self.case.previous_agent].keys())
            probabilities = list(self.transition_probabilities[tuple(prefix)][self.case.previous_agent].values())

        # ===== DECISION LOGIC: HOLISTIC OPTIMIZATION OR DEFAULT SIMULATION =====
        if self.model.params.get('execution_type') == 'greedy_optimize':
            
            # --- Retrieve optimizer parameters from the model ---
            progress_scores = self.model.params.get('activity_progress_scores', {})
            max_time = self.model.params.get('max_time_per_step', 1.0)
            weights = self.model.params.get('optimizer_weights', {'time': 0.5, 'progress': 0.5})

            best_choice = {
                'activity': None,
                'agent_id': None,
                'score': float('inf'), 
            }

            # === HOLISTIC EVALUATION LOOP: Iterate through all (activity, agent) pairs ===
            for possible_activity in activity_list:
                if possible_activity == 'zzz_end':
                    continue
                
                capable_agents = [key for key, value in self.agent_activity_mapping.items() if possible_activity in value]
                
                for agent_id in capable_agents:
                    # --- This is the core of Solution 3: Evaluate each specific pair ---
                    agent_availability = self.model.agents_busy_until.get(agent_id, self.model.params['start_timestamp'])
                    start_time = max(case.current_timestamp, agent_availability)
                    
                    activity_duration_dist = self.model.activity_durations_dict[agent_id][possible_activity]
                    work_duration_seconds = activity_duration_dist.mean

                    # --- Use calendar-aware logic to get the REAL end time for this pair ---
                    agent_calendar = self.model.calendars.get(agent_id)
                    if agent_calendar and work_duration_seconds > 0:
                        wall_clock_duration_seconds = agent_calendar.find_idle_time(start_time, work_duration_seconds)
                        end_time = start_time + pd.Timedelta(seconds=wall_clock_duration_seconds)
                    else:
                        end_time = start_time + pd.Timedelta(seconds=work_duration_seconds)
                    
                    # --- Calculate the final score for THIS SPECIFIC PAIR ---
                    duration_for_score = (end_time - case.current_timestamp).total_seconds()
                    normalized_time = min(1.0, duration_for_score / max_time)
                    progress_score = progress_scores.get(possible_activity, 0)
                    total_score = (weights['time'] * normalized_time) + (weights['progress'] * (1 - progress_score))

                    if total_score < best_choice['score']:
                        best_choice['activity'] = possible_activity
                        best_choice['agent_id'] = agent_id
                        best_choice['score'] = total_score
            
            # --- After checking all pairs, commit to the best one ---
            if best_choice['activity'] is None:
                next_activity = 'zzz_end'
                chosen_agent_id = None
            else:
                next_activity = best_choice['activity']
                chosen_agent_id = best_choice['agent_id']
                
            if next_activity == 'zzz_end':
                return None, True

            self.new_activity_index = self.activities.index(next_activity)
            # Return a list with ONLY the chosen agent. The rest of the system will handle it.
            potential_agents = [9999, chosen_agent_id]
            return potential_agents, False

        else: # Default simulation behavior (remains unchanged)
            next_activity = random.choices(activity_list, weights=probabilities, k=1)[0]
            if next_activity == 'zzz_end':
                return None, True
            
            self.new_activity_index = self.activities.index(next_activity)
            potential_agents = [key for key, value in self.agent_activity_mapping.items() if next_activity in value]
            potential_agents.insert(0, 9999)
            return potential_agents, False