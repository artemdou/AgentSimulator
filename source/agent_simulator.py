import os
from source.train_test_split import split_data
from source.discovery import discover_simulation_parameters
from source.greedy_simulation import simulate_process_greedy
from source.simulation import simulate_process


import pandas as pd

class AgentSimulatorGreedy:
    def __init__(self, params):
        self.params = params

        # This makes it fully configurable from the outside.
        default_weights = {
            'progress': 0.1,
            'task_cost': 0.3,
            'wait_cost': 0.2,
            'load': 0.2,
            'new_to_case': 0.2,
            'new_to_process': 0.0
        }
        # The user can now pass 'optimizer_weights' in the params dict.
        self.params['optimizer_weights'] = self.params.get('optimizer_weights', default_weights)


    def execute_pipeline(self):
        self.df_train, self.df_test, self.num_cases_to_simulate, self.df_val, self.num_cases_to_simulate_val = self._split_log()
        
        # discover basic simulation parameters
        self.df_train, self.simulation_parameters = discover_simulation_parameters(
            self.df_train, 
            self.df_test, 
            self.df_val, 
            self.data_dir, 
            self.num_cases_to_simulate, 
            self.num_cases_to_simulate_val,
            self.params['determine_automatically'],
            self.params['central_orchestration'],
            self.params['discover_extr_delays']
        )
        self.simulation_parameters['execution_type'] = self.params['execution_type']

        # ===== START OF NEW LOGIC =====
        # If we are in optimization mode, discover the progress-based scoring parameters
        if self.params['execution_type'] == 'greedy':
            print("Discovering progress-based scoring parameters for optimization...")
            
            # 1. Calculate Activity Progress Scores
            # Create a step number for each event within its case
            df_temp = self.df_train.copy()
            df_temp['step_number'] = df_temp.groupby('case_id')['start_timestamp'].rank(method='first', ascending=True)
            # Calculate the average step number for each activity
            avg_steps = df_temp.groupby('activity_name')['step_number'].mean()
            # Normalize to a 0-1 score
            max_avg_step = avg_steps.max()
            activity_progress_scores = (avg_steps / max_avg_step).to_dict()
            self.simulation_parameters['activity_progress_scores'] = activity_progress_scores
            print(f"  - Discovered progress scores for {len(activity_progress_scores)} activities.")
            print(activity_progress_scores)

            # # 2. Calculate Normalization Factors
            # df_temp['duration_seconds'] = (pd.to_datetime(df_temp['end_timestamp']) - pd.to_datetime(df_temp['start_timestamp'])).dt.total_seconds()
            # # Use the 95th percentile as a robust "max time" to avoid outliers
            # max_time_per_step = df_temp['duration_seconds'].quantile(0.95)
            # # Handle case where max_time is 0 to avoid division by zero
            # self.simulation_parameters['max_time_per_step'] = max_time_per_step if max_time_per_step > 0 else 1.0
            # print(f"  - Set max_time_per_step for normalization to {self.simulation_parameters['max_time_per_step']:.2f} seconds.")

            # ===== START: NEW WAIT COST DISCOVERY LOGIC =====
            self.simulation_parameters['cost_of_delay_per_hour'] = self.params.get('cost_of_delay_per_hour', 0)

            if self.simulation_parameters['cost_of_delay_per_hour'] > 0:
                # To normalize wait_cost, we need a sense of "max wait time".
                # We can derive this from the inter-arrival times of cases.
                # A long wait is relative to how often new cases arrive.
                case_arrivals = self.df_train.groupby('case_id')['start_timestamp'].min().sort_values()
                inter_arrival_times = case_arrivals.diff().dt.total_seconds().dropna()
                
                # Use the 95th percentile as a robust "max wait time" to avoid outliers
                max_wait_seconds = inter_arrival_times.quantile(0.95)
                
                # Calculate the max wait cost for normalization
                max_wait_cost = (max_wait_seconds / 3600) * self.simulation_parameters['cost_of_delay_per_hour']
                self.simulation_parameters['max_wait_cost'] = max_wait_cost if max_wait_cost > 0 else 1.0
                print(f"  - Set max_wait_cost for normalization to {self.simulation_parameters['max_wait_cost']:.2f}.")

            # ===== START: NEW LOAD BALANCING DISCOVERY LOGIC =====
            # We need a normalization factor for agent load.
            # A good proxy for "max load" is the duration of a long-running case.
            case_durations = df_temp.groupby('case_id').apply(
                lambda x: (x['end_timestamp'].max() - x['start_timestamp'].min()).total_seconds()
            )
            # Use the 95th percentile as a robust "max load" to avoid outliers
            max_load_seconds = case_durations.quantile(0.95)
            self.simulation_parameters['max_load_seconds'] = max_load_seconds if max_load_seconds > 0 else 1.0
            print(f"  - Set max_load_seconds for normalization to {self.simulation_parameters['max_load_seconds'] / 3600:.2f} hours.")
            # ===== END: NEW LOAD BALANCING DISCOVERY LOGIC =====


            # 3. Pass agents' hourly costs
            self.simulation_parameters['agent_costs'] = self.params.get('agent_costs', {})

            # 4. Pass optimizer weights to the simulation
            self.simulation_parameters['optimizer_weights'] = self.params['optimizer_weights']

        # I commended out
        # print(f"agent to resource: {self.simulation_parameters['agent_to_resource']}")

        # simulate process
        simulate_process_greedy(self.df_train, self.simulation_parameters, self.data_dir, self.params['num_simulations'])

    def _split_log(self):
        """
        Split the log into training, testing and validation data.
        """
        def get_validation_data(df):
            df_sorted = df.sort_values(by=['case_id', 'start_timestamp'])
            total_cases = df_sorted['case_id'].nunique()
            twenty_percent = int(total_cases * 0.2)
            last_20_percent_case_ids = df_sorted['case_id'].unique()[-twenty_percent:]
            df_val = df_sorted[df_sorted['case_id'].isin(last_20_percent_case_ids)]
            
            return df_val
        
        file_name = os.path.splitext(os.path.basename(self.params['PATH_LOG']))[0]
        if self.params['determine_automatically']:
            print("Choice for architecture and extraneous delays will be determined automatically")
            file_name_extension = 'main_results'
        else:
            if self.params['central_orchestration']:
                file_name_extension = 'orchestrated'
            else:
                file_name_extension = 'autonomous'
        if self.params['train_and_test']:
            df_train, df_test, num_cases_to_simulate = split_data(self.params['PATH_LOG'], self.params['column_names'], self.params['PATH_LOG_test'])
        else:
            df_train, df_test, num_cases_to_simulate = split_data(self.params['PATH_LOG'], self.params['column_names'])

        self.data_dir = os.path.join(os.getcwd(), "simulated_data", file_name, file_name_extension)

        df_val = get_validation_data(df_train)
        num_cases_to_simulate_val = len(set(df_val['case_id']))

        return df_train, df_test, num_cases_to_simulate, df_val, num_cases_to_simulate_val


class AgentSimulator:
    def __init__(self, params):
        self.params = params

        if 'optimizer_weights' not in self.params:
            self.params['optimizer_weights'] = {
                'progress': 0.1,
                'task_cost': 0.3,
                'wait_cost': 0.2,
                'load': 0.2,
                'new_to_case': 0.2,
                'new_to_process': 0.0
            }


    def execute_pipeline(self):
        self.df_train, self.df_test, self.num_cases_to_simulate, self.df_val, self.num_cases_to_simulate_val = self._split_log()
        
        # discover basic simulation parameters
        self.df_train, self.simulation_parameters = discover_simulation_parameters(
            self.df_train, 
            self.df_test, 
            self.df_val, 
            self.data_dir, 
            self.num_cases_to_simulate, 
            self.num_cases_to_simulate_val,
            self.params['determine_automatically'],
            self.params['central_orchestration'],
            self.params['discover_extr_delays']
        )
        self.simulation_parameters['execution_type'] = self.params['execution_type']

        # ===== START OF NEW LOGIC =====
        # If we are in optimization mode, discover the progress-based scoring parameters
        if self.params['execution_type'] == 'greedy':
            print("Something went wrong")
            
            # 1. Calculate Activity Progress Scores
            # Create a step number for each event within its case
            df_temp = self.df_train.copy()
            df_temp['step_number'] = df_temp.groupby('case_id')['start_timestamp'].rank(method='first', ascending=True)
            # Calculate the average step number for each activity
            avg_steps = df_temp.groupby('activity_name')['step_number'].mean()
            # Normalize to a 0-1 score
            max_avg_step = avg_steps.max()
            activity_progress_scores = (avg_steps / max_avg_step).to_dict()
            self.simulation_parameters['activity_progress_scores'] = activity_progress_scores
            print(f"  - Discovered progress scores for {len(activity_progress_scores)} activities.")
            print(activity_progress_scores)

            # # 2. Calculate Normalization Factors
            # df_temp['duration_seconds'] = (pd.to_datetime(df_temp['end_timestamp']) - pd.to_datetime(df_temp['start_timestamp'])).dt.total_seconds()
            # # Use the 95th percentile as a robust "max time" to avoid outliers
            # max_time_per_step = df_temp['duration_seconds'].quantile(0.95)
            # # Handle case where max_time is 0 to avoid division by zero
            # self.simulation_parameters['max_time_per_step'] = max_time_per_step if max_time_per_step > 0 else 1.0
            # print(f"  - Set max_time_per_step for normalization to {self.simulation_parameters['max_time_per_step']:.2f} seconds.")

            # ===== START: NEW WAIT COST DISCOVERY LOGIC =====
            self.simulation_parameters['cost_of_delay_per_hour'] = self.params.get('cost_of_delay_per_hour', 0)

            if self.simulation_parameters['cost_of_delay_per_hour'] > 0:
                # To normalize wait_cost, we need a sense of "max wait time".
                # We can derive this from the inter-arrival times of cases.
                # A long wait is relative to how often new cases arrive.
                case_arrivals = self.df_train.groupby('case_id')['start_timestamp'].min().sort_values()
                inter_arrival_times = case_arrivals.diff().dt.total_seconds().dropna()
                
                # Use the 95th percentile as a robust "max wait time" to avoid outliers
                max_wait_seconds = inter_arrival_times.quantile(0.95)
                
                # Calculate the max wait cost for normalization
                max_wait_cost = (max_wait_seconds / 3600) * self.simulation_parameters['cost_of_delay_per_hour']
                self.simulation_parameters['max_wait_cost'] = max_wait_cost if max_wait_cost > 0 else 1.0
                print(f"  - Set max_wait_cost for normalization to {self.simulation_parameters['max_wait_cost']:.2f}.")

            # ===== START: NEW LOAD BALANCING DISCOVERY LOGIC =====
            # We need a normalization factor for agent load.
            # A good proxy for "max load" is the duration of a long-running case.
            case_durations = df_temp.groupby('case_id').apply(
                lambda x: (x['end_timestamp'].max() - x['start_timestamp'].min()).total_seconds()
            )
            # Use the 95th percentile as a robust "max load" to avoid outliers
            max_load_seconds = case_durations.quantile(0.95)
            self.simulation_parameters['max_load_seconds'] = max_load_seconds if max_load_seconds > 0 else 1.0
            print(f"  - Set max_load_seconds for normalization to {self.simulation_parameters['max_load_seconds'] / 3600:.2f} hours.")
            # ===== END: NEW LOAD BALANCING DISCOVERY LOGIC =====


            # 3. Pass agents' hourly costs
            self.simulation_parameters['agent_costs'] = self.params.get('agent_costs', {})

            # 4. Pass optimizer weights to the simulation
            self.simulation_parameters['optimizer_weights'] = self.params['optimizer_weights']

        # I commended out
        # print(f"agent to resource: {self.simulation_parameters['agent_to_resource']}")

        # simulate process
        simulate_process(self.df_train, self.simulation_parameters, self.data_dir, self.params['num_simulations'])

    def _split_log(self):
        """
        Split the log into training, testing and validation data.
        """
        def get_validation_data(df):
            df_sorted = df.sort_values(by=['case_id', 'start_timestamp'])
            total_cases = df_sorted['case_id'].nunique()
            twenty_percent = int(total_cases * 0.2)
            last_20_percent_case_ids = df_sorted['case_id'].unique()[-twenty_percent:]
            df_val = df_sorted[df_sorted['case_id'].isin(last_20_percent_case_ids)]
            
            return df_val
        
        file_name = os.path.splitext(os.path.basename(self.params['PATH_LOG']))[0]
        if self.params['determine_automatically']:
            print("Choice for architecture and extraneous delays will be determined automatically")
            file_name_extension = 'main_results'
        else:
            if self.params['central_orchestration']:
                file_name_extension = 'orchestrated'
            else:
                file_name_extension = 'autonomous'
        if self.params['train_and_test']:
            df_train, df_test, num_cases_to_simulate = split_data(self.params['PATH_LOG'], self.params['column_names'], self.params['PATH_LOG_test'])
        else:
            df_train, df_test, num_cases_to_simulate = split_data(self.params['PATH_LOG'], self.params['column_names'])

        self.data_dir = os.path.join(os.getcwd(), "simulated_data", file_name, file_name_extension)

        df_val = get_validation_data(df_train)
        num_cases_to_simulate_val = len(set(df_val['case_id']))

        return df_train, df_test, num_cases_to_simulate, df_val, num_cases_to_simulate_val
