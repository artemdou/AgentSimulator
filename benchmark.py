import argparse
import warnings
import pandas as pd
import numpy as np
import os
import json # Import json to read the costs file

from source.agent_simulator import AgentSimulator

# --- NEW: A comprehensive metrics function, adapted from optimize_pro.py ---
def calculate_all_metrics(df_log, resource_costs, resource_to_agent_map):
    """Calculates three key metrics: total cost, avg cycle time, and avg waiting time."""
    if df_log.empty or len(df_log) < 2: 
        return float('inf'), float('inf'), float('inf')
    
    # Ensure timestamps are in datetime format with timezone awareness
    df_log['start_timestamp'] = pd.to_datetime(df_log['start_timestamp'], format='mixed', utc=True)
    df_log['end_timestamp'] = pd.to_datetime(df_log['end_timestamp'], format='mixed', utc=True)
    
    # 1. Calculate Total Cost
    # Calculate duration of each event in hours
    df_log['duration_hours'] = (df_log['end_timestamp'] - df_log['start_timestamp']).dt.total_seconds() / 3600
    
    # The simulated log has 'resource' and 'agent' columns. The costs file is keyed by resource name.
    # We can map the cost directly using the 'resource' column.
    df_log['cost_per_hour'] = df_log['resource'].astype(str).map(resource_costs).fillna(0)
    df_log['event_cost'] = df_log['duration_hours'] * df_log['cost_per_hour']
    total_cost = df_log['event_cost'].sum()

    # 2. Calculate Average Cycle Time
    case_times = df_log.groupby('case_id').agg(
        case_start=('start_timestamp', 'min'),
        case_end=('end_timestamp', 'max')
    )
    case_times['cycle_time_seconds'] = (case_times['case_end'] - case_times['case_start']).dt.total_seconds()
    avg_cycle_time_seconds = case_times['cycle_time_seconds'].mean()

    # 3. Calculate Average Waiting Time
    df_log_sorted = df_log.sort_values(by=['case_id', 'start_timestamp'])
    # Calculate waiting time as the gap between the previous event's end and the current event's start within a case
    df_log_sorted['previous_end'] = df_log_sorted.groupby('case_id')['end_timestamp'].shift(1)
    df_log_sorted['waiting_time_seconds'] = (df_log_sorted['start_timestamp'] - df_log_sorted['previous_end']).dt.total_seconds().fillna(0)
    # Ensure waiting time is not negative (can happen with parallel activities)
    df_log_sorted['waiting_time_seconds'] = df_log_sorted['waiting_time_seconds'].clip(lower=0)
    avg_waiting_time_seconds = df_log_sorted['waiting_time_seconds'].mean()

    return total_cost, avg_cycle_time_seconds, avg_waiting_time_seconds


def main(args):
    """Main function to run benchmark simulations."""
    warnings.filterwarnings("ignore")

    # --- NEW: Load resource costs ---
    try:
        with open(args.costs_path, 'r') as f:
            resource_costs = json.load(f)
        print(f"Successfully loaded resource costs from {args.costs_path}")
    except FileNotFoundError:
        print(f"Error: Costs file not found at {args.costs_path}. Please provide a valid path using --costs_path.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse the JSON costs file at {args.costs_path}.")
        return

    train_and_test = not bool(args.log_path)
    column_names = {
        args.case_id: 'case_id',
        args.activity_name: 'activity_name',
        args.resource_name: 'resource',
        args.end_timestamp: 'end_timestamp',
        args.start_timestamp: 'start_timestamp'
    }
    
    PATH_LOG = args.train_path if train_and_test else args.log_path
    
    params = {
        'discover_extr_delays': args.extr_delays,
        'discover_parallel_work': False,
        'central_orchestration': args.central_orchestration,
        'determine_automatically': args.determine_automatically,
        'PATH_LOG': PATH_LOG,
        'PATH_LOG_test': args.test_path,
        'train_and_test': train_and_test,
        'column_names': column_names,
        'num_simulations': args.num_simulations,
        'execution_type': 'original' # Use original learned probabilities
    }

    print("--- Starting Baseline Simulation ---")
    simulator = (params)
    
    # This runs the full pipeline and saves the logs
    simulator.execute_pipeline()
    
    print("\n--- Calculating Baseline Metrics ---")
    
    # --- NEW: Lists to store all three metrics ---
    total_costs = []
    cycle_times = []
    waiting_times = []
    
    simulated_log_directory = simulator.data_dir
    # The simulator discovers this mapping, so we can retrieve it for our metrics function
    resource_to_agent_map = simulator.simulation_parameters.get('agent_to_resource', {})
    if resource_to_agent_map:
        # The cost map is keyed by resource name, but our metric function can use the agent ID map if needed
        agent_to_resource_map = {v: k for k, v in resource_to_agent_map.items()}

    for i in range(args.num_simulations):
        log_path = os.path.join(simulated_log_directory, f"simulated_log_{i}.csv")
        if os.path.exists(log_path):
            df_simulated = pd.read_csv(log_path)
            
            # --- NEW: Call the new, comprehensive metrics function ---
            cost, cycle_time, wait_time = calculate_all_metrics(df_simulated, resource_costs, agent_to_resource_map)
            
            total_costs.append(cost)
            cycle_times.append(cycle_time)
            waiting_times.append(wait_time)
            
            print(f"Sim-run {i}: Cost=${cost:,.2f}, Cycle Time={cycle_time/3600:.2f}h, Wait Time={wait_time/3600:.2f}h")
        else:
            print(f"Warning: Could not find simulated log at {log_path}")

    # --- NEW: Embellished final printout ---
    if cycle_times:
        mean_cost = np.mean(total_costs)
        std_cost = np.std(total_costs)
        mean_cycle_time_s = np.mean(cycle_times)
        std_cycle_time_s = np.std(cycle_times)
        mean_wait_time_s = np.mean(waiting_times)
        std_wait_time_s = np.std(waiting_times)

        print("\n--- Baseline Simulation Results (Averaged over all runs) ---")
        print(f"Number of simulation runs: {len(cycle_times)}")
        print("\n[COST]")
        print(f"  > Average Total Cost: ${mean_cost:,.2f} (Std Dev: ${std_cost:,.2f})")
        print("\n[TIME]")
        print(f"  > Average Cycle Time: {mean_cycle_time_s / 3600:.2f} hours ({mean_cycle_time_s:.2f} seconds)")
        print(f"  > Standard Deviation: {std_cycle_time_s / 3600:.2f} hours ({std_cycle_time_s:.2f} seconds)")
        print("\n[WAITING]")
        print(f"  > Average Waiting Time: {mean_wait_time_s / 3600:.2f} hours ({mean_wait_time_s:.2f} seconds)")
        print(f"  > Standard Deviation: {std_wait_time_s / 3600:.2f} hours ({std_wait_time_s:.2f} seconds)")
        print("------------------------------------------------------------")
    else:
        print("No simulation logs were found to calculate metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark the AgentSimulator with comprehensive metrics.')
    # Use the same arguments as simulate.py for consistency
    parser.add_argument('--log_path', help='Path to single log file', required=True)
    parser.add_argument('--train_path', help='Path to training log file')
    parser.add_argument('--test_path', help='Path to test log file')
    parser.add_argument('--case_id', help='Case ID column name', required=True)
    parser.add_argument('--activity_name', help='Activity name column', required=True)
    parser.add_argument('--resource_name', help='Resource column name', required=True)
    parser.add_argument('--end_timestamp', help='End timestamp column name', required=True)
    parser.add_argument('--start_timestamp', help='Start timestamp column name', required=True)
    
    # --- NEW: Added costs_path argument ---
    parser.add_argument('--costs_path', default='costs.json', help='Path to the JSON file with resource costs per hour.', required=True)
    
    parser.add_argument('--extr_delays', action='store_true', help='Enable delay extraction')
    parser.add_argument('--central_orchestration', action='store_true', help='Enable central orchestration')
    parser.add_argument('--determine_automatically', action='store_true', help='Enable automatic determination of simulation parameters')
    parser.add_argument('--num_simulations', type=int, default=10, help='Number of simulations to run for benchmark')
    
    parsed_args = parser.parse_args()
    main(parsed_args)