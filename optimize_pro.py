# =============================================================================
# --- Circle Time, Wait Time, Agent Costs ---
# =============================================================================

import argparse
import warnings
import pandas as pd
import numpy as np
import os
import copy
import random
import json
import multiprocessing

# --- DEAP library for Genetic Algorithm ---
from deap import base, creator, tools, algorithms

# --- Import necessary components from your source code ---
from source.agent_simulator import AgentSimulator
from source.discovery import discover_simulation_parameters
from source.simulation import BusinessProcessModel, Case

# =============================================================================
# --- Core Simulation and Metric Calculation Functions ---
# =============================================================================

def calculate_metrics(df_log, resource_costs, resource_to_agent_map=None):
    """Calculates three metrics: total cost, avg cycle time, and avg waiting time."""
    if df_log.empty or len(df_log) < 2: 
        return float('inf'), float('inf'), float('inf')
    
    df_log['start_timestamp'] = pd.to_datetime(df_log['start_timestamp'], format='mixed', utc=True)
    df_log['end_timestamp'] = pd.to_datetime(df_log['end_timestamp'], format='mixed', utc=True)
    
    df_log['duration_hours'] = (df_log['end_timestamp'] - df_log['start_timestamp']).dt.total_seconds() / 3600
    
    if 'agent' not in df_log.columns and resource_to_agent_map is not None:
        agent_to_resource_map = {v: k for k, v in resource_to_agent_map.items()}
        df_log['agent'] = df_log['resource'].map(agent_to_resource_map)

    df_log['cost_per_hour'] = df_log['agent'].astype(str).map(resource_costs).fillna(0)
    df_log['event_cost'] = df_log['duration_hours'] * df_log['cost_per_hour']
    total_cost = df_log['event_cost'].sum()

    case_times = df_log.groupby('case_id').agg(start=('start_timestamp', 'min'), end=('end_timestamp', 'max'))
    case_times['cycle_time_seconds'] = (case_times['end'] - case_times['start']).dt.total_seconds()
    avg_cycle_time_seconds = case_times['cycle_time_seconds'].mean()

    df_log = df_log.sort_values(by=['case_id', 'start_timestamp'])
    df_log['previous_end'] = df_log.groupby('case_id')['end_timestamp'].shift(1)
    df_log['waiting_time_seconds'] = (df_log['start_timestamp'] - df_log['previous_end']).dt.total_seconds().fillna(0)
    df_log['waiting_time_seconds'] = df_log['waiting_time_seconds'].clip(lower=0)
    avg_waiting_time_seconds = df_log['waiting_time_seconds'].mean()

    return total_cost, avg_cycle_time_seconds, avg_waiting_time_seconds

def run_single_simulation(df_train, sim_params):
    """Runs a single simulation and returns the resulting log."""
    local_sim_params = copy.deepcopy(sim_params)
    start_timestamp = local_sim_params['case_arrival_times'][0]
    local_sim_params['start_timestamp'] = start_timestamp
    local_sim_params['case_arrival_times'] = local_sim_params['case_arrival_times'][1:]
    
    business_process_model = BusinessProcessModel(df_train, local_sim_params)
    case_id = 0
    case_ = Case(case_id=case_id, start_timestamp=start_timestamp)
    cases = [case_]
    
    while business_process_model.sampled_case_starting_times:
        business_process_model.step(cases)
        
    simulated_log = pd.DataFrame(business_process_model.simulated_events)
    if not simulated_log.empty:
        simulated_log['resource'] = simulated_log['agent'].map(local_sim_params['agent_to_resource'])
    
    return simulated_log

def advanced_fitness_function(individual_policy_dict, base_sim_params, df_train, runs_per_fitness, resource_costs):
    """Parallel-safe fitness function returning three objectives."""
    current_sim_params = copy.deepcopy(base_sim_params)
    current_sim_params['agent_transition_probabilities_autonomous'] = individual_policy_dict
    
    costs, cycle_times, waiting_times = [], [], []
    for _ in range(runs_per_fitness):
        simulated_log = run_single_simulation(df_train, current_sim_params)
        total_cost, avg_cycle_time, avg_waiting_time = calculate_metrics(simulated_log, resource_costs)
        costs.append(total_cost)
        cycle_times.append(avg_cycle_time)
        waiting_times.append(avg_waiting_time)
    
    return np.mean(costs), np.mean(cycle_times), np.mean(waiting_times)

# =============================================================================
# --- GA Helper Functions & Operators ---
# =============================================================================

def identify_and_split_policy(full_policy):
    """Splits the policy into fixed parts (1 outcome) and variable parts (>1 outcome)."""
    fixed_policy = {}
    variable_policy = {}
    for agent_from, activities in full_policy.items():
        fixed_policy.setdefault(agent_from, {})
        variable_policy.setdefault(agent_from, {})
        
        for activity_from, outcomes in activities.items():
            num_outcomes = sum(len(act_outcomes) for act_outcomes in outcomes.values())
            if num_outcomes > 1:
                variable_policy[agent_from][activity_from] = copy.deepcopy(outcomes)
            else:
                fixed_policy[agent_from][activity_from] = copy.deepcopy(outcomes)
                
    return fixed_policy, variable_policy

def merge_policies(fixed_policy, variable_policy):
    """Recombines the fixed and variable policies for a full policy to simulate."""
    full_policy = copy.deepcopy(fixed_policy)
    for agent_from, activities in variable_policy.items():
        for activity_from, outcomes in activities.items():
            full_policy[agent_from][activity_from] = outcomes
    return full_policy

def fitness_wrapper(individual_variable_policy, fixed_policy, base_sim_params, df_train, runs_per_fitness, resource_costs):
    """A top-level, picklable wrapper for the fitness function."""
    full_policy = merge_policies(fixed_policy, individual_variable_policy)
    return advanced_fitness_function(
        full_policy, base_sim_params, df_train, runs_per_fitness, resource_costs
    )

def create_random_individual(template_policy):
    """Creates a completely random (but validly structured) policy for population diversity."""
    random_policy = copy.deepcopy(template_policy)
    for agent_from, activities in random_policy.items():
        for activity_from, outcomes in activities.items():
            all_transitions = [(agent_to, act_to) for agent_to, acts in outcomes.items() for act_to in acts]
            if not all_transitions: continue
            
            new_probs = [random.random() for _ in all_transitions]
            total = sum(new_probs)
            normalized_probs = [p / total for p in new_probs] if total > 0 else [1.0/len(new_probs)] * len(new_probs)
            
            i = 0
            for agent_to, acts in outcomes.items():
                for act_to in acts:
                    random_policy[agent_from][activity_from][agent_to][act_to] = normalized_probs[i]
                    i += 1
    return random_policy

def decision_point_crossover(ind1, ind2):
    """Crossover by swapping entire decision strategies between parents."""
    child1, child2 = copy.deepcopy(ind1), copy.deepcopy(ind2)
    for agent_from, activities in child1.items():
        if agent_from in child2:
            for activity_from in activities:
                if activity_from in child2[agent_from]:
                    if random.random() < 0.5:
                        child1[agent_from][activity_from], child2[agent_from][activity_from] = \
                            child2[agent_from][activity_from], child1[agent_from][activity_from]
    return child1, child2

def cost_aware_mutate(individual_policy_dict, resource_costs, indpb=0.2):
    """Mutates by shifting probability from a high-cost route to a lower-cost one."""
    policy = individual_policy_dict
    if random.random() > indpb: return policy,
    
    potential_mutations = []
    for agent_from, activities in policy.items():
        for activity_from, outcomes in activities.items():
            if sum(len(acts) for acts in outcomes.values()) > 1:
                 potential_mutations.append((agent_from, activity_from))
    
    if not potential_mutations: return policy,

    agent_from, activity_from = random.choice(potential_mutations)
    decision_point = policy[agent_from][activity_from]

    options = []
    for agent_to, act_outcomes in decision_point.items():
        cost = resource_costs.get(str(agent_to), float('inf'))
        prob = sum(act_outcomes.values())
        options.append({"agent_to": agent_to, "cost": cost, "prob": prob})

    if len(options) < 2: return policy,
    
    options.sort(key=lambda x: x['cost'], reverse=True)
    highest_cost_route = options[0]
    lower_cost_candidates = [opt for opt in options if opt['agent_to'] != highest_cost_route['agent_to']]
    if not lower_cost_candidates: return policy,

    lower_cost_route = random.choice(lower_cost_candidates)
    
    prob_to_move = highest_cost_route['prob'] * random.uniform(0.1, 0.5)
    if prob_to_move == 0: return policy,

    high_cost_agent_acts = decision_point[highest_cost_route['agent_to']]
    low_cost_agent_acts = decision_point[lower_cost_route['agent_to']]
    
    for act, prob in high_cost_agent_acts.items():
        if highest_cost_route['prob'] > 0:
            high_cost_agent_acts[act] -= prob_to_move * (prob / highest_cost_route['prob'])

    total_low_prob = lower_cost_route['prob']
    if total_low_prob > 0:
        for act, prob in low_cost_agent_acts.items():
            low_cost_agent_acts[act] += prob_to_move * (prob / total_low_prob)
    else:
        num_acts = len(low_cost_agent_acts)
        if num_acts > 0:
            for act in low_cost_agent_acts:
                low_cost_agent_acts[act] += prob_to_move / num_acts
                
    return policy,

def random_scramble_mutate(individual_policy_dict, indpb=0.1):
    """Mutates a random decision point by re-assigning all probabilities randomly."""
    policy = individual_policy_dict
    if random.random() > indpb: return policy,
    
    decision_points = [(af, actf) for af, acts in policy.items() for actf in acts.keys() if sum(len(o) for o in acts[actf].values()) > 1]
    if not decision_points: return policy,
    
    agent_from, activity_from = random.choice(decision_points)
    decision_point = policy[agent_from][activity_from]

    all_transitions = [(agent_to, act_to) for agent_to, acts in decision_point.items() for act_to in acts]
    if len(all_transitions) < 2: return policy,

    new_probs = [random.random() for _ in all_transitions]
    total = sum(new_probs)
    normalized_probs = [p / total for p in new_probs] if total > 0 else [1.0/len(all_transitions)] * len(all_transitions)
    
    i = 0
    for agent_to, acts in decision_point.items():
        for act_to in acts:
            policy[agent_from][activity_from][agent_to][act_to] = normalized_probs[i]
            i += 1
            
    return policy,

# =============================================================================
# --- Main Execution Logic ---
# =============================================================================

def main(args):
    warnings.filterwarnings("ignore")
    try:
        with open(args.costs_path, 'r') as f:
            resource_costs = json.load(f)
        print(f"Successfully loaded resource costs from {args.costs_path}")
    except FileNotFoundError:
        print(f"Error: Costs file not found at {args.costs_path}"); return

    # --- 1. SETUP & BASELINE DISCOVERY ---
    column_names = {
        args.case_id: 'case_id', args.activity_name: 'activity_name',
        args.resource_name: 'resource', args.end_timestamp: 'end_timestamp',
        args.start_timestamp: 'start_timestamp'
    }
    params = {
        'PATH_LOG': args.log_path, 'train_and_test': False, 'column_names': column_names,
        'num_simulations': 1, 'central_orchestration': False,
        'determine_automatically': False, 'discover_extr_delays': False, 'execution_type': 'original'
    }
    print("--- Step 1: Discovering Baseline Simulation Parameters ---")
    simulator = AgentSimulator(params)
    df_train, df_test, num_cases, df_val, num_val_cases = simulator._split_log()
    df_train, baseline_parameters = discover_simulation_parameters(
        df_train, df_test, df_val, simulator.data_dir, num_cases, num_val_cases,
        determine_automatically=params['determine_automatically'],
        central_orchestration=params['central_orchestration'],
        discover_extr_delays=params['discover_extr_delays']
    )
    original_policy = baseline_parameters['agent_transition_probabilities_autonomous']
    
    # --- 1.5. PRUNE THE SEARCH SPACE ---
    print("\n--- Step 1.5: Pruning Search Space for Efficient Optimization ---")
    fixed_policy, variable_policy = identify_and_split_policy(original_policy)
    num_original_dps = sum(len(acts) for acts in original_policy.values())
    num_variable_dps = sum(len(acts) for acts in variable_policy.values())
    print(f"Reduced search space from {num_original_dps} to {num_variable_dps} variable decision points.")
    
    # --- 2. BASELINE EVALUATION ---
    print("\n--- Step 2: Evaluating Baseline Performance ---")
    
    ground_truth_cost, ground_truth_time, ground_truth_wait = calculate_metrics(
        df_test.copy(), resource_costs, baseline_parameters['agent_to_resource']
    )
    print("Ground Truth (from Test Log):")
    print(f"  -> Total Cost: ${ground_truth_cost:,.2f}")
    print(f"  -> Avg Cycle Time: {ground_truth_time / 3600:.2f} hours")
    print(f"  -> Avg Wait Time: {ground_truth_wait / 3600:.2f} hours")

    # Important: Use a deepcopy to prevent the baseline evaluation from consuming the arrival times
    baseline_eval_params = copy.deepcopy(baseline_parameters)
    baseline_cost, baseline_time, baseline_wait = advanced_fitness_function(
        original_policy, baseline_eval_params, df_train, 
        args.runs_per_fitness, resource_costs
    )
    print("\nSimulated Baseline (Original Policy):")
    print(f"  -> Total Cost: ${baseline_cost:,.2f}")
    print(f"  -> Avg Cycle Time: {baseline_time / 3600:.2f} hours")
    print(f"  -> Avg Wait Time: {baseline_wait / 3600:.2f} hours")
    
    # --- 3. GA OPTIMIZATION ---
    print("\n--- Step 3: Starting Multi-Objective Optimization ---")
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", dict, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    
    pool = None
    if args.n_cores > 1:
        print(f"Setting up parallel processing with {args.n_cores} cores.")
        pool = multiprocessing.Pool(processes=args.n_cores)
        toolbox.register("map", pool.map)

    toolbox.register("individual", lambda: creator.Individual(copy.deepcopy(variable_policy)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", 
                     fitness_wrapper,
                     fixed_policy=fixed_policy,
                     base_sim_params=baseline_parameters,
                     df_train=df_train,
                     runs_per_fitness=args.runs_per_fitness,
                     resource_costs=resource_costs)
                     
    toolbox.register("mate", decision_point_crossover)
    def multi_mutate(individual):
        if random.random() < 0.5:
            return cost_aware_mutate(individual, resource_costs)
        else:
            return random_scramble_mutate(individual)
    toolbox.register("mutate", multi_mutate)
    toolbox.register("select", tools.selNSGA2)

    print("Creating a diverse initial population...")
    population = toolbox.population(n=args.pop_size)
    num_random = args.pop_size // 2
    for i in range(num_random):
        population[i] = creator.Individual(create_random_individual(variable_policy))
    for i in range(num_random, args.pop_size):
        population[i], = toolbox.mutate(population[i])
    population[0] = creator.Individual(copy.deepcopy(variable_policy))

    pareto_front = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    
    algorithms.eaMuPlusLambda(population, toolbox, mu=args.pop_size, lambda_=args.pop_size,
                                cxpb=args.cx_prob, mutpb=args.mut_prob,
                                ngen=args.num_gen, stats=stats, 
                                halloffame=pareto_front, verbose=True)

    if pool is not None: pool.close(); pool.join()

    # --- 4. FINAL RESULTS ANALYSIS ---
    print("\n--- Step 4: Optimization Finished ---")
    print(f"Found {len(pareto_front)} non-dominated solutions (the Pareto Front).")

    print("\nBaseline Performance (Simulated):")
    print(f"  -> Cost: ${baseline_cost:,.2f}, Time: {baseline_time/3600:.2f}h, Wait: {baseline_wait/3600:.2f}h")
    
    print("\n--- Pareto Front Solutions ---")
    print("      Cost ($)   | Time (hours) | Wait (hours)")
    print("-----------------|--------------|---------------")
    
    best_solutions = []
    for i, solution in enumerate(pareto_front):
        cost, time, wait = solution.fitness.values
        print(f"Sol {i+1}: ${cost: >10,.2f} | {time/3600: >10.2f}   | {wait/3600: >10.2f}")
        full_solution_policy = merge_policies(fixed_policy, solution)
        best_solutions.append({'policy': full_solution_policy, 'cost': cost, 'time': time, 'wait': wait})

    if not best_solutions:
        print("\nNo solutions found on the Pareto front. Exiting.")
        return

    costs = np.array([s['cost'] for s in best_solutions])
    times = np.array([s['time'] for s in best_solutions])
    waits = np.array([s['wait'] for s in best_solutions])
    
    norm_costs = (costs - costs.min()) / (costs.ptp() or 1)
    norm_times = (times - times.min()) / (times.ptp() or 1)
    norm_waits = (waits - waits.min()) / (waits.ptp() or 1)
    distances = np.sqrt(norm_costs**2 + norm_times**2 + norm_waits**2)
    balanced_solution = best_solutions[np.argmin(distances)]

    print("\n--- Most Balanced Solution ---")
    best_policy = balanced_solution['policy']
    cost, time, wait = balanced_solution['cost'], balanced_solution['time'], balanced_solution['wait']
    print(f"Cost: ${cost:,.2f}, Time: {time/3600:.2f}h, Wait: {wait/3600:.2f}h")
    
    cost_reduction = (baseline_cost - cost) / baseline_cost * 100
    time_reduction = (baseline_time - time) / baseline_time * 100
    wait_reduction = (baseline_wait - wait) / baseline_wait * 100
    print(f"\nCompared to Simulated Baseline:")
    print(f"  -> Cost Reduction: {cost_reduction:.2f}%")
    print(f"  -> Time Reduction: {time_reduction:.2f}%")
    print(f"  -> Wait Reduction: {wait_reduction:.2f}%")

    best_policy_path = os.path.join(simulator.data_dir, 'best_advanced_policy.json')
    def convert_keys_to_str(obj):
        if isinstance(obj, dict): return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
        return obj
    with open(best_policy_path, 'w') as f:
        json.dump(convert_keys_to_str(best_policy), f, indent=4)
    print(f"\nBest balanced policy saved to {best_policy_path}")

    # --- 5. NEW: FINAL LOG GENERATION ---
    print("\n--- Step 5: Generating and Saving Log for Most Balanced Solution ---")
    final_run_params = copy.deepcopy(baseline_parameters)
    final_run_params['agent_transition_probabilities_autonomous'] = best_policy
    
    print(f"Generating a representative log using the best policy...")
    # It's important to use a deepcopy here so the final run doesn't affect other potential future steps
    best_policy_log = run_single_simulation(df_train, copy.deepcopy(final_run_params))
    
    log_path = os.path.join(simulator.data_dir, 'best_balanced_policy_log.csv')
    best_policy_log.to_csv(log_path, index=False)
    print(f"Log for the best balanced policy saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced Multi-Objective Optimization for Cost, Time, and Waiting Time.')
    parser.add_argument('--log_path', required=True, help='Path to the event log file.')
    parser.add_argument('--case_id', required=True, help='Column name for case ID.')
    parser.add_argument('--activity_name', required=True, help='Column name for activity name.')
    parser.add_argument('--resource_name', required=True, help='Column name for resource/agent.')
    parser.add_argument('--end_timestamp', required=True, help='Column name for activity end time.')
    parser.add_argument('--start_timestamp', required=True, help='Column name for activity start time.')
    parser.add_argument('--costs_path', default='agent_costs.json', help='Path to the JSON file with resource costs per hour.')
    parser.add_argument('--pop_size', type=int, default=50, help='Population size for the GA.')
    parser.add_argument('--num_gen', type=int, default=100, help='Number of generations for the GA.')
    parser.add_argument('--runs_per_fitness', type=int, default=10, help='Number of simulation runs to average for each fitness evaluation.')
    parser.add_argument('--cx_prob', type=float, default=0.6, help='Crossover probability.')
    parser.add_argument('--mut_prob', type=float, default=0.4, help='Mutation probability.')
    parser.add_argument('--n_cores', type=int, default=1, help='Number of CPU cores to use for parallel evaluation.')
    
    parsed_args = parser.parse_args()
    main(parsed_args)




# =============================================================================
# --- Wait Time and Agent Costs ---
# =============================================================================
# import argparse
# import warnings
# import pandas as pd
# import numpy as np
# import os
# import copy
# import random
# import json
# import multiprocessing

# # --- DEAP library for Genetic Algorithm ---
# from deap import base, creator, tools, algorithms

# # --- Import necessary components from your source code ---
# from source.agent_simulator import AgentSimulator
# from source.discovery import discover_simulation_parameters
# from source.simulation import BusinessProcessModel, Case

# # =============================================================================
# # --- Core Simulation and Metric Calculation Functions ---
# # =============================================================================

# def calculate_metrics(df_log, resource_costs, resource_to_agent_map=None):
#     if df_log.empty or len(df_log) < 2:
#         return float('inf'), float('inf'), float('inf')
    
#     df_log['start_timestamp'] = pd.to_datetime(df_log['start_timestamp'], utc=True)
#     df_log['end_timestamp'] = pd.to_datetime(df_log['end_timestamp'], utc=True)
    
#     df_log['duration_hours'] = (df_log['end_timestamp'] - df_log['start_timestamp']).dt.total_seconds() / 3600
    
#     if 'agent' not in df_log.columns and resource_to_agent_map is not None:
#         agent_to_resource = {v: k for k, v in resource_to_agent_map.items()}
#         df_log['agent'] = df_log['resource'].map(agent_to_resource)

#     df_log['cost_per_hour'] = df_log['agent'].astype(str).map(resource_costs).fillna(0)
#     df_log['event_cost'] = df_log['duration_hours'] * df_log['cost_per_hour']
#     total_cost = df_log['event_cost'].sum()

#     case_times = df_log.groupby('case_id').agg(start=('start_timestamp','min'), end=('end_timestamp','max'))
#     case_times['cycle_time_seconds'] = (case_times['end'] - case_times['start']).dt.total_seconds()
#     avg_cycle_time_seconds = case_times['cycle_time_seconds'].mean()

#     df_log = df_log.sort_values(['case_id', 'start_timestamp'])
#     df_log['prev_end'] = df_log.groupby('case_id')['end_timestamp'].shift(1)
#     df_log['wait_sec'] = (df_log['start_timestamp'] - df_log['prev_end']).dt.total_seconds().clip(lower=0).fillna(0)
#     avg_waiting_time_seconds = df_log['wait_sec'].mean()

#     return total_cost, avg_cycle_time_seconds, avg_waiting_time_seconds

# def run_single_simulation(df_train, sim_params):
#     local = copy.deepcopy(sim_params)
#     start_ts = local['case_arrival_times'][0]
#     local['start_timestamp'] = start_ts
#     local['case_arrival_times'] = local['case_arrival_times'][1:]
    
#     bpm = BusinessProcessModel(df_train, local)
#     case = Case(case_id=0, start_timestamp=start_ts)
#     cases = [case]
#     while bpm.sampled_case_starting_times:
#         bpm.step(cases)
    
#     log = pd.DataFrame(bpm.simulated_events)
#     if not log.empty:
#         log['resource'] = log['agent'].map(local['agent_to_resource'])
#     return log

# def advanced_fitness_function(individual_policy, base_sim_params, df_train, runs_per_fitness, resource_costs):
#     current = copy.deepcopy(base_sim_params)
#     current['agent_transition_probabilities_autonomous'] = individual_policy
#     costs, waits = [], []

#     for _ in range(runs_per_fitness):
#         log = run_single_simulation(df_train, current)
#         cost, cycle, wait = calculate_metrics(log, resource_costs)
#         costs.append(cost)
#         waits.append(wait)

#     return np.mean(costs), np.mean(waits)

# def identify_and_split_policy(full_policy):
#     fixed, variable = {}, {}
#     for a_from, acts in full_policy.items():
#         fixed.setdefault(a_from, {}); variable.setdefault(a_from, {})
#         for act_from, outcomes in acts.items():
#             total_outcomes = sum(len(v) for v in outcomes.values())
#             if total_outcomes > 1:
#                 variable[a_from][act_from] = copy.deepcopy(outcomes)
#             else:
#                 fixed[a_from][act_from] = copy.deepcopy(outcomes)
#     return fixed, variable

# def merge_policies(fixed_policy, variable_policy):
#     merged = copy.deepcopy(fixed_policy)
#     for a_from, acts in variable_policy.items():
#         merged[a_from].update(copy.deepcopy(acts))
#     return merged

# def fitness_wrapper(ind_var, fixed_policy, base_sim_params, df_train, runs_per_fitness, resource_costs):
#     full = merge_policies(fixed_policy, ind_var)
#     return advanced_fitness_function(full, base_sim_params, df_train, runs_per_fitness, resource_costs)

# def create_random_individual(template_policy):
#     rnd = copy.deepcopy(template_policy)
#     for a_from, acts in rnd.items():
#         for act_from, outcomes in acts.items():
#             all_transitions = [(ag_to, at) for ag_to, target in outcomes.items() for at in target]
#             if not all_transitions:
#                 continue
#             probs = np.random.rand(len(all_transitions))
#             probs /= probs.sum()
#             idx = 0
#             for ag_to, target in outcomes.items():
#                 for at in target:
#                     rnd[a_from][act_from][ag_to][at] = probs[idx]
#                     idx += 1
#     return rnd

# def decision_point_crossover(ind1, ind2):
#     c1, c2 = copy.deepcopy(ind1), copy.deepcopy(ind2)
#     for a_from, acts in c1.items():
#         if a_from in c2:
#             for act_from in acts:
#                 if act_from in c2[a_from] and random.random() < 0.5:
#                     c1[a_from][act_from], c2[a_from][act_from] = c2[a_from][act_from], c1[a_from][act_from]
#     return c1, c2

# def cost_aware_mutate(ind, resource_costs, indpb=0.2):
#     if random.random() > indpb:
#         return ind,
#     pts = [(a_from, af) for a_from, acts in ind.items() for af, outcomes in acts.items() if sum(len(v) for v in outcomes.values()) > 1]
#     if not pts:
#         return ind,
#     a_from, act_from = random.choice(pts)
#     dp = ind[a_from][act_from]
#     opts = []
#     for ag_to, outcomes in dp.items():
#         cost = resource_costs.get(str(ag_to), float('inf'))
#         total = sum(outcomes.values())
#         opts.append({'agent_to': ag_to, 'cost': cost, 'prob': total})
#     if len(opts) < 2:
#         return ind,
#     opts.sort(key=lambda x: x['cost'], reverse=True)
#     hi = opts[0]; choices = [o for o in opts if o['agent_to'] != hi['agent_to']]
#     lo = random.choice(choices) if choices else None
#     if not lo or hi['prob'] <= 0:
#         return ind,
#     move = hi['prob'] * random.uniform(0.1, 0.5)
#     hc = dp[hi['agent_to']]; lc = dp[lo['agent_to']]
#     for act, p in hc.items():
#         hc[act] -= move * (p / hi['prob'])
#     tot_low = sum(lc.values())
#     if tot_low > 0:
#         for act, p in lc.items():
#             lc[act] += move * (p / tot_low)
#     else:
#         for act in lc:
#             lc[act] += move / len(lc)
#     return ind,

# def random_scramble_mutate(ind, indpb=0.1):
#     if random.random() > indpb:
#         return ind,
#     pts = [(a_from, af) for a_from, acts in ind.items() for af, outcomes in acts.items() if sum(len(v) for v in outcomes.values()) > 1]
#     if not pts:
#         return ind,
#     a_from, act_from = random.choice(pts)
#     dp = ind[a_from][act_from]
#     all_trans = [(ag_to, at) for ag_to, target in dp.items() for at in target]
#     if len(all_trans) < 2:
#         return ind,
#     probs = np.random.rand(len(all_trans))
#     probs /= probs.sum()
#     idx = 0
#     for ag_to, target in dp.items():
#         for at in target:
#             dp[ag_to][at] = probs[idx]
#             idx += 1
#     return ind,

# def main(args):
#     warnings.filterwarnings("ignore")
#     try:
#         resource_costs = json.load(open(args.costs_path))
#     except FileNotFoundError:
#         print(f"Error: Costs file not found at {args.costs_path}")
#         return

#     col_map = {
#         args.case_id: 'case_id', args.activity_name: 'activity_name',
#         args.resource_name: 'resource', args.end_timestamp: 'end_timestamp',
#         args.start_timestamp: 'start_timestamp'
#     }
#     params = {
#         'PATH_LOG': args.log_path, 'train_and_test': False,
#         'column_names': col_map, 'num_simulations': 1,
#         'central_orchestration': False, 'determine_automatically': False,
#         'discover_extr_delays': False, 'execution_type': 'original'
#     }

#     print("--- Step 1: Discover Baseline Parameters ---")
#     sim = AgentSimulator(params)
#     df_train, df_test, n_cases, df_val, n_val = sim._split_log()
#     df_train, base_params = discover_simulation_parameters(
#         df_train, df_test, df_val, sim.data_dir, n_cases, n_val,
#         determine_automatically=False, central_orchestration=False,
#         discover_extr_delays=False
#     )
#     orig_policy = base_params['agent_transition_probabilities_autonomous']
#     fixed_policy, variable_policy = identify_and_split_policy(orig_policy)
#     print(f"Pruned decision points: {sum(len(v) for v in orig_policy.values())} → {sum(len(v) for v in variable_policy.values())}")

#     print("\n--- Step 2: Baseline Evaluation ---")
#     gt_cost, gt_time, gt_wait = calculate_metrics(df_test.copy(), resource_costs, base_params['agent_to_resource'])
#     print(f"Ground Truth → Cost: ${gt_cost:,.2f}, Cycle Time: {gt_time/3600:.2f}h, Wait: {gt_wait/3600:.2f}h")

#     baseline_eval_params = copy.deepcopy(base_params)
#     base_cost, base_wait = advanced_fitness_function(orig_policy, baseline_eval_params, df_train, args.runs_per_fitness, resource_costs)
#     print(f"Simulated Baseline → Cost: ${base_cost:,.2f}, Avg Wait: {base_wait/3600:.2f}h (Cycle Time not optimized)")

#     print("\n--- Step 3: GA Optimization (Cost & Wait) ---")
#     creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
#     creator.create("Individual", dict, fitness=creator.FitnessMulti)

#     toolbox = base.Toolbox()
#     pool = None
#     if args.n_cores > 1:
#         pool = multiprocessing.Pool(args.n_cores)
#         toolbox.register("map", pool.map)

#     toolbox.register("individual", lambda: creator.Individual(copy.deepcopy(variable_policy)))
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#     toolbox.register("evaluate", fitness_wrapper,
#                      fixed_policy=fixed_policy,
#                      base_sim_params=base_params,
#                      df_train=df_train,
#                      runs_per_fitness=args.runs_per_fitness,
#                      resource_costs=resource_costs)
#     toolbox.register("mate", decision_point_crossover)
#     toolbox.register("mutate", lambda ind: cost_aware_mutate(ind, resource_costs) if random.random() < 0.5 else random_scramble_mutate(ind))
#     toolbox.register("select", tools.selNSGA2)

#     pop = toolbox.population(n=args.pop_size)
#     half = args.pop_size // 2
#     for i in range(half):
#         pop[i] = creator.Individual(create_random_individual(variable_policy))
#     for i in range(half, args.pop_size):
#         pop[i], = toolbox.mutate(pop[i])
#     pop[0] = creator.Individual(copy.deepcopy(variable_policy))

#     pareto = tools.ParetoFront()
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean, axis=0)
#     stats.register("min", np.min, axis=0)

#     algorithms.eaMuPlusLambda(pop, toolbox,
#                               mu=args.pop_size, lambda_=args.pop_size,
#                               cxpb=args.cx_prob, mutpb=args.mut_prob,
#                               ngen=args.num_gen, stats=stats,
#                               halloffame=pareto, verbose=True)
#     if pool: pool.close(); pool.join()

#     print("\n--- Step 4: Optimization Done ---")
#     print(f"Found {len(pareto)} Pareto-optimal solutions.")

#     best_solutions = []
#     print("\nPareto Front Results:")
#     print(f"{'Sol':<5}{'Cost($)':>12}{'Wait(h)':>12}{'Cycle(h)':>12}")
#     print("-" * 41)
#     for idx, sol in enumerate(pareto):
#         cost, wait = sol.fitness.values
#         full = merge_policies(fixed_policy, sol)
#         sim_params = copy.deepcopy(base_params)
#         sim_params['agent_transition_probabilities_autonomous'] = full
#         log = run_single_simulation(df_train, sim_params)
#         _, cycle, _ = calculate_metrics(log, resource_costs)
#         print(f"{idx+1:<5}{cost:>12.2f}{wait/3600:>12.2f}{cycle/3600:>12.2f}")
#         best_solutions.append({'policy': full, 'cost': cost, 'wait': wait, 'time': cycle})

#     # Most balanced using all three for analysis
#     arr_cost = np.array([s['cost'] for s in best_solutions])
#     arr_wait = np.array([s['wait'] for s in best_solutions])
#     arr_time = np.array([s['time'] for s in best_solutions])
#     nc = (arr_cost - arr_cost.min()) / (arr_cost.ptp() or 1)
#     nw = (arr_wait - arr_wait.min()) / (arr_wait.ptp() or 1)
#     nt = (arr_time - arr_time.min()) / (arr_time.ptp() or 1)
#     dist = np.sqrt(nc**2 + nw**2 + nt**2)
#     best = best_solutions[np.argmin(dist)]

#     print("\nMost Balanced Solution (post hoc including Cycle Time):")
#     print(f"Cost: ${best['cost']:,.2f}, Wait: {best['wait']/3600:.2f}h, Cycle: {best['time']/3600:.2f}h")
#     print("\nNote: Cycle Time was **reported**, not optimized.")
    
#     json.dump({str(k): v for k, v in best['policy'].items()},
#               open(os.path.join(sim.data_dir, 'best_advanced_policy.json'), 'w'), indent=4)
#     log = run_single_simulation(df_train, {**base_params, 'agent_transition_probabilities_autonomous': best['policy']})
#     log.to_csv(os.path.join(sim.data_dir, 'best_balanced_policy_log.csv'), index=False)
#     print("Best policy and its log saved.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Multi-Objective GA (Cost & Wait, reporting Cycle).')
#     parser.add_argument('--log_path', required=True)
#     parser.add_argument('--case_id', required=True)
#     parser.add_argument('--activity_name', required=True)
#     parser.add_argument('--resource_name', required=True)
#     parser.add_argument('--end_timestamp', required=True)
#     parser.add_argument('--start_timestamp', required=True)
#     parser.add_argument('--costs_path', default='agent_costs.json')
#     parser.add_argument('--pop_size', type=int, default=50)
#     parser.add_argument('--num_gen', type=int, default=100)
#     parser.add_argument('--runs_per_fitness', type=int, default=10)
#     parser.add_argument('--cx_prob', type=float, default=0.6)
#     parser.add_argument('--mut_prob', type=float, default=0.4)
#     parser.add_argument('--n_cores', type=int, default=1)
#     args = parser.parse_args()
#     main(args)
