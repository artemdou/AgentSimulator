# =============================================================================
# --- User Defined Multi-Objective Optimization for Business Processes ---
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
import datetime

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
    """Calculates five metrics: total cost, avg cycle time, avg waiting time, avg agents per case, and total agents used."""
    if df_log.empty or len(df_log) < 2: 
        return float('inf'), float('inf'), float('inf'), float('inf'), float('inf')
    
    df_log['start_timestamp'] = pd.to_datetime(df_log['start_timestamp'], format='mixed', utc=True)
    df_log['end_timestamp'] = pd.to_datetime(df_log['end_timestamp'], format='mixed', utc=True)
    
    # Map resource to agent if needed
    if 'agent' not in df_log.columns and resource_to_agent_map is not None:
        agent_to_resource_map = resource_to_agent_map
        resource_to_agent_map_rev = {v: k for k, v in agent_to_resource_map.items()}
        df_log['agent'] = df_log['resource'].map(resource_to_agent_map_rev)

    # 1. Total Cost
    df_log['duration_hours'] = (df_log['end_timestamp'] - df_log['start_timestamp']).dt.total_seconds() / 3600
    df_log['cost_per_hour'] = df_log['agent'].astype(str).map(resource_costs).fillna(0)
    df_log['event_cost'] = df_log['duration_hours'] * df_log['cost_per_hour']
    total_cost = df_log['event_cost'].sum()

    # 2. Average Cycle Time
    case_times = df_log.groupby('case_id').agg(start=('start_timestamp', 'min'), end=('end_timestamp', 'max'))
    case_times['cycle_time_seconds'] = (case_times['end'] - case_times['start']).dt.total_seconds()
    avg_cycle_time_seconds = case_times['cycle_time_seconds'].mean()

    # 3. Average Waiting Time
    df_log = df_log.sort_values(by=['case_id', 'start_timestamp'])
    df_log['previous_end'] = df_log.groupby('case_id')['end_timestamp'].shift(1)
    df_log['waiting_time_seconds'] = (df_log['start_timestamp'] - df_log['previous_end']).dt.total_seconds().fillna(0)
    df_log['waiting_time_seconds'] = df_log['waiting_time_seconds'].clip(lower=0)
    avg_waiting_time_seconds = df_log['waiting_time_seconds'].mean()

    # 4. Average Agents per Case
    agents_per_case = df_log.groupby('case_id')['agent'].nunique()
    avg_agents_per_case = agents_per_case.mean()

    # 5. Total Agents Used
    total_agents_used = df_log['agent'].nunique()

    return total_cost, avg_cycle_time_seconds, avg_waiting_time_seconds, avg_agents_per_case, total_agents_used

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
    """Parallel-safe fitness function returning five objectives."""
    current_sim_params = copy.deepcopy(base_sim_params)
    current_sim_params['agent_transition_probabilities_autonomous'] = individual_policy_dict
    
    results = []
    for _ in range(runs_per_fitness):
        simulated_log = run_single_simulation(df_train, current_sim_params)
        metrics = calculate_metrics(simulated_log, resource_costs, base_sim_params['agent_to_resource'])
        results.append(metrics)
    
    return np.mean(results, axis=0)

# =============================================================================
# --- GA Helper Functions & Operators ---
# =============================================================================

def identify_and_split_policy(full_policy):
    """Splits the policy into fixed parts (1 outcome) and variable parts (>1 outcome)."""
    fixed_policy, variable_policy = {}, {}
    for agent_from, activities in full_policy.items():
        fixed_policy.setdefault(agent_from, {})
        variable_policy.setdefault(agent_from, {})
        
        for activity_from, outcomes in activities.items():
            # A decision point is variable if there's more than one potential (agent_to, activity_to) transition
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

def fitness_wrapper(individual_variable_policy, fixed_policy, base_sim_params, df_train, runs_per_fitness, resource_costs, objectives_to_eval):
    """A top-level, picklable wrapper for the fitness function that filters objectives."""
    full_policy = merge_policies(fixed_policy, individual_variable_policy)
    all_metrics = advanced_fitness_function(
        full_policy, base_sim_params, df_train, runs_per_fitness, resource_costs
    )
    
    metric_map = {
        'cost': all_metrics[0],
        'time': all_metrics[1],
        'wait': all_metrics[2],
        'agents_per_case': all_metrics[3],
        'total_agents': all_metrics[4]
    }
    
    return tuple(metric_map[obj] for obj in objectives_to_eval)


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

# --- NEW, GENERALIZED MUTATION OPERATORS ---
def guided_mutate(individual_policy_dict, base_sim_params, selected_objectives, indpb=0.3):
    """
    (Refinement) A generalized 'intelligent' mutation. It picks a guidable objective
    (cost, time, wait) and shifts a small amount of probability from a provably
    'worse' choice to a provably 'better' one.
    """
    policy = individual_policy_dict
    if random.random() > indpb:
        return policy,

    potential_mutations = []
    for agent_from, activities in policy.items():
        for activity_from, outcomes in activities.items():
            # We need at least two different agents to choose from for a meaningful mutation
            if len(outcomes) > 1:
                potential_mutations.append((agent_from, activity_from))

    if not potential_mutations:
        return policy,
    agent_from, activity_from = random.choice(potential_mutations)
    decision_point = policy[agent_from][activity_from]

    # Only apply guidance for objectives where we have a clear local heuristic.
    guidable_objectives = [obj for obj in selected_objectives if obj in ['cost', 'time', 'wait']]
    if not guidable_objectives:
        return policy,
        
    objective_to_improve = random.choice(guidable_objectives)
    
    options = []
    resource_map = base_sim_params.get('agent_to_resource', {})
    cost_map = base_sim_params.get('resource_costs', {})
    duration_map = base_sim_params.get('activity_durations_dict', {})

    for agent_to, act_outcomes in decision_point.items():
        score = float('inf')
        if objective_to_improve == 'cost':
            resource_name = resource_map.get(agent_to)
            score = cost_map.get(str(resource_name), float('inf'))

        elif objective_to_improve in ['time', 'wait']:
            if agent_to in duration_map:
                agent_durations = [d.mean for d in duration_map[agent_to].values() if hasattr(d, 'mean')]
                score = np.mean(agent_durations) if agent_durations else 0.0
            else:
                score = 0.0 # Instantaneous tasks are very good for time/wait
        
        total_prob = sum(act_outcomes.values())
        if total_prob > 1e-6:
            options.append({"agent_to": agent_to, "score": score, "prob": total_prob})

    if len(options) < 2:
        return policy,

    # Sort from best (lowest score) to worst (highest score)
    options.sort(key=lambda x: x['score']) 
    best_choice = options[0]
    worst_choice = options[-1]

    if best_choice['agent_to'] == worst_choice['agent_to'] or best_choice['score'] == worst_choice['score']:
        return policy,

    # Move a random portion of probability from the worst to the best
    prob_to_move = worst_choice['prob'] * random.uniform(0.1, 0.5)
    if prob_to_move <= 0:
        return policy,

    worst_agent_acts = decision_point[worst_choice['agent_to']]
    best_agent_acts = decision_point[best_choice['agent_to']]

    # Decrease probability from the worst choice, proportionally
    if worst_choice['prob'] > 0:
        for act, prob in worst_agent_acts.items():
            worst_agent_acts[act] -= prob_to_move * (prob / worst_choice['prob'])

    # Increase probability for the best choice, proportionally
    total_best_prob = best_choice['prob']
    if total_best_prob > 0:
        for act, prob in best_agent_acts.items():
            best_agent_acts[act] += prob_to_move * (prob / total_best_prob)
    else: # If the best choice had 0 probability, distribute new probability evenly
        num_acts = len(best_agent_acts)
        if num_acts > 0:
            for act in best_agent_acts:
                best_agent_acts[act] += prob_to_move / num_acts

    return policy,

def greedy_local_search_mutate(individual_policy_dict, base_sim_params, selected_objectives, indpb=0.1):
    """
    (Exploitation) An aggressive "hill-climbing" mutation. At a random decision point,
    it moves ALL probability from the worst agent to the best agent for a given objective.
    """
    policy = individual_policy_dict
    if random.random() > indpb:
        return policy,

    potential_mutations = []
    for agent_from, activities in policy.items():
        for activity_from, outcomes in activities.items():
            if len(outcomes) > 1:
                potential_mutations.append((agent_from, activity_from))

    if not potential_mutations: return policy,
    agent_from, activity_from = random.choice(potential_mutations)
    decision_point = policy[agent_from][activity_from]

    guidable_objectives = [obj for obj in selected_objectives if obj in ['cost', 'time', 'wait']]
    if not guidable_objectives: return policy,
    objective_to_improve = random.choice(guidable_objectives)
    
    # Scoring logic is identical to guided_mutate
    options = []
    resource_map = base_sim_params.get('agent_to_resource', {})
    cost_map = base_sim_params.get('resource_costs', {})
    duration_map = base_sim_params.get('activity_durations_dict', {})
    for agent_to, act_outcomes in decision_point.items():
        score = float('inf')
        if objective_to_improve == 'cost':
            resource_name = resource_map.get(agent_to)
            score = cost_map.get(str(resource_name), float('inf'))
        elif objective_to_improve in ['time', 'wait']:
            if agent_to in duration_map:
                agent_durations = [d.mean for d in duration_map[agent_to].values() if hasattr(d, 'mean')]
                score = np.mean(agent_durations) if agent_durations else 0.0
            else:
                score = 0.0
        total_prob = sum(act_outcomes.values())
        if total_prob > 1e-6:
            options.append({"agent_to": agent_to, "score": score, "prob": total_prob})

    if len(options) < 2: return policy,

    options.sort(key=lambda x: x['score'])
    best_choice = options[0]
    worst_choice = options[-1]

    if best_choice['agent_to'] == worst_choice['agent_to'] or best_choice['score'] == worst_choice['score']:
        return policy,

    # --- THE AGGRESSIVE ACTION ---
    prob_to_move = worst_choice['prob']
    if prob_to_move <= 0: return policy,

    worst_agent_acts = decision_point[worst_choice['agent_to']]
    best_agent_acts = decision_point[best_choice['agent_to']]

    # Set all outcomes for the worst choice to zero.
    for act in worst_agent_acts:
        worst_agent_acts[act] = 0.0
    
    # Give all that probability to the best choice, distributed proportionally.
    total_best_prob = best_choice['prob']
    if total_best_prob > 0:
        for act, prob in best_agent_acts.items():
            best_agent_acts[act] += prob_to_move * (prob / total_best_prob)
    else:
        num_acts = len(best_agent_acts)
        if num_acts > 0:
            for act in best_agent_acts:
                best_agent_acts[act] += prob_to_move / num_acts

    return policy,

def random_scramble_mutate(individual_policy_dict, indpb=0.1):
    """
    (Exploration) Mutates a random decision point by re-assigning all probabilities
    randomly, maintaining diversity and escaping local optima.
    """
    policy = individual_policy_dict
    if random.random() > indpb:
        return policy,
    
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
    dataset = args.log_path.split('/')[-1].split('.')[0]
    # Create a unique output directory for this specific run
    base_output_folder = f"optimization_runs/{dataset}"
    run_name = (
        f"pop{args.pop_size}_gen{args.num_gen}_runs{args.runs_per_fitness}_"
        f"obj_{args.objectives.replace(',', '_')}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir = os.path.join(base_output_folder, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- All results for this run will be saved in: {output_dir} ---")

    # Initialize a list to hold summary text for the final report
    summary_lines = [f"Run Summary for: {run_name}\n{'='*40}"]
    
    try:
        with open(args.costs_path, 'r') as f:
            resource_costs = {str(k): v for k, v in json.load(f).items()}
        print(f"Successfully loaded resource costs from {args.costs_path}")
    except FileNotFoundError:
        print(f"Error: Costs file not found at {args.costs_path}"); return

    all_possible_objectives = ['cost', 'time', 'wait', 'agents_per_case', 'total_agents']
    selected_objectives = args.objectives.split(',')
    if not all(obj in all_possible_objectives for obj in selected_objectives):
        print(f"Error: Invalid objective specified. Choose from: {', '.join(all_possible_objectives)}")
        return
    num_objectives = len(selected_objectives)
    
    run_params_text = [
        "\n--- Run Parameters ---", f"Log Path: {args.log_path}", f"Objectives: {args.objectives}",
        f"Population Size: {args.pop_size}", f"Generations: {args.num_gen}",
        f"Runs per Fitness Eval: {args.runs_per_fitness}", f"Crossover Probability: {args.cx_prob}",
        f"Mutation Probability: {args.mut_prob}", f"Cores Used: {args.n_cores}"
    ]
    for line in run_params_text: print(line); summary_lines.append(line)
        
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
    print("\n--- Step 1: Discovering Baseline Simulation Parameters ---")
    simulator = AgentSimulator(params)
    df_train, df_test, num_cases, df_val, num_val_cases = simulator._split_log()
    df_train, baseline_parameters = discover_simulation_parameters(
        df_train, df_test, df_val, simulator.data_dir, num_cases, num_val_cases,
        determine_automatically=params['determine_automatically'],
        central_orchestration=params['central_orchestration'],
        discover_extr_delays=params['discover_extr_delays']
    )
    baseline_parameters['resource_costs'] = resource_costs
    original_policy = baseline_parameters['agent_transition_probabilities_autonomous']
    
    def convert_keys_to_str(obj):
        if isinstance(obj, dict): return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer, np.floating)): return obj.item()
        return obj
    
    original_policy_path = os.path.join(output_dir, 'original_policy.json')
    with open(original_policy_path, 'w') as f:
        json.dump(convert_keys_to_str(original_policy), f, indent=4)
    print(f"Original discovered policy saved to {original_policy_path}")

    print("\n--- Step 1.5: Pruning Search Space for Efficient Optimization ---")
    fixed_policy, variable_policy = identify_and_split_policy(original_policy)
    num_original_dps = sum(len(acts) for acts in original_policy.values())
    num_variable_dps = sum(len(acts) for acts in variable_policy.values())
    print(f"Reduced search space from {num_original_dps} to {num_variable_dps} variable decision points.")
    
    print("\n--- Step 2: Evaluating Baseline Performance ---")
    gt_cost, gt_time, gt_wait, gt_apc, gt_ta = calculate_metrics(df_test.copy(), resource_costs, baseline_parameters['agent_to_resource'])
    
    gt_header = "Ground Truth (from Test Log):"
    gt_line = f"  -> Cost: ${gt_cost:,.2f}, Time: {gt_time/3600:.2f}h, Wait: {gt_wait/3600:.2f}h, Agents/Case: {gt_apc:.2f}, Total Agents: {gt_ta:.0f}"
    print(gt_header); print(gt_line)
    summary_lines.extend(["\n--- Baseline Performance ---", gt_header, gt_line])

    baseline_eval_params = copy.deepcopy(baseline_parameters)
    base_cost, base_time, base_wait, base_apc, base_ta = advanced_fitness_function(
        original_policy, baseline_eval_params, df_train, args.runs_per_fitness, resource_costs
    )
    
    base_header = "\nSimulated Baseline (Original Policy):"
    base_line = f"  -> Cost: ${base_cost:,.2f}, Time: {base_time/3600:.2f}h, Wait: {base_wait/3600:.2f}h, Agents/Case: {base_apc:.2f}, Total Agents: {base_ta:.0f}"
    print(base_header); print(base_line)
    summary_lines.extend([base_header, base_line])

    print(f"\n--- Step 3: Starting Multi-Objective Optimization for {', '.join(selected_objectives)} ---")
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * num_objectives)
    creator.create("Individual", dict, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    
    pool = None
    if args.n_cores > 1:
        print(f"Setting up parallel processing with {args.n_cores} cores.")
        pool = multiprocessing.Pool(processes=args.n_cores)
        toolbox.register("map", pool.map)

    toolbox.register("individual", lambda: creator.Individual(copy.deepcopy(variable_policy)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", fitness_wrapper, fixed_policy=fixed_policy, base_sim_params=baseline_parameters,
                     df_train=df_train, runs_per_fitness=args.runs_per_fitness,
                     resource_costs=resource_costs, objectives_to_eval=selected_objectives)
                     
    toolbox.register("mate", decision_point_crossover)
    
    def combined_mutate(individual):
        mutation_choice = random.random()
        if mutation_choice < 0.6:  # 60% chance for guided refinement
            return guided_mutate(individual, base_sim_params=baseline_parameters, selected_objectives=selected_objectives, indpb=1.0)
        elif mutation_choice < 0.8: # 20% chance for aggressive exploitation
            return greedy_local_search_mutate(individual, base_sim_params=baseline_parameters, selected_objectives=selected_objectives, indpb=1.0)
        else: # 20% chance for random scrambling to maintain diversity
            return random_scramble_mutate(individual, indpb=1.0)

    toolbox.register("mutate", combined_mutate)
    toolbox.register("select", tools.selNSGA2)

    print("Creating a diverse initial population...")
    population = toolbox.population(n=args.pop_size)
    num_random = args.pop_size // 2
    for i in range(num_random):
        population[i] = creator.Individual(create_random_individual(variable_policy))
    # The rest are mutated versions of the original to seed the search
    for i in range(num_random, args.pop_size):
        population[i], = toolbox.mutate(population[i])
    population[0] = creator.Individual(copy.deepcopy(variable_policy)) # Ensure original is present

    pareto_front = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    
    algorithms.eaMuPlusLambda(population, toolbox, mu=args.pop_size, lambda_=args.pop_size,
                                cxpb=args.cx_prob, mutpb=args.mut_prob,
                                ngen=args.num_gen, stats=stats, 
                                halloffame=pareto_front, verbose=True)

    if pool is not None: pool.close(); pool.join()

    print("\n--- Step 4: Optimization Finished ---")
    pareto_line = f"Found {len(pareto_front)} non-dominated solutions (the Pareto Front)."
    print(pareto_line)
    summary_lines.extend(["\n--- Optimization Results ---", pareto_line])

    baseline_metrics = {
        'cost': base_cost, 'time': base_time, 'wait': base_wait, 
        'agents_per_case': base_apc, 'total_agents': base_ta
    }
    
    pareto_header = "\n--- Pareto Front Solutions ---"
    header = " | ".join([f"{obj.replace('_', ' ').title():<15}" for obj in selected_objectives])
    header_line = f"Sol | {header}"
    separator = "----|-" + "-" * (len(header) + (len(selected_objectives)-1)*2)
    print(pareto_header); print(header_line); print(separator)
    summary_lines.extend([pareto_header, header_line, separator])

    best_solutions = []
    if not pareto_front:
        no_sol_line = "Pareto front is empty. Cannot select a best solution."
        print(no_sol_line); summary_lines.append(no_sol_line)
    else:
        for i, solution in enumerate(pareto_front):
            full_policy = merge_policies(fixed_policy, solution)
            # Re-evaluate with more runs for a more stable final metric
            all_sol_metrics = advanced_fitness_function(full_policy, copy.deepcopy(baseline_parameters), df_train, args.runs_per_fitness * 2, resource_costs)
            
            best_solutions.append({'policy': full_policy, 'metrics': all_sol_metrics})
            
            print_values = []
            for j, obj_name in enumerate(selected_objectives):
                val = solution.fitness.values[j]
                if 'time' in obj_name or 'wait' in obj_name: val /= 3600
                elif obj_name == 'total_agents': val = round(val)
                print_values.append(f"{val:<15.2f}")
            sol_line = f"{(i+1):<3} | {' | '.join(print_values)}"
            print(sol_line); summary_lines.append(sol_line)

    if not best_solutions:
        print("\nNo solutions found on the Pareto front. Exiting.")
        summary_path = os.path.join(output_dir, 'run_summary.txt')
        with open(summary_path, 'w') as f: f.write('\n'.join(summary_lines))
        print(f"Run summary saved to {summary_path}")
        return
        
    print(f"\nSelecting 'most balanced' solution using the optimized objectives: {', '.join(selected_objectives)}")

    metric_to_idx = {name: i for i, name in enumerate(all_possible_objectives)}
    selection_indices = [metric_to_idx[name] for name in selected_objectives]
    
    all_metrics_array = np.array([s['metrics'] for s in best_solutions])
    selection_metrics_array = all_metrics_array[:, selection_indices]

    # Normalize metrics to a 0-1 scale to find the most balanced solution
    ptp_values = selection_metrics_array.ptp(axis=0)
    denominator = np.where(ptp_values == 0, 1, ptp_values)
    norm_metrics = (selection_metrics_array - selection_metrics_array.min(axis=0)) / denominator
    
    distances = np.linalg.norm(norm_metrics, axis=1)
    balanced_solution = best_solutions[np.argmin(distances)]

    balanced_header = "\n--- Most Balanced Solution (based on optimized metrics) ---"
    best_policy = balanced_solution['policy']
    cost, time, wait, apc, ta = balanced_solution['metrics']
    balanced_line = f"Cost: ${cost:,.2f}, Time: {time/3600:.2f}h, Wait: {wait/3600:.2f}h, Agents/Case: {apc:.2f}, Total Agents: {ta:.0f}"
    print(balanced_header); print(balanced_line)
    summary_lines.extend([balanced_header, balanced_line])
    
    comparison_header = "\nCompared to Simulated Baseline:"
    cost_redux = f"  -> Cost Reduction: {(baseline_metrics['cost'] - cost) / baseline_metrics['cost'] * 100:.2f}%" if baseline_metrics['cost'] > 0 else "N/A"
    time_redux = f"  -> Time Reduction: {(baseline_metrics['time'] - time) / baseline_metrics['time'] * 100:.2f}%" if baseline_metrics['time'] > 0 else "N/A"
    wait_redux = f"  -> Wait Reduction: {(baseline_metrics['wait'] - wait) / baseline_metrics['wait'] * 100:.2f}%" if baseline_metrics['wait'] > 0 else "N/A"
    apc_redux = f"  -> Agents/Case Reduction: {(baseline_metrics['agents_per_case'] - apc) / baseline_metrics['agents_per_case'] * 100:.2f}%" if baseline_metrics['agents_per_case'] > 0 else "N/A"
    ta_redux = f"  -> Total Agents Reduction: {(baseline_metrics['total_agents'] - ta) / baseline_metrics['total_agents'] * 100:.2f}%" if baseline_metrics['total_agents'] > 0 else "N/A"
    print(comparison_header); print(cost_redux); print(time_redux); print(wait_redux); print(apc_redux); print(ta_redux)
    summary_lines.extend([comparison_header, cost_redux, time_redux, wait_redux, apc_redux, ta_redux])

    best_policy_path = os.path.join(output_dir, 'best_advanced_policy.json')
    with open(best_policy_path, 'w') as f:
        json.dump(convert_keys_to_str(best_policy), f, indent=4)
    best_policy_line = f"\nBest balanced policy saved to {best_policy_path}"
    print(best_policy_line); summary_lines.append(best_policy_line)

    print("\n--- Step 5: Generating and Saving Log for Most Balanced Solution ---")
    final_run_params = copy.deepcopy(baseline_parameters)
    final_run_params['agent_transition_probabilities_autonomous'] = best_policy
    
    print(f"Generating a representative log using the best policy...")
    best_policy_log = run_single_simulation(df_train, copy.deepcopy(final_run_params))
    
    log_path = os.path.join(output_dir, 'best_balanced_policy_log.csv')
    best_policy_log.to_csv(log_path, index=False)
    log_path_line = f"Log for the best balanced policy saved to: {log_path}"
    print(log_path_line); summary_lines.append(log_path_line)
    
    summary_path = os.path.join(output_dir, 'run_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nRun summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced Multi-Objective Optimization for Business Process Simulation.')
    parser.add_argument('--log_path', required=True, help='Path to the event log file.')
    parser.add_argument('--case_id', required=True, help='Column name for case ID.')
    parser.add_argument('--activity_name', required=True, help='Column name for activity name.')
    parser.add_argument('--resource_name', required=True, help='Column name for resource/agent.')
    parser.add_argument('--end_timestamp', required=True, help='Column name for activity end time.')
    parser.add_argument('--start_timestamp', required=True, help='Column name for activity start time.')
    parser.add_argument('--costs_path', default='agent_costs.json', help='Path to the JSON file with resource costs per hour.')
    
    parser.add_argument('--objectives', type=str, default='cost,time,wait', 
                        help="Comma-separated list of objectives to optimize. "
                             "Choose from: cost, time, wait, agents_per_case, total_agents")
    
    parser.add_argument('--pop_size', type=int, default=50, help='Population size for the GA.')
    parser.add_argument('--num_gen', type=int, default=100, help='Number of generations for the GA.')
    parser.add_argument('--runs_per_fitness', type=int, default=10, help='Number of simulation runs to average for each fitness evaluation.')
    parser.add_argument('--cx_prob', type=float, default=0.6, help='Crossover probability.')
    parser.add_argument('--mut_prob', type=float, default=0.4, help='Mutation probability.')
    parser.add_argument('--n_cores', type=int, default=1, help='Number of CPU cores to use for parallel evaluation.')
    
    parsed_args = parser.parse_args()
    main(parsed_args)