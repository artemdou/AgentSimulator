import pandas as pd
import pickle
import argparse
from state import Case, Agent
from simulation import Simulation
from utils import (
    discover_post_conditions,
    extract_all_successors,
    mine_concurrent_activities,
    extract_xor_groups_from_cooccurrence,
    discover_prerequisites_from_log,
)
from activity_rules import ActivityRules


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def prepare_inputs(sim_param_path, raw_logs_path):
    """
    Loads and prepares simulation parameters from a file path.

    Parameters:
    - sim_params: dict with simulation data (must include 'case_arrival_times')
    - agent_ids: list of agent IDs
    - durations: dict of activity durations by agent
    - calendars: dict of agent calendars
    - rules: ActivityRules object
    - xor: dict of XOR group decisions
    - max_cases: number of successful cases to simulate

    Returns:
    - sim_params, agent_ids, durations, calendars, rules, xor
    """
    with open(sim_param_path, 'rb') as f:
        sim_params = pickle.load(f)
    raw_log = pd.read_csv(raw_logs_path, compression='gzip')

    transition_dict = sim_params['transition_probabilities']
    flat_map = extract_all_successors({p: a for p, a in transition_dict.items() if len(p) == 1})
    concurrent_map = mine_concurrent_activities(raw_log)
    xor = extract_xor_groups_from_cooccurrence(flat_map, concurrent_map)
    post_conditions = discover_post_conditions(raw_log)
    prerequisites = discover_prerequisites_from_log(raw_log, activity_col='activity', case_col='case_id', order_by='end_time')

    # print("================================")
    # print("XOR RULES:", xor)
    # print("POST CONDITIONS:", post_conditions)
    # print("================================")

    rules = ActivityRules(
        prerequisites=prerequisites,
        post_conditions=post_conditions,
        transition_dict=transition_dict,
        xor_rules=xor
    )

    agent_ids = list(sim_params['activity_durations_dict'].keys())
    durations = sim_params['activity_durations_dict']
    calendars = sim_params['res_calendars']

    possible_cases = len(sim_params['case_arrival_times']) - 1

    return sim_params, agent_ids, durations, calendars, rules, xor, possible_cases


# def run_simulation(sim_param_path, raw_logs_path, max_cases=5, max_steps=100):
#     sim_params, agent_ids, durations, calendars, rules, xor, possible_cases = prepare_inputs(sim_param_path, raw_logs_path)
    
#     all_logs = []
#     successful_cases = 0
#     case_counter = 0
    

#     while (successful_cases < max_cases) and (case_counter <= possible_cases):
#         print(f"\n🚀 Starting simulation for case {case_counter}")

#         arrival_time = sim_params['case_arrival_times'][case_counter]
#         case = Case(str(case_counter), arrival_time, xor_decisions={})
#         case.performed = []
#         case.current_time = arrival_time

#         agents = []
#         for agent_id in agent_ids:
#             capable_acts = {
#                 act for act, dist in durations[agent_id].items()
#                 if dist and not isinstance(dist, list)
#             }
#             agents.append(Agent(agent_id, capable_activities=capable_acts, calendar=calendars[agent_id]))

#         sim = Simulation(
#             agents=agents,
#             cases=[case],
#             rules=rules,
#             durations=durations,
#             case_arrivals={str(case_counter): arrival_time}
#         )

#         step = 0
#         while not case.done and step < max_steps:
#             if not sim.tick(calendars):
#                 print(f"⚠️ No activity executed at tick {step}, aborting case {case_counter}")
#                 break
#             step += 1

#         if case.done:
#             print(f"✅ Case {case_counter} completed successfully with {len(sim.log)} log entries.")
#             all_logs.extend([entry.to_dict() for entry in sim.log])
#             successful_cases += 1
#         else:
#             print(f"❌ Case {case_counter} did not complete. Retrying with new case.")

#         print("Performed activities: ", case.performed)
#         case_counter += 1

#     df_simulated_log = pd.DataFrame(all_logs)
#     print("\n✅ Simulation completed for all cases.")
#     return df_simulated_log

def run_simulation(sim_param_path, raw_logs_path, max_cases=5, max_steps=10000):
    sim_params, agent_ids, durations, calendars, rules, xor, possible_cases = prepare_inputs(sim_param_path, raw_logs_path)

    all_logs = []
    case_id = 0

    # --- Step 1: Create shared Agent pool --- #
    agents = []
    for agent_id in agent_ids:
        capable_acts = {
            act for act, dist in durations[agent_id].items()
            if dist and not isinstance(dist, list)
        }
        agent = Agent(agent_id, capable_activities=capable_acts, calendar=calendars[agent_id])
        # agent.available_at = pd.Timestamp.min.tz_localize("UTC")  # globally available from the start
        agents.append(agent)

    # --- Step 2: Simulate each case one at a time --- #
    for arrival_time in sim_params['case_arrival_times']:
        if int(case_id) >= max_cases:
            break

        print(f"\n🚀 Starting simulation for case {case_id}")

        case = Case(str(case_id), arrival_time, xor_decisions={})
        case.performed = []
        case.current_time = arrival_time

        sim = Simulation(
            agents=agents,  # shared across all cases
            cases=[case],   # only simulate this case
            rules=rules,
            durations=durations,
            case_arrivals={str(case_id): arrival_time}
        )

        step = 0
        while not case.done and step < max_steps:
            if not sim.tick(calendars):
                print(f"⚠️ No activity executed at tick {step}, aborting case {case_id}")
                break
            step += 1

        if case.done:
            print(f"✅ Case {case_id} completed successfully with {len(sim.log)} log entries.")
            all_logs.extend([entry.to_dict() for entry in sim.log])
        else:
            print(f"❌ Case {case_id} did not complete. Retrying with new case.")

        print("Performed activities: ", case.performed)
        case_id += 1

    df_simulated_log = pd.DataFrame(all_logs)
    print("\n✅ Simulation completed for all cases.")
    return df_simulated_log



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_param_path", required=True, help="Path to sim_params pickle file")
    parser.add_argument("--raw_logs_path", required=True, help="Path to raw logs file")
    parser.add_argument("--max_cases", type=int, default=5, help="Number of successful cases to simulate")
    parser.add_argument("--max_steps", type=int, default=100, help="Max steps per case before aborting")

    args = parser.parse_args()

    df = run_simulation(args.sim_param_path, max_cases=args.max_cases, max_steps=args.max_steps, raw_logs_path=args.raw_logs_path)
    df.to_csv("simulated_log.csv", index=False)
    print("📝 Saved simulated log to simulated_log.csv")
