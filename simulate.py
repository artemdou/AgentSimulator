import argparse
import warnings

from source.agent_simulator import AgentSimulator, AgentSimulatorGreedy
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process event log parameters')
    
    # File paths
    parser.add_argument('--log_path', help='Path to single log file, which is split into train and test')
    parser.add_argument('--train_path', help='Path to training log file')
    parser.add_argument('--test_path', help='Path to test log file')
    
    # Column names
    parser.add_argument('--case_id', help='Case ID column name')
    parser.add_argument('--activity_name', help='Activity name column')
    parser.add_argument('--resource_name', help='Resource column name')
    parser.add_argument('--end_timestamp', help='End timestamp column name')
    parser.add_argument('--start_timestamp', help='Start timestamp column name')
    
    # Hyperparameters
    parser.add_argument('--extr_delays', action='store_true', help='Enable delay extraction')
    parser.add_argument('--central_orchestration', action='store_true', help='Enable central orchestration')
    parser.add_argument('--determine_automatically', action='store_true', help='Enable automatic determination of simulation parameters')

    # Simulation parameters
    parser.add_argument('--num_simulations', type=int, default=10, help='Number of simulations to run')
    parser.add_argument('--execution_type', choices=['greedy', 'original'], default='original',
                    help="Sets the execution mode")
    parser.add_argument('--weights', type=str, default='{}', help='JSON string of optimizer weights')

    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    
    # Process arguments
    train_and_test = not bool(args.log_path)
    column_names = {
        args.case_id: 'case_id',
        args.activity_name: 'activity_name',
        args.resource_name: 'resource',
        args.end_timestamp: 'end_timestamp',
        args.start_timestamp: 'start_timestamp'
    }
    
    # Set paths
    PATH_LOG = args.train_path if train_and_test else args.log_path
    PATH_LOG_test = args.test_path
    
    # Feature flags
    discover_extr_delays = discover_delays = args.extr_delays
    central_orchestration = args.central_orchestration
    determine_automatically = args.determine_automatically

    params = {
        'discover_extr_delays': discover_extr_delays,
        'discover_parallel_work': False,
        'central_orchestration': central_orchestration,
        'determine_automatically': determine_automatically,
        'PATH_LOG': PATH_LOG,
        'PATH_LOG_test': PATH_LOG_test,
        'train_and_test': train_and_test,
        'column_names': column_names,
        'num_simulations': args.num_simulations,
        'execution_type': args.execution_type,
        'agent_costs': {
                "Clerk-000006": 90,
                "Clerk-000001": 30,
                "Applicant-000001": 0,
                "Clerk-000007": 30,
                "Clerk-000004": 90,
                "Clerk-000003": 60,
                "Clerk-000008": 30,
                "Senior Officer-000002": 150,
                "Appraiser-000002": 90,
                "AML Investigator-000002": 110,
                "Appraiser-000001": 90,
                "Loan Officer-000002": 95,
                "AML Investigator-000001": 110,
                "Loan Officer-000001": 95,
                "Loan Officer-000004": 105,
                "Clerk-000002": 30,
                "Loan Officer-000003": 105,
                "Senior Officer-000001": 150,
                "Clerk-000005": 90
                },
        'cost_of_delay_per_hour': 500,
    }

    if params.get('execution_type') == 'greedy':
        simulator = AgentSimulatorGreedy(params)
        print("-- Begining Greedy Selection --")
    else:
        simulator = AgentSimulator(params)
    simulator.execute_pipeline()