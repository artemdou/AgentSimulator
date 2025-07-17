import pandas as pd
import os
import numpy as np
import argparse
from datetime import datetime
import sys

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================

LOG_NAME = 'LoanApp.csv'
DEFAULT_BASELINE_DIR = 'main_results'
DEFAULT_OPTIMIZED_DIR = 'greedy_optimized'

AGENT_COSTS = {
    "Clerk-000006": 90, "Clerk-000001": 30, "Applicant-000001": 0,
    "Clerk-000007": 30, "Clerk-000004": 90, "Clerk-000003": 60,
    "Clerk-000008": 30, "Senior Officer-000002": 150, "Appraiser-000002": 90,
    "AML Investigator-000002": 110, "Appraiser-000001": 90, "Loan Officer-000002": 95,
    "AML Investigator-000001": 110, "Loan Officer-000001": 95, "Loan Officer-000004": 105,
    "Clerk-000002": 30, "Loan Officer-000003": 105, "Senior Officer-000001": 150,
    "Clerk-000005": 90
}

# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def enrich_log_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches a log DataFrame with all necessary calculated columns."""
    if df.empty: return df
    df_copy = df.copy()
    df_copy['start_timestamp'] = pd.to_datetime(df_copy['start_timestamp'], format='mixed')
    df_copy['end_timestamp'] = pd.to_datetime(df_copy['end_timestamp'], format='mixed')
    
    # Cycle Time related (total duration of case)
    case_starts = df_copy.groupby('case_id')['start_timestamp'].transform('min')
    case_ends = df_copy.groupby('case_id')['end_timestamp'].transform('max')
    df_copy['cycle_time'] = (case_ends - case_starts).dt.total_seconds()

    # Wait Time related (idle time between activities)
    df_sorted = df_copy.sort_values(by=['case_id', 'start_timestamp']).copy()
    df_sorted['previous_end_timestamp'] = df_sorted.groupby('case_id')['end_timestamp'].shift(1)
    df_sorted['wait_time'] = (df_sorted['start_timestamp'] - df_sorted['previous_end_timestamp']).dt.total_seconds().clip(lower=0)
    
    # Task Cost related
    df_sorted['work_duration'] = (df_sorted['end_timestamp'] - df_sorted['start_timestamp']).dt.total_seconds()
    df_sorted['cost_per_hour'] = df_sorted['resource'].map(AGENT_COSTS).fillna(0)
    df_sorted['task_cost'] = (df_sorted['work_duration'] / 3600) * df_sorted['cost_per_hour']
    
    return df_sorted

def calculate_gini_coefficient(df: pd.DataFrame) -> float:
    """Calculates the Gini coefficient for resource workload in a log."""
    if df.empty: return 0.0
    workload = df.groupby('resource')['work_duration'].sum()
    if workload.empty: return 0.0
    values = workload.sort_values().to_numpy()
    n = len(values)
    if n < 2 or np.sum(values) == 0: return 0.0
    i = np.arange(1, n + 1)
    numerator = np.sum((2 * i - n - 1) * values)
    denominator = n * np.sum(values)
    return numerator / denominator if denominator > 0 else 0.0

def calculate_avg_agents_per_case(df: pd.DataFrame) -> float:
    """Calculates the average number of unique agents used per case."""
    return df.groupby('case_id')['resource'].nunique().mean() if not df.empty else 0.0

def calculate_total_unique_agents(df: pd.DataFrame) -> int:
    """Calculates the total number of unique agents used in the entire log."""
    return df['resource'].nunique() if not df.empty else 0

# ==============================================================================
# --- 3. REPORTING FUNCTION ---
# ==============================================================================

def generate_report(baseline_df, optimized_df, metric_col, metric_name, unit, higher_is_better=False, is_case_metric=True):
    """Generates and prints a comparison report for a given metric."""
    print("-" * 60)
    print(f"METRIC: {metric_name.upper()}")
    print("-" * 60)

    # --- Per-Log Analysis ---
    if is_case_metric:
        # For metrics like time/cost, we average the per-case values for each log
        baseline_log_agg = baseline_df.groupby(['log_num', 'case_id'])[metric_col].sum().groupby('log_num').mean()
        optimized_log_agg = optimized_df.groupby(['log_num', 'case_id'])[metric_col].sum().groupby('log_num').mean()
    else:
        # For metrics like Gini/Agent Count, we calculate it once per log
        baseline_log_agg = baseline_df.groupby('log_num').apply(metric_col)
        optimized_log_agg = optimized_df.groupby('log_num').apply(metric_col)

    log_comparison = pd.concat([baseline_log_agg, optimized_log_agg], axis=1, keys=['Baseline', 'Optimized']).fillna(0)
    
    if higher_is_better:
        log_comparison['Improvement'] = ((log_comparison['Optimized'] - log_comparison['Baseline']) / log_comparison['Baseline'].replace(0, np.nan)) * 100
    else:
        log_comparison['Improvement'] = ((log_comparison['Baseline'] - log_comparison['Optimized']) / log_comparison['Baseline'].replace(0, np.nan)) * 100

    print("\n[Analysis 1: Per-Log Comparison (Avg. per Case)]\n")
    print(log_comparison.to_string(formatters={'Baseline': '{:,.2f}'.format, 'Optimized': '{:,.2f}'.format, 'Improvement': '{:,.2f}%'.format}))
    
    # --- Per-Case Analysis ---
    if is_case_metric:
        baseline_case_agg = baseline_df.groupby('case_id')[metric_col].sum()
        optimized_case_agg = optimized_df.groupby('case_id')[metric_col].sum()
        
        case_comparison = pd.concat([baseline_case_agg, optimized_case_agg], axis=1, keys=['Baseline', 'Optimized']).fillna(0)

        if higher_is_better:
            case_comparison['Improvement'] = ((case_comparison['Optimized'] - case_comparison['Baseline']) / case_comparison['Baseline'].replace(0, np.nan)) * 100
        else:
            case_comparison['Improvement'] = ((case_comparison['Baseline'] - case_comparison['Optimized']) / case_comparison['Baseline'].replace(0, np.nan)) * 100

        print("\n\n[Analysis 2: Per-Case Sum (Aggregated Across all Logs)]\n")
        print(case_comparison.to_string(formatters={'Baseline': '{:,.2f}'.format, 'Optimized': '{:,.2f}'.format, 'Improvement': '{:,.2f}%'.format}))

    # --- Overall Summary ---
    overall_base = log_comparison['Baseline'].mean()
    overall_opt = log_comparison['Optimized'].mean()
    
    if higher_is_better:
        overall_improvement = ((overall_opt - overall_base) / overall_base) * 100 if overall_base != 0 else 0
    else:
        overall_improvement = ((overall_base - overall_opt) / overall_base) * 100 if overall_base != 0 else 0

    print("\n\n--- Overall Summary ---")
    print(f"  Avg Baseline ({unit}):  {overall_base:,.2f}")
    print(f"  Avg Optimized ({unit}): {overall_opt:,.2f}")
    print(f"  Avg Improvement:      {overall_improvement:,.2f}%")
    print("-" * 60 + "\n")
    
    return overall_improvement

# ==============================================================================
# --- 4. MAIN SCRIPT LOGIC ---
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark simulation results.')
    parser.add_argument('--baseline', default=DEFAULT_BASELINE_DIR, help='Name of the baseline results directory.')
    parser.add_argument('--optimized', default=DEFAULT_OPTIMIZED_DIR, help='Name of the optimized results directory.')
    parser.add_argument('--weights', type=str, default="Not specified", help='String representation of the optimizer weights used.')
    args = parser.parse_args()

    base_path = os.path.join('simulated_data', LOG_NAME)
    baseline_dir = os.path.join(base_path, args.baseline)
    optimized_dir = os.path.join(base_path, args.optimized)

    if not os.path.exists(baseline_dir) or not os.path.exists(optimized_dir):
        print(f"FATAL ERROR: Could not find directories.\n  Baseline: '{baseline_dir}'\n  Optimized: '{optimized_dir}'")
        sys.exit(1)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"benchmark_report_{timestamp_str}.txt"
    
    class Tee(object):
        def __init__(self, *files): self.files = files
        def write(self, obj):
            for f in self.files: f.write(obj); f.flush()
        def flush(self):
            for f in self.files: f.flush()

    original_stdout = sys.stdout
    with open(report_filename, 'w') as report_file:
        sys.stdout = Tee(sys.stdout, report_file)
        
        print("=" * 60)
        print(f"BENCHMARKING REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"Baseline Directory:  {args.baseline}")
        print(f"Optimized Directory: {args.optimized}")
        print(f"Optimizer Weights:   {args.weights}\n")

        try:
            baseline_dfs = [pd.read_csv(os.path.join(baseline_dir, f)).assign(log_num=i) for i, f in enumerate(sorted(os.listdir(baseline_dir))) if f.startswith('simulated_log_')]
            optimized_dfs = [pd.read_csv(os.path.join(optimized_dir, f)).assign(log_num=i) for i, f in enumerate(sorted(os.listdir(optimized_dir))) if f.startswith('simulated_log_')]
            
            df_base_all = enrich_log_data(pd.concat(baseline_dfs, ignore_index=True))
            df_opt_all = enrich_log_data(pd.concat(optimized_dfs, ignore_index=True))
        except Exception as e:
            print(f"FATAL ERROR during data loading: {e}")
            sys.exit(1)

        summary = {}
        # Correctly call generate_report for all case-level metrics
        summary['Cycle Time'] = generate_report(df_base_all, df_opt_all, 'cycle_time', 'Cycle Time (Case Duration)', 'seconds', is_case_metric=True)
        summary['Wait Time'] = generate_report(df_base_all, df_opt_all, 'wait_time', 'Total Wait Time per Case', 'seconds', is_case_metric=True)
        summary['Task Cost'] = generate_report(df_base_all, df_opt_all, 'task_cost', 'Total Task Cost per Case', '$', is_case_metric=True)
        
        # Log-level metrics
        summary['Gini Improvement'] = generate_report(df_base_all, df_opt_all, calculate_gini_coefficient, 'Load Balancing (Gini)', 'coeff.', is_case_metric=False)
        summary['Avg Agents Per Case'] = generate_report(df_base_all, df_opt_all, calculate_avg_agents_per_case, 'Avg. Unique Agents Per Case', 'agents', is_case_metric=False)
        summary['Total Unique Agents'] = generate_report(df_base_all, df_opt_all, calculate_total_unique_agents, 'Total Unique Agents Used', 'agents', is_case_metric=False)

    sys.stdout = original_stdout
    print("\n" + "="*50)
    print("       QUICK SUMMARY OF RESULTS")
    print("="*50)
    print(f"  Cycle Time Improvement:           {summary.get('Cycle Time', 0):>8.2f}%")
    print(f"  Wait Time Reduction:              {summary.get('Wait Time', 0):>8.2f}%")
    print(f"  Task Cost Reduction:              {summary.get('Task Cost', 0):>8.2f}%")
    print(f"  Load Balancing (Gini) Impr.:     {summary.get('Gini Improvement', 0):>8.2f}%")
    print(f"  Avg Agents/Case Reduction:        {summary.get('Avg Agents Per Case', 0):>8.2f}%")
    print(f"  Total Unique Agents Reduction:    {summary.get('Total Unique Agents', 0):>8.2f}%")
    print("="*50)
    print(f"\nFull detailed report saved to: {report_filename}")

if __name__ == '__main__':
    main()