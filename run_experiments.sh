#!/bin/bash

# =================================================================
# --- Overnight Optimization Experiment Script ---
# =================================================================

echo "Starting overnight experiment run..."
start_time=$(date)

# --- Common Parameters ---
LOG_PATH="raw_data/LoanApp.csv.gz"
CASE_ID="case_id"
ACTIVITY="activity"
RESOURCE="resource"
START_TS="start_time"
END_TS="end_time"
COSTS_PATH="costs.json"
# Use smaller GA params for pairwise tests to finish faster
POP_SIZE=50
NUM_GEN=30
N_CORES=7

# Create a directory for the results
mkdir -p experiment_results

# =================================================================
# --- Phase 1: Stability Test ---
# This is a bit tricky with the current script, so we'll simulate it by
# just running the baseline evaluation with high repetitions.
# For a true stability test, you would modify the script to run just the
# fitness function on one policy many times. But this is a good proxy.
# =================================================================
echo "\nPHASE 1: STABILITY TEST"
echo "Running baseline policy with high repetitions to check metric stability..."
# We run the GA for 0 generations, which just evaluates the initial population.
# We set runs_per_fitness high to see the variance.
python optimize_pro.py \
    --log_path "$LOG_PATH" --case_id "$CASE_ID" --activity_name "$ACTIVITY" \
    --resource_name "$RESOURCE" --end_timestamp "$END_TS" --start_timestamp "$START_TS" \
    --costs_path "$COSTS_PATH" --objectives "cost" \
    --pop_size 1 --num_gen 0 --runs_per_fitness 30 --n_cores 1 \
    > experiment_results/stability_test.log 2>&1

echo "Stability test finished. Analyze stability_test.log for variance."


# =================================================================
# --- Phase 2: Pairwise Correlation Test ---
# Assuming 'wait' is unstable and 'agents_per_case'/'total_agents' are correlated,
# we focus on the most promising pairs. You can add more pairs if needed.
# =================================================================
echo "\nPHASE 2: PAIRWISE CORRELATION TESTS"

# Define the pairs of stable metrics to test
# Let's assume 'wait' was too noisy and we test the other 4.
declare -a pairs=(
    "cost,time"
    "cost,agents_per_case"
    "cost,total_agents"
    "time,agents_per_case"
    "time,total_agents"
    "agents_per_case,total_agents"
)

# Loop through the pairs and run the optimization
for pair in "${pairs[@]}"
do
    echo "--- Running test for objectives: $pair ---"
    # Create a clean filename for the log
    filename=$(echo "$pair" | tr ',' '_')
    
    python optimize_pro.py \
        --log_path "$LOG_PATH" --case_id "$CASE_ID" --activity_name "$ACTIVITY" \
        --resource_name "$RESOURCE" --end_timestamp "$END_TS" --start_timestamp "$START_TS" \
        --costs_path "$COSTS_PATH" --objectives "$pair" \
        --pop_size "$POP_SIZE" --num_gen "$NUM_GEN" --n_cores "$N_CORES" \
        > "experiment_results/pairwise_$filename.log" 2>&1
    
    echo "Finished test for $pair. Log saved to experiment_results/pairwise_$filename.log"
done

# =================================================================
echo "\nAll experiments finished!"
end_time=$(date)
echo "Started: $start_time"
echo "Ended:   $end_time"
echo "Results are in the 'experiment_results' directory."