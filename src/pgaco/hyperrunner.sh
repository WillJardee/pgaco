#!/usr/bin/env bash

# Array of hyperparameters
declare -a RHO=("0.10" "0.25")
declare -a LEARNING_RATE=("10" "100" "1000")
declare -a POP_SIZE=("10")

# Constants
RUNS=1
ALPHA=1
BETA=2
# GRAPHNAME="att48.tsp"
GRAPHNAME=1000
ITERS=1000

# Trap SIGINT (Ctrl+C) and ensure all child processes are killed
trap 'echo "Terminating all child processes..."; kill 0' SIGINT

# Loop through all combinations of hyperparameters
for rho in "${RHO[@]}"; do
  for learning_rate in "${LEARNING_RATE[@]}"; do
    for pop_size in "${POP_SIZE[@]}"; do
      python plot_run.py $RUNS $rho $ALPHA $BETA $pop_size $GRAPHNAME $ITERS $learning_rate "True"&
    done
  done
done

# Wait for all child processes to complete
wait

echo "Processes completed"

