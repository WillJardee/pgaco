#!/usr/bin/env bash

# Array of hyperparameters
declare -a RHO=("0.10")
declare -a LEARNING_RATE=("0.01" "0.1" "1")
declare -a POP_SIZE=("10")

# Constants
RUNS=1
ALPHA=1
BETA=2
# GRAPHNAME="att48.tsp"
GRAPHNAME=100
ITERS=100

# Trap SIGINT (Ctrl+C) and ensure all child processes are killed
trap 'echo "Terminating all child processes..."; kill 0' SIGINT

# Loop through all combinations of hyperparameters
for rho in "${RHO[@]}"; do
  for learning_rate in "${LEARNING_RATE[@]}"; do
    for pop_size in "${POP_SIZE[@]}"; do
      python plot_run.py $RUNS $rho $ALPHA $BETA $pop_size $GRAPHNAME $ITERS $learning_rate "False"&
    done
  done
done

# Wait for all child processes to complete
wait

echo "Processes completed"

