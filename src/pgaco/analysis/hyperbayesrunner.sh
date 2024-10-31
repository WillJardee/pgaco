#!/usr/bin/env bash

GRAPHNAME=att48.tsp

# Trap SIGINT (Ctrl+C) and ensure all child processes are killed
trap 'echo "Terminating all child processes..."; kill 0' SIGINT

# Loop through all combinations of hyperparameters
python bayesian_tuning_aco.py "$GRAPHNAME" > aco_tuning.txt &
python bayesian_tuning_pgaco-log.py "$GRAPHNAME" > pgaco-log_tuning.txt &
python bayesian_tuning_pgaco-ratio.py "$GRAPHNAME" > pgaco-ratio_tuning.txt &
python bayesian_tuning_pgaco-ratio-clip.py "$GRAPHNAME" > pgaco-ratio-clip_tunign.txt &

# Wait for all child processes to complete
wait

echo "Processes completed"

