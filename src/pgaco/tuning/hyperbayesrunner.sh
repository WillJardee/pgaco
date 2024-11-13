#!/usr/bin/env bash

# Trap SIGINT (Ctrl+C) and ensure all child processes are killed
trap 'echo "Terminating all child processes..."; kill 0' SIGINT

# Loop through all combinations of hyperparameters
python tuning_aco.py &
python tuning_acopg-clip.py &
python tuning_acopg.py &
python tuning_acosgd-l2.py &
python tuning_acosgd.py &
python tuning_adaco.py &
python tuning_antq.py &
python tuning_minmaxaco.py &

# Wait for all child processes to complete
wait

echo "Processes completed"

