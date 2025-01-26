#!/usr/bin/env bash

PYTHON="python3"

# Function to kill all child processes when the script is terminated
cleanup() {
  echo "Killing all child processes..."
  kill $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7
}

# Set up a trap to catch exit signals and run the cleanup function
trap cleanup EXIT

$PYTHON ACO.py &
PID1=$!
wait

$PYTHON ACOPPO.py &
PID2=$!
wait

$PYTHON ACOPG.py &
PID3=$!
wait

$PYTHON MMACO.py &
PID4=$!
wait

$PYTHON ACOSGD.py &
PID5=$!
wait

$PYTHON ADACO.py &
PID6=$!
wait

# Notify when the scripts are finished
notify-send "Python Script Finished" "The pr76 script completed!" -i dialog-information

