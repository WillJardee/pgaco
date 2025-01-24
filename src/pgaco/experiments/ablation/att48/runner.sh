#!/usr/bin/env bash

PYTHON="python3"

# Function to kill all child processes when the script is terminated
cleanup() {
  echo "Killing all child processes..."
  kill $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 $PID9 $PID10 $PID11 $PID12
}

# Set up a trap to catch exit signals and run the cleanup function
trap cleanup EXIT

# Start the first set of processes
$PYTHON ACO_replay.py &
PID1=$!
$PYTHON ACO_replay_big.py &
PID2=$!
$PYTHON ACO_replay_none.py &
PID3=$!
wait

# Start the second set of processes
$PYTHON ACOPG_replay.py &
PID4=$!
$PYTHON ACOPG_replay_big.py &
PID5=$!
$PYTHON ACOPG_replay_none.py &
PID6=$!
wait

# Start the third set of processes
$PYTHON ACOPPO_replay.py &
PID7=$!
$PYTHON ACOPPO_replay_big.py &
PID8=$!
$PYTHON ACOPPO_replay_none.py &
PID9=$!
wait

# Start the fourth set of processes
$PYTHON ADACO_replay.py &
PID10=$!
$PYTHON ADACO_replay_big.py &
PID11=$!
$PYTHON ADACO_replay_none.py &
PID12=$!
wait

# Start the fifth set of processes
$PYTHON MMACO_replay.py &
PID13=$!
$PYTHON MMACO_replay_big.py &
PID14=$!
$PYTHON MMACO_replay_none.py &
PID15=$!
wait

# Start the sixth set of processes
$PYTHON ACOSGD_replay.py &
PID16=$!
$PYTHON ACOSGD_replay_big.py &
PID17=$!
$PYTHON ACOSGD_replay_none.py &
PID18=$!
wait

# Notify when the scripts are finished
notify-send "Python Script Finished" "The pr76 script completed!" -i dialog-information

