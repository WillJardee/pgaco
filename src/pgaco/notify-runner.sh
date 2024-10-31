#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT=$1

# Run the Python script
python3 "$PYTHON_SCRIPT"

# Capture the exit status
EXIT_STATUS=$?

# Check the exit status and send a notification
if [ $EXIT_STATUS -eq 0 ]; then
    notify-send "Python Script Finished" "The script completed successfully!" -i dialog-information
else
    notify-send "Python Script Failed" "The script encountered an error. Exit code: $EXIT_STATUS" -i dialog-error
fi
