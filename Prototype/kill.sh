#!/bin/bash

# Find all PIDs of running streamlit processes
PIDS=$(ps aux | grep streamlit | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No streamlit processes found."
else
    echo "Killing streamlit processes: $PIDS"
    kill -9 $PID

fi
