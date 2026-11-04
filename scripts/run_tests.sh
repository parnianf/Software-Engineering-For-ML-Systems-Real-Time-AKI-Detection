#!/bin/bash
set -e
project_root=$(pwd)
export PYTHONPATH="$PYTHONPATH:$project_root/src"
export LOG_LEVEL="CRITICAL"
export PERSISTENCE_PATH="/tmp/patients.pkl"
export JOURNAL_PATH="/tmp/journal.jsonl"
echo "--- Running Tests ---"
python -m unittest discover -b tests
echo "--- Tests Complete ---"