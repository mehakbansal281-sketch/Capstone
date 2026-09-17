#!/bin/bash
# Test Automation Script for ML Business Solution

set -e

echo "=========================================="
echo " Running Automated Unit & Integration Tests"
echo "=========================================="

export PYTHONPATH=.

if [ -f ".venv/Scripts/pytest.exe" ]; then
    .venv/Scripts/pytest.exe tests/ -v --tb=short
elif [ -f ".venv/bin/pytest" ]; then
    .venv/bin/pytest tests/ -v --tb=short
else
    pytest tests/ -v --tb=short
fi

echo "=========================================="
echo " All Unit Tests Passed Successfully!"
echo "=========================================="
