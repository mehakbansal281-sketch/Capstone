@echo off
echo ==========================================
echo  Running Automated Unit ^& Integration Tests
echo ==========================================

set PYTHONPATH=.

if exist ".venv\Scripts\pytest.exe" (
    .venv\Scripts\pytest.exe tests\ -v --tb=short
) else (
    pytest tests\ -v --tb=short
)

if %ERRORLEVEL% EQU 0 (
    echo ==========================================
    echo  All Unit Tests Passed Successfully!
    echo ==========================================
) else (
    echo ==========================================
    echo  Test Failure Detected!
    echo ==========================================
    exit /b %ERRORLEVEL%
)
