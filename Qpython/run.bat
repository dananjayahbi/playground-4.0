@echo off
echo Activating the Conda environment...
call conda activate Deeplearn
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate the Conda environment.
    exit /b %ERRORLEVEL%
)
echo Starting the application...
python test.py
pause