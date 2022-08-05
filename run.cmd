@echo off
cd /d %~dp0

set data_dir=config.ini
set python_env_name=getsteps

call activate %python_env_name%

echo [START]%time:~0,8%
echo Running...

python GetSteps.py %data_dir%

echo [END]%time:~0,8% 
echo.
echo Press enter key

pause >nul
