@echo off
cd /d %~dp0

set data_dir=config.ini

rem set python_env_name=[env name]
rem call activate %python_env_name%

echo [START]%time:~0,8%
echo Running...

python "getsteps_ini.py" %data_dir%

echo [END]%time:~0,8% 
echo.
echo Press enter key

pause >nul
