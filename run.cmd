@echo off
cd /d %~dp0

set data_dir=config.ini

@REM venvを使いたい方は下のコメントを省く
@REM call venv/Scripts/activate

echo [START]%time:~0,8%
echo Running...

python "getsteps_ini.py" %data_dir%

echo [END]%time:~0,8% 
echo.
echo Press enter key

pause >nul
