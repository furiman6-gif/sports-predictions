@echo off
setlocal
set SCRIPT_DIR=%~dp0
set TRAIN_SEASONS=%~1
if "%TRAIN_SEASONS%"=="" set TRAIN_SEASONS=5
set LOG_PATH=%SCRIPT_DIR%csv\train_backtest_%TRAIN_SEASONS%sezony.log
echo Trening modelu i backtest na final_2_projekt.csv
echo Sezony treningowe: %TRAIN_SEASONS%
echo Log: %LOG_PATH%
python "%SCRIPT_DIR%..\main.py" --history-csv "%SCRIPT_DIR%csv\final_2_projekt.csv" --upcoming-csv "%SCRIPT_DIR%csv\all_seasons_upcoming_project.csv" --output-dir "%SCRIPT_DIR%csv" --train-seasons %TRAIN_SEASONS% > "%LOG_PATH%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
