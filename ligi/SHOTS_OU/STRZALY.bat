@echo off
cd /d "%~dp0"
echo ========================================
echo  SHOTS O/U - 21 lig, 3 strony, 3 linie
echo ========================================
python train_shots.py
echo.
pause
