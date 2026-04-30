@echo off
cd /d "%~dp0"
echo ========================================
echo  KROK 1: Sciaganie fixtures z API
echo ========================================
python fetch_fixtures.py
echo.
echo ========================================
echo  KROK 2: Dopisywanie 22 lig
echo ========================================
python append_fixtures_22.py
echo.
echo ========================================
echo  KROK 3: Trening i predykcje (22 ligi)
echo ========================================
python auto_future_optimizer.py
echo.
echo GOTOWE! Wyniki w: auto_outputs_future\
pause
