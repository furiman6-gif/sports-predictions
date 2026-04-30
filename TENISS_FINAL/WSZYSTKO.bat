@echo off
setlocal
set ROOT=%~dp0
set STATS=%ROOT%stats\
set LOG=%ROOT%logs\wszystko.log

if not exist "%ROOT%logs\" mkdir "%ROOT%logs\"

echo ============================================
echo  TENISS_FINAL — PEŁNY PIPELINE + TRENING
echo  %DATE% %TIME%
echo ============================================
echo [%DATE% %TIME%] START >> "%LOG%"

echo.
echo ============================================
echo  KROK 1/2: Pipeline dzienny (xlsx + scraper2 + extended)
echo ============================================
python "%STATS%pipeline_dzienny.py"
if errorlevel 1 goto :fail

echo.
echo ============================================
echo  KROK 2/2: Trening + predykcje
echo ============================================
python "%ROOT%gbm4_tenis.py"
if errorlevel 1 goto :fail

echo.
echo ============================================
echo  GOTOWE - predykcje w: outputs_tenis\
echo ============================================
echo [%DATE% %TIME%] SUKCES >> "%LOG%"

if exist "%ROOT%outputs_tenis\tenis_ALL\future_predictions.csv" (
    echo.
    echo === FUTURE PREDICTIONS ===
    type "%ROOT%outputs_tenis\tenis_ALL\future_predictions.csv"
)

pause
exit /b 0

:fail
echo [%DATE% %TIME%] BLAD >> "%LOG%"
echo.
echo BLAD - sprawdz log: %LOG%
pause
exit /b 1
