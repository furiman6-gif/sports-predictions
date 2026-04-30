@echo off
setlocal
set ROOT=%~dp0

echo ============================================
echo  TENISS_FINAL — trening modelu
echo ============================================
echo.
echo Tryb splitu [auto / 80_10_10 / 75_12_5_12_5] (Enter=auto):
echo Nawierzchnia [Hard/Clay/Grass/Enter=wszystkie]:
echo.

python "%ROOT%gbm4_tenis.py"
if errorlevel 1 goto :fail

echo.
echo Model wytrenowany. Wyniki w outputs_tenis\
pause
exit /b 0

:fail
echo BLAD podczas treningu.
pause
exit /b 1
