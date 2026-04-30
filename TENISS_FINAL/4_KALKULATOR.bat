@echo off
setlocal
set ROOT=%~dp0

echo ============================================
echo  TENISS_FINAL — kalkulator zakładów
echo ============================================
echo.

python "%ROOT%kalkulator_zakładów.py"
pause
