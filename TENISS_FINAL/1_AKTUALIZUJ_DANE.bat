@echo off
setlocal
set ROOT=%~dp0
set STATS=%ROOT%stats\
set LOG=%ROOT%logs\aktualizacja.log

if not exist "%ROOT%logs\" mkdir "%ROOT%logs\"

echo ============================================
echo  TENISS_FINAL — aktualizacja danych
echo  %DATE% %TIME%
echo ============================================

echo [%DATE% %TIME%] START >> "%LOG%"

echo.
echo [1/1] Pobieranie nowych meczow + dopisanie do CSV...
python "%STATS%aktualizuj_przyrostowo.py"
if errorlevel 1 (
    echo BLAD >> "%LOG%"
    goto :fail
)
echo OK >> "%LOG%"

echo.
echo [%DATE% %TIME%] SUKCES >> "%LOG%"
echo ============================================
echo  Dane zaktualizowane. Uruchom 2_TRENUJ.bat
echo ============================================
exit /b 0

:fail
echo [%DATE% %TIME%] BLAD >> "%LOG%"
echo.
echo BLAD. Sprawdz log: %LOG%
pause
exit /b 1
