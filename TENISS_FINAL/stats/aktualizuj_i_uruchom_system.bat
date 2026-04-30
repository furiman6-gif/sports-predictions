@echo off
setlocal
set SCRIPT_DIR=%~dp0

echo [1/5] Pobieranie bazy i live update
python "%SCRIPT_DIR%przygotuj_baze_i_live.py"
if errorlevel 1 goto :fail

echo [2/5] Aktualizacja wersji zwyklej
call "%SCRIPT_DIR%pobierz_i_dopisz_zwykla.bat" --skip-download --skip-live
if errorlevel 1 goto :fail

echo [3/5] Aktualizacja wersji rozszerzonej
call "%SCRIPT_DIR%pobierz_i_dopisz_rozszerzona.bat" --skip-download --skip-live
if errorlevel 1 goto :fail

echo [4/5] Budowanie plikow match-ready
python "%SCRIPT_DIR%..\run_match_ready_generation.py"
if errorlevel 1 goto :fail

echo [5/5] Uruchamianie systemu i rankingu value
python "%SCRIPT_DIR%..\main.py"
if errorlevel 1 goto :fail

echo Gotowe
exit /b 0

:fail
echo Wystapil blad
pause
exit /b 1
