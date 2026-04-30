@echo off
setlocal
set SCRIPT_DIR=%~dp0
python "%SCRIPT_DIR%pobierz_i_dopisz_zwykla.py" %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
