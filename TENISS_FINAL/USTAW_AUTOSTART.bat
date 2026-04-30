@echo off
setlocal
set ROOT=%~dp0
set TASK_NAME=TenissFinal_Aktualizacja
set BAT_FILE=%ROOT%1_AKTUALIZUJ_DANE.bat

echo ============================================
echo  Ustawianie automatycznej aktualizacji
echo  Zadanie: codziennie o 07:00
echo ============================================
echo.

:: Usuń stare zadanie jeśli istnieje
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

:: Utwórz nowe zadanie — codziennie o 07:00
schtasks /create ^
  /tn "%TASK_NAME%" ^
  /tr "cmd /c \"%BAT_FILE%\"" ^
  /sc daily ^
  /st 07:00 ^
  /ru "%USERNAME%" ^
  /rl highest ^
  /f

if errorlevel 1 (
    echo BLAD podczas tworzenia zadania.
    pause
    exit /b 1
)

echo.
echo Zadanie utworzone: "%TASK_NAME%"
echo Uruchamia sie codziennie o 07:00
echo.
echo Aby zmienic godzine:
echo   schtasks /change /tn "%TASK_NAME%" /st 09:00
echo.
echo Aby wylaczyc:
echo   schtasks /delete /tn "%TASK_NAME%" /f
echo.
echo Aby uruchomic teraz recznie:
echo   schtasks /run /tn "%TASK_NAME%"
echo.

pause
