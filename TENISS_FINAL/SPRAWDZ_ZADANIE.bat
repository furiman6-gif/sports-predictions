@echo off
echo Status zadania automatycznej aktualizacji:
echo.
schtasks /query /tn "TenissFinal_Aktualizacja" /fo list /v 2>nul || echo Zadanie nie istnieje. Uruchom USTAW_AUTOSTART.bat
echo.
echo Ostatnie linie logu:
echo.
if exist "%~dp0logs\aktualizacja.log" (
    powershell -command "Get-Content '%~dp0logs\aktualizacja.log' -Tail 20"
) else (
    echo Brak logu - zadanie jeszcze nie uruchamiano.
)
pause
