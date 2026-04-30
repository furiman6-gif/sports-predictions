@echo off
setlocal
cd /d "%~dp0.."
python ligi\auto_future_optimizer.py
if errorlevel 1 (
  echo BLAD: auto_future_optimizer.py zakonczyl sie bledem.
  exit /b 1
)
echo GOTOWE: wyniki sa w folderze ligi\auto_outputs_future
exit /b 0
