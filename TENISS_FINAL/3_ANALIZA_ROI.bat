@echo off
setlocal
set ROOT=%~dp0

echo ============================================
echo  TENISS_FINAL — analiza ROI (test 2024-2026)
echo ============================================
echo.
echo Ksiazki: Avg / Max / B365 / PS
echo.

python "%ROOT%analysis_odds_roi.py" --odds Avg
echo.
python "%ROOT%analysis_odds_roi.py" --odds Max
pause
