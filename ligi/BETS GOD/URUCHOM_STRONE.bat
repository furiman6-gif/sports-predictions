@echo off
cd /d "%~dp0"
echo ========================================
echo  BETS GOD - dashboard Streamlit
echo  Wszystkie 17 targetow (FTR/O25/BTTS/kartki/
echo  faule/strzaly/rozne/gole H/A...)
echo  Otwiera sie w przegladarce: localhost:8502
echo ========================================
python -m streamlit run app.py --server.port 8502
pause
