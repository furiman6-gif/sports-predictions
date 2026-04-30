@echo off
cd /d "%~dp0"
echo ========================================
echo  SHOTS O/U - dashboard Streamlit
echo  Otwiera sie w przegladarce: localhost:8501
echo ========================================
python -m streamlit run app.py
pause
