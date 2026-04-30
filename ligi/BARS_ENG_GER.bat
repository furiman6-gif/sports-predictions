@echo off
cd /d "%~dp0"

echo [1/6] England - Premier League...
python bars5.py "England\Premier_League\wszystkie_sezony.csv"

echo [2/6] England - Championship...
python bars5.py "England\Championship\wszystkie_sezony.csv"

echo [3/6] England - League One...
python bars5.py "England\League_One\wszystkie_sezony.csv"

echo [4/6] England - League Two...
python bars5.py "England\League_Two\wszystkie_sezony.csv"

echo [5/6] Germany - Bundesliga 1...
python bars5.py "Germany\Bundesliga_1\wszystkie_sezony.csv"

echo [6/6] Germany - Bundesliga 2...
python bars5.py "Germany\Bundesliga_2\wszystkie_sezony.csv"

echo.
echo GOTOWE!
pause
