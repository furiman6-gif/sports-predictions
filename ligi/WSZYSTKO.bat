@echo off
cd /d "%~dp0"

echo ========================================
echo  KROK 1/4: Sciaganie fixtures z API
echo ========================================
python fetch_fixtures.py

echo.
echo ========================================
echo  KROK 2/4: Dopisywanie przyszlych meczow do 22 lig
echo ========================================
python append_fixtures_22.py

echo.
echo ========================================
echo  KROK 3/4: Rankingi (bars6 + Siamese) dla 22 lig
echo ========================================

echo [1/22] Belgium - First Division A...
python bars6.py "Belgium\First_Division_A\wszystkie_sezony.csv"
echo [2/22] England - Premier League...
python bars6.py "England\Premier_League\wszystkie_sezony.csv"
echo [3/22] England - Championship...
python bars6.py "England\Championship\wszystkie_sezony.csv"
echo [4/22] England - League One...
python bars6.py "England\League_One\wszystkie_sezony.csv"
echo [5/22] England - League Two...
python bars6.py "England\League_Two\wszystkie_sezony.csv"
echo [6/22] England - Conference...
python bars6.py "England\Conference\wszystkie_sezony.csv"
echo [7/22] France - Ligue 1...
python bars6.py "France\Ligue_1\wszystkie_sezony.csv"
echo [8/22] France - Ligue 2...
python bars6.py "France\Ligue_2\wszystkie_sezony.csv"
echo [9/22] Germany - Bundesliga 1...
python bars6.py "Germany\Bundesliga_1\wszystkie_sezony.csv"
echo [10/22] Germany - Bundesliga 2...
python bars6.py "Germany\Bundesliga_2\wszystkie_sezony.csv"
echo [11/22] Greece - Super League...
python bars6.py "Greece\Super_League\wszystkie_sezony.csv"
echo [12/22] Italy - Serie A...
python bars6.py "Italy\Serie_A\wszystkie_sezony.csv"
echo [13/22] Italy - Serie B...
python bars6.py "Italy\Serie_B\wszystkie_sezony.csv"
echo [14/22] Netherlands - Eredivisie...
python bars6.py "Netherlands\Eredivisie\wszystkie_sezony.csv"
echo [15/22] Portugal - Primeira Liga...
python bars6.py "Portugal\Primeira_Liga\wszystkie_sezony.csv"
echo [16/22] Scotland - Premiership...
python bars6.py "Scotland\Premiership\wszystkie_sezony.csv"
echo [17/22] Scotland - Championship...
python bars6.py "Scotland\Championship\wszystkie_sezony.csv"
echo [18/22] Scotland - League One...
python bars6.py "Scotland\League_One\wszystkie_sezony.csv"
echo [19/22] Scotland - League Two...
python bars6.py "Scotland\League_Two\wszystkie_sezony.csv"
echo [20/22] Spain - La Liga...
python bars6.py "Spain\La_Liga\wszystkie_sezony.csv"
echo [21/22] Spain - Segunda Division...
python bars6.py "Spain\Segunda_Division\wszystkie_sezony.csv"
echo [22/22] Turkey - Super Lig...
python bars6.py "Turkey\Super_Lig\wszystkie_sezony.csv"

echo.
echo ========================================
echo  KROK 4/4: Trening + predykcje (22 ligi)
echo ========================================
python auto_future_optimizer.py

echo.
echo ========================================
echo  GOTOWE - typy w: auto_outputs_future\
echo ========================================
python typy_dzis.py
pause
