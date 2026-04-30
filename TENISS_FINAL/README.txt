============================================
  TENISS_FINAL — system predykcji tenisowych
============================================

PIERWSZE URUCHOMIENIE:
1. Skopiuj pliki .xls/.xlsx z historycznymi danymi do folderu stats/
   (pobierz z tennis-data.co.uk lub skopiuj z poprzedniego projektu)
2. Skopiuj plik .env (lub uzupełnij .env.example i zmień nazwę na .env)
3. Zainstaluj zależności:
   pip install -r requirements.txt

CODZIENNY WORKFLOW:
1. 1_AKTUALIZUJ_DANE.bat   — pobiera nowe mecze, buduje CSV
2. 2_TRENUJ.bat            — trenuje model, generuje predykcje
3. 3_ANALIZA_ROI.bat       — sprawdza ROI na danych testowych
4. 4_KALKULATOR.bat        — wpisujesz kursy → dostaje stawkę

PLIKI WYNIKOWE (po treningu):
  outputs_tenis/tenis_ALL/
    future_predictions.csv   ← zakłady na nadchodzące mecze
    test_predictions.csv     ← backtest
    feature_importance.csv
    calibration_bins.csv
    calibrator.pkl
    roi_analysis_Avg.csv     ← po uruchomieniu 3_ANALIZA_ROI

STRUKTURA:
  gbm4_tenis.py              — główny model (LightGBM)
  config.py                  — konfiguracja
  .env                       — klucze API, bankroll, Kelly
  core/                      — warstwy danych i skrapowania
  stats/                     — skrypty aktualizacji danych
  analysis_odds_roi.py       — analiza ROI per bin
  analysis_total_games.py    — over/under gemy
  kalkulator_zakładów.py     — kalkulator edge + Kelly
