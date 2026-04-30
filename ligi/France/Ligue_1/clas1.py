import os
import re
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import label_binarize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    log_loss, f1_score, classification_report, roc_curve, top_k_accuracy_score
)

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 220)

# ============================================================================
# CONFIG
# ============================================================================
DEFAULT_DATA_CSV = r"final_feature_engineered_v2.csv"
SEED = 42
np.random.seed(SEED)

OPTUNA_TRIALS_LGB = 12
OPTUNA_TRIALS_XGB = 12
OPTUNA_TRIALS_CAT = 10

SAVE_DASHBOARD = True
SAVE_JSON = True
SAVE_CSV = True
MODEL_PKL = "football_3class_models_v2.pkl"

RESULT_TO_CLASS = {"L": 0, "D": 1, "W": 2}
CLASS_TO_RESULT = {v: k for k, v in RESULT_TO_CLASS.items()}
CLASS_NAMES = {0: "LOSS", 1: "DRAW", 2: "WIN"}
CLASS_NAMES_PL = {0: "PRZEGRANA", 1: "REMIS", 2: "WYGRANA"}
N_CLASSES = 3

C_WIN = "#1a9641"
C_DRAW = "#fdae61"
C_LOSS = "#d7191c"
C_PRIMARY = "#2c7bb6"
C_GRAY = "#636363"
CLASS_COLORS = {0: C_LOSS, 1: C_DRAW, 2: C_WIN}

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
})


# ============================================================================
# HELPERS
# ============================================================================
def print_header(text, char="=", width=88):
    print(f"\n{char * width}")
    padding = max(0, (width - len(text) - 2) // 2)
    print(f"{char * padding} {text} {char * padding}")
    print(f"{char * width}\n")


def print_subheader(text, char="-", width=70):
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}")


def sanitize_feature_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    original_columns = df.columns.tolist()
    new_columns = []
    name_mapping = {}

    for col in original_columns:
        new_col = re.sub(r'[\[\]\{\}\":,\'\s]', "_", str(col))
        new_col = re.sub(r"_+", "_", new_col).strip("_")
        if not new_col:
            new_col = f"feature_{len(new_columns)}"

        base_name = new_col
        counter = 1
        while new_col in new_columns:
            new_col = f"{base_name}_{counter}"
            counter += 1

        new_columns.append(new_col)
        name_mapping[new_col] = col

    df_clean = df.copy()
    df_clean.columns = new_columns
    return df_clean, name_mapping


def multiclass_brier_score(y_true, y_proba):
    y_onehot = label_binarize(y_true, classes=[0, 1, 2])
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def detect_date_column(df: pd.DataFrame):
    possible_date_cols = ["date", "Date", "DATE", "match_date", "MatchDate", "datetime", "Datetime"]
    for c in possible_date_cols:
        if c in df.columns:
            return c
    return None


def detect_season_column(df: pd.DataFrame):
    possible_season_cols = ["season", "Season", "SEASON"]
    for c in possible_season_cols:
        if c in df.columns:
            return c
    return None


def extract_season(df):
    season_col = detect_season_column(df)
    if season_col is not None:
        return df[season_col]

    date_col = detect_date_column(df)
    if date_col is not None:
        tmp_date = pd.to_datetime(df[date_col], errors="coerce")
        year = tmp_date.dt.year
        month = tmp_date.dt.month
        season_start = year.where(month >= 8, year - 1)
        return season_start.astype(str) + "/" + (season_start + 1).astype(str).str[-2:]

    return None


def split_by_date_fallback(df, date_col: str):
    df = df.sort_values(date_col).reset_index(drop=True)
    unique_dates = np.array(sorted(pd.to_datetime(df[date_col]).dropna().unique()))
    n = len(unique_dates)

    if n < 6:
        raise ValueError("❌ Za mało unikalnych dat do sensownego splitu future")

    i1 = max(1, int(n * 0.70))
    i2 = max(i1 + 1, int(n * 0.85))
    i2 = min(i2, n - 1)

    train_dates = set(unique_dates[:i1])
    cal_dates = set(unique_dates[i1:i2])
    hold_dates = set(unique_dates[i2:])

    train_mask = df[date_col].isin(train_dates)
    cal_mask = df[date_col].isin(cal_dates)
    hold_mask = df[date_col].isin(hold_dates)

    return train_mask, cal_mask, hold_mask


def get_user_input_seasons(df):
    print_header("KONFIGURACJA PODZIAŁU DANYCH V2", "▓", 88)

    seasons = extract_season(df)
    if seasons is None:
        print("⚠️ Brak kolumny season i brak kolumny date/Date.")
        print("   Użyję fallbacku strict split po dacie, jeśli data istnieje po normalizacji.")
        return None, None, None, None

    df["_season"] = seasons
    unique_seasons = sorted(df["_season"].dropna().unique())
    season_counts = df["_season"].value_counts().sort_index()

    print(f"\n📅 Znalezione sezony ({len(unique_seasons)}):\n")
    print(f"{'Nr':<5} {'Sezon':<15} {'Liczba meczów':<15}")
    print("-" * 36)
    for i, season in enumerate(unique_seasons, 1):
        print(f"{i:<5} {str(season):<15} {int(season_counts.get(season, 0)):<15,}")

    print("\n" + "=" * 50)
    print(f"SUMA: {len(df):,} meczów z wynikiem")
    print("=" * 50)

    while True:
        print("\nWybierz ile OSTATNICH sezonów użyć (min 3):")
        print("  - ostatni sezon = STRICT HOLDOUT")
        print("  - przedostatni = CALIBRATION / TUNING")
        print("  - wcześniejsze = TRAIN")
        print("  - wpisz 0 żeby użyć fallbacku po dacie")

        try:
            user_input = input("\n➤ Ile sezonów: ").strip()
            if user_input == "0":
                return None, None, None, None

            n_seasons = int(user_input)
            if n_seasons < 3 or n_seasons > len(unique_seasons):
                print(f"❌ Wpisz liczbę od 3 do {len(unique_seasons)}")
                continue

            selected = unique_seasons[-n_seasons:]
            holdout_season = [selected[-1]]
            calibration_season = [selected[-2]]
            train_seasons = list(selected[:-2])

            train_n = df[df["_season"].isin(train_seasons)].shape[0]
            cal_n = df[df["_season"].isin(calibration_season)].shape[0]
            hold_n = df[df["_season"].isin(holdout_season)].shape[0]

            print("\n" + "=" * 50)
            print("WYBRANY PODZIAŁ")
            print("=" * 50)
            print(f"TRAIN       : {train_seasons} -> {train_n:,} meczów")
            print(f"CALIBRATION : {calibration_season} -> {cal_n:,} meczów")
            print(f"HOLDOUT     : {holdout_season} -> {hold_n:,} meczów")

            conf = input("\nPotwierdzasz? (T/n): ").strip().lower()
            if conf in ("", "t", "tak", "y", "yes"):
                return train_seasons, calibration_season, holdout_season, selected

        except ValueError:
            print("❌ Wpisz liczbę całkowitą.")
        except KeyboardInterrupt:
            print("\n❌ Anulowano.")
            raise SystemExit


def expected_calibration_error_multiclass(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    eces = []
    for cls in range(y_prob.shape[1]):
        y_bin = (y_true == cls).astype(int)
        p = y_prob[:, cls]
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (p >= bins[i]) & (p <= bins[i + 1])
            else:
                mask = (p >= bins[i]) & (p < bins[i + 1])
            if not mask.any():
                continue
            acc = y_bin[mask].mean()
            conf = p[mask].mean()
            ece += mask.sum() * abs(acc - conf)
        ece /= len(p)
        eces.append(ece)
    return float(np.mean(eces))


def evaluate_multiclass(y_true, proba, pred=None):
    if pred is None:
        pred = proba.argmax(axis=1)
    return {
        "logloss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
        "brier": float(multiclass_brier_score(y_true, proba)),
        "auc_ovr": float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted")),
        "top2_accuracy": float(top_k_accuracy_score(y_true, proba, k=2, labels=[0, 1, 2])),
        "ece_mean": float(expected_calibration_error_multiclass(y_true, proba, n_bins=10)),
    }


def actual_accuracy_by_probability_bins(y_true: np.ndarray, proba_col: np.ndarray, positive_class: int, bin_width: float = 0.05):
    y_true = np.asarray(y_true)
    proba_col = np.asarray(proba_col)

    bins = np.arange(0, 1 + bin_width, bin_width)
    rows = []

    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]

        if i == len(bins) - 2:
            mask = (proba_col >= left) & (proba_col <= right)
        else:
            mask = (proba_col >= left) & (proba_col < right)

        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bin_left": float(left),
                "bin_right": float(right),
                "bin_label": f"{left:.2f}-{right:.2f}",
                "count": 0,
                "avg_pred": np.nan,
                "actual_rate": np.nan,
            })
            continue

        avg_pred = float(proba_col[mask].mean())
        actual_rate = float((y_true[mask] == positive_class).mean())

        rows.append({
            "bin_left": float(left),
            "bin_right": float(right),
            "bin_label": f"{left:.2f}-{right:.2f}",
            "count": n,
            "avg_pred": avg_pred,
            "actual_rate": actual_rate,
        })

    return pd.DataFrame(rows)


def print_bin_summary(title, df_bins):
    print(f"\n📊 {title}")
    print(f"{'Bin':<12} {'Count':>8} {'AvgPred':>10} {'ActualHit':>12}")
    print("-" * 46)
    for _, r in df_bins.iterrows():
        avg_pred = "-" if pd.isna(r["avg_pred"]) else f"{r['avg_pred']:.2%}"
        actual_rate = "-" if pd.isna(r["actual_rate"]) else f"{r['actual_rate']:.2%}"
        print(f"{r['bin_label']:<12} {int(r['count']):>8} {avg_pred:>10} {actual_rate:>12}")


def fit_temperature_multiclass(p_raw: np.ndarray, y_true: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    p_raw = np.clip(np.asarray(p_raw, dtype=float), 1e-7, 1 - 1e-7)
    logits = np.log(p_raw)

    def _apply(T):
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def objective(T):
        p = _apply(T)
        return float(log_loss(y_true, p, labels=[0, 1, 2]))

    res = minimize_scalar(objective, bounds=(0.5, 3.0), method="bounded")
    T_opt = float(res.x)

    def transform(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        z = np.log(p) / T_opt
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    return T_opt, transform


def fit_isotonic_ovr_multiclass(p_raw: np.ndarray, y_true: np.ndarray):
    calibrators = []
    p_cal = np.zeros_like(p_raw)

    for cls in range(p_raw.shape[1]):
        iso = IsotonicRegression(out_of_bounds="clip")
        y_bin = (y_true == cls).astype(int)
        iso.fit(p_raw[:, cls], y_bin)
        p_cal[:, cls] = iso.predict(p_raw[:, cls])
        calibrators.append(iso)

    p_cal = p_cal / p_cal.sum(axis=1, keepdims=True)

    def transform(p):
        p = np.asarray(p, dtype=float)
        out = np.zeros_like(p)
        for cls, iso in enumerate(calibrators):
            out[:, cls] = iso.predict(p[:, cls])
        out = out / out.sum(axis=1, keepdims=True)
        return out

    return calibrators, transform


def optimize_blend_weights_multiclass(y_true: np.ndarray, probas: list[np.ndarray], names: list[str]):
    y_true = np.asarray(y_true, dtype=int)

    def objective(w):
        w = np.clip(np.asarray(w, dtype=float), 0, None)
        if w.sum() == 0:
            return 999.0
        w = w / w.sum()
        p = np.zeros_like(probas[0], dtype=float)
        for wi, pi in zip(w, probas):
            p += wi * pi
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(log_loss(y_true, p, labels=[0, 1, 2]))

    x0 = np.ones(len(probas)) / len(probas)
    bounds = [(0, 1)] * len(probas)
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)

    if res.success:
        w = np.clip(res.x, 0, None)
        w = w / w.sum()
    else:
        w = x0

    blend = np.zeros_like(probas[0], dtype=float)
    for wi, pi in zip(w, probas):
        blend += wi * pi

    return w, blend


# ============================================================================
# OPTUNA
# ============================================================================
def tune_lgb(X_train, y_train, X_cal, y_cal):
    default = dict(
        n_estimators=700, learning_rate=0.015, num_leaves=127,
        min_child_samples=10, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.05, reg_lambda=0.8, objective="multiclass", num_class=3
    )
    if not _OPTUNA_AVAILABLE or OPTUNA_TRIALS_LGB <= 0:
        return default

    print_subheader("OPTUNA — LIGHTGBM")

    def objective(trial):
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        p = np.clip(model.predict_proba(X_cal), 1e-7, 1 - 1e-7)
        score = float(log_loss(y_cal, p, labels=[0, 1, 2]))
        print(f"  trial {trial.number+1}/{OPTUNA_TRIALS_LGB} | logloss={score:.5f}")
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS_LGB)
    best = study.best_params
    best["objective"] = "multiclass"
    best["num_class"] = 3
    return best


def tune_xgb(X_train, y_train, X_cal, y_cal):
    default = dict(
        n_estimators=700, learning_rate=0.02, max_depth=6,
        min_child_weight=6, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.05, reg_lambda=0.8, objective="multi:softprob", num_class=3
    )
    if not _OPTUNA_AVAILABLE or OPTUNA_TRIALS_XGB <= 0:
        return default

    print_subheader("OPTUNA — XGBOOST")

    def objective(trial):
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 15),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "random_state": SEED,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        p = np.clip(model.predict_proba(X_cal), 1e-7, 1 - 1e-7)
        score = float(log_loss(y_cal, p, labels=[0, 1, 2]))
        print(f"  trial {trial.number+1}/{OPTUNA_TRIALS_XGB} | logloss={score:.5f}")
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS_XGB)
    best = study.best_params
    best["objective"] = "multi:softprob"
    best["num_class"] = 3
    return best


def tune_cat(X_train, y_train, X_cal, y_cal):
    default = dict(
        iterations=700, learning_rate=0.02, depth=7, l2_leaf_reg=4,
        bootstrap_type="Bernoulli", subsample=0.85, loss_function="MultiClass"
    )
    if not _OPTUNA_AVAILABLE or OPTUNA_TRIALS_CAT <= 0:
        return default

    print_subheader("OPTUNA — CATBOOST")

    def objective(trial):
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Bayesian"])
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 10.0),
            "bootstrap_type": bootstrap_type,
            "loss_function": "MultiClass",
            "random_seed": SEED,
            "verbose": 0,
            "thread_count": -1,
        }
        if bootstrap_type == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.7, 1.0)
        else:
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        p = np.clip(model.predict_proba(X_cal), 1e-7, 1 - 1e-7)
        score = float(log_loss(y_cal, p, labels=[0, 1, 2]))
        print(f"  trial {trial.number+1}/{OPTUNA_TRIALS_CAT} | logloss={score:.5f}")
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS_CAT)
    best = study.best_params
    best["loss_function"] = "MultiClass"
    return best


# ============================================================================
# PIPELINE
# ============================================================================
def run_rating_pipeline_v2(csv_path):
    print_header("FOOTBALL 3-CLASS PRODUCTION PIPELINE V2", "█", 88)
    print(f"📂 Plik danych: {csv_path}")
    print(f"🎯 Cel: 3 klasy - LOSS / DRAW / WIN")
    print(f"🧪 Setup: TRAIN -> CALIBRATION -> STRICT HOLDOUT -> FUTURE")

    # ------------------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------------------
    print_subheader("1. WCZYTYWANIE DANYCH")
    df_full = pd.read_csv(csv_path, low_memory=False)
    print(f"✅ Wczytano: {df_full.shape[0]:,} wierszy × {df_full.shape[1]} kolumn")

    if "Date" in df_full.columns and "date" not in df_full.columns:
        df_full["date"] = df_full["Date"]
    if "Season" in df_full.columns and "season" not in df_full.columns:
        df_full["season"] = df_full["Season"]

    df_future = pd.DataFrame()

    if "re_ult" in df_full.columns:
        df_full["result_raw"] = df_full["re_ult"].astype(str).str.upper().str.strip()
        valid_results = {"W", "D", "L"}

        future_mask = ~df_full["result_raw"].isin(valid_results)
        df_future = df_full[future_mask].copy()
        df = df_full[df_full["result_raw"].isin(valid_results)].copy()

        df["Label"] = df["result_raw"].map(RESULT_TO_CLASS)
        print(f"✅ Mecze z wynikiem: {len(df):,}")
        print(f"🔮 Mecze przyszłe   : {len(df_future):,}")

    elif "home_pts" in df_full.columns and "away_pts" in df_full.columns:
        future_mask = df_full["home_pts"].isna() | df_full["away_pts"].isna()
        df_future = df_full[future_mask].copy()
        df = df_full[~future_mask].copy()

        conditions = [
            df["home_pts"] < df["away_pts"],
            df["home_pts"] == df["away_pts"],
            df["home_pts"] > df["away_pts"]
        ]
        df["Label"] = np.select(conditions, [0, 1, 2], default=-1)
        df = df[df["Label"] >= 0].copy()
        print(f"✅ Mecze z wynikiem: {len(df):,}")
        print(f"🔮 Mecze przyszłe   : {len(df_future):,}")
    else:
        raise ValueError("❌ Brak kolumny docelowej: re_ult lub home_pts/away_pts")

    date_col = detect_date_column(df)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    future_date_col = detect_date_column(df_future)
    if future_date_col is not None:
        df_future[future_date_col] = pd.to_datetime(df_future[future_date_col], errors="coerce")

    print("\n📊 Rozkład klas docelowych:")
    for cls in [0, 1, 2]:
        cnt = (df["Label"] == cls).sum()
        pct = cnt / len(df) * 100
        print(f"   {cls} ({CLASS_NAMES[cls]:>4} / {CLASS_NAMES_PL[cls]:<10}): {cnt:,} ({pct:.2f}%)")

    # ------------------------------------------------------------------------
    # 2. SPLIT CONFIG
    # ------------------------------------------------------------------------
    train_seasons, calibration_season, holdout_season, selected_seasons = get_user_input_seasons(df)

    use_date_fallback = False
    if selected_seasons is None:
        print("\n⚠️ Brak sezonowego splitu — przechodzę na fallback strict split po dacie.")
        use_date_fallback = True

        date_col = detect_date_column(df)
        if date_col is None:
            raise ValueError("❌ Brak season i brak date/Date — nie da się zrobić future split")

        df = df.sort_values(date_col).reset_index(drop=True)
        train_mask, cal_mask, hold_mask = split_by_date_fallback(df, date_col)

        train_seasons = ["DATE_FALLBACK_TRAIN"]
        calibration_season = ["DATE_FALLBACK_CAL"]
        holdout_season = ["DATE_FALLBACK_HOLDOUT"]

    else:
        df = df[df["_season"].isin(selected_seasons)].copy()
        if len(df_future) > 0 and "_season" not in df_future.columns:
            fut_seasons = extract_season(df_future)
            if fut_seasons is not None:
                df_future["_season"] = fut_seasons

        train_mask = df["_season"].isin(train_seasons)
        cal_mask = df["_season"].isin(calibration_season)
        hold_mask = df["_season"].isin(holdout_season)

    print_subheader("2. PODZIAŁ V2")
    print(f"TRAIN       : {train_seasons}")
    print(f"CALIBRATION : {calibration_season}")
    print(f"HOLDOUT     : {holdout_season}")

    # ------------------------------------------------------------------------
    # 3. FEATURES
    # ------------------------------------------------------------------------
    print_subheader("3. PRZYGOTOWANIE CECH")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"Label", "home_pts", "away_pts", "game_id", "target", "_season"}
    feat_cols = [c for c in num_cols if c not in drop_cols]

    print(f"📊 Liczba cech numerycznych: {len(feat_cols)}")
    print(f"📊 Pierwsze 25 cech: {', '.join(feat_cols[:25])}")

    df_features = df[feat_cols].copy()
    df_features, name_mapping = sanitize_feature_names(df_features)
    feat_cols_clean = list(df_features.columns)

    train_med = df_features.loc[train_mask].median()
    df_features = df_features.fillna(train_med)

    X_tr = df_features.loc[train_mask].values
    y_tr = df.loc[train_mask, "Label"].values.astype(int)

    X_cal = df_features.loc[cal_mask].values
    y_cal = df.loc[cal_mask, "Label"].values.astype(int)

    X_ho = df_features.loc[hold_mask].values
    y_ho = df.loc[hold_mask, "Label"].values.astype(int)

    X_future = None
    if len(df_future) > 0:
        available_future_cols = [c for c in feat_cols if c in df_future.columns]
        missing_future_cols = [c for c in feat_cols if c not in df_future.columns]

        if len(available_future_cols) > 0:
            df_future_features = df_future[available_future_cols].copy()
            for mc in missing_future_cols:
                df_future_features[mc] = np.nan
            df_future_features = df_future_features[feat_cols]
            df_future_features, _ = sanitize_feature_names(df_future_features)
            df_future_features.columns = feat_cols_clean
            df_future_features = df_future_features.fillna(train_med)
            X_future = df_future_features.values

    print(f"\n📚 Train       : {len(X_tr):,}")
    print(f"🧪 Calibration : {len(X_cal):,}")
    print(f"🧱 Holdout     : {len(X_ho):,}")
    if X_future is not None:
        print(f"🔮 Future      : {len(X_future):,}")

    # ------------------------------------------------------------------------
    # 4. TUNE
    # ------------------------------------------------------------------------
    print_subheader("4. STROJENIE HIPERPARAMETRÓW")
    lgb_params = tune_lgb(X_tr, y_tr, X_cal, y_cal)
    xgb_params = tune_xgb(X_tr, y_tr, X_cal, y_cal)
    cat_params = tune_cat(X_tr, y_tr, X_cal, y_cal)

    # ------------------------------------------------------------------------
    # 5. TRAIN BASE MODELS
    # ------------------------------------------------------------------------
    print_subheader("5. TRENING MODELI BAZOWYCH")

    print("\n🔄 Trening final LightGBM...")
    lgb_m = lgb.LGBMClassifier(**lgb_params, random_state=SEED, n_jobs=-1, verbose=-1)
    lgb_m.fit(X_tr, y_tr)

    print("\n🔄 Trening final XGBoost...")
    xgb_m = xgb.XGBClassifier(
        **xgb_params,
        random_state=SEED,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        verbosity=0
    )
    xgb_m.fit(X_tr, y_tr)

    print("\n🔄 Trening final CatBoost...")
    cat_m = CatBoostClassifier(
        **cat_params,
        random_seed=SEED,
        verbose=0,
        thread_count=-1
    )
    cat_m.fit(X_tr, y_tr)

    # ------------------------------------------------------------------------
    # 6. CALIBRATION SET PREDICTIONS
    # ------------------------------------------------------------------------
    print_subheader("6. BLEND + KALIBRACJA NA CALIBRATION SET")

    p_cal_lgb = np.clip(lgb_m.predict_proba(X_cal), 1e-7, 1 - 1e-7)
    p_cal_xgb = np.clip(xgb_m.predict_proba(X_cal), 1e-7, 1 - 1e-7)
    p_cal_cat = np.clip(cat_m.predict_proba(X_cal), 1e-7, 1 - 1e-7)

    base_cal_models = {
        "LightGBM": p_cal_lgb,
        "XGBoost": p_cal_xgb,
        "CatBoost": p_cal_cat,
    }

    print("\n📊 Calibration-set metrics (base models):")
    base_cal_metrics = []
    for name, p in base_cal_models.items():
        m = evaluate_multiclass(y_cal, p)
        base_cal_metrics.append({"Model": name, **m})
        print(
            f"{name:<12} | "
            f"logloss={m['logloss']:.5f} | "
            f"brier={m['brier']:.5f} | "
            f"auc={m['auc_ovr']:.5f} | "
            f"acc={m['accuracy']:.5f}"
        )

    probas_list = [p_cal_lgb, p_cal_xgb, p_cal_cat]
    names_list = ["LightGBM", "XGBoost", "CatBoost"]
    blend_weights, p_cal_blend = optimize_blend_weights_multiclass(y_cal, probas_list, names_list)

    print("\n📊 Optymalne wagi blendu:")
    for n, w in zip(names_list, blend_weights):
        print(f"   {n:<10}: {w:.4f}")

    blend_uncal_metrics = evaluate_multiclass(y_cal, p_cal_blend)
    print(
        f"\nBlend(uncal) on calibration | "
        f"logloss={blend_uncal_metrics['logloss']:.5f} | "
        f"brier={blend_uncal_metrics['brier']:.5f} | "
        f"auc={blend_uncal_metrics['auc_ovr']:.5f} | "
        f"acc={blend_uncal_metrics['accuracy']:.5f}"
    )

    print("\n🔄 Wybór kalibratora...")
    T_opt, temp_fn = fit_temperature_multiclass(p_cal_blend, y_cal)
    iso_models, iso_fn = fit_isotonic_ovr_multiclass(p_cal_blend, y_cal)

    cal_candidates = {
        "raw": p_cal_blend,
        "temp": temp_fn(p_cal_blend),
        "isotonic_ovr": iso_fn(p_cal_blend),
    }

    cal_candidate_metrics = {}
    for cname, cp in cal_candidates.items():
        cal_candidate_metrics[cname] = evaluate_multiclass(y_cal, cp)
        m = cal_candidate_metrics[cname]
        print(
            f"   {cname:<12} | "
            f"logloss={m['logloss']:.5f} | "
            f"brier={m['brier']:.5f} | "
            f"ece={m['ece_mean']:.5f}"
        )

    best_cal_name = min(cal_candidate_metrics.keys(), key=lambda k: cal_candidate_metrics[k]["logloss"])
    print(f"\n✅ Wybrany kalibrator: {best_cal_name}")

    if best_cal_name == "temp":
        cal_transform = temp_fn
        calibration_artifact = {"type": "temperature", "T": T_opt}
    elif best_cal_name == "isotonic_ovr":
        cal_transform = iso_fn
        calibration_artifact = {"type": "isotonic_ovr", "models": iso_models}
    else:
        cal_transform = lambda p: p
        calibration_artifact = {"type": "raw"}

    # ------------------------------------------------------------------------
    # 7. HOLDOUT EVAL
    # ------------------------------------------------------------------------
    print_subheader("7. STRICT HOLDOUT EVALUATION")

    p_ho_lgb = np.clip(lgb_m.predict_proba(X_ho), 1e-7, 1 - 1e-7)
    p_ho_xgb = np.clip(xgb_m.predict_proba(X_ho), 1e-7, 1 - 1e-7)
    p_ho_cat = np.clip(cat_m.predict_proba(X_ho), 1e-7, 1 - 1e-7)

    base_hold_models = {
        "LightGBM": p_ho_lgb,
        "XGBoost": p_ho_xgb,
        "CatBoost": p_ho_cat,
    }

    print("\n📊 Holdout metrics (base models):")
    hold_metrics_table = []
    for name, p in base_hold_models.items():
        m = evaluate_multiclass(y_ho, p)
        hold_metrics_table.append({"Model": name, **m})
        print(
            f"{name:<12} | "
            f"logloss={m['logloss']:.5f} | "
            f"brier={m['brier']:.5f} | "
            f"auc={m['auc_ovr']:.5f} | "
            f"acc={m['accuracy']:.5f}"
        )

    p_ho_blend_uncal = (
        blend_weights[0] * p_ho_lgb +
        blend_weights[1] * p_ho_xgb +
        blend_weights[2] * p_ho_cat
    )
    p_ho_blend_uncal = np.clip(p_ho_blend_uncal, 1e-7, 1 - 1e-7)
    p_ho_blend_cal = np.clip(cal_transform(p_ho_blend_uncal), 1e-7, 1 - 1e-7)

    hold_blend_uncal_metrics = evaluate_multiclass(y_ho, p_ho_blend_uncal)
    hold_blend_cal_metrics = evaluate_multiclass(y_ho, p_ho_blend_cal)

    hold_metrics_table.append({"Model": "Blend (uncal)", **hold_blend_uncal_metrics})
    hold_metrics_table.append({"Model": "Blend (cal)", **hold_blend_cal_metrics})

    print("\n📊 Holdout metrics (blend):")
    for name, m in [("Blend (uncal)", hold_blend_uncal_metrics), ("Blend (cal)", hold_blend_cal_metrics)]:
        print(
            f"{name:<12} | "
            f"logloss={m['logloss']:.5f} | "
            f"brier={m['brier']:.5f} | "
            f"auc={m['auc_ovr']:.5f} | "
            f"acc={m['accuracy']:.5f} | "
            f"f1_macro={m['f1_macro']:.5f} | "
            f"ece={m['ece_mean']:.5f}"
        )

    y_pred_hold = p_ho_blend_cal.argmax(axis=1)

    # ------------------------------------------------------------------------
    # 8. FEATURE IMPORTANCE
    # ------------------------------------------------------------------------
    print_subheader("8. FEATURE IMPORTANCE")
    lgb_imp = pd.Series(lgb_m.feature_importances_, index=feat_cols_clean)
    xgb_imp = pd.Series(xgb_m.feature_importances_, index=feat_cols_clean)
    cat_imp = pd.Series(cat_m.get_feature_importance(), index=feat_cols_clean)

    imp_df = pd.concat([lgb_imp, xgb_imp, cat_imp], axis=1)
    imp_df.columns = ["LightGBM", "XGBoost", "CatBoost"]
    imp_df = imp_df.fillna(0)

    norm = imp_df.div(imp_df.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    imp_df["Mean"] = norm.mean(axis=1)
    imp_df["Std"] = norm.std(axis=1)
    imp_df["Original"] = [name_mapping.get(f, f) for f in imp_df.index]

    top30 = imp_df.nlargest(30, "Mean")

    print(f"\n{'Rank':<6} {'Feature':<45} {'Mean':<10}")
    print("-" * 70)
    for i, (_, row) in enumerate(top30.iterrows(), 1):
        print(f"{i:<6} {row['Original'][:43]:<45} {row['Mean']:<10.4f}")

    if SAVE_CSV:
        imp_df_save = imp_df.reset_index()
        imp_df_save.columns = ["Feature_Clean", "LightGBM", "XGBoost", "CatBoost", "Mean", "Std", "Original_Feature"]
        imp_df_save = imp_df_save.sort_values("Mean", ascending=False)
        imp_df_save.to_csv("feature_importance_full_v2.csv", index=False)

    # ------------------------------------------------------------------------
    # 9. FUTURE PREDICTIONS
    # ------------------------------------------------------------------------
    print_subheader("9. FUTURE PREDICTIONS")
    future_predictions_df = None
    if X_future is not None and len(X_future) > 0:
        pf_lgb = np.clip(lgb_m.predict_proba(X_future), 1e-7, 1 - 1e-7)
        pf_xgb = np.clip(xgb_m.predict_proba(X_future), 1e-7, 1 - 1e-7)
        pf_cat = np.clip(cat_m.predict_proba(X_future), 1e-7, 1 - 1e-7)

        pf_blend = (
            blend_weights[0] * pf_lgb +
            blend_weights[1] * pf_xgb +
            blend_weights[2] * pf_cat
        )
        pf_blend = np.clip(pf_blend, 1e-7, 1 - 1e-7)
        pf_blend_cal = np.clip(cal_transform(pf_blend), 1e-7, 1 - 1e-7)

        y_pred_future = pf_blend_cal.argmax(axis=1)
        confidence_future = pf_blend_cal.max(axis=1)
        entropy_future = -np.sum(pf_blend_cal * np.log(pf_blend_cal + 1e-10), axis=1)

        future_predictions_df = pd.DataFrame({
            "predicted": y_pred_future,
            "predicted_label": [CLASS_NAMES[y] for y in y_pred_future],
            "p_loss": pf_blend_cal[:, 0],
            "p_draw": pf_blend_cal[:, 1],
            "p_win": pf_blend_cal[:, 2],
            "confidence": confidence_future,
            "entropy": entropy_future,
        })

        for col in ["date", "Date", "home", "HomeTeam", "away", "AwayTeam", "_season"]:
            if col in df_future.columns:
                new_col = "season" if col == "_season" else col
                future_predictions_df[new_col] = df_future.reset_index(drop=True)[col].values

        print(f"✅ Future predictions gotowe: {len(future_predictions_df):,}")
        print(f"📊 Średnia pewność: {future_predictions_df['confidence'].mean():.4f}")

        if SAVE_CSV:
            future_predictions_df.to_csv("future_predictions_v2.csv", index=False)

    # ------------------------------------------------------------------------
    # 10. HOLDOUT OUTPUTS
    # ------------------------------------------------------------------------
    print_subheader("10. HOLDOUT OUTPUTS")
    holdout_predictions_df = pd.DataFrame({
        "actual": y_ho,
        "actual_label": [CLASS_NAMES[y] for y in y_ho],
        "predicted": y_pred_hold,
        "predicted_label": [CLASS_NAMES[y] for y in y_pred_hold],
        "correct": (y_ho == y_pred_hold).astype(int),
        "p_loss": p_ho_blend_cal[:, 0],
        "p_draw": p_ho_blend_cal[:, 1],
        "p_win": p_ho_blend_cal[:, 2],
        "confidence": p_ho_blend_cal.max(axis=1),
        "entropy": -np.sum(p_ho_blend_cal * np.log(p_ho_blend_cal + 1e-10), axis=1),
    })

    hold_df = df.loc[hold_mask].reset_index(drop=True)
    for col in ["date", "Date", "home", "HomeTeam", "away", "AwayTeam", "_season"]:
        if col in hold_df.columns:
            new_col = "season" if col == "_season" else col
            holdout_predictions_df[new_col] = hold_df[col].values

    if SAVE_CSV:
        holdout_predictions_df.to_csv("holdout_predictions_v2.csv", index=False)

    # ------------------------------------------------------------------------
    # 11. CONFUSION / REPORT
    # ------------------------------------------------------------------------
    print_subheader("11. HOLDOUT CLASSIFICATION REPORT")
    cm = confusion_matrix(y_ho, y_pred_hold, labels=[0, 1, 2])

    print(f"\n                           PREDICTED")
    print(f"                    LOSS      DRAW      WIN")
    print(f"          LOSS     {cm[0][0]:>6}    {cm[0][1]:>6}    {cm[0][2]:>6}")
    print(f"   ACTUAL DRAW     {cm[1][0]:>6}    {cm[1][1]:>6}    {cm[1][2]:>6}")
    print(f"          WIN      {cm[2][0]:>6}    {cm[2][1]:>6}    {cm[2][2]:>6}")

    print("\n📊 Classification Report (HOLDOUT):")
    print(classification_report(y_ho, y_pred_hold, target_names=[CLASS_NAMES[i] for i in range(N_CLASSES)]))

    # ------------------------------------------------------------------------
    # 11B. ACTUAL ACCURACY BY 5% PROBABILITY BINS
    # ------------------------------------------------------------------------
    print_subheader("11B. FAKTYCZNA SKUTECZNOŚĆ CO 5% BIN - HOME/WIN I LOSS")

    win_bins_df = actual_accuracy_by_probability_bins(
        y_true=y_ho,
        proba_col=p_ho_blend_cal[:, 2],
        positive_class=2,
        bin_width=0.05
    )

    loss_bins_df = actual_accuracy_by_probability_bins(
        y_true=y_ho,
        proba_col=p_ho_blend_cal[:, 0],
        positive_class=0,
        bin_width=0.05
    )

    print_bin_summary("WIN / HOME — faktyczna skuteczność w binach 5%", win_bins_df)
    print_bin_summary("LOSS — faktyczna skuteczność w binach 5%", loss_bins_df)

    if SAVE_CSV:
        win_bins_df.to_csv("holdout_actual_accuracy_bins_win_5pct.csv", index=False)
        loss_bins_df.to_csv("holdout_actual_accuracy_bins_loss_5pct.csv", index=False)

    # ------------------------------------------------------------------------
    # 12. DASHBOARD
    # ------------------------------------------------------------------------
    if SAVE_DASHBOARD:
        print_subheader("12. DASHBOARD HOLDOUT")
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        for cls in range(N_CLASSES):
            ax1.hist(p_ho_blend_cal[:, cls], bins=40, alpha=0.5, color=CLASS_COLORS[cls], label=f"P({CLASS_NAMES[cls]})", density=True)
        ax1.set_title("Holdout Probability Distributions")
        ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        confidence_hold = p_ho_blend_cal.max(axis=1)
        ax2.hist(confidence_hold[y_ho == y_pred_hold], bins=40, alpha=0.6, color=C_WIN, label="Correct", density=True)
        ax2.hist(confidence_hold[y_ho != y_pred_hold], bins=40, alpha=0.6, color=C_LOSS, label="Incorrect", density=True)
        ax2.set_title("Holdout Confidence")
        ax2.legend()

        ax3 = fig.add_subplot(gs[0, 2])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=ax3,
            xticklabels=[CLASS_NAMES[i] for i in range(N_CLASSES)],
            yticklabels=[CLASS_NAMES[i] for i in range(N_CLASSES)],
        )
        ax3.set_title("Holdout Confusion Matrix")

        y_onehot = label_binarize(y_ho, classes=[0, 1, 2])
        for cls in range(N_CLASSES):
            ax = fig.add_subplot(gs[1, cls])
            for name, proba, color in [
                ("LGB", p_ho_lgb, C_PRIMARY),
                ("XGB", p_ho_xgb, C_LOSS),
                ("CAT", p_ho_cat, C_WIN),
                ("BLEND", p_ho_blend_cal, "black"),
            ]:
                fpr, tpr, _ = roc_curve(y_onehot[:, cls], proba[:, cls])
                auc = roc_auc_score(y_onehot[:, cls], proba[:, cls])
                ax.plot(fpr, tpr, label=f"{name} ({auc:.3f})", color=color, lw=1.5)
            ax.plot([0, 1], [0, 1], "k:", lw=1)
            ax.set_title(f"ROC Holdout - {CLASS_NAMES[cls]}")
            ax.legend(fontsize=7)

        ax7 = fig.add_subplot(gs[2, 0])
        top15 = imp_df.nlargest(15, "Mean")
        ax7.barh(range(len(top15)), top15["Mean"].values, color=C_PRIMARY)
        ax7.set_yticks(range(len(top15)))
        ax7.set_yticklabels([x[:28] for x in top15["Original"]], fontsize=8)
        ax7.set_title("Top 15 Features")
        ax7.invert_yaxis()

        ax8 = fig.add_subplot(gs[2, 1])
        entropy_hold = -np.sum(p_ho_blend_cal * np.log(p_ho_blend_cal + 1e-10), axis=1)
        ax8.hist(entropy_hold, bins=50, color=C_GRAY, edgecolor="white")
        ax8.axvline(entropy_hold.mean(), color="red", linestyle="--", lw=2, label=f"Mean={entropy_hold.mean():.3f}")
        ax8.set_title("Holdout Entropy")
        ax8.legend()

        ax9 = fig.add_subplot(gs[2, 2])
        bars = ["LGB", "XGB", "CAT", "Blend"]
        vals = [
            evaluate_multiclass(y_ho, p_ho_lgb)["logloss"],
            evaluate_multiclass(y_ho, p_ho_xgb)["logloss"],
            evaluate_multiclass(y_ho, p_ho_cat)["logloss"],
            hold_blend_cal_metrics["logloss"],
        ]
        ax9.bar(bars, vals, color=[C_PRIMARY, C_LOSS, C_WIN, "black"])
        ax9.set_title("Holdout LogLoss")
        ax9.tick_params(axis="x", rotation=20)

        split_title = f"Train={train_seasons} | Cal={calibration_season} | Holdout={holdout_season}"
        plt.suptitle(f"Football 3-Class Production V2\n{split_title}", fontsize=14, y=1.01)
        plt.savefig("evaluation_dashboard_3class_v2.png", bbox_inches="tight", dpi=150)
        plt.show()

    # ------------------------------------------------------------------------
    # 13. SAVE MODEL BUNDLE
    # ------------------------------------------------------------------------
    print_subheader("13. ZAPIS MODELU")
    bundle = {
        "lgb": lgb_m,
        "xgb": xgb_m,
        "cat": cat_m,
        "median": train_med,
        "feat_cols": feat_cols_clean,
        "feat_cols_original": feat_cols,
        "name_mapping": name_mapping,
        "blend_weights": blend_weights,
        "calibration_name": best_cal_name,
        "calibration_artifact": calibration_artifact,
        "class_names": CLASS_NAMES,
        "n_classes": N_CLASSES,
        "train_seasons": train_seasons,
        "calibration_season": calibration_season,
        "holdout_season": holdout_season,
        "holdout_metrics": hold_blend_cal_metrics,
        "lgb_params": lgb_params,
        "xgb_params": xgb_params,
        "cat_params": cat_params,
    }

    with open(MODEL_PKL, "wb") as f:
        pickle.dump(bundle, f)
    print(f"💾 Model zapisany do '{MODEL_PKL}'")

    # ------------------------------------------------------------------------
    # 14. SAVE REPORT JSON
    # ------------------------------------------------------------------------
    if SAVE_JSON:
        print_subheader("14. ZAPIS RAPORTU JSON")
        report = {
            "setup": {
                "train_seasons": train_seasons,
                "calibration_season": calibration_season,
                "holdout_season": holdout_season,
                "future_n": int(len(df_future)),
                "used_date_fallback": bool(use_date_fallback),
            },
            "base_calibration_metrics": base_cal_metrics,
            "blend_weights": {n: float(w) for n, w in zip(names_list, blend_weights)},
            "calibration_candidates_on_cal": cal_candidate_metrics,
            "chosen_calibrator": best_cal_name,
            "holdout_metrics_table": hold_metrics_table,
            "holdout_blend_uncal": hold_blend_uncal_metrics,
            "holdout_blend_cal": hold_blend_cal_metrics,
            "top_features": top30["Original"].tolist(),
            "holdout_actual_accuracy_bins_win_5pct": win_bins_df.to_dict(orient="records"),
            "holdout_actual_accuracy_bins_loss_5pct": loss_bins_df.to_dict(orient="records"),
        }
        with open("report_v2.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("💾 Raport zapisany do 'report_v2.json'")

    # ------------------------------------------------------------------------
    # 15. FINAL SUMMARY
    # ------------------------------------------------------------------------
    print_header("PODSUMOWANIE KOŃCOWE V2", "█", 88)
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                   FOOTBALL 3-CLASS PRODUCTION PIPELINE V2                           ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║ 📚 TRAIN       : {str(train_seasons):<64} ║
║ 🧪 CALIBRATION : {str(calibration_season):<64} ║
║ 🧱 HOLDOUT     : {str(holdout_season):<64} ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║ 📊 Train samples       : {len(X_tr):>10,}                                                   ║
║ 📊 Calibration samples : {len(X_cal):>10,}                                                   ║
║ 📊 Holdout samples     : {len(X_ho):>10,}                                                   ║
║ 🔮 Future samples      : {len(df_future):>10,}                                                   ║
║ 📈 Features            : {len(feat_cols):>10,}                                                   ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║ ✅ HOLDOUT FINAL (BLEND CALIBRATED)                                                     ║
║   • LogLoss     : {hold_blend_cal_metrics['logloss']:<10.5f}                                               ║
║   • Brier       : {hold_blend_cal_metrics['brier']:<10.5f}                                               ║
║   • AUC-OVR     : {hold_blend_cal_metrics['auc_ovr']:<10.5f}                                               ║
║   • Accuracy    : {hold_blend_cal_metrics['accuracy']:<10.5f}                                               ║
║   • F1-macro    : {hold_blend_cal_metrics['f1_macro']:<10.5f}                                               ║
║   • Top2 Acc    : {hold_blend_cal_metrics['top2_accuracy']:<10.5f}                                               ║
║   • ECE mean    : {hold_blend_cal_metrics['ece_mean']:<10.5f}                                               ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║ 🧪 Wybrany kalibrator: {best_cal_name:<58} ║
║ 📁 Model PKL         : {MODEL_PKL:<58} ║
║ 📁 Holdout CSV       : {"holdout_predictions_v2.csv":<58} ║
║ 📁 Future CSV        : {"future_predictions_v2.csv":<58} ║
║ 📁 Dashboard         : {"evaluation_dashboard_3class_v2.png":<58} ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
""")

    if "_season" in df.columns:
        df.drop("_season", axis=1, inplace=True)

    return bundle


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print_header("FOOTBALL 3-CLASS PRODUCTION SYSTEM V2", "█", 88)
    print("""
    ⚽ Production-grade V2
    - TRAIN / CALIBRATION / STRICT HOLDOUT / FUTURE
    - Optuna
    - Weighted blend
    - Calibration chosen on calibration set
    - Final metrics only on holdout
    - Fallback split po dacie, jeśli brak season
    - Faktyczna skuteczność co 5% bin dla WIN/HOME i LOSS
    """)

    if os.path.exists(DEFAULT_DATA_CSV):
        run_rating_pipeline_v2(DEFAULT_DATA_CSV)
    else:
        print(f"❌ Nie znaleziono pliku: {DEFAULT_DATA_CSV}")
        user_path = input("\n📁 Wpisz ścieżkę do pliku CSV: ").strip()
        if user_path and os.path.exists(user_path):
            run_rating_pipeline_v2(user_path)
        else:
            print("❌ Nieprawidłowa ścieżka.")