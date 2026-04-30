import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    log_loss,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import lightgbm as lgb
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# =====================================
# KONFIG
# =====================================
CSV_PATH = "merged_data_Tylko_Cechy_Rankingowe.csv"
DATE_COL = "Date"
TARGET_COL = "result"

CLASS_ORDER = ["W", "D", "L"]
CLASS_TO_INT = {"W": 0, "D": 1, "L": 2}
INT_TO_CLASS = {0: "W", 1: "D", 2: "L"}

OUTPUT_BASE_DIR = "outputs_single"

LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.02,
    "n_estimators": 5000,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 60,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
    "min_split_gain": 0.0,
    "random_state": 42,
    "n_jobs": -1
}


# =====================================
# POMOCNICZE
# =====================================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_date(df, date_col=DATE_COL):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return df


def infer_season_from_date(date_series):
    """
    Sezon od lipca:
    2023-08 -> sezon 2023
    2024-03 -> sezon 2023
    """
    years = date_series.dt.year
    months = date_series.dt.month
    season = np.where(months >= 7, years, years - 1)
    return season.astype(int)


def encode_target(df, target_col=TARGET_COL):
    df = df.copy()
    df = df[df[target_col].isin(CLASS_TO_INT.keys())].copy()
    df["target"] = df[target_col].map(CLASS_TO_INT)
    return df


def get_feature_columns(df):
    exclude_cols = {
        DATE_COL, "Season", TARGET_COL, "target", "HomeTeam", "AwayTeam"
    }
    return [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]


def filter_features(train_df, valid_df, test_df, feature_cols,
                    missing_threshold=0.50, corr_threshold=0.995):
    """
    Czyści cechy:
    - usuwa kolumny z dużym missingiem
    - usuwa stałe
    - usuwa jedną z bardzo silnie skorelowanych par
    """
    dropped_info = []

    # 1. missing + constant na train
    keep = []
    for col in feature_cols:
        s = train_df[col].replace([np.inf, -np.inf], np.nan)

        miss = s.isna().mean()
        nunique = s.nunique(dropna=False)

        if miss > missing_threshold:
            dropped_info.append((col, f"missing>{missing_threshold}"))
            continue

        if nunique <= 1:
            dropped_info.append((col, "constant"))
            continue

        keep.append(col)

    X_train = train_df[keep].replace([np.inf, -np.inf], np.nan).copy()
    X_valid = valid_df[keep].replace([np.inf, -np.inf], np.nan).copy()
    X_test = test_df[keep].replace([np.inf, -np.inf], np.nan).copy()

    # 2. wysokie korelacje tylko na train
    if len(keep) > 1:
        corr = X_train.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]

        for col in to_drop_corr:
            dropped_info.append((col, f"corr>{corr_threshold}"))

        keep = [c for c in keep if c not in to_drop_corr]

        X_train = X_train[keep]
        X_valid = X_valid[keep]
        X_test = X_test[keep]

    return X_train, X_valid, X_test, keep, dropped_info


def seasonal_split(unique_seasons, mode="auto"):
    seasons = list(sorted(unique_seasons))
    n = len(seasons)

    if n < 4:
        raise ValueError(f"Za mało sezonów do sensownego splitu: {n}. Minimum 4.")

    if mode == "80_10_10":
        n_test = max(1, round(n * 0.10))
        n_valid = max(1, round(n * 0.10))
        n_train = n - n_valid - n_test

    elif mode == "75_12_5_12_5":
        n_test = max(1, round(n * 0.125))
        n_valid = max(1, round(n * 0.125))
        n_train = n - n_valid - n_test

    else:  # auto
        if n <= 8:
            n_valid, n_test = 1, 1
            n_train = n - 2
        elif n <= 12:
            n_valid, n_test = 1, 1
            n_train = n - 2
        elif n <= 19:
            n_valid, n_test = 1, 2
            n_train = n - 3
        else:
            n_test = max(1, round(n * 0.10))
            n_valid = max(1, round(n * 0.10))
            n_train = n - n_valid - n_test

    if n_train < 2:
        raise ValueError(
            f"Za mało sezonów treningowych po splicie. "
            f"n={n}, train={n_train}, valid={n_valid}, test={n_test}"
        )

    train_seasons = seasons[:n_train]
    valid_seasons = seasons[n_train:n_train + n_valid]
    test_seasons = seasons[n_train + n_valid:]

    return train_seasons, valid_seasons, test_seasons


def multiclass_brier_score(y_true, proba, n_classes=3):
    y_onehot = np.eye(n_classes)[y_true]
    return np.mean(np.sum((proba - y_onehot) ** 2, axis=1))


def per_class_brier(y_true, proba, class_idx):
    y_bin = (y_true == class_idx).astype(int)
    p = proba[:, class_idx]
    return np.mean((p - y_bin) ** 2)


def calibration_table_binary(y_true_binary, pred_prob, step=0.05):
    bins = np.arange(0.0, 1.0 + step, step)
    rows = []

    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]

        if i == len(bins) - 2:
            mask = (pred_prob >= left) & (pred_prob <= right)
        else:
            mask = (pred_prob >= left) & (pred_prob < right)

        n = mask.sum()

        if n == 0:
            rows.append({
                "bin_left": left,
                "bin_right": right,
                "count": 0,
                "avg_pred": np.nan,
                "actual_rate": np.nan,
                "diff": np.nan
            })
            continue

        avg_pred = pred_prob[mask].mean()
        actual_rate = y_true_binary[mask].mean()

        rows.append({
            "bin_left": left,
            "bin_right": right,
            "count": int(n),
            "avg_pred": float(avg_pred),
            "actual_rate": float(actual_rate),
            "diff": float(actual_rate - avg_pred)
        })

    return pd.DataFrame(rows)


def expected_calibration_error(calib_df):
    df = calib_df.dropna().copy()
    if df.empty:
        return np.nan
    total = df["count"].sum()
    return np.sum((df["count"] / total) * np.abs(df["actual_rate"] - df["avg_pred"]))


def plot_calibration(calib_df, class_name, save_path):
    df = calib_df.dropna().copy()

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    plt.plot(df["avg_pred"], df["actual_rate"], marker="o", label=class_name)

    for _, row in df.iterrows():
        plt.annotate(
            str(int(row["count"])),
            (row["avg_pred"], row["actual_rate"]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=8
        )

    plt.xlabel("Średnie przewidywane prawdopodobieństwo")
    plt.ylabel("Rzeczywisty odsetek trafień")
    plt.title(f"Calibration - {class_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def plot_probability_histogram(pred_prob, class_name, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(pred_prob, bins=20, edgecolor="black", alpha=0.85)
    plt.xlabel("Przewidywane prawdopodobieństwo")
    plt.ylabel("Liczba meczów")
    plt.title(f"Histogram prawdopodobieństw - {class_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def plot_conf_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def plot_feature_importance(model, feature_names, save_path, top_n=30):
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 9))
    sns.barplot(data=imp_df, x="importance", y="feature", orient="h")
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Nie znaleziono pliku: {CSV_PATH}")
        return

    n_last_seasons = int(input("Ile ostatnich sezonów użyć? "))
    split_mode = input("Tryb splitu [auto / 80_10_10 / 75_12_5_12_5] (Enter=auto): ").strip()
    if split_mode == "":
        split_mode = "auto"

    out_dir_name = Path(CSV_PATH).stem + "_stronger_lgbm"
    output_dir = os.path.join(OUTPUT_BASE_DIR, out_dir_name)
    ensure_dir(output_dir)

    # Wczytanie
    df = pd.read_csv(CSV_PATH)
    df = parse_date(df, DATE_COL)
    df = encode_target(df, TARGET_COL)

    if df.empty:
        print("Brak danych po parsowaniu daty / targetu.")
        return

    # Sezon
    df["Season"] = infer_season_from_date(df[DATE_COL])
    all_seasons = sorted(df["Season"].unique())

    if len(all_seasons) < n_last_seasons:
        print(f"Plik ma tylko {len(all_seasons)} sezonów, a podałeś {n_last_seasons}.")
        return

    selected_seasons = all_seasons[-n_last_seasons:]
    df = df[df["Season"].isin(selected_seasons)].copy()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    train_seasons, valid_seasons, test_seasons = seasonal_split(
        sorted(df["Season"].unique()),
        mode=split_mode
    )

    print("\n===== SPLIT =====")
    print(f"Plik: {CSV_PATH}")
    print(f"Sezony użyte: {selected_seasons[0]} -> {selected_seasons[-1]}")
    print(f"TRAIN seasons: {train_seasons}")
    print(f"VALID seasons: {valid_seasons}")
    print(f"TEST  seasons: {test_seasons}")

    train_df = df[df["Season"].isin(train_seasons)].copy()
    valid_df = df[df["Season"].isin(valid_seasons)].copy()
    test_df = df[df["Season"].isin(test_seasons)].copy()

    feature_cols = get_feature_columns(df)

    X_train, X_valid, X_test, feature_cols, dropped_info = filter_features(
        train_df, valid_df, test_df, feature_cols,
        missing_threshold=0.50,
        corr_threshold=0.995
    )

    y_train = train_df["target"].values
    y_valid = valid_df["target"].values
    y_test = test_df["target"].values

    print("\n===== DANE =====")
    print(f"TRAIN shape: {X_train.shape}")
    print(f"VALID shape: {X_valid.shape}")
    print(f"TEST  shape: {X_test.shape}")
    print(f"Feature count after cleaning: {len(feature_cols)}")
    print(f"Dropped columns: {len(dropped_info)}")

    dropped_df = pd.DataFrame(dropped_info, columns=["feature", "reason"])
    dropped_df.to_csv(os.path.join(output_dir, "dropped_features.csv"), index=False)

    # Model
    model = LGBMClassifier(**LGBM_PARAMS)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )

    print(f"\nBest iteration: {model.best_iteration_}")

    # Predykcje
    valid_proba = model.predict_proba(X_valid, num_iteration=model.best_iteration_)
    test_proba = model.predict_proba(X_test, num_iteration=model.best_iteration_)

    valid_pred = np.argmax(valid_proba, axis=1)
    test_pred = np.argmax(test_proba, axis=1)

    # Metryki
    valid_logloss = log_loss(y_valid, valid_proba, labels=[0, 1, 2])
    test_logloss = log_loss(y_test, test_proba, labels=[0, 1, 2])

    valid_brier = multiclass_brier_score(y_valid, valid_proba, n_classes=3)
    test_brier = multiclass_brier_score(y_test, test_proba, n_classes=3)

    valid_acc = accuracy_score(y_valid, valid_pred)
    test_acc = accuracy_score(y_test, test_pred)

    test_brier_w = per_class_brier(y_test, test_proba, 0)
    test_brier_d = per_class_brier(y_test, test_proba, 1)
    test_brier_l = per_class_brier(y_test, test_proba, 2)

    # Calibration
    ece_map = {}
    for cls_idx, cls_name in enumerate(CLASS_ORDER):
        y_bin = (y_test == cls_idx).astype(int)
        p_cls = test_proba[:, cls_idx]

        calib_df = calibration_table_binary(y_bin, p_cls, step=0.05)
        ece_map[cls_name] = expected_calibration_error(calib_df)

        calib_df.to_csv(os.path.join(output_dir, f"calibration_bins_{cls_name}.csv"), index=False)

        plot_calibration(
            calib_df,
            class_name=cls_name,
            save_path=os.path.join(output_dir, f"calibration_{cls_name}.png")
        )

        plot_probability_histogram(
            p_cls,
            class_name=cls_name,
            save_path=os.path.join(output_dir, f"hist_prob_{cls_name}.png")
        )

    # Wykresy
    plot_conf_matrix(
        y_test,
        test_pred,
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )

    plot_feature_importance(
        model,
        feature_cols,
        save_path=os.path.join(output_dir, "feature_importance.png"),
        top_n=30
    )

    # Feature importance csv
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    # Predykcje testowe
    base_cols = [c for c in [DATE_COL, "HomeTeam", "AwayTeam", TARGET_COL, "Season"] if c in test_df.columns]
    pred_df = test_df[base_cols].copy()
    pred_df["pred_W"] = test_proba[:, 0]
    pred_df["pred_D"] = test_proba[:, 1]
    pred_df["pred_L"] = test_proba[:, 2]
    pred_df["pred_class"] = [INT_TO_CLASS[i] for i in test_pred]
    pred_df["max_prob"] = test_proba.max(axis=1)
    pred_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    # Summary
    summary = pd.DataFrame([{
        "file": Path(CSV_PATH).name,
        "n_selected_seasons": n_last_seasons,
        "train_seasons": "|".join(map(str, train_seasons)),
        "valid_seasons": "|".join(map(str, valid_seasons)),
        "test_seasons": "|".join(map(str, test_seasons)),
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
        "n_features": len(feature_cols),
        "best_iteration": model.best_iteration_,
        "valid_logloss": valid_logloss,
        "test_logloss": test_logloss,
        "valid_brier_multiclass": valid_brier,
        "test_brier_multiclass": test_brier,
        "valid_accuracy": valid_acc,
        "test_accuracy": test_acc,
        "test_brier_W": test_brier_w,
        "test_brier_D": test_brier_d,
        "test_brier_L": test_brier_l,
        "ece_W": ece_map["W"],
        "ece_D": ece_map["D"],
        "ece_L": ece_map["L"]
    }])

    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    print("\n===== WYNIKI =====")
    print(summary.T)
    print(f"\nWyniki zapisane do: {output_dir}")


if __name__ == "__main__":
    main()