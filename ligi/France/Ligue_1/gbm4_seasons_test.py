import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from gbm4 import (
    CSV_PATH, DATE_COL, TARGET_COL, CLASS_TO_INT, CLASS_ORDER,
    parse_date, infer_season_from_date, seasonal_split, split_known_future,
    build_team_match_history, add_advanced_rolling_features,
    merge_form_features_to_match, add_h2h_features,
    compute_elo, compute_xg_elo, compute_goals_elo, compute_glicko,
    compute_elo_home_away, compute_xg_elo_home_away,
    compute_goals_elo_home_away, compute_glicko_home_away,
    get_feature_columns, filter_features,
    multiclass_brier_score, per_class_brier, expected_calibration_error,
    calibration_table_binary
)

# best config from grid30: GOSS
BEST_CFG = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.02,
    "n_estimators": 5000,
    "num_leaves": 63,
    "min_child_samples": 60,
    "subsample": 1.0,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
    "boosting_type": "goss",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# also test baseline for comparison
BASELINE_CFG = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.02,
    "n_estimators": 5000,
    "num_leaves": 63,
    "min_child_samples": 60,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def prepare_and_run(n_last_seasons, cfg, split_mode="auto"):
    df_all = pd.read_csv(CSV_PATH)
    df_all = parse_date(df_all)
    df_all["Season"] = infer_season_from_date(df_all[DATE_COL])

    all_seasons = sorted(df_all["Season"].unique())
    if len(all_seasons) < n_last_seasons:
        return None

    selected = all_seasons[-n_last_seasons:]
    df_all = df_all[df_all["Season"].isin(selected)].copy().reset_index(drop=True)

    known_df, future_df = split_known_future(df_all)

    all_df = pd.concat(
        [known_df.drop(columns=["target"], errors="ignore"), future_df],
        ignore_index=True
    ).sort_values(DATE_COL).reset_index(drop=True)
    all_df["Season"] = infer_season_from_date(all_df[DATE_COL])

    long_df = build_team_match_history(all_df, use_xg=True)
    long_df = add_advanced_rolling_features(long_df)
    full = merge_form_features_to_match(all_df, long_df)
    full = add_h2h_features(full)
    full = compute_elo(full)
    full = compute_goals_elo(full)
    full = compute_glicko(full)
    full = compute_elo_home_away(full)
    full = compute_goals_elo_home_away(full)
    full = compute_glicko_home_away(full)
    full = compute_xg_elo(full)
    full = compute_xg_elo_home_away(full)

    known_df = full[full[TARGET_COL].isin(CLASS_TO_INT.keys())].copy().reset_index(drop=True)
    known_df["target"] = known_df[TARGET_COL].map(CLASS_TO_INT)

    try:
        train_s, valid_s, test_s = seasonal_split(known_df["Season"].unique(), split_mode)
    except ValueError:
        return None

    train_df = known_df[known_df["Season"].isin(train_s)].copy()
    valid_df = known_df[known_df["Season"].isin(valid_s)].copy()
    test_df = known_df[known_df["Season"].isin(test_s)].copy()

    feature_cols = get_feature_columns(known_df)
    X_train, X_valid, X_test, _, final_feats, _ = filter_features(
        train_df, valid_df, test_df,
        pd.DataFrame(columns=known_df.columns), feature_cols
    )

    y_train = train_df["target"].values
    y_valid = valid_df["target"].values
    y_test = test_df["target"].values

    model = LGBMClassifier(**cfg)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )

    best_iter = model.best_iteration_
    valid_proba = model.predict_proba(X_valid, num_iteration=best_iter)
    test_proba = model.predict_proba(X_test, num_iteration=best_iter)

    v_ll = log_loss(y_valid, valid_proba, labels=[0, 1, 2])
    t_ll = log_loss(y_test, test_proba, labels=[0, 1, 2])
    v_brier = multiclass_brier_score(y_valid, valid_proba)
    t_brier = multiclass_brier_score(y_test, test_proba)
    v_acc = accuracy_score(y_valid, np.argmax(valid_proba, axis=1))
    t_acc = accuracy_score(y_test, np.argmax(test_proba, axis=1))

    ece_vals = {}
    for cls_idx, cls_name in enumerate(CLASS_ORDER):
        y_bin = (y_test == cls_idx).astype(int)
        p_cls = test_proba[:, cls_idx]
        calib_df = calibration_table_binary(y_bin, p_cls, step=0.05)
        ece_vals[cls_name] = expected_calibration_error(calib_df)

    return {
        "n_seasons": n_last_seasons,
        "train_seasons": "|".join(map(str, train_s)),
        "valid_seasons": "|".join(map(str, valid_s)),
        "test_seasons": "|".join(map(str, test_s)),
        "n_train": len(y_train),
        "n_valid": len(y_valid),
        "n_test": len(y_test),
        "n_features": len(final_feats),
        "best_iter": best_iter,
        "valid_logloss": round(v_ll, 6),
        "test_logloss": round(t_ll, 6),
        "valid_brier": round(v_brier, 6),
        "test_brier": round(t_brier, 6),
        "valid_acc": round(v_acc, 4),
        "test_acc": round(t_acc, 4),
        "ece_H": round(ece_vals["H"], 6),
        "ece_D": round(ece_vals["D"], 6),
        "ece_A": round(ece_vals["A"], 6),
    }


def main():
    # check how many seasons available
    df_all = pd.read_csv(CSV_PATH)
    df_all = parse_date(df_all)
    df_all["Season"] = infer_season_from_date(df_all[DATE_COL])
    all_seasons = sorted(df_all["Season"].unique())
    max_s = len(all_seasons)
    print(f"Dostepne sezony: {all_seasons} (lacznie {max_s})")

    season_range = list(range(4, max_s + 1))
    print(f"Testuje n_last_seasons od 4 do {max_s}...\n")

    results_goss = []
    results_base = []

    for n in season_range:
        print(f"--- n_last_seasons={n} ---")

        print(f"  GOSS...", end=" ", flush=True)
        r = prepare_and_run(n, BEST_CFG)
        if r:
            r["model"] = "GOSS"
            results_goss.append(r)
            print(f"v_ll={r['valid_logloss']:.4f}  t_ll={r['test_logloss']:.4f}  t_acc={r['test_acc']:.3f}  t_brier={r['test_brier']:.4f}")
        else:
            print("SKIP")

        print(f"  Baseline...", end=" ", flush=True)
        r = prepare_and_run(n, BASELINE_CFG)
        if r:
            r["model"] = "baseline"
            results_base.append(r)
            print(f"v_ll={r['valid_logloss']:.4f}  t_ll={r['test_logloss']:.4f}  t_acc={r['test_acc']:.3f}  t_brier={r['test_brier']:.4f}")
        else:
            print("SKIP")

    df_goss = pd.DataFrame(results_goss)
    df_base = pd.DataFrame(results_base)
    df_all_res = pd.concat([df_goss, df_base], ignore_index=True)

    out_dir = Path(__file__).parent / "outputs_single"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all_res.to_csv(out_dir / "seasons_test_report.csv", index=False)

    print("\n" + "=" * 110)
    print("RAPORT - GOSS (najlepsza konfiguracja)")
    print("=" * 110)
    cols = ["n_seasons", "n_train", "n_valid", "n_test", "n_features", "best_iter",
            "valid_logloss", "test_logloss", "valid_brier", "test_brier", "valid_acc", "test_acc",
            "ece_H", "ece_D", "ece_A"]
    print(df_goss[cols].to_string(index=False))

    print("\n" + "=" * 110)
    print("RAPORT - BASELINE")
    print("=" * 110)
    print(df_base[cols].to_string(index=False))

    # comparison
    print("\n" + "=" * 110)
    print("POROWNANIE GOSS vs BASELINE (test_logloss)")
    print("=" * 110)
    merged = df_goss[["n_seasons", "test_logloss", "test_acc", "test_brier"]].rename(
        columns={"test_logloss": "goss_ll", "test_acc": "goss_acc", "test_brier": "goss_brier"}
    ).merge(
        df_base[["n_seasons", "test_logloss", "test_acc", "test_brier"]].rename(
            columns={"test_logloss": "base_ll", "test_acc": "base_acc", "test_brier": "base_brier"}
        ),
        on="n_seasons"
    )
    merged["ll_diff"] = merged["goss_ll"] - merged["base_ll"]
    merged["acc_diff"] = merged["goss_acc"] - merged["base_acc"]
    merged["winner_ll"] = np.where(merged["ll_diff"] < 0, "GOSS", np.where(merged["ll_diff"] > 0, "BASE", "TIE"))
    print(merged.to_string(index=False))

    # best overall
    print("\n" + "-" * 80)
    best_goss = df_goss.loc[df_goss["test_logloss"].idxmin()]
    best_base = df_base.loc[df_base["test_logloss"].idxmin()]
    print(f"Najlepszy GOSS:     n={int(best_goss['n_seasons'])}  t_ll={best_goss['test_logloss']:.4f}  t_acc={best_goss['test_acc']:.3f}  t_brier={best_goss['test_brier']:.4f}")
    print(f"Najlepszy Baseline: n={int(best_base['n_seasons'])}  t_ll={best_base['test_logloss']:.4f}  t_acc={best_base['test_acc']:.3f}  t_brier={best_base['test_brier']:.4f}")


if __name__ == "__main__":
    main()
