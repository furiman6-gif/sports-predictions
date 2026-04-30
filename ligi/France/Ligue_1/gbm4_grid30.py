import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# reuse all functions from gbm4
sys.path.insert(0, str(Path(__file__).parent))
from gbm4 import (
    CSV_PATH, DATE_COL, TARGET_COL, CLASS_TO_INT, INT_TO_CLASS, CLASS_ORDER,
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

# ---------- 30 configurations ----------
CONFIGS = [
    # 1: baseline (current)
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 3.0},
    # 2: lower lr
    {"learning_rate": 0.01, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 3.0},
    # 3: higher lr
    {"learning_rate": 0.05, "n_estimators": 3000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 3.0},
    # 4: fewer leaves
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 31, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 3.0},
    # 5: more leaves
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 127, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 3.0},
    # 6: high regularization
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 1.0, "reg_lambda": 10.0},
    # 7: low regularization
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.0, "reg_lambda": 0.0},
    # 8: more min_child_samples
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 100,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 3.0},
    # 9: fewer min_child_samples
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 30,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 3.0},
    # 10: lower subsample
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 0.65, "colsample_bytree": 0.65, "reg_alpha": 0.1, "reg_lambda": 3.0},
    # 11: high subsample
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 0.95, "colsample_bytree": 0.95, "reg_alpha": 0.1, "reg_lambda": 3.0},
    # 12: lr=0.03, leaves=47
    {"learning_rate": 0.03, "n_estimators": 4000, "num_leaves": 47, "min_child_samples": 50,
     "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 0.3, "reg_lambda": 5.0},
    # 13: aggressive (deep trees, low reg)
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 127, "min_child_samples": 20,
     "subsample": 0.90, "colsample_bytree": 0.90, "reg_alpha": 0.0, "reg_lambda": 0.5},
    # 14: conservative (shallow, high reg)
    {"learning_rate": 0.01, "n_estimators": 5000, "num_leaves": 15, "min_child_samples": 100,
     "subsample": 0.70, "colsample_bytree": 0.70, "reg_alpha": 2.0, "reg_lambda": 10.0},
    # 15: balanced combo 1
    {"learning_rate": 0.015, "n_estimators": 5000, "num_leaves": 47, "min_child_samples": 80,
     "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 0.5, "reg_lambda": 5.0},
    # 16: balanced combo 2
    {"learning_rate": 0.025, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 40,
     "subsample": 0.85, "colsample_bytree": 0.75, "reg_alpha": 0.2, "reg_lambda": 2.0},
    # 17: very low lr + many trees
    {"learning_rate": 0.005, "n_estimators": 8000, "num_leaves": 31, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 3.0},
    # 18: high lr fast
    {"learning_rate": 0.08, "n_estimators": 2000, "num_leaves": 31, "min_child_samples": 60,
     "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 0.5, "reg_lambda": 5.0},
    # 19: colsample low
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.50, "reg_alpha": 0.1, "reg_lambda": 3.0},
    # 20: leaves=95, mid reg
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 95, "min_child_samples": 50,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.5, "reg_lambda": 5.0},
    # 21: very shallow trees
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 7, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 3.0},
    # 22: dart boosting
    {"learning_rate": 0.02, "n_estimators": 2000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 3.0,
     "boosting_type": "dart", "drop_rate": 0.1},
    # 23: goss boosting
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 1.0, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 3.0,
     "boosting_type": "goss"},
    # 24: max_depth limited
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 3.0,
     "max_depth": 6},
    # 25: max_depth=4
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 3.0,
     "max_depth": 4},
    # 26: feature_fraction_bynode
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 1.0, "reg_alpha": 0.1, "reg_lambda": 3.0,
     "feature_fraction_bynode": 0.5},
    # 27: min_child_samples=10, high reg
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 10,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 2.0, "reg_lambda": 8.0},
    # 28: lr=0.04, leaves=47, balanced
    {"learning_rate": 0.04, "n_estimators": 3000, "num_leaves": 47, "min_child_samples": 70,
     "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 0.3, "reg_lambda": 4.0},
    # 29: extra reg + path_smooth
    {"learning_rate": 0.02, "n_estimators": 5000, "num_leaves": 63, "min_child_samples": 60,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.5, "reg_lambda": 5.0,
     "path_smooth": 1.0},
    # 30: big leaves, high child, low col
    {"learning_rate": 0.015, "n_estimators": 5000, "num_leaves": 95, "min_child_samples": 100,
     "subsample": 0.75, "colsample_bytree": 0.60, "reg_alpha": 1.0, "reg_lambda": 7.0},
]


def prepare_data(n_last_seasons=12, split_mode="auto", use_xg=True):
    df_all = pd.read_csv(CSV_PATH)
    df_all = parse_date(df_all)
    df_all["Season"] = infer_season_from_date(df_all[DATE_COL])

    all_seasons = sorted(df_all["Season"].unique())
    selected_seasons = all_seasons[-n_last_seasons:]
    df_all = df_all[df_all["Season"].isin(selected_seasons)].copy().reset_index(drop=True)

    known_df, future_df = split_known_future(df_all)

    all_df = pd.concat(
        [known_df.drop(columns=["target"], errors="ignore"), future_df],
        ignore_index=True
    ).sort_values(DATE_COL).reset_index(drop=True)
    all_df["Season"] = infer_season_from_date(all_df[DATE_COL])

    long_df_all = build_team_match_history(all_df, use_xg=use_xg)
    long_df_all = add_advanced_rolling_features(long_df_all)
    full = merge_form_features_to_match(all_df, long_df_all)
    full = add_h2h_features(full)
    full = compute_elo(full)
    full = compute_goals_elo(full)
    full = compute_glicko(full)
    full = compute_elo_home_away(full)
    full = compute_goals_elo_home_away(full)
    full = compute_glicko_home_away(full)
    if use_xg:
        full = compute_xg_elo(full)
        full = compute_xg_elo_home_away(full)

    known_df = full[full[TARGET_COL].isin(CLASS_TO_INT.keys())].copy().reset_index(drop=True)
    known_df["target"] = known_df[TARGET_COL].map(CLASS_TO_INT)

    if len(future_df) > 0:
        future_df = full[~full[TARGET_COL].isin(CLASS_TO_INT.keys())].copy().reset_index(drop=True)
    else:
        future_df = pd.DataFrame(columns=known_df.columns)

    train_s, valid_s, test_s = seasonal_split(known_df["Season"].unique(), split_mode)

    train_df = known_df[known_df["Season"].isin(train_s)].copy()
    valid_df = known_df[known_df["Season"].isin(valid_s)].copy()
    test_df = known_df[known_df["Season"].isin(test_s)].copy()

    feature_cols = get_feature_columns(known_df)
    X_train, X_valid, X_test, X_future, final_feats, dropped = filter_features(
        train_df, valid_df, test_df, future_df, feature_cols
    )

    y_train = train_df["target"].values
    y_valid = valid_df["target"].values
    y_test = test_df["target"].values

    return X_train, X_valid, X_test, y_train, y_valid, y_test, final_feats


def run_config(cfg_id, cfg, X_train, X_valid, X_test, y_train, y_valid, y_test):
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "subsample_freq": 1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        **cfg
    }

    model = LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
    )

    best_iter = model.best_iteration_
    valid_proba = model.predict_proba(X_valid, num_iteration=best_iter)
    test_proba = model.predict_proba(X_test, num_iteration=best_iter)

    valid_pred = np.argmax(valid_proba, axis=1)
    test_pred = np.argmax(test_proba, axis=1)

    v_ll = log_loss(y_valid, valid_proba, labels=[0, 1, 2])
    t_ll = log_loss(y_test, test_proba, labels=[0, 1, 2])
    v_brier = multiclass_brier_score(y_valid, valid_proba)
    t_brier = multiclass_brier_score(y_test, test_proba)
    v_acc = accuracy_score(y_valid, valid_pred)
    t_acc = accuracy_score(y_test, test_pred)

    t_brier_h = per_class_brier(y_test, test_proba, 0)
    t_brier_d = per_class_brier(y_test, test_proba, 1)
    t_brier_a = per_class_brier(y_test, test_proba, 2)

    ece_vals = {}
    for cls_idx, cls_name in enumerate(CLASS_ORDER):
        y_bin = (y_test == cls_idx).astype(int)
        p_cls = test_proba[:, cls_idx]
        calib_df = calibration_table_binary(y_bin, p_cls, step=0.05)
        ece_vals[cls_name] = expected_calibration_error(calib_df)

    return {
        "cfg_id": cfg_id,
        "best_iter": best_iter,
        "valid_logloss": round(v_ll, 6),
        "test_logloss": round(t_ll, 6),
        "valid_brier": round(v_brier, 6),
        "test_brier": round(t_brier, 6),
        "valid_acc": round(v_acc, 4),
        "test_acc": round(t_acc, 4),
        "test_brier_H": round(t_brier_h, 6),
        "test_brier_D": round(t_brier_d, 6),
        "test_brier_A": round(t_brier_a, 6),
        "ece_H": round(ece_vals["H"], 6),
        "ece_D": round(ece_vals["D"], 6),
        "ece_A": round(ece_vals["A"], 6),
    }


def main():
    print("Przygotowywanie danych (12 sezonow, auto split, xG=tak)...")
    X_train, X_valid, X_test, y_train, y_valid, y_test, feats = prepare_data(
        n_last_seasons=12, split_mode="auto", use_xg=True
    )
    print(f"  train={len(y_train)}, valid={len(y_valid)}, test={len(y_test)}, features={len(feats)}")
    print(f"\nUruchamiam 30 konfiguracji...\n")

    results = []
    for i, cfg in enumerate(CONFIGS):
        label = f"[{i+1:2d}/30]"
        short = ", ".join(f"{k}={v}" for k, v in cfg.items()
                          if k not in ("objective", "num_class", "random_state", "n_jobs"))
        print(f"{label} {short[:90]}...")
        res = run_config(i + 1, cfg, X_train, X_valid, X_test, y_train, y_valid, y_test)
        results.append(res)
        print(f"       -> valid_ll={res['valid_logloss']:.4f}  test_ll={res['test_logloss']:.4f}  "
              f"test_acc={res['test_acc']:.3f}  test_brier={res['test_brier']:.4f}")

    df = pd.DataFrame(results)

    # add config descriptions
    cfg_descs = []
    for i, cfg in enumerate(CONFIGS):
        parts = []
        for k, v in cfg.items():
            if k in ("objective", "num_class", "random_state", "n_jobs"):
                continue
            parts.append(f"{k}={v}")
        cfg_descs.append("; ".join(parts))
    df["config"] = cfg_descs

    # save
    out_dir = Path(__file__).parent / "outputs_single"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grid30_report.csv"
    df.to_csv(out_path, index=False)

    # print report
    print("\n" + "=" * 120)
    print("RAPORT - 30 KONFIGURACJI LightGBM (Ligue 1)")
    print("=" * 120)

    # sort by valid_logloss
    df_sorted = df.sort_values("valid_logloss").reset_index(drop=True)
    df_sorted.index = df_sorted.index + 1  # rank from 1

    cols_print = ["cfg_id", "valid_logloss", "test_logloss", "valid_brier", "test_brier",
                  "valid_acc", "test_acc", "ece_H", "ece_D", "ece_A", "best_iter"]

    print("\nRANKING wg valid_logloss:")
    print(df_sorted[cols_print].to_string())

    print("\n" + "-" * 80)
    print("TOP 5 (valid_logloss):")
    for rank, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
        print(f"  #{rank} cfg={int(row['cfg_id']):2d}  "
              f"v_ll={row['valid_logloss']:.4f}  t_ll={row['test_logloss']:.4f}  "
              f"t_acc={row['test_acc']:.3f}  t_brier={row['test_brier']:.4f}")
        print(f"         {df_sorted.loc[_, 'config'] if _ in df_sorted.index else cfg_descs[int(row['cfg_id'])-1]}")

    print("\n" + "-" * 80)
    print("TOP 5 (test_logloss):")
    df_by_test = df.sort_values("test_logloss").head(5)
    for rank, (_, row) in enumerate(df_by_test.iterrows(), 1):
        print(f"  #{rank} cfg={int(row['cfg_id']):2d}  "
              f"v_ll={row['valid_logloss']:.4f}  t_ll={row['test_logloss']:.4f}  "
              f"t_acc={row['test_acc']:.3f}  t_brier={row['test_brier']:.4f}")

    print("\n" + "-" * 80)
    print("TOP 5 (test_accuracy):")
    df_by_acc = df.sort_values("test_acc", ascending=False).head(5)
    for rank, (_, row) in enumerate(df_by_acc.iterrows(), 1):
        print(f"  #{rank} cfg={int(row['cfg_id']):2d}  "
              f"v_ll={row['valid_logloss']:.4f}  t_ll={row['test_logloss']:.4f}  "
              f"t_acc={row['test_acc']:.3f}  t_brier={row['test_brier']:.4f}")

    print("\n" + "-" * 80)
    print("TOP 5 (test_brier - nizszy = lepszy):")
    df_by_brier = df.sort_values("test_brier").head(5)
    for rank, (_, row) in enumerate(df_by_brier.iterrows(), 1):
        print(f"  #{rank} cfg={int(row['cfg_id']):2d}  "
              f"v_ll={row['valid_logloss']:.4f}  t_ll={row['test_logloss']:.4f}  "
              f"t_acc={row['test_acc']:.3f}  t_brier={row['test_brier']:.4f}")

    # baseline comparison
    baseline = df[df["cfg_id"] == 1].iloc[0]
    print("\n" + "=" * 80)
    print("POROWNANIE Z BASELINE (cfg #1):")
    print(f"  Baseline: v_ll={baseline['valid_logloss']:.4f}  t_ll={baseline['test_logloss']:.4f}  "
          f"t_acc={baseline['test_acc']:.3f}  t_brier={baseline['test_brier']:.4f}")

    better_v = df[df["valid_logloss"] < baseline["valid_logloss"]]
    better_t = df[df["test_logloss"] < baseline["test_logloss"]]
    print(f"  Lepszych wg valid_logloss: {len(better_v)}/29")
    print(f"  Lepszych wg test_logloss:  {len(better_t)}/29")

    print(f"\nZapisano CSV: {out_path}")


if __name__ == "__main__":
    main()
