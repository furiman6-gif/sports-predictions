import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gbm4


def compute_stat_elo(df, home_col, away_col, prefix, k=20, initial=1500, scale=1.0, lower_is_better=False):
    ratings = {}
    elo_h, elo_a = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        rh = ratings.get(home, initial)
        ra = ratings.get(away, initial)
        elo_h.append(rh)
        elo_a.append(ra)

        hv = row.get(home_col)
        av = row.get(away_col)
        if pd.isna(hv) or pd.isna(av):
            continue

        diff = (av - hv) if lower_is_better else (hv - av)
        sh = 1 / (1 + 10 ** (-diff / scale))
        eh = 1 / (1 + 10 ** ((ra - rh) / 400))

        ratings[home] = rh + k * (sh - eh)
        ratings[away] = ra + k * ((1 - sh) - (1 - eh))

    df = df.copy()
    df[f"{prefix}_elo_home"] = elo_h
    df[f"{prefix}_elo_away"] = elo_a
    df[f"{prefix}_elo_diff"] = df[f"{prefix}_elo_home"] - df[f"{prefix}_elo_away"]
    return df


def compute_stat_elo_home_away(df, home_col, away_col, prefix, k=20, initial=1500, scale=1.0, lower_is_better=False):
    h_ratings = {}
    a_ratings = {}
    eh_list, ea_list = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        rh = h_ratings.get(home, initial)
        ra = a_ratings.get(away, initial)
        eh_list.append(rh)
        ea_list.append(ra)

        hv = row.get(home_col)
        av = row.get(away_col)
        if pd.isna(hv) or pd.isna(av):
            continue

        diff = (av - hv) if lower_is_better else (hv - av)
        sh = 1 / (1 + 10 ** (-diff / scale))
        exp_h = 1 / (1 + 10 ** ((ra - rh) / 400))

        h_ratings[home] = rh + k * (sh - exp_h)
        a_ratings[away] = ra + k * ((1 - sh) - (1 - exp_h))

    df = df.copy()
    df[f"{prefix}_elo_H_home"] = eh_list
    df[f"{prefix}_elo_A_away"] = ea_list
    df[f"{prefix}_elo_HA_diff"] = df[f"{prefix}_elo_H_home"] - df[f"{prefix}_elo_A_away"]
    return df


def add_stat_elo_features(df):
    df = df.copy()

    if "HS" in df.columns and "AS" in df.columns:
        df = compute_stat_elo(df, "HS", "AS", "shots", k=18, scale=5.0)
        df = compute_stat_elo_home_away(df, "HS", "AS", "shots", k=18, scale=5.0)
    if "HST" in df.columns and "AST" in df.columns:
        df = compute_stat_elo(df, "HST", "AST", "sot", k=18, scale=2.0)
        df = compute_stat_elo_home_away(df, "HST", "AST", "sot", k=18, scale=2.0)
    if "HC" in df.columns and "AC" in df.columns:
        df = compute_stat_elo(df, "HC", "AC", "corners", k=16, scale=2.0)
        df = compute_stat_elo_home_away(df, "HC", "AC", "corners", k=16, scale=2.0)
    if "HF" in df.columns and "AF" in df.columns:
        df = compute_stat_elo(df, "HF", "AF", "fouls", k=14, scale=5.0, lower_is_better=True)
        df = compute_stat_elo_home_away(df, "HF", "AF", "fouls", k=14, scale=5.0, lower_is_better=True)
    if "HY" in df.columns and "AY" in df.columns:
        df = compute_stat_elo(df, "HY", "AY", "yellows", k=12, scale=1.0, lower_is_better=True)
        df = compute_stat_elo_home_away(df, "HY", "AY", "yellows", k=12, scale=1.0, lower_is_better=True)
    if "HR" in df.columns and "AR" in df.columns:
        df = compute_stat_elo(df, "HR", "AR", "reds", k=10, scale=0.5, lower_is_better=True)
        df = compute_stat_elo_home_away(df, "HR", "AR", "reds", k=10, scale=0.5, lower_is_better=True)

    return df


def split_known_future_by_target(df, target_mode):
    df = df.copy()

    if target_mode == "FTR":
        known_mask = df[gbm4.TARGET_COL].isin(gbm4.CLASS_TO_INT.keys())
        known_df = df[known_mask].copy().reset_index(drop=True)
        future_df = df[~known_mask].copy().reset_index(drop=True)
        known_df["target"] = known_df[gbm4.TARGET_COL].map(gbm4.CLASS_TO_INT)
        return known_df, future_df, "multiclass"

    gh = pd.to_numeric(df.get("FTHG"), errors="coerce")
    ga = pd.to_numeric(df.get("FTAG"), errors="coerce")
    known_mask = gh.notna() & ga.notna()
    known_df = df[known_mask].copy().reset_index(drop=True)
    future_df = df[~known_mask].copy().reset_index(drop=True)

    gh_k = pd.to_numeric(known_df.get("FTHG"), errors="coerce")
    ga_k = pd.to_numeric(known_df.get("FTAG"), errors="coerce")

    if target_mode == "O25":
        known_df["target"] = ((gh_k + ga_k) > 2.5).astype(int)
    elif target_mode == "BTTS":
        known_df["target"] = ((gh_k > 0) & (ga_k > 0)).astype(int)
    else:
        raise ValueError(f"Nieznany target_mode: {target_mode}")

    return known_df, future_df, "binary"


def binary_labels(target_mode):
    if target_mode == "O25":
        return "U", "O"
    if target_mode == "BTTS":
        return "NO", "YES"
    return "0", "1"


def main():
    if not os.path.exists(gbm4.CSV_PATH):
        print(f"Nie znaleziono pliku: {gbm4.CSV_PATH}")
        return

    n_last_seasons = int(input("Ile ostatnich sezonów użyć? "))
    split_mode = input("Tryb splitu [auto / 80_10_10 / 75_12_5_12_5] (Enter=auto): ").strip()
    if split_mode == "":
        split_mode = "auto"
    n_features_input = input("Ile features użyć? (Enter=wszystkie): ").strip()
    n_features_limit = int(n_features_input) if n_features_input else None
    use_xg_input = input("Użyć statystyk xG? [t/n] (Enter=tak): ").strip().lower()
    use_xg = use_xg_input not in ("n", "nie", "no")
    target_mode = input("Target [FTR / O25 / BTTS] (Enter=FTR): ").strip().upper()
    if target_mode == "":
        target_mode = "FTR"
    if target_mode not in ("FTR", "O25", "BTTS"):
        print(f"Nieznany target: {target_mode} (używam FTR)")
        target_mode = "FTR"

    output_dir = os.path.join(
        gbm4.OUTPUT_BASE_DIR,
        Path(gbm4.CSV_PATH).stem + f"_FULL_FORM_SHOTS_SOT_XG_H2H_STAT_ELO_{target_mode}"
    )
    gbm4.ensure_dir(output_dir)

    df_all = pd.read_csv(gbm4.CSV_PATH)
    df_all = gbm4.parse_date(df_all)
    df_all["Season"] = gbm4.infer_season_from_date(df_all[gbm4.DATE_COL])

    all_seasons = sorted(df_all["Season"].unique())
    if len(all_seasons) < n_last_seasons:
        print(f"Plik ma tylko {len(all_seasons)} sezonów, a podałeś {n_last_seasons}")
        return

    selected_seasons = all_seasons[-n_last_seasons:]
    df_all = df_all[df_all["Season"].isin(selected_seasons)].copy().reset_index(drop=True)

    known_df, future_df, task_kind = split_known_future_by_target(df_all, target_mode)

    all_df = pd.concat(
        [known_df.drop(columns=["target"], errors="ignore"), future_df],
        ignore_index=True
    ).sort_values(gbm4.DATE_COL).reset_index(drop=True)
    all_df["Season"] = gbm4.infer_season_from_date(all_df[gbm4.DATE_COL])

    long_df_all = gbm4.build_team_match_history(all_df, use_xg=use_xg)
    long_df_all = gbm4.add_advanced_rolling_features(long_df_all)
    full_with_features = gbm4.merge_form_features_to_match(all_df, long_df_all)
    full_with_features = gbm4.add_h2h_features(full_with_features)
    full_with_features = gbm4.compute_elo(full_with_features)
    full_with_features = gbm4.compute_goals_elo(full_with_features)
    full_with_features = gbm4.compute_glicko(full_with_features)
    full_with_features = gbm4.compute_elo_home_away(full_with_features)
    full_with_features = gbm4.compute_goals_elo_home_away(full_with_features)
    full_with_features = gbm4.compute_glicko_home_away(full_with_features)
    full_with_features = add_stat_elo_features(full_with_features)
    if use_xg:
        full_with_features = gbm4.compute_xg_elo(full_with_features)
        full_with_features = gbm4.compute_xg_elo_home_away(full_with_features)

    if task_kind == "multiclass":
        known_mask = full_with_features[gbm4.TARGET_COL].isin(gbm4.CLASS_TO_INT.keys())
        known_df = full_with_features[known_mask].copy().reset_index(drop=True)
        known_df["target"] = known_df[gbm4.TARGET_COL].map(gbm4.CLASS_TO_INT)
        if len(future_df) > 0:
            future_df = full_with_features[~known_mask].copy().reset_index(drop=True)
        else:
            future_df = pd.DataFrame(columns=known_df.columns)
    else:
        gh = pd.to_numeric(full_with_features.get("FTHG"), errors="coerce")
        ga = pd.to_numeric(full_with_features.get("FTAG"), errors="coerce")
        known_mask = gh.notna() & ga.notna()
        known_df = full_with_features[known_mask].copy().reset_index(drop=True)
        gh_k = pd.to_numeric(known_df.get("FTHG"), errors="coerce")
        ga_k = pd.to_numeric(known_df.get("FTAG"), errors="coerce")
        if target_mode == "O25":
            known_df["target"] = ((gh_k + ga_k) > 2.5).astype(int)
        else:
            known_df["target"] = ((gh_k > 0) & (ga_k > 0)).astype(int)
        if len(future_df) > 0:
            future_df = full_with_features[~known_mask].copy().reset_index(drop=True)
        else:
            future_df = pd.DataFrame(columns=known_df.columns)

    train_s, valid_s, test_s = gbm4.seasonal_split(known_df["Season"].unique(), split_mode)

    train_df = known_df[known_df["Season"].isin(train_s)].copy()
    valid_df = known_df[known_df["Season"].isin(valid_s)].copy()
    test_df = known_df[known_df["Season"].isin(test_s)].copy()

    feature_cols = gbm4.get_feature_columns(known_df)
    X_train, X_valid, X_test, X_future, final_feats, dropped = gbm4.filter_features(
        train_df, valid_df, test_df, future_df, feature_cols
    )

    y_train = train_df["target"].values
    y_valid = valid_df["target"].values
    y_test = test_df["target"].values

    if n_features_limit and n_features_limit < len(final_feats):
        print(f"\n--- Auto-select: wybieram top {n_features_limit} z {len(final_feats)} features ---")
        if task_kind == "multiclass":
            selector = gbm4.LGBMClassifier(
                objective="multiclass", num_class=3, learning_rate=0.05,
                n_estimators=300, num_leaves=31, random_state=42, n_jobs=-1
            )
            selector_eval_metric = "multi_logloss"
        else:
            selector = gbm4.LGBMClassifier(
                objective="binary", learning_rate=0.05,
                n_estimators=300, num_leaves=31, random_state=42, n_jobs=-1
            )
            selector_eval_metric = "binary_logloss"
        selector.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                      eval_metric=selector_eval_metric,
                      callbacks=[gbm4.lgb.early_stopping(30), gbm4.lgb.log_evaluation(0)])
        fi = pd.Series(selector.feature_importances_, index=final_feats)
        top_feats = fi.nlargest(n_features_limit).index.tolist()

        X_train = X_train[top_feats]
        X_valid = X_valid[top_feats]
        X_test = X_test[top_feats]
        X_future = X_future[top_feats] if len(X_future) > 0 else X_future
        final_feats = top_feats
        print(f"--- Wybrano: {final_feats[:10]}{'...' if len(final_feats) > 10 else ''} ---\n")

    dropped_df = pd.DataFrame(dropped, columns=["feature", "reason"])
    dropped_df.to_csv(os.path.join(output_dir, "dropped_features.csv"), index=False)

    if task_kind == "multiclass":
        lgbm_params = dict(gbm4.LGBM_PARAMS)
        model_eval_metric = "multi_logloss"
    else:
        lgbm_params = dict(gbm4.LGBM_PARAMS)
        lgbm_params.pop("num_class", None)
        lgbm_params["objective"] = "binary"
        model_eval_metric = "binary_logloss"

    model = gbm4.LGBMClassifier(**lgbm_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=model_eval_metric,
        callbacks=[
            gbm4.lgb.early_stopping(150),
            gbm4.lgb.log_evaluation(50)
        ]
    )

    best_iter = model.best_iteration_

    valid_proba = model.predict_proba(X_valid, num_iteration=best_iter)
    test_proba = model.predict_proba(X_test, num_iteration=best_iter)

    if task_kind == "multiclass":
        valid_pred = np.argmax(valid_proba, axis=1)
        test_pred = np.argmax(test_proba, axis=1)

        valid_logloss = gbm4.log_loss(y_valid, valid_proba, labels=[0, 1, 2])
        test_logloss = gbm4.log_loss(y_test, test_proba, labels=[0, 1, 2])

        valid_brier = gbm4.multiclass_brier_score(y_valid, valid_proba, n_classes=3)
        test_brier = gbm4.multiclass_brier_score(y_test, test_proba, n_classes=3)

        valid_acc = gbm4.accuracy_score(y_valid, valid_pred)
        test_acc = gbm4.accuracy_score(y_test, test_pred)

        test_brier_h = gbm4.per_class_brier(y_test, test_proba, 0)
        test_brier_d = gbm4.per_class_brier(y_test, test_proba, 1)
        test_brier_a = gbm4.per_class_brier(y_test, test_proba, 2)

        prec, rec, f1, supp = gbm4.precision_recall_fscore_support(
            y_test,
            test_pred,
            labels=[0, 1, 2],
            zero_division=0
        )

        class_metrics_df = pd.DataFrame({
            "class": ["win", "draw", "loss"],
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": supp
        })

        ece_map = {}
        calib_print_tables = {}

        for cls_idx, cls_name in enumerate(gbm4.CLASS_ORDER):
            y_bin = (y_test == cls_idx).astype(int)
            p_cls = test_proba[:, cls_idx]
            calib_df = gbm4.calibration_table_binary(y_bin, p_cls, step=0.05)
            calib_df.to_csv(os.path.join(output_dir, f"calibration_bins_{cls_name}.csv"), index=False)
            ece_map[cls_name] = gbm4.expected_calibration_error(calib_df)
            calib_print_tables[cls_name] = calib_df
    else:
        neg_label, pos_label = binary_labels(target_mode)
        valid_p = valid_proba[:, 1]
        test_p = test_proba[:, 1]
        valid_pred = (valid_p >= 0.5).astype(int)
        test_pred = (test_p >= 0.5).astype(int)

        valid_logloss = gbm4.log_loss(y_valid, valid_proba, labels=[0, 1])
        test_logloss = gbm4.log_loss(y_test, test_proba, labels=[0, 1])

        valid_brier = float(np.mean((valid_p - y_valid) ** 2))
        test_brier = float(np.mean((test_p - y_test) ** 2))

        valid_acc = gbm4.accuracy_score(y_valid, valid_pred)
        test_acc = gbm4.accuracy_score(y_test, test_pred)

        test_brier_h = np.nan
        test_brier_d = np.nan
        test_brier_a = np.nan

        prec, rec, f1, supp = gbm4.precision_recall_fscore_support(
            y_test,
            test_pred,
            labels=[0, 1],
            zero_division=0
        )

        class_metrics_df = pd.DataFrame({
            "class": [neg_label, pos_label],
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": supp
        })

        calib_df = gbm4.calibration_table_binary(y_test, test_p, step=0.05)
        calib_df.to_csv(os.path.join(output_dir, f"calibration_bins_{pos_label}.csv"), index=False)
        ece_map = {pos_label: gbm4.expected_calibration_error(calib_df)}
        calib_print_tables = {pos_label: calib_df}

    fi_df = pd.DataFrame({
        "feature": final_feats,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    base_cols = [c for c in [gbm4.DATE_COL, "HomeTeam", "AwayTeam", "FTHG", "FTAG", gbm4.TARGET_COL, "Season"] if c in test_df.columns]
    pred_df = test_df[base_cols].copy()
    if task_kind == "multiclass":
        pred_df["pred_H"] = test_proba[:, 0]
        pred_df["pred_D"] = test_proba[:, 1]
        pred_df["pred_A"] = test_proba[:, 2]
        pred_df["pred_class"] = [gbm4.INT_TO_CLASS[i] for i in test_pred]
        pred_df["max_prob"] = test_proba.max(axis=1)
    else:
        neg_label, pos_label = binary_labels(target_mode)
        pred_df[f"pred_{neg_label}"] = test_proba[:, 0]
        pred_df[f"pred_{pos_label}"] = test_proba[:, 1]
        pred_df["pred_class"] = [pos_label if i == 1 else neg_label for i in test_pred]
        pred_df["max_prob"] = np.maximum(test_proba[:, 0], test_proba[:, 1])
    pred_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    future_pred_df = pd.DataFrame()
    if len(future_df) > 0:
        future_proba = model.predict_proba(X_future, num_iteration=best_iter)
        if task_kind == "multiclass":
            future_pred = np.argmax(future_proba, axis=1)
        else:
            future_pred = (future_proba[:, 1] >= 0.5).astype(int)

        future_base_cols = [c for c in [gbm4.DATE_COL, "HomeTeam", "AwayTeam", "Season"] if c in future_df.columns]
        future_pred_df = future_df[future_base_cols].copy()
        if task_kind == "multiclass":
            future_pred_df["pred_H"] = future_proba[:, 0]
            future_pred_df["pred_D"] = future_proba[:, 1]
            future_pred_df["pred_A"] = future_proba[:, 2]
            future_pred_df["pred_class"] = [gbm4.INT_TO_CLASS[i] for i in future_pred]
            future_pred_df["max_prob"] = future_proba.max(axis=1)
        else:
            neg_label, pos_label = binary_labels(target_mode)
            future_pred_df[f"pred_{neg_label}"] = future_proba[:, 0]
            future_pred_df[f"pred_{pos_label}"] = future_proba[:, 1]
            future_pred_df["pred_class"] = [pos_label if i == 1 else neg_label for i in future_pred]
            future_pred_df["max_prob"] = np.maximum(future_proba[:, 0], future_proba[:, 1])
        future_pred_df.to_csv(os.path.join(output_dir, "future_predictions.csv"), index=False)

    summary = pd.DataFrame([{
        "file": Path(gbm4.CSV_PATH).name,
        "target_mode": target_mode,
        "task_kind": task_kind,
        "n_selected_seasons": n_last_seasons,
        "train_seasons": "|".join(map(str, train_s)),
        "valid_seasons": "|".join(map(str, valid_s)),
        "test_seasons": "|".join(map(str, test_s)),
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
        "n_future": len(future_df),
        "n_features": len(final_feats),
        "best_iteration": best_iter,
        "valid_logloss": valid_logloss,
        "test_logloss": test_logloss,
        "valid_brier": valid_brier,
        "test_brier": test_brier,
        "valid_accuracy": valid_acc,
        "test_accuracy": test_acc,
        "test_brier_H": test_brier_h,
        "test_brier_D": test_brier_d,
        "test_brier_A": test_brier_a,
        "ece_H": ece_map.get("H", np.nan),
        "ece_D": ece_map.get("D", np.nan),
        "ece_A": ece_map.get("A", np.nan),
        "ece_pos": next(iter(ece_map.values())) if len(ece_map) == 1 else np.nan
    }])
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    gbm4.print_separator("PODSUMOWANIE MODELU")
    print(summary.T.to_string(header=False))

    gbm4.print_separator("PRECISION / RECALL / F1")
    print(class_metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if task_kind == "multiclass":
        gbm4.print_separator("KALIBRACJA - WIN (H)")
        print(calib_print_tables["H"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        gbm4.print_separator("KALIBRACJA - DRAW (D)")
        print(calib_print_tables["D"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        gbm4.print_separator("KALIBRACJA - LOSS (A)")
        print(calib_print_tables["A"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        neg_label, pos_label = binary_labels(target_mode)
        gbm4.print_separator(f"KALIBRACJA - {pos_label}")
        print(calib_print_tables[pos_label].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    gbm4.print_separator("PREDYKCJE PRZYSZLYCH MECZOW (najblizszy tydzien)")
    if len(future_pred_df) > 0:
        today = pd.Timestamp.today().normalize()
        week_mask = (future_pred_df[gbm4.DATE_COL] >= today) & (future_pred_df[gbm4.DATE_COL] <= today + pd.Timedelta(days=7))
        week_df = future_pred_df[week_mask]
        if len(week_df) > 0:
            print(week_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        else:
            print("Brak meczow w ciagu najblizszych 7 dni.")
    else:
        print("Brak przyszlych meczow bez wyniku w CSV.")

    gbm4.print_separator("TOP 30 FEATURE IMPORTANCE")
    print(fi_df.head(30).to_string(index=False))

    gbm4.print_separator("ZAPIS")
    print(f"Zapisano do: {output_dir}")


if __name__ == "__main__":
    main()
