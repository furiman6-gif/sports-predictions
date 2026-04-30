import subprocess
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
MODEL_SCRIPT = ROOT / "gbm4_tenis.py"
OUT_DIR = ROOT / "outputs_tenis" / "tenis_ALL"

# OOT windows: mniejsza liczba sezonów przesuwa test dalej wstecz.
WINDOWS = [12, 16, 20, 25]
TARGET_THRESHOLDS = [0.55, 0.60, 0.65, 0.69, 0.70]


def run_model(n_last_seasons: int) -> None:
    inputs = f"{n_last_seasons}\nauto\n\n\n"
    proc = subprocess.run(
        ["python", str(MODEL_SCRIPT)],
        input=inputs,
        text=True,
        cwd=str(ROOT),
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError(f"Run failed for n_last_seasons={n_last_seasons}")


def load_reports():
    summary = pd.read_csv(OUT_DIR / "summary.csv").iloc[0].to_dict()
    value = pd.read_csv(OUT_DIR / "value_report.csv")
    bank = pd.read_csv(OUT_DIR / "bankroll_report.csv")
    return summary, value, bank


def one_window_result(n_last_seasons: int) -> pd.DataFrame:
    run_model(n_last_seasons)
    summary, value_report, bankroll_report = load_reports()

    value_cut = value_report[value_report["threshold"].isin(TARGET_THRESHOLDS)].copy()
    bank_cut = bankroll_report[bankroll_report["threshold"].isin(TARGET_THRESHOLDS)].copy()
    merged = value_cut.merge(bank_cut, on=["threshold", "bets"], how="left")

    for k in ["train_seasons", "valid_seasons", "test_seasons", "calib_used", "test_ece_cal", "test_accuracy", "test_auc"]:
        merged[k] = summary.get(k, np.nan)
    merged["n_last_seasons"] = n_last_seasons
    merged["clv_available"] = 0
    return merged[
        [
            "n_last_seasons",
            "train_seasons",
            "valid_seasons",
            "test_seasons",
            "threshold",
            "bets",
            "hit_rate_pct",
            "roi_pct",
            "final_bankroll",
            "roi_on_bankroll_pct",
            "max_drawdown_pct",
            "test_ece_cal",
            "test_accuracy",
            "test_auc",
            "calib_used",
            "clv_available",
        ]
    ].sort_values("threshold")


def build_recommendation(agg: pd.DataFrame) -> pd.DataFrame:
    if len(agg) == 0:
        return pd.DataFrame()

    cand = agg.copy()
    # Scoring:
    # - agressive: najwyzsze srednie ROI
    # - stable: dodatnie ROI avg, niegleboki najgorszy okres i niski sredni DD
    cand["aggressive_score"] = cand["roi_pct_avg"]
    cand["stable_score"] = (
        cand["roi_pct_avg"]
        - 0.5 * np.maximum(0.0, -cand["roi_pct_min"])
        - 0.05 * cand["max_dd_avg"]
    )

    aggressive_row = cand.sort_values(
        ["aggressive_score", "roi_pct_min", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]

    stable_pool = cand[
        (cand["roi_pct_avg"] > 0)
        & (cand["roi_pct_min"] > -2.0)
        & (cand["max_dd_avg"] < 20.0)
    ]
    if len(stable_pool) > 0:
        stable_row = stable_pool.sort_values(
            ["stable_score", "roi_pct_avg", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
    else:
        stable_row = cand.sort_values(
            ["stable_score", "roi_pct_avg", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]

    rec = pd.DataFrame(
        [
            {
                "profile": "aggressive",
                "recommended_threshold": float(aggressive_row["threshold"]),
                "roi_pct_avg": round(float(aggressive_row["roi_pct_avg"]), 2),
                "roi_pct_min": round(float(aggressive_row["roi_pct_min"]), 2),
                "max_dd_avg": round(float(aggressive_row["max_dd_avg"]), 2),
                "rule": "max avg ROI",
            },
            {
                "profile": "stable",
                "recommended_threshold": float(stable_row["threshold"]),
                "roi_pct_avg": round(float(stable_row["roi_pct_avg"]), 2),
                "roi_pct_min": round(float(stable_row["roi_pct_min"]), 2),
                "max_dd_avg": round(float(stable_row["max_dd_avg"]), 2),
                "rule": "positive avg ROI + mild worst window + low drawdown",
            },
        ]
    )
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-existing",
        action="store_true",
        help="Nie uruchamia modeli ponownie, tylko liczy agregaty/rekomendacje z istniejacych CSV.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "walkforward_report.csv"

    if args.from_existing:
        report = pd.read_csv(report_path)
    else:
        rows = []
        for n in WINDOWS:
            print(f"[walk-forward] run n_last_seasons={n}")
            rows.append(one_window_result(n))
        report = pd.concat(rows, ignore_index=True)
        report.to_csv(report_path, index=False)

    agg = (
        report.groupby("threshold", as_index=False)
        .agg(
            windows=("n_last_seasons", "count"),
            roi_pct_avg=("roi_pct", "mean"),
            roi_pct_min=("roi_pct", "min"),
            roi_pct_max=("roi_pct", "max"),
            bankroll_roi_avg=("roi_on_bankroll_pct", "mean"),
            max_dd_avg=("max_drawdown_pct", "mean"),
        )
        .sort_values("threshold")
    )
    agg_path = OUT_DIR / "walkforward_aggregate.csv"
    agg.to_csv(agg_path, index=False)

    rec = build_recommendation(agg)
    rec_path = OUT_DIR / "recommended_threshold.csv"
    rec.to_csv(rec_path, index=False)

    txt_path = OUT_DIR / "recommended_threshold.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        if len(rec) == 0:
            f.write("No recommendation available.\n")
        else:
            for _, r in rec.iterrows():
                f.write(
                    f"{r['profile']}: threshold={r['recommended_threshold']:.2f} | "
                    f"roi_avg={r['roi_pct_avg']:.2f}% | roi_min={r['roi_pct_min']:.2f}% | "
                    f"max_dd_avg={r['max_dd_avg']:.2f}% | rule={r['rule']}\n"
                )

    print("\n[walk-forward] saved:")
    print(report_path)
    print(agg_path)
    print(rec_path)
    print(txt_path)


if __name__ == "__main__":
    main()
