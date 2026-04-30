"""
analysis_odds_roi.py
====================
Porównuje predykcje modelu z kursami bukmacherów.
Pokazuje ROI i edge per 5%-bin prawdopodobieństwa (flat bet 1 jednostka).

Uruchomienie:
    python analysis_odds_roi.py [--surface ALL/Hard/Clay/Grass] [--odds Avg/Max/B365/PS]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR  = Path(__file__).parent
PREDS_DIR   = SCRIPT_DIR / "outputs_tenis"
ODDS_CSV    = SCRIPT_DIR / "stats" / "csv" / "final_2_projekt.csv"

ODDS_BOOKS = {
    "Avg":  ("AvgW",  "AvgL"),
    "Max":  ("MaxW",  "MaxL"),
    "B365": ("B365W", "B365L"),
    "PS":   ("PSW",   "PSL"),
}


def load_and_merge(surface: str, book: str) -> pd.DataFrame:
    surf_key = surface.upper() if surface else "ALL"
    preds_path = PREDS_DIR / f"tenis_{surf_key}" / "test_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Brak pliku predykcji: {preds_path}")

    preds = pd.read_csv(preds_path)
    preds["Date"] = pd.to_datetime(preds["Date"], format="mixed", errors="coerce")

    odds_df = pd.read_csv(ODDS_CSV, low_memory=False)
    odds_df["Date"] = pd.to_datetime(odds_df["Date"], format="mixed", errors="coerce")

    w_col, l_col = ODDS_BOOKS[book]
    keep = ["Date", "Winner", "Loser", w_col, l_col]
    odds_df = odds_df[keep].dropna(subset=[w_col, l_col])

    # Case 1: P1 = Winner
    c1 = preds.merge(odds_df, left_on=["Date", "P1", "P2"],
                     right_on=["Date", "Winner", "Loser"], how="inner")
    c1["P1_odds"] = c1[w_col]
    c1["P2_odds"] = c1[l_col]

    # Case 2: P1 = Loser (P2 = Winner)
    c2 = preds.merge(odds_df, left_on=["Date", "P2", "P1"],
                     right_on=["Date", "Winner", "Loser"], how="inner")
    c2["P1_odds"] = c2[l_col]
    c2["P2_odds"] = c2[w_col]

    df = pd.concat([c1, c2], ignore_index=True)
    df = df.drop_duplicates(subset=["Date", "P1", "P2"])

    # Prawdopodobieństwo i kurs przewidywanego zwycięzcy
    df["pred_prob"] = np.where(df["pred"] == "P1", df["prob_P1_wins"], df["prob_P2_wins"])
    df["pred_odds"] = np.where(df["pred"] == "P1", df["P1_odds"], df["P2_odds"])

    # Implied probability bukmachera (bez marginesu)
    df["bookie_prob_P1"] = 1.0 / df["P1_odds"]
    df["bookie_prob_P2"] = 1.0 / df["P2_odds"]
    margin = df["bookie_prob_P1"] + df["bookie_prob_P2"]
    df["bookie_prob_pred"] = np.where(
        df["pred"] == "P1",
        df["bookie_prob_P1"] / margin,
        df["bookie_prob_P2"] / margin,
    )

    # Edge = przewaga modelu nad bukmacherem (po usunięciu marginesu)
    df["edge"] = df["pred_prob"] - df["bookie_prob_pred"]

    # Zysk z flat-bet 1 jednostki na przewidywanego zwycięzcę
    df["profit"] = np.where(df["correct"] == 1,
                            df["pred_odds"] - 1,
                            -1.0)
    return df


def roi_report(df: pd.DataFrame, book: str) -> None:
    total_bets   = len(df)
    total_profit = df["profit"].sum()
    total_roi    = total_profit / total_bets * 100
    overall_acc  = df["correct"].mean() * 100
    avg_edge     = df["edge"].mean() * 100

    print(f"\nKsiążka: {book} | Meczów z kursami: {total_bets}")
    print(f"Ogólne: accuracy={overall_acc:.1f}%  ROI={total_roi:+.2f}%  avg_edge={avg_edge:+.2f}%\n")

    # Biny po 5% prawdopodobieństwa przewidywanego zwycięzcy (0.50–1.00)
    bins   = np.arange(0.50, 1.01, 0.05)
    labels = [f"{b:.2f}-{b+0.05:.2f}" for b in bins[:-1]]
    df["bin"] = pd.cut(df["pred_prob"], bins=bins, labels=labels, right=False, include_lowest=True)

    rows = []
    for lbl in labels:
        g = df[df["bin"] == lbl]
        if len(g) == 0:
            continue
        roi  = g["profit"].sum() / len(g) * 100
        acc  = g["correct"].mean() * 100
        edge = g["edge"].mean() * 100
        avg_odds = g["pred_odds"].mean()
        rows.append({
            "bin_prob":  lbl,
            "n_bets":    len(g),
            "accuracy":  f"{acc:.1f}%",
            "avg_odds":  f"{avg_odds:.2f}",
            "avg_edge":  f"{edge:+.2f}%",
            "ROI":       f"{roi:+.2f}%",
            "profit":    f"{g['profit'].sum():+.2f}",
        })

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))

    # Tylko bety z edge > 0
    pos_edge = df[df["edge"] > 0]
    if len(pos_edge) > 0:
        roi_pos = pos_edge["profit"].sum() / len(pos_edge) * 100
        print(f"\nTylko bety z edge>0: n={len(pos_edge)}, ROI={roi_pos:+.2f}%")

    # Tylko bety z edge > 3%
    pos_edge3 = df[df["edge"] > 0.03]
    if len(pos_edge3) > 0:
        roi_pos3 = pos_edge3["profit"].sum() / len(pos_edge3) * 100
        print(f"Tylko bety z edge>3%: n={len(pos_edge3)}, ROI={roi_pos3:+.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", default="ALL",
                        help="ALL / Hard / Clay / Grass (default: ALL)")
    parser.add_argument("--odds", default="Avg",
                        choices=list(ODDS_BOOKS.keys()),
                        help="Bukmacher: Avg / Max / B365 / PS (default: Avg)")
    args = parser.parse_args()

    df = load_and_merge(args.surface, args.odds)
    roi_report(df, args.odds)

    # Zapisz szczegóły
    out_dir = PREDS_DIR / f"tenis_{args.surface.upper()}"
    out_path = out_dir / f"roi_analysis_{args.odds}.csv"
    df[["Date","P1","P2","Surface","pred","pred_prob","pred_odds",
        "bookie_prob_pred","edge","correct","profit"]].to_csv(out_path, index=False)
    print(f"\nSzczegóły zapisane: {out_path}")


if __name__ == "__main__":
    main()
