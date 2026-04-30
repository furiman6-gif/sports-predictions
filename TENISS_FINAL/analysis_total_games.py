"""
analysis_total_games.py
=======================
Analiza zakładów over/under na łączną liczbę gemów.

Tryb interaktywny (jeden mecz):
    python analysis_total_games.py

Tryb batch (wszystkie przyszłe mecze z future_predictions.csv):
    python analysis_total_games.py --batch --surface ALL

Tryb kalibracja (jak model radził sobie w teście 2024-2026):
    python analysis_total_games.py --calibrate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
CSV_PATH   = SCRIPT_DIR / "stats" / "csv" / "5stats_projekt.csv"

# Błąd modelu z backtestów (MAE per format)
MODEL_MAE = {"bo3": 4.95, "bo5": 7.62, "all": 5.45}

# ── Rozkłady historyczne total_games (z danych 2001-2026) ──────────────────────
def load_distributions() -> dict:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    set_cols = ["W1","L1","W2","L2","W3","L3","W4","L4","W5","L5"]
    for c in set_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["total_games"] = df[set_cols].sum(axis=1)
    df["best_of"]     = pd.to_numeric(df["Best of"], errors="coerce")
    df = df[df["total_games"] > 5].dropna(subset=["Surface","best_of"])

    dists = {}
    for surf in ["Hard","Clay","Grass","all"]:
        for bo in [3, 5]:
            sub = df[df["best_of"] == bo]
            if surf != "all":
                sub = sub[sub["Surface"] == surf]
            if len(sub) < 50:
                continue
            key = f"{surf}_bo{bo}"
            dists[key] = {
                "mean": sub["total_games"].mean(),
                "std":  sub["total_games"].std(),
                "p25":  sub["total_games"].quantile(0.25),
                "p50":  sub["total_games"].median(),
                "p75":  sub["total_games"].quantile(0.75),
                "n":    len(sub),
            }
    return dists


def prob_over(pred: float, line: float, mae: float) -> float:
    """
    Szacunkowe P(actual > line) zakładając błąd modelu ~normalny z sigma=mae*1.25.
    """
    sigma = mae * 1.25
    return float(1 - stats.norm.cdf(line + 0.5, loc=pred, scale=sigma))


def expected_value(prob_win: float, odds: float) -> float:
    return prob_win * (odds - 1) - (1 - prob_win)


def analyze_match(pred: float, line: float, best_of: int,
                  odds_over: float, odds_under: float,
                  surface: str = "all") -> None:
    mae    = MODEL_MAE["bo5"] if best_of == 5 else MODEL_MAE["bo3"]
    p_over = prob_over(pred, line, mae)
    p_under = 1 - p_over

    ev_over  = expected_value(p_over,  odds_over)
    ev_under = expected_value(p_under, odds_under)

    gap = pred - line

    print(f"\n{'='*55}")
    print(f"Predykcja modelu : {pred:.1f} gemów")
    print(f"Linia bukmachera : {line:.1f} gemów  (BO{best_of}, {surface})")
    print(f"Gap (pred - linia): {gap:+.1f} gemów")
    print(f"MAE modelu dla BO{best_of}: ±{mae:.1f}")
    print(f"{'─'*55}")
    print(f"P(over {line})  = {p_over*100:.1f}%   kurs: {odds_over}  EV: {ev_over:+.3f}")
    print(f"P(under {line}) = {p_under*100:.1f}%   kurs: {odds_under}  EV: {ev_under:+.3f}")
    print(f"{'─'*55}")

    if ev_over > 0 and ev_over > ev_under:
        conf = "SILNA" if abs(gap) > mae * 0.8 else "SŁABA"
        print(f"→ BET OVER  (EV={ev_over:+.3f}, pewność: {conf})")
    elif ev_under > 0 and ev_under > ev_over:
        conf = "SILNA" if abs(gap) > mae * 0.8 else "SŁABA"
        print(f"→ BET UNDER (EV={ev_under:+.3f}, pewność: {conf})")
    else:
        print("→ PASS — brak wartości")
    print(f"{'='*55}\n")


def calibrate() -> None:
    """Pokazuje jak model radził sobie w 2024-2026 per bin przesunięcia od mediany."""
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    set_cols = ["W1","L1","W2","L2","W3","L3","W4","L4","W5","L5"]
    for c in set_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["total_games"] = df[set_cols].sum(axis=1)
    df["best_of"]     = pd.to_numeric(df["Best of"], errors="coerce")
    test = df[(df["Date"].dt.year >= 2024) & (df["total_games"] > 5)].copy()

    print("\n=== ROZKŁAD TOTAL GAMES W TEŚCIE (2024-2026) ===\n")
    for bo in [3, 5]:
        sub = test[test["best_of"] == bo]
        if len(sub) == 0:
            continue
        print(f"BO{bo} ({len(sub)} meczów):")
        by_surf = sub.groupby("Surface")["total_games"].agg(
            n="count", mean="mean", std="std",
            p25=lambda x: x.quantile(0.25),
            median="median",
            p75=lambda x: x.quantile(0.75)
        ).round(1)
        print(by_surf.to_string())
        print()

    print("\n=== SKUTECZNOŚĆ O/U PRZY RÓŻNYCH PROGACH ODCHYLENIA ===")
    print("(symulacja: linia = mediana per surface+BO, bet gdy |pred-linia| > próg)")
    print("Brak zapisanych predykcji TG dla testu — uruchom gbm4_tenis.py i dodaj")
    print("pred_total_games do test_predictions.csv, aby zobaczyć pełną analizę.\n")


def batch_mode(surface: str) -> None:
    surf_key = surface.upper() if surface else "ALL"
    fut_path = SCRIPT_DIR / "outputs_tenis" / f"tenis_{surf_key}" / "future_predictions.csv"
    if not fut_path.exists():
        print(f"Brak pliku: {fut_path}")
        return

    fut = pd.read_csv(fut_path)
    if "pred_total_games" not in fut.columns:
        print("Brak kolumny pred_total_games w future_predictions.csv")
        return

    fut["best_of"] = fut.get("best_of", 3)
    print(f"\n{'='*70}")
    print(f"{'Mecz':<40} {'BO':>3} {'Pred':>6} {'Med':>6} {'Gap':>6}")
    print(f"{'─'*70}")

    dists = load_distributions()

    for _, row in fut.iterrows():
        p1   = str(row.get("P1",""))[:18]
        p2   = str(row.get("P2",""))[:18]
        surf = str(row.get("Surface","all"))
        bo   = int(row.get("best_of", 3))
        pred = float(row["pred_total_games"])
        key  = f"{surf}_bo{bo}"
        if key not in dists:
            key = f"all_bo{bo}"
        median = dists.get(key, {}).get("p50", 22)
        gap = pred - median
        match_str = f"{p1} vs {p2}"[:40]
        print(f"{match_str:<40} {bo:>3} {pred:>6.1f} {median:>6.1f} {gap:>+6.1f}")

    print(f"\nAby postawić: użyj trybu interaktywnego podając pred i linię bukmachera.")


def interactive_mode(dists: dict) -> None:
    print("\n=== KALKULATOR O/U GEMÓW ===")
    print("Podaj dane meczu (Enter = wyjście)\n")

    while True:
        try:
            pred_str = input("Predykcja modelu (gemów): ").strip()
            if not pred_str:
                break
            pred = float(pred_str)

            line_str = input("Linia bukmachera (np. 22.5): ").strip()
            if not line_str:
                break
            line = float(line_str)

            bo_str = input("Best of [3/5]: ").strip() or "3"
            best_of = int(bo_str)

            surf = input("Nawierzchnia [Hard/Clay/Grass/all]: ").strip() or "all"

            o_str = input("Kurs na OVER: ").strip() or "1.85"
            u_str = input("Kurs na UNDER: ").strip() or "1.85"
            odds_over  = float(o_str)
            odds_under = float(u_str)

            analyze_match(pred, line, best_of, odds_over, odds_under, surf)

            # Pokaż kontekst historyczny
            key = f"{surf}_bo{best_of}"
            if key not in dists:
                key = f"all_bo{best_of}"
            if key in dists:
                d = dists[key]
                print(f"Kontekst historyczny ({key}): "
                      f"mediana={d['p50']:.0f}, "
                      f"Q25={d['p25']:.0f}, Q75={d['p75']:.0f}  "
                      f"(n={d['n']:,})")

        except (ValueError, KeyboardInterrupt):
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch",     action="store_true", help="Batch mode: pokaż przyszłe mecze")
    parser.add_argument("--calibrate", action="store_true", help="Pokaż kalibrację modelu TG")
    parser.add_argument("--surface",   default="ALL")
    args = parser.parse_args()

    dists = load_distributions()

    if args.calibrate:
        calibrate()
    elif args.batch:
        batch_mode(args.surface)
    else:
        interactive_mode(dists)


if __name__ == "__main__":
    main()
