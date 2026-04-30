"""
kalkulator_zakładów.py
======================
Kalkulator wartości i rozmiaru stawki dla zakładów tenisowych.

Uruchomienie:
    python kalkulator_zakładów.py
"""
from __future__ import annotations

import math


# ── Konfiguracja ──────────────────────────────────────────────────────────────
BANKROLL       = 2000.0   # PLN — zmień na swój bankroll
KELLY_FRACTION = 0.25     # quarter Kelly (bezpieczne), zmień na 0.5 = half Kelly
MIN_EDGE       = 0.03     # minimalny edge żeby w ogóle rozważać zakład (3%)
MIN_AGREEMENT  = 0.00     # minimalna pewność modelu (0 = bez progu)
MAX_BET_PCT    = 0.05     # max 5% bankrollu na jeden zakład
# ─────────────────────────────────────────────────────────────────────────────


def implied_prob(odds: float) -> float:
    """Prawdopodobieństwo implikowane przez kurs (bez marginesu)."""
    return 1.0 / odds


def margin(odds1: float, odds2: float) -> float:
    """Marża bukmachera (vig)."""
    return (1 / odds1 + 1 / odds2 - 1) * 100


def fair_prob(odds_bet: float, odds_other: float) -> float:
    """Prawdopodobieństwo implikowane po usunięciu marży bukmachera."""
    total = 1 / odds_bet + 1 / odds_other
    return (1 / odds_bet) / total


def kelly_stake(prob: float, odds: float, bankroll: float,
                fraction: float = KELLY_FRACTION,
                max_pct: float = MAX_BET_PCT) -> float:
    """
    Kelly Criterion: f = (p*b - q) / b
    gdzie b = odds - 1, p = prob wygranej, q = 1 - p
    Zwraca stawkę w PLN (z limitem max_pct bankrollu).
    """
    b = odds - 1
    q = 1 - prob
    f = (prob * b - q) / b
    if f <= 0:
        return 0.0
    full_kelly = f * bankroll
    fractional = full_kelly * fraction
    max_allowed = bankroll * max_pct
    return min(fractional, max_allowed)


def expected_value(prob: float, odds: float) -> float:
    """EV na 1 PLN stawki."""
    return prob * (odds - 1) - (1 - prob)


def edge(model_prob: float, fair_bookie_prob: float) -> float:
    """Przewaga modelu nad bukmacherem (po usunięciu marży)."""
    return model_prob - fair_bookie_prob


def verdict(model_prob: float, odds_bet: float, odds_other: float,
            bankroll: float) -> dict:
    fair_p   = fair_prob(odds_bet, odds_other)
    e        = edge(model_prob, fair_p)
    ev       = expected_value(model_prob, odds_bet)
    stake    = kelly_stake(model_prob, odds_bet, bankroll)
    vig      = margin(odds_bet, odds_other)
    impl_p   = implied_prob(odds_bet)
    roi_pct  = ev * 100

    return {
        "model_prob":  model_prob,
        "impl_prob":   impl_p,
        "fair_prob":   fair_p,
        "edge":        e,
        "ev":          ev,
        "roi_pct":     roi_pct,
        "stake":       stake,
        "vig":         vig,
        "bet":         e >= MIN_EDGE and ev > 0 and model_prob >= MIN_AGREEMENT,
    }


def print_verdict(v: dict, odds: float, label: str) -> None:
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  ZAKŁAD: {label}  @  {odds}")
    print(f"{'─'*52}")
    print(f"  Prawdopodobieństwo modelu  : {v['model_prob']*100:>6.1f}%")
    print(f"  Implikowane przez kurs     : {v['impl_prob']*100:>6.1f}%")
    print(f"  Uczciwe (po usunięciu vig) : {v['fair_prob']*100:>6.1f}%")
    print(f"  Edge (model vs buk)        : {v['edge']*100:>+6.2f}%")
    print(f"  EV na 1 PLN                : {v['ev']:>+6.4f} PLN")
    print(f"  ROI                        : {v['roi_pct']:>+6.2f}%")
    print(f"  Marża bukmachera           : {v['vig']:>6.2f}%")
    print(f"{'─'*52}")

    if v["bet"]:
        print(f"  ✓ GRAĆ  —  stawka: {v['stake']:.2f} PLN")
        print(f"  (bankroll {BANKROLL:.0f} PLN, "
              f"{KELLY_FRACTION*100:.0f}% Kelly, "
              f"max {MAX_BET_PCT*100:.0f}%)")
    else:
        reasons = []
        if v["edge"] < MIN_EDGE:
            reasons.append(f"edge {v['edge']*100:.2f}% < min {MIN_EDGE*100:.0f}%")
        if v["ev"] <= 0:
            reasons.append("EV ujemne")
        print(f"  ✗ PASS  —  {', '.join(reasons)}")

    print(sep)


def confidence_label(prob: float) -> str:
    if prob >= 0.90: return "BARDZO WYSOKA (>90%)"
    if prob >= 0.75: return "WYSOKA (75-90%)"
    if prob >= 0.60: return "ŚREDNIA (60-75%)"
    if prob >= 0.50: return "NISKA (50-60%)"
    return "BRAK FAWORYTA"


def run_calculator() -> None:
    print("\n" + "=" * 52)
    print("  KALKULATOR ZAKŁADÓW TENISOWYCH")
    print(f"  Bankroll: {BANKROLL:.0f} PLN  |  "
          f"Kelly: {KELLY_FRACTION*100:.0f}%  |  "
          f"Min edge: {MIN_EDGE*100:.0f}%")
    print("=" * 52)
    print("  (Enter bez wartości = wyjście)\n")

    while True:
        try:
            # Gracze
            p1 = input("Gracz 1 (lub nazwa zakładu): ").strip()
            if not p1:
                break
            p2 = input("Gracz 2                    : ").strip()

            # Prawdopodobieństwa modelu
            prob_str = input(f"Prawdopod. modelu {p1} wygra [0-1 lub %]: ").strip()
            if not prob_str:
                break
            prob_val = float(prob_str.replace(",", ".").replace("%", ""))
            prob_p1  = prob_val / 100 if prob_val > 1 else prob_val
            prob_p2  = 1 - prob_p1

            print(f"\n  Pewność modelu: {confidence_label(max(prob_p1, prob_p2))}")

            # Kursy
            odds_p1 = float(input(f"Kurs na {p1:<20}: ").strip().replace(",", "."))
            odds_p2 = float(input(f"Kurs na {p2:<20}: ").strip().replace(",", "."))

            # Analiza obu stron
            v1 = verdict(prob_p1, odds_p1, odds_p2, BANKROLL)
            v2 = verdict(prob_p2, odds_p2, odds_p1, BANKROLL)

            print_verdict(v1, odds_p1, p1)
            print_verdict(v2, odds_p2, p2)

            # Podsumowanie
            print("\n  PODSUMOWANIE:")
            if v1["bet"] and v2["bet"]:
                # Oba mają edge — rzadko, ale możliwe (arbitraż)
                print(f"  ARB? Sprawdź — obie strony mają pozytywny edge")
                print(f"  P1: {v1['stake']:.2f} PLN  |  P2: {v2['stake']:.2f} PLN")
            elif v1["bet"]:
                print(f"  → ZAKŁAD na {p1}: {v1['stake']:.2f} PLN")
                pot_win = v1["stake"] * (odds_p1 - 1)
                print(f"     Potencjalny zysk: +{pot_win:.2f} PLN  "
                      f"| Strata przy przegranej: -{v1['stake']:.2f} PLN")
            elif v2["bet"]:
                print(f"  → ZAKŁAD na {p2}: {v2['stake']:.2f} PLN")
                pot_win = v2["stake"] * (odds_p2 - 1)
                print(f"     Potencjalny zysk: +{pot_win:.2f} PLN  "
                      f"| Strata przy przegranej: -{v2['stake']:.2f} PLN")
            else:
                print("  → PASS na oba — brak wartości")

            print()

            # Pytanie o kolejny mecz
            again = input("  Kolejny mecz? [Enter=tak / n=wyjście]: ").strip().lower()
            if again == "n":
                break
            print()

        except (ValueError, KeyboardInterrupt):
            print("\n  Wyjście.")
            break


if __name__ == "__main__":
    run_calculator()
