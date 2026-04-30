"""
BETS GOD dashboard — Streamlit
Wszystkie 17 targetow z auto_outputs_future/future_predictions_1row_all_targets.csv
Uruchom: python -m streamlit run app.py
"""
from pathlib import Path
from datetime import date
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BETS GOD", layout="wide", page_icon="💰")

# Sidebar szerszy + multiselect zwijany scrollem
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    min-width: 380px !important;
}
section[data-testid="stSidebar"] div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:first-child {
    flex-wrap: wrap !important;
    align-items: flex-start !important;
    padding: 4px !important;
}
section[data-testid="stSidebar"] div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    margin: 2px !important;
}
</style>
""", unsafe_allow_html=True)

DIR = Path(__file__).resolve().parent
SRC = DIR.parent / "auto_outputs_future" / "future_predictions_1row_all_targets.csv"
CALIB = DIR.parent / "auto_outputs_future" / "calibration_5pct_bins.csv"
FIX = DIR.parent / "fixtures_22_ligues_2025_26.csv"

# (target, czytelna_nazwa, kategoria, czy_OU)
_BASE_TARGETS = {
    "FTR":     ("Wynik (1X2)",                 "WYNIK",       False),
    "O25":     ("Gole — suma",                  "GOLE",        True),
    "BTTS":    ("BTTS (obie strzela)",         "GOLE",        False),
    "HGOOU":   ("Gole — gospodarz",             "GOLE H/A",    True),
    "AGOOU":   ("Gole — gosc",                  "GOLE H/A",    True),
    "CRD25":   ("Kartki — suma",                "KARTKI",      True),
    "HCRDOU":  ("Kartki — gospodarz",           "KARTKI H/A",  True),
    "ACRDOU":  ("Kartki — gosc",                "KARTKI H/A",  True),
    "FOULSOU": ("Faule — suma",                 "FAULE",       True),
    "HFOULOU": ("Faule — gospodarz",            "FAULE H/A",   True),
    "AFOULOU": ("Faule — gosc",                 "FAULE H/A",   True),
    "CORNOU":  ("Rozne — suma",                 "ROZNE",       True),
    "HCORNOU": ("Rozne — gospodarz",            "ROZNE H/A",   True),
    "ACORNOU": ("Rozne — gosc",                 "ROZNE H/A",   True),
    "SHTOU":   ("Strzaly celne — suma",         "STRZALY",     True),
    "HSHTOU":  ("Strzaly celne — gospodarz",    "STRZALY H/A", True),
    "ASHTOU":  ("Strzaly celne — gosc",         "STRZALY H/A", True),
}

CATEGORY_ORDER = ["WYNIK", "GOLE", "GOLE H/A", "KARTKI", "KARTKI H/A",
                  "FAULE", "FAULE H/A", "ROZNE", "ROZNE H/A",
                  "STRZALY", "STRZALY H/A"]


def _make_targets():
    out = {}
    for k, (label, cat, is_ou) in _BASE_TARGETS.items():
        thr = f"{k}_ou_threshold" if is_ou or k == "O25" else None
        out[k] = (f"{k}_pred_class", f"{k}_max_prob", f"{k}_test_acc", thr, label, cat, "BAZA")
        if is_ou or k == "O25":
            for suf, slabel in [("_LO", "linia -1"), ("_HI", "linia +1")]:
                kk = f"{k}{suf}"
                thr_kk = f"{kk}_ou_threshold"
                out[kk] = (f"{kk}_pred_class", f"{kk}_max_prob", f"{kk}_test_acc",
                           thr_kk, f"{label} [{slabel}]", cat, slabel)
    return out

TARGETS = _make_targets()


UNIT_BY_BASE = {
    "FTR":     "",
    "O25":     "GOLI",
    "BTTS":    "",
    "CRD25":   "KARTEK",
    "HCRDOU":  "KARTEK GOSPODARZA",
    "ACRDOU":  "KARTEK GOSCIA",
    "FOULSOU": "FAULI",
    "HFOULOU": "FAULI GOSPODARZA",
    "AFOULOU": "FAULI GOSCIA",
    "CORNOU":  "ROZNYCH",
    "HCORNOU": "ROZNYCH GOSPODARZA",
    "ACORNOU": "ROZNYCH GOSCIA",
    "SHTOU":   "STRZALOW CELNYCH",
    "HSHTOU":  "STRZALOW CELNYCH GOSPODARZA",
    "ASHTOU":  "STRZALOW CELNYCH GOSCIA",
    "HGOOU":   "GOLI GOSPODARZA",
    "AGOOU":   "GOLI GOSCIA",
}


def readable_typ(target_full: str, typ_short: str) -> str:
    """np. ('CRD25_LO', 'OVER 1.5') -> 'OVER 1.5 KARTEK'."""
    base = target_full
    if base.endswith("_LO"): base = base[:-3]
    elif base.endswith("_HI"): base = base[:-3]
    unit = UNIT_BY_BASE.get(base, "")
    if base == "FTR":
        return f"WYNIK: {typ_short}"
    if base == "BTTS":
        return f"BTTS: {typ_short}"
    if not typ_short:
        return ""
    parts = typ_short.split()
    side = parts[0]  # OVER / UNDER
    val = parts[1] if len(parts) > 1 else ""
    return f"{side} {val} {unit}".strip()


@st.cache_data(ttl=60)
def load_data():
    if not SRC.exists():
        return None, None, None
    df = pd.read_csv(SRC, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    fx = None
    if FIX.exists():
        fx = pd.read_csv(FIX, low_memory=False)
        if "Date" in fx.columns and "Time" in fx.columns:
            fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
            fx = fx[["Date", "HomeTeam", "AwayTeam", "Time"]]
    cal = None
    if CALIB.exists():
        cal = pd.read_csv(CALIB, low_memory=False)
    return df, fx, cal


def lookup_bin_acc(cal, league, target_mode, outcome, prob):
    """Zwroc actual_rate z 5% bina dla (league, target, outcome, prob).
    Fallback: jesli bin pusty/maly -> uzyj league=ALL."""
    if cal is None or pd.isna(prob):
        return None, None
    for lg_try in [league, "ALL"]:
        sub = cal[(cal["league"] == lg_try) &
                  (cal["target_mode"] == target_mode) &
                  (cal["outcome"] == outcome) &
                  (cal["bin_start"] <= prob) & (prob < cal["bin_end"] + 1e-9)]
        if not sub.empty and sub["n"].iloc[0] >= 5:
            return float(sub["actual_rate"].iloc[0]), int(sub["n"].iloc[0])
    return None, None


def fmt_pick(tname, cls, thr):
    if pd.isna(cls):
        return None
    cls = str(cls)
    if tname == "FTR":
        return {"H": "1 (gosp)", "D": "X (remis)", "A": "2 (gosc)"}.get(cls, cls)
    thr_str = "" if pd.isna(thr) else f" {thr}"
    if cls == "O":   return f"OVER{thr_str}"
    if cls == "U":   return f"UNDER{thr_str}"
    if cls == "YES": return "TAK"
    if cls == "NO":  return "NIE"
    return cls


def build_tips(df: pd.DataFrame, fx, cal) -> pd.DataFrame:
    out = []
    for _, r in df.iterrows():
        # godzina meczu
        t = ""
        if fx is not None:
            m = fx[(fx["Date"] == r["Date"]) &
                   (fx["HomeTeam"].str.contains(str(r["HomeTeam"])[:6], case=False, na=False) |
                    fx["AwayTeam"].str.contains(str(r["AwayTeam"])[:6], case=False, na=False))]
            if not m.empty:
                t = str(m.iloc[0]["Time"])
        liga_full = str(r.get("league", "")).split("::")[0]
        for tname, (cls_c, prob_c, acc_c, thr_c, label, cat, variant) in TARGETS.items():
            if cls_c not in r or pd.isna(r[cls_c]):
                continue
            prob = r.get(prob_c)
            acc_overall = r.get(acc_c)
            if pd.isna(prob):
                continue
            thr = r.get(thr_c) if thr_c and thr_c in r else None
            pick = fmt_pick(tname, r[cls_c], thr)
            outcome = str(r[cls_c])
            # bin-specific accuracy
            bin_acc, bin_n = lookup_bin_acc(cal, liga_full, tname, outcome, float(prob))
            acc_use = bin_acc if bin_acc is not None else (float(acc_overall) if pd.notna(acc_overall) else None)
            if acc_use is None:
                continue
            score = float(prob) * float(acc_use)
            out.append({
                "Date": r["Date"], "godz": t, "liga": liga_full,
                "mecz": f"{r['HomeTeam']} - {r['AwayTeam']}",
                "target": tname,
                "kategoria": cat,
                "wariant": variant,
                "rynek": label,
                "typ_pelny": readable_typ(tname, pick),
                "typ": pick,
                "prob": float(prob),
                "acc": float(acc_use),
                "acc_bin_n": bin_n,
                "acc_overall": float(acc_overall) if pd.notna(acc_overall) else None,
                "score": score,
            })
    return pd.DataFrame(out)


def color_prob(v):
    if pd.isna(v): return ""
    if v >= 0.80: return "background-color: #1b5e20; color: white; font-weight: bold"
    if v >= 0.70: return "background-color: #2e7d32; color: white"
    if v >= 0.60: return "background-color: #66bb6a; color: white"
    if v >= 0.50: return "background-color: #fff176"
    return "background-color: #ef5350; color: white"


def color_score(v):
    if pd.isna(v): return ""
    if v >= 0.60: return "background-color: #1b5e20; color: white; font-weight: bold"
    if v >= 0.45: return "background-color: #2e7d32; color: white"
    if v >= 0.30: return "background-color: #66bb6a; color: white"
    return ""


# ───────── load ─────────
df, fx, cal = load_data()
if df is None:
    st.error(f"Brak pliku: {SRC}\nOdpal najpierw `python auto_future_optimizer.py`")
    st.stop()

# ───────── sidebar ─────────
st.sidebar.title("💰 BETS GOD")
st.sidebar.caption(f"Plik: `{SRC.name}`")
st.sidebar.markdown("---")

dates = sorted(df["Date"].dropna().dt.date.unique())
default_date = date.today() if date.today() in dates else (dates[-1] if dates else date.today())
sel_date = st.sidebar.date_input("Data meczow", value=default_date,
                                  min_value=min(dates) if dates else date.today(),
                                  max_value=max(dates) if dates else date.today())
df_d = df[df["Date"].dt.date == sel_date]

min_prob = st.sidebar.slider("Min prob (%)", 50, 99, 65, 1) / 100
min_acc = st.sidebar.slider("Min accuracy (%)", 30, 95, 50, 1) / 100

st.sidebar.markdown("---")
CAT_LABELS = {
    "WYNIK": "🏆 WYNIK (1X2)",
    "GOLE": "⚽ GOLE (suma + BTTS)",
    "GOLE H/A": "⚽ GOLE — gospodarz / gosc",
    "KARTKI": "🟨 KARTKI — suma",
    "KARTKI H/A": "🟨 KARTKI — gospodarz / gosc",
    "FAULE": "🦵 FAULE — suma",
    "FAULE H/A": "🦵 FAULE — gospodarz / gosc",
    "ROZNE": "🚩 ROZNE — suma",
    "ROZNE H/A": "🚩 ROZNE — gospodarz / gosc",
    "STRZALY": "🎯 STRZALY CELNE — suma",
    "STRZALY H/A": "🎯 STRZALY CELNE — gospodarz / gosc",
}

# pogrupuj targety po kategorii
targets_by_cat = {cat: [] for cat in CATEGORY_ORDER}
for k, v in TARGETS.items():
    cat = v[5]
    if cat in targets_by_cat:
        targets_by_cat[cat].append(k)

st.sidebar.markdown("---")
ligi = sorted(set(str(l).split("::")[0] for l in df_d.get("league", []).unique()))
view_options = ["🏠 Strona glowna"] + [f"⚽ {l}" for l in ligi]
view = st.sidebar.radio("Widok", view_options)

# ───────── tips ─────────
tips = build_tips(df_d, fx, cal)
tips_filt_all = tips[
    (tips["prob"] >= min_prob)
    & (tips["acc"] >= min_acc)
].sort_values("score", ascending=False)

# ───────── strona glowna ─────────
if view == "🏠 Strona glowna":
    st.title("💰 BETS GOD")
    st.caption(f"Data: **{sel_date}** | Lig dzis: **{len(ligi)}** | Typow po filtrach: **{len(tips_filt_all)}**")
    st.markdown("---")

    st.subheader("Wybierz statystyke do przegladania:")

    # selectbox z placeholderem - na poczatku NIC nie pokazujemy
    cat_options = ["— wybierz kategorie —"] + [CAT_LABELS[c] for c in CATEGORY_ORDER if c in targets_by_cat and targets_by_cat[c]]
    chosen_cat_label = st.selectbox(" ", cat_options, label_visibility="collapsed")

    if chosen_cat_label == "— wybierz kategorie —":
        # nic nie pokazuj, daj kafelki podpowiedzi z liczba typow per kategoria
        st.info("👈 Wybierz statystyke z listy powyzej (np. KARTKI, STRZALY, GOLE) zeby zobaczyc typy.")
        cols = st.columns(4)
        for i, cat in enumerate(CATEGORY_ORDER):
            if cat not in targets_by_cat or not targets_by_cat[cat]: continue
            tk = targets_by_cat[cat]
            n = len(tips_filt_all[tips_filt_all["target"].isin(tk)])
            with cols[i % 4]:
                st.metric(CAT_LABELS[cat], f"{n} typow")
    else:
        # znajdz cat key
        cat_key = next(c for c in CATEGORY_ORDER if CAT_LABELS.get(c) == chosen_cat_label)
        cat_targets = targets_by_cat[cat_key]
        cat_tips = tips_filt_all[tips_filt_all["target"].isin(cat_targets)].copy()

        st.markdown(f"## {chosen_cat_label}")
        st.caption(f"{len(cat_tips)} typow po filtrach (≥{int(min_prob*100)}% prob i ≥{int(min_acc*100)}% acc)")

        if cat_tips.empty:
            st.warning("Brak typow spelniajacych filtry. Obniz suwak w sidebarze.")
        else:
            # top 5 jako kafelki COMBO
            best = cat_tips.head(5)
            cols = st.columns(len(best))
            for i, (_, b) in enumerate(best.iterrows()):
                with cols[i]:
                    st.metric(
                        label=f"{b['liga'][:20]}",
                        value=b["typ_pelny"],
                        delta=f"{b['score']*100:.1f}% ({b['prob']*100:.0f}p × {b['acc']*100:.0f}a)",
                    )
                    st.caption(f"⏰ {b['godz']} • {b['mecz']}")
            st.markdown("---")

            top_n = st.slider("Ile typow pokazac", 10, 300, 100, 10)
            show = cat_tips.head(top_n).copy()
            show["prob"] = (show["prob"] * 100).round(1)
            show["acc"] = (show["acc"] * 100).round(1)
            show["score"] = (show["score"] * 100).round(1)
            show["acc_overall"] = (show["acc_overall"] * 100).round(1)
            show = show[["godz", "liga", "mecz", "typ_pelny", "wariant", "prob", "acc", "acc_bin_n", "acc_overall", "score"]]
            show.columns = ["Godz", "Liga", "Mecz", "TYP", "Wariant", "Prob %", "Acc bin %", "Bin n", "Acc total %", "Score %"]
            st.dataframe(
                show.style.applymap(color_prob, subset=["Prob %"])
                          .applymap(color_prob, subset=["Acc bin %"])
                          .applymap(color_score, subset=["Score %"])
                          .format({"Prob %": "{:.1f}", "Acc bin %": "{:.1f}",
                                   "Acc total %": "{:.1f}", "Score %": "{:.1f}"}, na_rep="-"),
                use_container_width=True, height=600, hide_index=True,
            )
            st.caption("**TYP** = peln1 czytelne haslo (np. OVER 4.5 STRZALOW CELNYCH GOSCIA). **Acc bin %** = skutecznosc w 5%-binie gdzie wpadla prob.")
            st.download_button("📥 Pobierz CSV", show.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"bets_god_{cat_key}_{sel_date}.csv")

# ───────── per liga ─────────
else:
    liga = view.replace("⚽ ", "")
    st.title(f"⚽ {liga}")
    st.caption(f"Data: **{sel_date}**")

    df_l = df_d[df_d["league"].astype(str).str.startswith(liga)]
    if df_l.empty:
        st.warning("Brak meczow tej ligi w wybranej dacie.")
        st.stop()

    tips_l_all = tips[tips["liga"] == liga]
    tips_l = tips_l_all[(tips_l_all["prob"] >= min_prob) & (tips_l_all["acc"] >= min_acc)]

    # ACCURACY tiles pogrupowane po kategorii
    st.subheader("📊 ACCURACY MODELU NA TEJ LIDZE (per kategoria)")
    for cat in CATEGORY_ORDER:
        cat_targets = [k for k, v in TARGETS.items() if v[5] == cat]
        if not cat_targets: continue
        with st.expander(f"**{CAT_LABELS.get(cat, cat)}** ({len(cat_targets)} wariantow)", expanded=(cat in ("WYNIK", "GOLE"))):
            rows_per = 4
            for chunk_start in range(0, len(cat_targets), rows_per):
                chunk = cat_targets[chunk_start:chunk_start + rows_per]
                cols = st.columns(len(chunk))
                for i, tname in enumerate(chunk):
                    _, _, acc_c, thr_c, label, _, variant = TARGETS[tname]
                    with cols[i]:
                        if acc_c in df_l.columns:
                            a = df_l[acc_c].dropna()
                            if len(a):
                                thr_val = ""
                                if thr_c and thr_c in df_l.columns:
                                    tv = df_l[thr_c].dropna()
                                    if len(tv): thr_val = f" @ {tv.iloc[0]}"
                                st.metric(f"**{label}**{thr_val}", f"{a.iloc[0]*100:.1f}%",
                                          help=f"{tname} ({variant})")

    st.markdown("---")
    st.subheader("🎯 TYPY MECZOW — wybierz statystyke")
    cat_options_l = ["— wybierz kategorie —"] + [CAT_LABELS[c] for c in CATEGORY_ORDER if c in targets_by_cat and targets_by_cat[c]]
    chosen_cat_l = st.selectbox("Kategoria:", cat_options_l, key="cat_per_liga", label_visibility="collapsed")

    if chosen_cat_l == "— wybierz kategorie —":
        st.info("👆 Wybierz statystyke z listy zeby zobaczyc typy.")
        cols = st.columns(4)
        for i, cat in enumerate(CATEGORY_ORDER):
            if cat not in targets_by_cat or not targets_by_cat[cat]: continue
            tk = targets_by_cat[cat]
            n = len(tips_l[tips_l["target"].isin(tk)])
            with cols[i % 4]:
                st.metric(CAT_LABELS[cat], f"{n} typow")
    else:
        cat_key = next(c for c in CATEGORY_ORDER if CAT_LABELS.get(c) == chosen_cat_l)
        cat_tk = targets_by_cat[cat_key]
        show_t = tips_l[tips_l["target"].isin(cat_tk)].sort_values("score", ascending=False).copy()
        if show_t.empty:
            st.warning("Brak typow tej kategorii spelniajacych filtry.")
        else:
            for c in ["prob", "acc", "score", "acc_overall"]:
                if c in show_t.columns:
                    show_t[c] = pd.to_numeric(show_t[c], errors="coerce")
            show_t["prob"] = (show_t["prob"] * 100).round(1)
            show_t["acc"] = (show_t["acc"] * 100).round(1)
            show_t["acc_overall"] = (show_t["acc_overall"] * 100).round(1)
            show_t["score"] = (show_t["score"] * 100).round(1)
            show_t = show_t[["godz", "mecz", "typ_pelny", "wariant", "prob", "acc", "acc_bin_n", "acc_overall", "score"]]
            show_t.columns = ["Godz", "Mecz", "TYP", "Wariant", "Prob %", "Acc bin %", "Bin n", "Acc total %", "Score %"]
            st.dataframe(
                show_t.style.applymap(color_prob, subset=["Prob %"])
                          .applymap(color_prob, subset=["Acc bin %"])
                          .applymap(color_score, subset=["Score %"])
                          .format({"Prob %": "{:.1f}", "Acc bin %": "{:.1f}",
                                   "Acc total %": "{:.1f}", "Score %": "{:.1f}"}, na_rep="-"),
                use_container_width=True, height=600, hide_index=True,
            )
            st.download_button("📥 Pobierz CSV",
                               show_t.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{liga.replace('/', '_')}_{cat_key}_{sel_date}.csv")
