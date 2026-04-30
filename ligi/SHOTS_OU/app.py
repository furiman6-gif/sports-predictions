"""
SHOTS O/U dashboard — Streamlit
Uruchom: streamlit run app.py
"""
import re
from pathlib import Path
from datetime import date
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SHOTS O/U", layout="wide", page_icon="⚽")

DIR = Path(__file__).resolve().parent

# ───────── helpers ─────────
@st.cache_data(ttl=60)
def load_latest():
    files = sorted(DIR.glob("predykcje_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None, None, None
    # preferuj clf jesli jest
    clf = [f for f in files if "_clf" in f.name]
    pick = clf[0] if clf else files[0]
    df = pd.read_csv(pick, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    mode = "clf" if "_clf" in pick.name else "reg"
    return df, pick.name, mode


def parse_threshold(col: str):
    m = re.search(r"P_over_([\d\.]+)$", col)
    return float(m.group(1)) if m else None


def parse_target(col: str):
    if col.startswith("TOT_SHOTS"): return "TOT", "Total strzalow celnych"
    if col.startswith("H_SHOTS"):   return "H", "Strzaly gospodarzy"
    if col.startswith("A_SHOTS"):   return "A", "Strzaly gosci"
    return None, None


def get_acc(df: pd.DataFrame, target_prefix: str, line: float):
    col = f"{target_prefix}_SHOTS_acc_{line}"
    if col in df.columns:
        v = df[col].dropna()
        if len(v): return float(v.iloc[0])
    return None


def build_tips(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Rozwin tabele 1-row-per-mecz na 1-row-per-typ."""
    out = []
    for _, r in df.iterrows():
        for col in df.columns:
            t = parse_threshold(col)
            if t is None or not col.endswith(str(t)) or "P_over" not in col: continue
            tgt, _ = parse_target(col)
            if tgt is None: continue
            prob = r[col]
            if pd.isna(prob): continue
            pred_col = f"{tgt}_SHOTS_pred"
            avg_col = f"{tgt}_SHOTS_avg2y"
            sig_col = f"{tgt}_SHOTS_sigma"
            acc_col = f"{tgt}_SHOTS_acc_{t}"
            acc = float(r[acc_col]) if acc_col in df.columns and pd.notna(r.get(acc_col)) else None
            # OVER czy UNDER mocniejszy?
            side = "OVER" if prob >= 0.5 else "UNDER"
            p_pick = prob if side == "OVER" else (1 - prob)
            score = p_pick * acc if acc is not None else p_pick
            out.append({
                "Date": r["Date"], "liga": r["liga"],
                "mecz": f"{r['HomeTeam']} - {r['AwayTeam']}",
                "target": tgt, "linia": t, "typ": f"{side} {t}",
                "prob": p_pick,
                "acc": acc,
                "score": score,
                "pred": r.get(pred_col), "avg2y": r.get(avg_col), "sigma": r.get(sig_col),
            })
    return pd.DataFrame(out)


def color_prob(v):
    if pd.isna(v): return ""
    if v >= 0.75: return "background-color: #2e7d32; color: white; font-weight: bold"
    if v >= 0.65: return "background-color: #66bb6a; color: white"
    if v >= 0.55: return "background-color: #fff176"
    return "background-color: #ef5350; color: white"


# ───────── load ─────────
df, fname, mode = load_latest()
if df is None:
    st.error("Brak plikow predykcje_*.csv w SHOTS_OU/. Odpal najpierw `python train_shots.py`.")
    st.stop()

# ───────── sidebar ─────────
st.sidebar.title("⚽ SHOTS O/U")
st.sidebar.caption(f"Plik: `{fname}` ({mode.upper()})")
st.sidebar.markdown("---")

# data filter
dates = sorted(df["Date"].dropna().dt.date.unique())
default_date = date.today() if date.today() in dates else dates[0]
sel_date = st.sidebar.date_input("Data meczow", value=default_date,
                                  min_value=min(dates), max_value=max(dates))
df_d = df[df["Date"].dt.date == sel_date]

min_prob = st.sidebar.slider("Min prob (%)", 50, 99, 65, 1) / 100
st.sidebar.markdown("---")

# nawigacja
ligi = sorted(df_d["liga"].unique())
view = st.sidebar.radio("Widok", ["🏠 Strona glowna"] + [f"⚽ {l}" for l in ligi])

# ───────── strona glowna ─────────
tips = build_tips(df_d, mode)
tips_filt = tips[tips["prob"] >= min_prob].sort_values("score", ascending=False)

if view == "🏠 Strona glowna":
    st.title("🏆 NAJMOCNIEJSZE TYPY DNIA")
    st.caption(f"Data: **{sel_date}** | Lig z meczami: **{len(ligi)}** | Typow ≥{int(min_prob*100)}%: **{len(tips_filt)}**")

    if mode == "reg":
        st.info("⚠ Tryb REG (regresja) — kolumny `acc` puste. Dla accuracy odpal opcje [3] STRZALY DOKLADNIE.")

    top_n = st.slider("Ile typow pokazac", 10, 100, 30, 5)
    show = tips_filt.head(top_n).copy()
    for c in ["prob", "acc", "score", "pred", "avg2y", "sigma"]:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce")
    show["prob"] = (show["prob"] * 100).round(1)
    show["acc"] = (show["acc"] * 100).round(1)
    show["score"] = (show["score"] * 100).round(1)
    show["pred"] = show["pred"].round(2)
    show["avg2y"] = show["avg2y"].round(2)
    show = show[["Date", "liga", "mecz", "typ", "prob", "acc", "score", "pred", "avg2y"]]
    show.columns = ["Data", "Liga", "Mecz", "Typ", "Prob %", "Acc %", "Score %", "Pred", "Avg ligi"]

    st.dataframe(
        show.style.applymap(color_prob, subset=["Prob %"]).format({
            "Prob %": "{:.1f}", "Acc %": "{:.1f}", "Score %": "{:.1f}",
            "Pred": "{:.2f}", "Avg ligi": "{:.2f}",
        }, na_rep="-"),
        use_container_width=True, height=600, hide_index=True,
    )

    st.download_button("📥 Pobierz CSV", show.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"top_typy_{sel_date}.csv")

# ───────── per liga ─────────
else:
    liga = view.replace("⚽ ", "")
    st.title(f"⚽ {liga}")
    st.caption(f"Data: **{sel_date}**")

    df_l = df_d[df_d["liga"] == liga]
    tips_l = tips[(tips["liga"] == liga)]

    if df_l.empty:
        st.warning("Brak meczow tej ligi w wybranej dacie.")
        st.stop()

    # ACCURACY tiles
    st.subheader("📊 ACCURACY MODELU NA TEJ LIDZE")
    cols = st.columns(3)
    for i, (pref, label) in enumerate([("TOT", "TOTAL"), ("H", "HOME"), ("A", "AWAY")]):
        with cols[i]:
            st.markdown(f"### {label}")
            avg_col = f"{pref}_SHOTS_avg2y"
            avg = df_l[avg_col].dropna().iloc[0] if avg_col in df_l.columns and df_l[avg_col].notna().any() else None
            sig_col = f"{pref}_SHOTS_sigma"
            sig = df_l[sig_col].dropna().iloc[0] if sig_col in df_l.columns and df_l[sig_col].notna().any() else None
            if avg is not None: st.metric("Avg ligi 2 lata", f"{avg:.2f}")
            if sig is not None: st.metric("Sigma modelu", f"{sig:.2f}")
            # accuracy per linia (clf)
            for col in df_l.columns:
                if col.startswith(f"{pref}_SHOTS_acc_"):
                    line = col.replace(f"{pref}_SHOTS_acc_", "")
                    v = df_l[col].dropna()
                    if len(v):
                        st.metric(f"Acc @ {line}", f"{v.iloc[0]*100:.1f}%")

    st.markdown("---")
    st.subheader(f"🎯 TYPY MECZOW ({len(tips_l)} typow, filtr ≥{int(min_prob*100)}%)")

    show = tips_l[tips_l["prob"] >= min_prob].sort_values("score", ascending=False).copy()
    if show.empty:
        st.warning("Zaden typ nie spelnia progu. Obniz suwak w sidebarze.")
    else:
        for c in ["prob", "acc", "score", "pred", "avg2y", "sigma"]:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce")
        show["prob"] = (show["prob"] * 100).round(1)
        show["acc"] = (show["acc"] * 100).round(1)
        show["score"] = (show["score"] * 100).round(1)
        show["pred"] = show["pred"].round(2)
        show["avg2y"] = show["avg2y"].round(2)
        show["sigma"] = show["sigma"].round(2)
        show = show[["mecz", "typ", "target", "prob", "acc", "score", "pred", "avg2y", "sigma"]]
        show.columns = ["Mecz", "Typ", "Strona", "Prob %", "Acc %", "Score %", "Pred", "Avg ligi", "Sigma"]
        st.dataframe(
            show.style.applymap(color_prob, subset=["Prob %"]).format({
                "Prob %": "{:.1f}", "Acc %": "{:.1f}", "Score %": "{:.1f}",
                "Pred": "{:.2f}", "Avg ligi": "{:.2f}", "Sigma": "{:.2f}",
            }, na_rep="-"),
            use_container_width=True, height=600, hide_index=True,
        )
        st.download_button("📥 Pobierz CSV ligi", show.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{liga.replace('/', '_')}_{sel_date}.csv")
