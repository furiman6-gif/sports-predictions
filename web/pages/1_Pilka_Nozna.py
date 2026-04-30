import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Pilka Nozna", page_icon="⚽", layout="wide")
st.title("⚽ Predykcje piłkarskie")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ligi"

# --- Load data ---
@st.cache_data(ttl=3600)
def load_predictions():
    path = DATA_DIR / "future_predictions_all.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_wide():
    path = DATA_DIR / "future_predictions_1row_all_targets.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_best_settings():
    path = DATA_DIR / "best_settings.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


df = load_predictions()
df_wide = load_wide()
best = load_best_settings()

if df is None or df.empty:
    st.warning("Brak danych predykcji. Pipeline jeszcze nie wygenerował wyników.")
    st.stop()

# --- Filters ---
st.sidebar.header("Filtry")

leagues = sorted(df["league"].dropna().unique())
sel_leagues = st.sidebar.multiselect("Liga", leagues, default=leagues)

targets = sorted(df["target_mode"].dropna().unique())
sel_targets = st.sidebar.multiselect("Target", targets, default=["FTR"])

min_prob = st.sidebar.slider("Min prawdopodobieństwo", 0.0, 1.0, 0.50, 0.05)

# --- Filter ---
mask = (
    df["league"].isin(sel_leagues)
    & df["target_mode"].isin(sel_targets)
    & (df["max_prob"] >= min_prob)
)
filtered = df[mask].sort_values(["Date", "league"]).copy()

# --- Main table ---
st.subheader(f"Predykcje ({len(filtered)} meczów)")

display_cols = ["Date", "league", "HomeTeam", "AwayTeam", "target_mode", "pred_class", "max_prob"]
available = [c for c in display_cols if c in filtered.columns]

# Probabilities
prob_cols = [c for c in filtered.columns if c.startswith("pred_") and c not in ["pred_class"]]
available += [c for c in prob_cols if c in filtered.columns]

if not filtered.empty:
    show = filtered[available].copy()
    show["Date"] = show["Date"].dt.strftime("%Y-%m-%d")

    # Color by confidence
    st.dataframe(
        show.style.background_gradient(subset=["max_prob"], cmap="RdYlGn", vmin=0.3, vmax=0.8),
        use_container_width=True,
        height=500,
    )
else:
    st.info("Brak meczów spełniających kryteria.")

# --- Charts ---
if not filtered.empty:
    st.subheader("Rozkład pewności predykcji")
    fig = px.histogram(filtered, x="max_prob", nbins=20, color="target_mode",
                       title="Histogram prawdopodobieństw", barmode="overlay")
    fig.update_layout(xaxis_title="Prawdopodobieństwo", yaxis_title="Liczba meczów")
    st.plotly_chart(fig, use_container_width=True)

# --- Model performance ---
if best is not None and not best.empty:
    st.subheader("Jakość modeli (test set)")
    best_filtered = best[best["target_mode"].isin(sel_targets)]
    if not best_filtered.empty:
        perf_cols = ["league", "target_mode", "test_acc", "test_logloss", "n_test", "n_future"]
        perf_available = [c for c in perf_cols if c in best_filtered.columns]
        st.dataframe(
            best_filtered[perf_available].sort_values(["target_mode", "league"]),
            use_container_width=True,
        )

        fig2 = px.bar(best_filtered, x="league", y="test_acc", color="target_mode",
                      barmode="group", title="Accuracy per liga/target")
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
