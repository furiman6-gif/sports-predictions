import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Tenis", page_icon="🎾", layout="wide")
st.title("🎾 Predykcje tenisowe")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "tenis"

@st.cache_data(ttl=3600)
def load_predictions():
    path = DATA_DIR / "future_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_recommended():
    path = DATA_DIR / "future_predictions_recommended.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def load_summary():
    path = DATA_DIR / "summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


df = load_predictions()
df_rec = load_recommended()
summary = load_summary()

if df is None or df.empty:
    st.warning("Brak danych predykcji. Pipeline jeszcze nie wygenerował wyników.")
    st.stop()

# --- Filters ---
st.sidebar.header("Filtry")

surfaces = sorted(df["Surface"].dropna().unique()) if "Surface" in df.columns else []
sel_surfaces = st.sidebar.multiselect("Nawierzchnia", surfaces, default=surfaces)

tournaments = sorted(df["Tournament"].dropna().unique()) if "Tournament" in df.columns else []
sel_tournaments = st.sidebar.multiselect("Turniej", tournaments, default=tournaments)

min_prob = st.sidebar.slider("Min prob P1 wins", 0.0, 1.0, 0.0, 0.05)
only_bets = st.sidebar.checkbox("Tylko z bet_flag = 1", value=False)

# --- Filter ---
mask = pd.Series(True, index=df.index)
if sel_surfaces:
    mask &= df["Surface"].isin(sel_surfaces)
if sel_tournaments:
    mask &= df["Tournament"].isin(sel_tournaments)
if "prob_P1_wins" in df.columns:
    max_prob = df[["prob_P1_wins", "prob_P2_wins"]].max(axis=1)
    mask &= max_prob >= min_prob
if only_bets and "bet_flag" in df.columns:
    mask &= df["bet_flag"] == 1

filtered = df[mask].sort_values("Date").copy()

# --- Tabs ---
tab_all, tab_rec = st.tabs(["Wszystkie predykcje", "Rekomendowane zakłady"])

with tab_all:
    st.subheader(f"Predykcje ({len(filtered)} meczów)")

    display_cols = [
        "Date", "Tournament", "Surface", "Round",
        "P1", "P2", "P1Rank", "P2Rank",
        "prob_P1_wins", "prob_P2_wins", "pred",
        "pred_total_games",
        "pick_side", "pick_odds", "pick_edge_prob", "pick_ev", "bet_flag",
    ]
    available = [c for c in display_cols if c in filtered.columns]

    if not filtered.empty:
        show = filtered[available].copy()
        show["Date"] = show["Date"].dt.strftime("%Y-%m-%d")

        prob_col = "prob_P1_wins" if "prob_P1_wins" in show.columns else None
        if prob_col:
            st.dataframe(
                show.style.background_gradient(subset=[prob_col], cmap="RdYlGn", vmin=0.2, vmax=0.8),
                use_container_width=True,
                height=500,
            )
        else:
            st.dataframe(show, use_container_width=True, height=500)
    else:
        st.info("Brak meczów spełniających kryteria.")

    # Charts
    if not filtered.empty and "prob_P1_wins" in filtered.columns:
        st.subheader("Rozkład prawdopodobieństw")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(filtered, x="prob_P1_wins", nbins=20,
                               title="P(P1 wins)", color="Surface")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "pick_ev" in filtered.columns:
                fig2 = px.scatter(filtered, x="prob_P1_wins", y="pick_ev",
                                  color="Surface", hover_data=["P1", "P2"],
                                  title="EV vs Prawdopodobieństwo")
                st.plotly_chart(fig2, use_container_width=True)

with tab_rec:
    if df_rec is not None and not df_rec.empty:
        st.subheader(f"Rekomendowane zakłady ({len(df_rec)})")
        rec_cols = [
            "Date", "Tournament", "P1", "P2",
            "prob_P1_wins", "prob_P2_wins",
            "pick_side", "pick_odds", "pick_edge_prob", "pick_ev",
        ]
        rec_available = [c for c in rec_cols if c in df_rec.columns]
        show_rec = df_rec[rec_available].copy()
        if "Date" in show_rec.columns:
            show_rec["Date"] = pd.to_datetime(show_rec["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        st.dataframe(show_rec, use_container_width=True)
    else:
        st.info("Brak rekomendowanych zakładów.")

# --- Model summary ---
if summary is not None and not summary.empty:
    st.subheader("Model summary")
    st.dataframe(summary, use_container_width=True)
