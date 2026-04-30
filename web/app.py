import streamlit as st

st.set_page_config(
    page_title="Sports Predictions",
    page_icon="⚽",
    layout="wide",
)

st.title("Sports Predictions Dashboard")
st.markdown("Codzienne predykcje ML dla piłki nożnej i tenisa.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⚽ Piłka nożna")
    st.markdown("""
    - **22 ligi** europejskie
    - **5 targetów**: FTR, BTTS, O2.5, kartki, strzały celne
    - Model: LightGBM + Massey/Colley ratings
    """)
    st.page_link("pages/1_Pilka_Nozna.py", label="Przejdź do predykcji ➜")

with col2:
    st.markdown("### 🎾 Tenis")
    st.markdown("""
    - **ATP** — wszystkie turnieje
    - Predykcje: zwycięzca + total games
    - Model: LightGBM + Glicko + ELO + charting stats
    """)
    st.page_link("pages/2_Tenis.py", label="Przejdź do predykcji ➜")

st.markdown("---")
st.caption("Dane aktualizowane codziennie o 08:00 CET przez GitHub Actions.")
