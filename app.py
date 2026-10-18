import streamlit as st

from config import MODE_HISTORY
from src.ai.embeddings import warm_up as warm_up_embeddings
from src.data.filters import apply_filters
from src.ui.ai_panel import render_ai_tabs
from src.ui.forecast_panel import render_forecast
from src.ui.history import render_history_page
from src.ui.ui import (
    handle_snapshot_refresh,
    render_charts,
    render_downloads,
    render_lake_sidebar,
    render_map,
    render_metrics,
    render_sidebar,
    render_snapshot_bar,
    render_station_detail,
)

st.set_page_config(
    page_title="Vélib' Dashboard",
    page_icon="",
    layout="wide",
)

st.title("Vélib' Dashboard Paris")

# Modèle d'embeddings du RAG chargé en arrière-plan pendant que l'utilisateur regarde la page
warm_up_embeddings()

handle_snapshot_refresh()
mode, filters, days = render_sidebar()
render_lake_sidebar()

if mode == MODE_HISTORY:
    st.header("📚 Historique — tous les snapshots du lakehouse")
    render_history_page(days)
    render_ai_tabs()
    st.stop()

st.header("⏱️ Temps réel — snapshot de l'API")
df = render_snapshot_bar(key="top")
if df is None or df.empty:
    st.stop()

df_filtered = apply_filters(df, *filters)
st.sidebar.info(f"{len(df_filtered)} stations après filtres")

render_metrics(df_filtered)
render_station_detail(df_filtered)
render_map(df_filtered)
render_forecast()
render_charts(df_filtered)
render_ai_tabs()
render_downloads(df_filtered)
