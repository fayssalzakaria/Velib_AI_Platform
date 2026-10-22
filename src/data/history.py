"""
history.py — État du lakehouse affiché dans la barre latérale.
"""
import streamlit as st

from src.data import lake


@st.cache_data(ttl=300, show_spinner=False)
def get_lake_status() -> dict:
    """Résumé du silver, mis en cache 5 minutes : la barre latérale est redessinée à chaque interaction."""
    return lake.lake_status()
