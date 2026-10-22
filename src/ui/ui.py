"""
ui.py — Mode Temps réel : barre latérale, snapshot de session, carte, graphiques et téléchargements.

Le snapshot est pris par session et partagé avec les onglets IA (render_snapshot_bar).
"""
from datetime import datetime

import pandas as pd
import plotly.express as px
import pydeck as pdk
import pytz
import streamlit as st

from config import MODE_HISTORY, MODE_REALTIME, PARIS_TIMEZONE
from src.data import lake
from src.data.filters import search_stations
from src.data.data_loader import fetch_api_snapshot
from src.data.history import get_lake_status


def render_sidebar():
    """Choix du mode, puis période (Historique) ou filtres (Temps réel) : renvoie (mode, filtres, jours)."""
    st.sidebar.title("Configuration")

    mode = st.sidebar.radio(
        "Mode",
        [MODE_REALTIME, MODE_HISTORY],
        index=0,
        captions=[
            "L'état du réseau à l'instant où vous chargez la page",
            "Statistiques sur tous les snapshots collectés",
        ],
    )
    st.sidebar.divider()

    if mode == MODE_HISTORY:
        from src.ui.history import render_period_selector
        return mode, None, render_period_selector()

    st.sidebar.subheader("Filtres")
    filtre_type = st.sidebar.selectbox(
        "Type de velo",
        ["Tous", "Mecaniques uniquement", "Electriques uniquement"],
    )
    filtre_etat = st.sidebar.selectbox(
        "Etat des stations",
        ["Tous", "Disponibles", "Vides", "Pleines"],
        index=0,
    )
    filtre_min_velos = st.sidebar.slider(
        "Minimum de velos disponibles",
        min_value=0,
        max_value=50,
        value=0,
    )
    return mode, (filtre_type, filtre_etat, filtre_min_velos), None


def get_session_snapshot(refresh: bool = False) -> pd.DataFrame | None:
    """
    Snapshot propre à la session : pris à l'arrivée sur la page, puis à chaque rafraîchissement.
    API injoignable : on garde le snapshot précédent, à défaut le dernier du lakehouse, sinon None.
    """
    if not refresh and "snapshot_df" in st.session_state:
        return st.session_state.snapshot_df

    try:
        with st.spinner("Snapshot de l'API Vélib' en cours..."):
            df = fetch_api_snapshot()
        st.session_state.snapshot_df = df
        st.session_state.snapshot_at = datetime.now(pytz.timezone(PARIS_TIMEZONE))
        st.session_state.snapshot_source = "API Open Data Vélib'"
        return df
    except Exception as e:
        error = e

    if "snapshot_df" in st.session_state:
        st.warning(f"API Vélib' momentanément injoignable ({type(error).__name__}) : snapshot précédent conservé.")
        return st.session_state.snapshot_df

    fallback = lake.latest_snapshot()
    if not fallback.empty:
        st.warning(
            f"API Vélib' injoignable ({type(error).__name__}) : affichage du dernier snapshot collecté "
            f"par le pipeline. Réessayez avec le bouton « Récupérer »."
        )
        st.session_state.snapshot_df = fallback
        st.session_state.snapshot_at = fallback["ingested_at"].iloc[0].to_pydatetime()
        st.session_state.snapshot_source = f"lakehouse, collecté par {fallback['ingested_by'].iloc[0]}"
        return fallback

    st.error(f"API Vélib' indisponible et aucun snapshot dans le lakehouse : {error}")
    return None


SNAPSHOT_BUTTON_PREFIX = "snapshot_refresh_"


def handle_snapshot_refresh() -> None:
    """
    À appeler en tout début de page : un clic sur n'importe quel bouton « Récupérer » reprend le snapshot
    avant que la page ne soit dessinée, pour que tous les bandeaux affichent le même.
    """
    if any(v for k, v in st.session_state.items() if str(k).startswith(SNAPSHOT_BUTTON_PREFIX)):
        get_session_snapshot(refresh=True)


def _snapshot_age() -> str:
    seconds = (datetime.now(pytz.timezone(PARIS_TIMEZONE)) - st.session_state.snapshot_at).total_seconds()
    if seconds < 60:
        return "à l'instant"
    if seconds < 3600:
        return f"il y a {int(seconds // 60)} min"
    return f"il y a {int(seconds // 3600)} h {int(seconds % 3600 // 60):02d}"


def render_snapshot_bar(key: str, label: str = "Données temps réel") -> pd.DataFrame | None:
    """
    Bandeau « heure et âge du snapshot » + bouton pour en récupérer un nouveau.
    `key` distingue les boutons quand le bandeau apparaît deux fois sur la page.
    """
    col_info, col_button = st.columns([3, 1])
    with col_button:
        # Pas de traitement ici : le clic est lu en début de page par handle_snapshot_refresh
        st.button(
            "📡 Récupérer un nouveau snapshot",
            key=f"{SNAPSHOT_BUTTON_PREFIX}{key}",
            type="primary",
            use_container_width=True,
        )
    df = get_session_snapshot()
    if df is not None:
        col_info.info(
            f"**{label}** : snapshot du {st.session_state.snapshot_at:%d/%m/%Y à %H:%M:%S} "
            f"({_snapshot_age()}) — {len(df)} stations, source : {st.session_state.snapshot_source}"
        )
    return df


def render_metrics(df_filtered):
    """Indicateurs clés du snapshot filtré."""
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Stations", len(df_filtered))
    col2.metric("Velos dispo", f"{int(df_filtered['numbikesavailable'].sum()):,}")
    col3.metric("Dont electriques", f"{int(df_filtered['ebike'].sum()):,}")
    col4.metric("Bornes dispo", f"{int(df_filtered['numdocksavailable'].sum()):,}")
    col5.metric("Vides / pleines", f"{int(df_filtered['is_empty'].sum())} / {int(df_filtered['is_full'].sum())}")
    col6.metric("Remplissage", f"{df_filtered['bike_ratio'].mean():.1%}")
    st.divider()

def render_map(df_filtered):
    st.subheader("Carte des stations")

    if "lat" in df_filtered.columns and df_filtered["lat"].notna().any():
        map_df = df_filtered[
            ["name", "lat", "lon", "numbikesavailable", "ebike", "numdocksavailable"]
        ].dropna(subset=["lat", "lon"]).copy()
        map_df["lat"] = map_df["lat"].astype(float)
        map_df["lon"] = map_df["lon"].astype(float)
        # Rouge : aucun vélo ; vert : au moins un vélo
        map_df["color"] = [
            [163, 45, 45, 210] if bikes == 0 else [29, 158, 117, 190]
            for bikes in map_df["numbikesavailable"]
        ]

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=45,
            radius_min_pixels=3,
            radius_max_pixels=12,
            pickable=True,
        )
        tooltip = {
            "html": "<b>{name}</b><br/>Vélos : {numbikesavailable} (dont {ebike} électriques)"
                    "<br/>Bornes libres : {numdocksavailable}",
            "style": {"fontSize": "13px"},
        }
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(latitude=48.8566, longitude=2.3522, zoom=11.5),
                tooltip=tooltip,
                map_provider="carto",
                map_style="light",
            ),
            use_container_width=True,
        )
        n_empty = int((map_df["numbikesavailable"] == 0).sum())
        st.caption(f"🟢 vélos disponibles · 🔴 aucun vélo ({n_empty} stations) — survolez une station pour le détail")
    else:
        st.info("Coordonnees non disponibles pour ce snapshot.")

    st.divider()

def render_charts(df_filtered):
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top 10 mieux fournies")
        top10 = df_filtered.nlargest(10, "numbikesavailable")[
            ["name", "numbikesavailable", "ebike"]
        ]
        fig = px.bar(
            top10,
            x="numbikesavailable",
            y="name",
            orientation="h",
            color="ebike",
            color_continuous_scale="teal",
            labels={
                "numbikesavailable": "Velos",
                "name": "Station",
                "ebike": "Electriques",
            },
        )
        fig.update_layout(yaxis={"autorange": "reversed"}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Top 10 plus vides")
        empty10 = df_filtered.nsmallest(10, "numbikesavailable")[
            ["name", "numbikesavailable"]
        ]
        fig2 = px.bar(
            empty10,
            x="numbikesavailable",
            y="name",
            orientation="h",
            color="numbikesavailable",
            color_continuous_scale="reds",
            labels={"numbikesavailable": "Velos", "name": "Station"},
        )
        fig2.update_layout(yaxis={"autorange": "reversed"}, height=400)
        st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Types de velos")
        bike_data = pd.DataFrame(
            {
                "Type": ["Mecaniques", "Electriques"],
                "Nombre": [
                    int(df_filtered["mechanical"].sum()),
                    int(df_filtered["ebike"].sum()),
                ],
            }
        )
        fig3 = px.pie(
            bike_data,
            values="Nombre",
            names="Type",
            color_discrete_sequence=["#185FA5", "#1D9E75"],
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.subheader("Etat des stations")
        status_data = pd.DataFrame(
            {
                "Etat": ["Vides", "Pleines", "Partielles"],
                "Nombre": [
                    int(df_filtered["is_empty"].sum()),
                    int(df_filtered["is_full"].sum()),
                    len(df_filtered)
                    - int(df_filtered["is_empty"].sum())
                    - int(df_filtered["is_full"].sum()),
                ],
            }
        )
        fig4 = px.pie(
            status_data,
            values="Nombre",
            names="Etat",
            color_discrete_sequence=["#A32D2D", "#1D9E75", "#EF9F27"],
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

def render_downloads(df_filtered):
    st.subheader("Telechargements")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        if st.button("Generer rapport PDF", use_container_width=True):
            with st.spinner("Generation du rapport..."):
                from src.reports.report_generator import generate_pdf_report
                pdf_bytes = generate_pdf_report(df_filtered)
            st.download_button(
                "Telecharger le PDF",
                data=pdf_bytes,
                file_name=f"velib_rapport_{datetime.now(pytz.timezone('Europe/Paris')).strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with col_dl2:
        csv_bytes = df_filtered.to_csv(index=False, sep=";").encode("utf-8")
        st.download_button(
            "Telecharger CSV",
            data=csv_bytes,
            file_name="velib_latest.csv",
            mime="text/csv",
            use_container_width=True,
        )

def render_station_detail(df_filtered):
    st.subheader("Recherche de station")

    search = st.text_input(
        "Nom de la station",
        placeholder="Ex: Bastille, Nation, Mairie de Clichy...",
        key="station_search",
    )

    if not search:
        return

    results = search_stations(df_filtered, search)

    if results.empty:
        st.warning(f"Aucune station trouvee pour '{search}'")
        return

    # Plusieurs stations : on montre combien et on laisse choisir (rien de présélectionné)
    station_names = sorted(results["name"].unique().tolist())
    if len(station_names) > 1:
        bikes = results.groupby("name")["numbikesavailable"].first()
        station_name = st.selectbox(
            f"{len(station_names)} stations trouvées pour « {search} »",
            station_names,
            index=None,
            placeholder="Choisir une station…",
            format_func=lambda name: f"{name} — {int(bikes[name])} vélos",
        )
        if station_name is None:
            st.divider()
            return
    else:
        station_name = station_names[0]

    row = results[results["name"] == station_name].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Velos dispo", int(row["numbikesavailable"]))
    c2.metric("Dont electriques", int(row["ebike"]))
    c3.metric("Bornes dispo", int(row["numdocksavailable"]))
    c4.metric("Remplissage", f"{row['bike_ratio']:.0%}")

    if row["is_empty"]:
        st.error("Station vide en ce moment")
    elif row["is_full"]:
        st.warning("Station pleine en ce moment")
    else:
        st.success("Station disponible")

    st.caption("Évolution de cette station dans le temps : mode 📚 Historique.")
    st.divider()


def render_lake_sidebar():
    """État du lakehouse dans la barre latérale : dit si le mode Historique a des données."""
    st.sidebar.divider()
    st.sidebar.subheader("Lakehouse")

    if not lake.is_configured():
        st.sidebar.info("Non configuré — mode temps réel seul.")
        return

    status = get_lake_status()
    if not status.get("available"):
        st.sidebar.warning("Inaccessible ou vide — mode temps réel seul.")
        return

    st.sidebar.success(
        f"{status['snapshots']} snapshots — dernier à "
        f"{status['last_snapshot']:%H:%M} ({status['last_ingested_by']})"
    )
