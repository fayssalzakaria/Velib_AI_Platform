"""
history.py — Mode Historique : statistiques sur les snapshots collectés dans le lakehouse.

Tout vient du silver : une ligne par snapshot pour les courbes, une ligne par station
pour les classements et la carte. Lectures mises en cache 10 minutes (collecte horaire).
"""
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from src.data import lake
from src.data.filters import search_stations

PERIODS = {"24 heures": 1, "7 jours": 7, "30 jours": 30, "Tout l'historique": None}


@st.cache_data(ttl=600, show_spinner=False)
def _network_history(days):
    return lake.network_history(days)


@st.cache_data(ttl=600, show_spinner=False)
def _station_stats(days):
    return lake.station_stats(days)


@st.cache_data(ttl=600, show_spinner=False)
def _station_history(name, hours):
    return lake.station_history(name, hours)


def render_period_selector() -> int | None:
    label = st.sidebar.radio("Période analysée", list(PERIODS), index=1)
    return PERIODS[label]


def _render_overview(history: pd.DataFrame) -> None:
    first, last = history["ingested_at"].min(), history["ingested_at"].max()
    peak = history.loc[history["empty_stations"].idxmax()]
    collectors = history["ingested_by"].value_counts()

    st.caption(
        f"{len(history)} snapshots du {first:%d/%m/%Y %H:%M} au {last:%d/%m/%Y %H:%M} — collectés par : "
        + ", ".join(f"{who} ({n})" for who, n in collectors.items())
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Snapshots", len(history))
    c2.metric("Vélos dispo (moyenne)", f"{history['total_bikes'].mean():,.0f}")
    c3.metric("Stations vides (moyenne)", f"{history['empty_stations'].mean():.0f}")
    c4.metric(
        "Pic de stations vides",
        int(peak["empty_stations"]),
        help=f"Snapshot du {peak['ingested_at']:%d/%m %H:%M}",
    )
    c5.metric("Remplissage moyen", f"{history['avg_bike_ratio'].mean():.1%}")
    st.divider()


def _render_trends(history: pd.DataFrame) -> None:
    st.subheader("Évolution du réseau")
    st.caption(f"1 point = 1 snapshot ({len(history)} points)")

    col_bikes, col_empty = st.columns(2)
    for col, column, title, color in [
        (col_bikes, "total_bikes", "Vélos disponibles", "#1D9E75"),
        (col_empty, "empty_stations", "Stations vides", "#A32D2D"),
    ]:
        fig = px.line(
            history,
            x="ingested_at",
            y=column,
            markers=True,
            labels={"ingested_at": "Snapshot", column: title},
            color_discrete_sequence=[color],
            hover_data={"ingested_by": True},
        )
        fig.update_layout(title=title, height=280, margin=dict(t=40, b=10))
        col.plotly_chart(fig, use_container_width=True)

    # Heures critiques : stations vides en moyenne, par heure de la journée, sur toute la période
    by_hour = (
        history.assign(heure=history["ingested_at"].dt.hour)
        .groupby("heure", as_index=False)
        .agg(stations_vides=("empty_stations", "mean"), snapshots=("snapshot_id", "count"))
    )
    fig = px.bar(
        by_hour,
        x="heure",
        y="stations_vides",
        hover_data={"snapshots": True},
        labels={"heure": "Heure de la journée", "stations_vides": "Stations vides (moyenne)"},
        color="stations_vides",
        color_continuous_scale="reds",
    )
    fig.update_layout(title="Heures critiques", height=280, margin=dict(t=40, b=10), coloraxis_showscale=False)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()


def _render_rankings(stats: pd.DataFrame, n_snapshots: int) -> None:
    st.subheader("Stations problématiques")
    # Stations vues dans moins de la moitié des snapshots (nouvelles, fermées) : pourcentages trop fragiles
    reliable = stats[stats["n_snapshots"] >= max(1, n_snapshots // 2)]
    st.caption(f"% des snapshots où la station était vide ou pleine — {len(reliable)} stations suivies")

    col_empty, col_full = st.columns(2)
    for col, column, title, scale in [
        (col_empty, "pct_empty", "Le plus souvent vides", "reds"),
        (col_full, "pct_full", "Le plus souvent pleines", "oranges"),
    ]:
        top = reliable.nlargest(10, column)
        fig = px.bar(
            top,
            x=column,
            y="name",
            orientation="h",
            color=column,
            color_continuous_scale=scale,
            hover_data={"n_snapshots": True, "avg_bikes": ":.1f"},
            labels={column: "% des snapshots", "name": "Station"},
        )
        fig.update_layout(title=title, height=380, yaxis={"autorange": "reversed"}, coloraxis_showscale=False)
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        col.plotly_chart(fig, use_container_width=True)
    st.divider()


def _render_history_map(stats: pd.DataFrame) -> None:
    st.subheader("Carte : fréquence des stations vides")
    map_df = stats.dropna(subset=["lat", "lon"]).copy()
    if map_df.empty:
        st.info("Coordonnées non disponibles pour cette période.")
        return

    # Vert (jamais vide) -> rouge (vide la moitié du temps ou plus)
    level = (map_df["pct_empty"] * 2).clip(0, 1)
    map_df["color"] = [
        [int(29 + (163 - 29) * x), int(158 + (45 - 158) * x), int(117 + (45 - 117) * x), 200] for x in level
    ]
    map_df["pct_empty_txt"] = (map_df["pct_empty"] * 100).round(0).astype(int).astype(str) + " %"
    map_df["avg_bikes_txt"] = map_df["avg_bikes"].round(1).astype(str)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df[["name", "lat", "lon", "color", "pct_empty_txt", "avg_bikes_txt", "n_snapshots"]],
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=45,
        radius_min_pixels=3,
        radius_max_pixels=12,
        pickable=True,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=48.8566, longitude=2.3522, zoom=11.5),
            tooltip={
                "html": "<b>{name}</b><br/>Vide {pct_empty_txt} du temps"
                        "<br/>{avg_bikes_txt} vélos en moyenne ({n_snapshots} snapshots)",
                "style": {"fontSize": "13px"},
            },
            map_provider="carto",
            map_style="light",
        ),
        use_container_width=True,
    )
    st.caption("🟢 jamais vide → 🔴 vide la moitié du temps ou plus — survolez une station pour le détail")
    st.divider()


def _render_station_history(stats: pd.DataFrame, days: int | None) -> None:
    st.subheader("Historique d'une station")
    search = st.text_input("Nom de la station", placeholder="Ex: Bastille, Nation, Clichy...", key="history_search")
    if not search:
        return

    results = search_stations(stats, search)
    if results.empty:
        st.warning(f"Aucune station trouvée pour '{search}'")
        return

    names = sorted(results["name"].unique().tolist())
    if len(names) > 1:
        station_name = st.selectbox(
            f"{len(names)} stations trouvées pour « {search} »",
            names,
            index=None,
            placeholder="Choisir une station…",
            key="history_station",
        )
        if station_name is None:
            return
    else:
        station_name = names[0]

    row = results[results["name"] == station_name].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Snapshots observés", int(row["n_snapshots"]))
    c2.metric("Vide", f"{row['pct_empty']:.0%} du temps")
    c3.metric("Pleine", f"{row['pct_full']:.0%} du temps")
    c4.metric("Vélos en moyenne", f"{row['avg_bikes']:.1f}")

    hours = 24 * days if days else 24 * 3650  # « Tout l'historique » : 10 ans suffisent
    curve = _station_history(station_name, hours)
    if curve.empty:
        return
    fig = px.line(
        curve,
        x="run_at",
        y=["numbikesavailable", "ebike", "mechanical"],
        markers=True,
        labels={"run_at": "Snapshot", "value": "Nombre", "variable": "Type"},
        color_discrete_map={"numbikesavailable": "#1D9E75", "ebike": "#185FA5", "mechanical": "#EF9F27"},
    )
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_history_page(days: int | None) -> None:
    """Page du mode Historique ; days=None couvre tout l'historique."""
    if not lake.is_configured():
        st.info("Lakehouse non configuré (LAKE_ROOT) : le mode Historique n'est pas disponible.")
        return

    history = _network_history(days)
    if history.empty:
        st.warning("Aucun snapshot dans le lakehouse pour cette période (ou lakehouse inaccessible).")
        return

    stats = _station_stats(days)

    if st.button("🔄 Rafraîchir l'historique"):
        st.cache_data.clear()
        st.rerun()

    _render_overview(history)
    _render_trends(history)
    if not stats.empty:
        _render_rankings(stats, len(history))
        _render_history_map(stats)
        _render_station_history(stats, days)

        st.download_button(
            "Télécharger les statistiques par station (CSV)",
            data=stats.sort_values("pct_empty", ascending=False).to_csv(index=False, sep=";").encode("utf-8"),
            file_name="velib_stats_stations.csv",
            mime="text/csv",
        )
