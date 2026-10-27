"""
forecast_panel.py — Prévisions LightGBM de 1 à 6 heures : risque de station vide ou pleine, vélos attendus.

Affiche aussi la fiche des modèles publiés, leur suivi en production (lu dans le gold) et le
panneau MLOps (cycle de vie, historique des entraînements, lancement d'un entraînement).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src.data import lake
from src.data.filters import search_stations
from src.ml import forecast
from src.ui.mlops_panel import render_mlops

QUESTIONS = {
    "vide": {"risk": "risk_empty", "now": "is_empty_now", "label": "Risque d'être vide",
             "action": "À réapprovisionner : stations encore fournies mais à fort risque d'être vides"},
    "pleine": {"risk": "risk_full", "now": "is_full_now", "label": "Risque d'être pleine",
               "action": "À délester : stations avec des bornes libres mais à fort risque d'être pleines"},
}


@st.cache_data(ttl=600, show_spinner=False)
def _predictions():
    return forecast.predict()


@st.cache_data(ttl=600, show_spinner=False)
def _status():
    return forecast.model_status()


@st.cache_data(ttl=600, show_spinner=False)
def _monitoring(table: str) -> pd.DataFrame:
    """Table de suivi des modèles écrite dans le gold par le job quotidien."""
    return lake.query(f"SELECT * FROM read_parquet('{lake.lake_root()}/gold/{table}.parquet')")


def _risk_map(frame: pd.DataFrame, risk: str, label: str) -> None:
    """Carte des stations colorées du vert (risque nul) au rouge (risque certain)."""
    map_df = frame.dropna(subset=["lat", "lon", risk]).copy()
    map_df["color"] = [[int(29 + 134 * r), int(158 - 113 * r), int(117 - 72 * r), 200] for r in map_df[risk]]
    map_df["risk_txt"] = (map_df[risk] * 100).round(0).astype(int).astype(str) + " %"
    map_df["bikes_txt"] = map_df.apply(
        lambda r: "—" if pd.isna(r["bikes_q50"]) else f"{r['bikes_q50']:.0f} (entre {r['bikes_q10']:.0f} et {r['bikes_q90']:.0f})",
        axis=1,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[pdk.Layer(
                "ScatterplotLayer",
                data=map_df[["name", "lat", "lon", "color", "risk_txt", "bikes_txt"]],
                get_position="[lon, lat]", get_fill_color="color", get_radius=45,
                radius_min_pixels=3, radius_max_pixels=12, pickable=True,
            )],
            initial_view_state=pdk.ViewState(latitude=48.8566, longitude=2.3522, zoom=11.5),
            tooltip={"html": f"<b>{{name}}</b><br/>{label} : {{risk_txt}}<br/>Vélos attendus : {{bikes_txt}}",
                     "style": {"fontSize": "13px"}},
            map_provider="carto", map_style="light",
        ),
        use_container_width=True,
    )
    st.caption(f"🟢 faible → 🔴 élevé — {label.lower()}")


def _station_outlook(predictions: pd.DataFrame) -> None:
    """Pour une station : vélos attendus heure par heure, avec l'intervalle de 80 %."""
    search = st.text_input("Prévision pour une station", placeholder="Ex : Nation, Bastille…", key="forecast_search")
    if not search:
        return
    matches = search_stations(predictions, search)
    if matches.empty:
        st.warning(f"Aucune station trouvée pour '{search}'")
        return
    names = sorted(matches["name"].unique())
    name = names[0] if len(names) == 1 else st.selectbox(
        f"{len(names)} stations trouvées", names, index=None, placeholder="Choisir une station…", key="forecast_station")
    if name is None:
        return

    station = matches[matches["name"] == name].sort_values("horizon")
    hours = station["ts_target"].dt.tz_convert("Europe/Paris").dt.strftime("%H:%M")
    fig = go.Figure([
        go.Scatter(x=hours, y=station["bikes_q90"], line=dict(width=0), showlegend=False, hoverinfo="skip"),
        go.Scatter(x=hours, y=station["bikes_q10"], fill="tonexty", line=dict(width=0),
                   fillcolor="rgba(29,158,117,0.2)", name="Intervalle 80 %"),
        go.Scatter(x=hours, y=station["bikes_q50"], mode="lines+markers", line=dict(color="#1D9E75"),
                   name="Vélos attendus"),
    ])
    fig.update_layout(title=f"{name} : vélos attendus", height=300, margin=dict(t=40, b=10),
                      yaxis_title="Vélos", xaxis_title="Heure")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame({
        "Heure": hours.values,
        "Vélos attendus": station.apply(lambda r: f"{r['bikes_q50']:.0f} (entre {r['bikes_q10']:.0f} et {r['bikes_q90']:.0f})"
                                        if pd.notna(r["bikes_q50"]) else "—", axis=1).values,
        "Risque vide": (station["risk_empty"] * 100).round(0).map(lambda v: "—" if pd.isna(v) else f"{v:.0f} %").values,
        "Risque pleine": (station["risk_full"] * 100).round(0).map(lambda v: "—" if pd.isna(v) else f"{v:.0f} %").values,
    }), hide_index=True, use_container_width=True)


def _model_card(model: dict) -> None:
    """Performances par horizon face aux baselines, pour chaque question publiée."""
    st.caption(
        f"Version {model['version']} — entraînée sur {model['days_of_history']} jours, testée sur les jours "
        f"suivants (jamais vus à l'entraînement). Publiées : {', '.join(model['promoted'])}."
    )
    horizons = [f"{h}h" for h in model["horizons"]]
    for family, title, metric, better in (("empty", "Vide — AUC", "auc", "↑"), ("full", "Pleine — AUC", "auc", "↑"),
                                          ("bikes", "Nombre de vélos — erreur moyenne (vélos)", "mae", "↓")):
        evaluation = model["families"][family]["evaluation"]
        rows = [{"Méthode": label, **{h: evaluation[key][h][metric] for h in horizons}}
                for key, label in (("model", "LightGBM"), ("persistence", "Persistance"), ("profile", "Habitudes"))]
        st.markdown(f"**{title} {better}** par horizon")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    coverage = model["families"]["bikes"]["evaluation"]["model"]
    st.caption("Intervalle de 80 % : part des valeurs réelles dedans, par horizon : "
               + " · ".join(f"{h} {coverage[h]['coverage_80']:.0%}" for h in horizons))

    importance = pd.DataFrame(list(model["feature_importance_gain"].items()), columns=["Variable", "Importance"])
    fig = px.bar(importance, x="Importance", y="Variable", orientation="h")
    fig.update_layout(title="Ce qui pèse le plus (modèle « vide »)", height=420,
                      yaxis={"autorange": "reversed"}, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _one_point_per_day(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Le suivi est calculé par version : un jour de changement de modèle compte deux lignes par horizon.
    Pour les courbes, moyenne pondérée par le nombre de prévisions, un point par jour et par horizon.
    """
    def mean(group, column):
        known = group[column].notna()
        return np.average(group.loc[known, column], weights=group.loc[known, "predictions"]) if known.any() else np.nan

    def weighted(group):
        return pd.Series({c: mean(group, c) for c in ("auc_empty", "mae_bikes")} | {"predictions": group["predictions"].sum()})
    return daily.groupby(["date", "horizon"]).apply(weighted, include_groups=False).reset_index()


def _production_performance(model: dict) -> None:
    """Les modèles face à la réalité : prévisions enregistrées à chaque collecte, comparées chaque jour aux observations."""
    daily = _monitoring("model_monitoring_daily")
    if daily.empty:
        st.info("Pas encore de suivi en production : les prévisions sont enregistrées à chaque collecte "
                "et comparées à la réalité chaque jour.")
        return

    daily = _one_point_per_day(daily).assign(horizon=lambda d: d["horizon"].astype(int).astype(str) + " h")
    col_auc, col_mae = st.columns(2)
    fig = px.line(daily, x="date", y="auc_empty", color="horizon", markers=True,
                  labels={"date": "Jour", "auc_empty": "AUC (vide)", "horizon": "Horizon"})
    fig.update_layout(title="Vide : qualité jour par jour", height=300, margin=dict(t=40, b=10))
    col_auc.plotly_chart(fig, use_container_width=True)
    if daily["mae_bikes"].isna().all():
        col_mae.markdown("**Vélos : erreur jour par jour**")
        col_mae.info("Pas de suivi : le modèle vélos n'est pas en production (il ne bat pas encore les baselines).")
    else:
        fig = px.line(daily, x="date", y="mae_bikes", color="horizon", markers=True,
                      labels={"date": "Jour", "mae_bikes": "Erreur moyenne (vélos)", "horizon": "Horizon"})
        fig.update_layout(title="Vélos : erreur jour par jour", height=300, margin=dict(t=40, b=10))
        col_mae.plotly_chart(fig, use_container_width=True)

    col_calib, col_drift = st.columns(2)
    calibration = _monitoring("model_calibration")
    if not calibration.empty:
        fig = px.scatter(calibration, x="mean_predicted", y="observed_rate", size="predictions", color="question",
                         labels={"mean_predicted": "Risque annoncé", "observed_rate": "Fréquence réelle"})
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        fig.update_layout(title="Calibration (7 derniers jours)", height=320, margin=dict(t=40, b=10))
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        col_calib.plotly_chart(fig, use_container_width=True)
        col_calib.caption("Sur la diagonale : quand le modèle annonce 30 %, c'est vrai 30 % du temps.")

    drift = _monitoring("model_drift")
    if not drift.empty:
        col_drift.markdown("**Dérive des données (PSI, dernières 24 h vs entraînement)**")
        icons = {"stable": "🟢", "à surveiller": "🟠", "dérive": "🔴"}
        col_drift.dataframe(
            drift.assign(État=drift["status"].map(lambda v: f"{icons.get(v, '')} {v}"))[["feature", "psi", "État"]]
            .rename(columns={"feature": "Variable", "psi": "PSI"}),
            hide_index=True, use_container_width=True,
        )
        col_drift.caption("PSI < 0,1 : stable · 0,1 à 0,25 : à surveiller · > 0,25 : dérive, réentraînement conseillé.")


def render_forecast() -> None:
    """
    Section Prévisions. Un modèle n'est publié que s'il bat les baselines et le modèle en production,
    d'où les messages qui expliquent pourquoi rien n'est affiché.
    """
    st.subheader("🔮 Prévisions de 1 à 6 heures")
    if not lake.is_configured():
        st.info("Prévision indisponible : lakehouse non configuré.")
        st.divider()
        return

    predictions, info = _predictions()
    if predictions is None:
        status = _status()["status"]
        if status.get("status") == "skipped":
            st.info(f"Modèles pas encore entraînés : {status.get('reason')}.")
        elif status.get("status") == "test":
            st.info("Dernier entraînement : un test, ses modèles ne sont pas publiés dans l'application.")
        elif status.get("status") == "trained" and not status.get("promoted"):
            st.info(f"Dernier entraînement non publié : {status.get('decision', 'il ne bat pas les baselines')}.")
        else:
            st.info(info.get("reason", "Prévision indisponible."))
        render_mlops()
        st.divider()
        return

    col_h, col_q = st.columns([2, 1])
    horizon = col_h.select_slider("Horizon", options=info["horizons"], value=1, format_func=lambda h: f"dans {h} h")
    available = [q for q, spec in QUESTIONS.items() if predictions[spec["risk"]].notna().any()]
    question = col_q.radio("Question", available, horizontal=True, format_func=lambda q: f"station {q}")
    spec = QUESTIONS[question]

    at_h = predictions[predictions["horizon"] == horizon]
    target_time = at_h["ts_target"].iloc[0].tz_convert("Europe/Paris")
    st.caption(f"Prévision pour **{target_time:%H:%M}**, à partir du snapshot de {info['snapshot_at']:%d/%m %H:%M} "
               f"(collecte horaire du lakehouse) · prévisions {info.get('source', 'à la demande')}.")

    # Une station déjà vide (ou pleine) n'est pas une alerte : on classe celles qui risquent de le devenir
    candidates = at_h[at_h[spec["now"]] == 0]
    c1, c2, c3 = st.columns(3)
    # Les risques sont faibles (une station fournie se vide rarement en quelques heures) : un seuil à
    # 50 % donnerait toujours 0. La somme des probabilités est le nombre de stations attendu.
    c1.metric(f"Stations qui devraient devenir {question}s", f"≈ {candidates[spec['risk']].sum():.0f}",
              help="Somme des probabilités des stations qui ne le sont pas encore.")
    c2.metric(f"Déjà {question}s au dernier snapshot", int(at_h[spec["now"]].sum()))
    c3.metric("Vélos attendus sur le réseau",
              "—" if at_h["bikes_q50"].isna().all() else f"{at_h['bikes_q50'].sum():,.0f}")

    st.markdown(f"**{spec['action']}**")
    top = candidates.nlargest(15, spec["risk"]).assign(
        Risque=lambda d: (d[spec["risk"]] * 100).round(0).astype(int).astype(str) + " %",
        Vélos=lambda d: d["bikes"].astype(int),
        Attendus=lambda d: d.apply(lambda r: "—" if pd.isna(r["bikes_q50"])
                                   else f"{r['bikes_q50']:.0f} ({r['bikes_q10']:.0f}–{r['bikes_q90']:.0f})", axis=1),
    )[["name", "Vélos", "Attendus", "Risque"]].rename(columns={"name": "Station", "Vélos": "Vélos maintenant"})
    st.dataframe(top, hide_index=True, use_container_width=True)

    _risk_map(at_h, spec["risk"], spec["label"])
    _station_outlook(predictions)

    with st.expander("Fiche des modèles : performances par horizon face aux baselines"):
        _model_card(info["model"])
    with st.expander("📈 Performance en production (monitoring)"):
        _production_performance(info["model"])
    render_mlops(info.get("source"))
    st.divider()
