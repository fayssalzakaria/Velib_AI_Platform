"""
mlops_panel.py — Cycle de vie du modèle de prévision : modèle en production, historique des
entraînements (déclencheur, raison, décision champion / challenger) et lancement d'un entraînement.

Le bouton n'entraîne rien dans l'app (publique, CPU limité, lecture seule sur S3) : il déclenche
le workflow GitHub Actions train.yml, qui a ses propres droits. Il est protégé par un mot de passe.
Secrets Streamlit : ADMIN_PASSWORD, GITHUB_TRAIN_TOKEN (jeton limité au repo, droit « Actions : écriture »),
GITHUB_REPO (par défaut fayssalzakaria/Velib_AI_Platform).
"""
import hmac
import os

import pandas as pd
import requests
import streamlit as st

from src.ml.storage import model_uri, read_json

MODEL_NAME = "forecast"
DEFAULT_REPO = "fayssalzakaria/Velib_AI_Platform"
RULES = (
    "Réentraînement automatique chaque jour (après le monitoring) si : aucun modèle en production · "
    "modèle de plus de 7 jours · dérive d'une feature (PSI > 0,25) · AUC « vide » à 1 h en baisse de plus "
    "de 0,05. Le nouveau modèle n'est publié que s'il bat les baselines **et** le modèle en production."
)


@st.cache_data(ttl=300, show_spinner=False)
def _registry() -> dict:
    return {
        "published": read_json(model_uri(MODEL_NAME, "latest.json")),
        "status": read_json(model_uri(MODEL_NAME, "status.json")) or {},
        "runs": (read_json(model_uri(MODEL_NAME, "history.json")) or {}).get("runs", []),
    }


def trigger_training(min_days: int, publish: bool, reason: str) -> tuple[bool, str]:
    """Déclenche le workflow train.yml par l'API GitHub. Renvoie (succès, message)."""
    token = os.environ.get("GITHUB_TRAIN_TOKEN", "")
    if not token:
        return False, "Jeton GitHub non configuré (secret GITHUB_TRAIN_TOKEN)."
    repo = os.environ.get("GITHUB_REPO", DEFAULT_REPO)
    response = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/train.yml/dispatches",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"},
        json={"ref": "main", "inputs": {"min_days": str(min_days), "publish": str(publish).lower(),
                                        "reason": reason or "lancé depuis l'application"}},
        timeout=15,
    )
    if response.status_code == 204:  # GitHub ne renvoie pas d'identifiant de run, seulement « accepté »
        return True, (f"Entraînement lancé sur GitHub Actions (quelques minutes). Suivi : "
                      f"https://github.com/{repo}/actions/workflows/train.yml")
    return False, f"GitHub a refusé la demande ({response.status_code}) : {response.text[:200]}"


def _production(published: dict | None, source: str | None) -> None:
    if not published:
        st.info("Aucun modèle en production pour l'instant.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Version en production", published["version"])
    c2.metric("Entraînée sur", f"{published['days_of_history']} jours")
    c3.metric("Questions servies", ", ".join(published["promoted"]) or "—")
    c4.metric("Prévisions servies", source or "—")
    if published.get("decision"):
        st.caption(f"Décision lors de sa publication : {published['decision']}.")


def _history(runs: list[dict]) -> None:
    if not runs:
        st.caption("Aucun entraînement enregistré pour l'instant.")
        return
    table = pd.DataFrame([{
        "Version": r["version"],
        "Déclencheur": r.get("trigger", ""),
        "Raison": r.get("reason", "") or "—",
        "Jours": r.get("days_of_history"),
        "AUC vide": (r.get("scores") or {}).get("empty"),
        "AUC pleine": (r.get("scores") or {}).get("full"),
        "MAE vélos": (r.get("scores") or {}).get("bikes"),
        "Publié": "✅" if r.get("published") else "—",
        "Décision": r.get("decision", ""),
    } for r in runs])
    st.dataframe(table, hide_index=True, use_container_width=True)


def _launch_form() -> None:
    """Formulaire protégé : un entraînement par session au plus, pour éviter les doubles clics."""
    if st.session_state.get("training_launched"):
        st.success(st.session_state["training_launched"])
        return
    with st.form("launch_training"):
        st.markdown("**Lancer un entraînement** (administrateur)")
        col_days, col_publish = st.columns(2)
        min_days = col_days.number_input("Historique minimal (jours)", min_value=1, max_value=90, value=7)
        publish = col_publish.toggle("Publier si le modèle gagne", value=True,
                                     help="Sinon : entraînement de test, métriques enregistrées sans servir le modèle.")
        reason = st.text_input("Raison", placeholder="ex. nouvelles features, dérive constatée…")
        password = st.text_input("Mot de passe administrateur", type="password")
        if not st.form_submit_button("Lancer sur GitHub Actions"):
            return
    expected = os.environ.get("ADMIN_PASSWORD", "")
    if not expected or not hmac.compare_digest(password, expected):
        st.error("Mot de passe incorrect ou non configuré.")
        return
    ok, message = trigger_training(int(min_days), publish, reason)
    if ok:
        st.session_state["training_launched"] = message
        st.success(message)
    else:
        st.error(message)


def render_mlops(source: str | None = None) -> None:
    """Panneau MLOps, affiché sous les prévisions (même quand aucun modèle n'est encore publié)."""
    with st.expander("⚙️ MLOps : cycle de vie du modèle"):
        registry = _registry()
        st.markdown("**Modèle en production (champion)**")
        _production(registry["published"], source)
        status = registry["status"]
        if status:
            st.caption(f"Dernier essai : {status.get('status')} — "
                       f"{status.get('decision') or status.get('reason') or ''} (déclencheur : {status.get('trigger', '—')}).")
        st.markdown("**Historique des entraînements**")
        _history(registry["runs"])
        st.caption(RULES)
        # Bouton seulement s'il est configuré : sinon le réentraînement reste entièrement automatique
        if os.environ.get("ADMIN_PASSWORD") and os.environ.get("GITHUB_TRAIN_TOKEN"):
            _launch_form()
