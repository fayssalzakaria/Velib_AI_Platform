"""
filters.py — Recherche de stations par nom et filtres d'affichage du mode Temps réel.

Partagé par l'interface et l'IA, pour qu'une station soit retrouvée de la même façon partout.
"""
import math
import re
import unicodedata

import pandas as pd


def normalize_name(text: str) -> str:
    """Majuscules, sans accents ni ponctuation : « Châtelet - Rivoli » -> « CHATELET RIVOLI »."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()


def search_stations(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Stations dont le nom contient tous les mots de la recherche, sans tenir compte des accents ni de l'ordre."""
    words = normalize_name(query).split()
    if not words or df.empty:
        return df.iloc[0:0]
    names = df["name"].map(normalize_name)
    mask = pd.Series(True, index=df.index)
    for word in words:
        mask &= names.str.contains(word, regex=False)
    return df[mask]


# Mots de la question qui ne désignent jamais une station
_NOT_A_STATION = {
    "STATION", "STATIONS", "VELO", "VELOS", "VELIB", "BORNE", "BORNES", "COMBIEN", "ETAT", "QUEL", "QUELLE",
    "EST", "SERA", "DANS", "HEURE", "HEURES", "LIBRE", "LIBRES", "DISPO", "DISPONIBLE", "DISPONIBLES",
    "VIDE", "VIDES", "PLEINE", "PLEINES", "ELECTRIQUE", "ELECTRIQUES", "MECANIQUE", "MECANIQUES",
    "MAINTENANT", "ACTUELLEMENT", "AUJOURD", "HUI", "MATIN", "MIDI", "SOIR", "NUIT", "SEMAINE", "WEEK", "END",
    "SOUVENT", "HABITUDE", "PASSE", "QUOI", "POUR", "AVEC", "PARIS", "RESEAU",
}
# Un mot présent dans plus de stations que ça (« PLACE », « SAINT », « DES ») ne suffit pas à en désigner une
MAX_STATIONS_PER_WORD = 40


def stations_in_question(question: str, names) -> list[str]:
    """
    Stations citées dans une question en langage naturel, la plus probable en premier.
    Même normalisation que la recherche (accents, casse, ponctuation). Chaque mot de la question
    présent dans un nom rapporte d'autant plus qu'il est rare dans le réseau : « CLICHY » pèse plus
    que « RUE ». Renvoie toutes les stations ex æquo : « Bastille » désigne les 5 stations Bastille.
    """
    words = {w for w in normalize_name(question).split() if len(w) >= 3 and w not in _NOT_A_STATION}
    names = [n for n in dict.fromkeys(names) if n]
    if not words or not names:
        return []

    tokens = {name: set(normalize_name(name).split()) for name in names}
    frequency = {w: sum(w in t for t in tokens.values()) for w in words}
    rarity = {w: math.log(len(names) / n) for w, n in frequency.items() if n}
    distinctive = {w for w, n in frequency.items() if 0 < n <= MAX_STATIONS_PER_WORD}
    if not distinctive:
        return []

    scores = {name: sum(rarity[w] for w in tokens[name] & rarity.keys())
              for name in names if tokens[name] & distinctive}
    best = max(scores.values())
    # à score égal, le nom le plus court (le moins de mots non demandés) d'abord
    return sorted((n for n, sc in scores.items() if sc >= best - 1e-9), key=lambda n: len(tokens[n]))


def apply_filters(
    df: pd.DataFrame,
    filtre_type: str,
    filtre_etat: str,
    filtre_min_velos: int,
) -> pd.DataFrame:
    """Filtres de la barre latérale (type de vélo, état, minimum de vélos) ; les libellés sont ceux du selectbox."""
    df_filtered = df.copy()

    if filtre_type == "Mecaniques uniquement":
        df_filtered = df_filtered[df_filtered["mechanical"] > 0]
    elif filtre_type == "Electriques uniquement":
        df_filtered = df_filtered[df_filtered["ebike"] > 0]

    if filtre_etat == "Vides":
        df_filtered = df_filtered[df_filtered["is_empty"]]
    elif filtre_etat == "Pleines":
        df_filtered = df_filtered[df_filtered["is_full"]]
    elif filtre_etat == "Disponibles":
        df_filtered = df_filtered[
            (~df_filtered["is_empty"]) & (~df_filtered["is_full"])
        ]

    df_filtered = df_filtered[df_filtered["numbikesavailable"] >= filtre_min_velos]

    # Si le DataFrame couvre plusieurs snapshots, on ne garde que la ligne la plus récente par station
    if "snapshot_id" in df_filtered.columns:
        df_filtered = (
            df_filtered
            .sort_values("snapshot_id", ascending=False)
            .drop_duplicates(subset=["station_id"])
        )
    elif "run_at" in df_filtered.columns:
        df_filtered = (
            df_filtered
            .sort_values("run_at", ascending=False)
            .drop_duplicates(subset=["station_id"])
        )

    return df_filtered