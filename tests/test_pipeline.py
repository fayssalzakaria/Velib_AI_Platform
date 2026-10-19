"""Tests du pipeline lakehouse : transformation silver et idempotence du MERGE."""
from datetime import datetime, timezone

import pytest
from deltalake import DeltaTable

from pipeline.transform import snapshot_id_for, snapshot_time, to_silver

RECORDS = [
    {
        "stationcode": "11104", "name": "  Bastille -  Biscornet ", "capacity": 20,
        "numbikesavailable": 0, "numdocksavailable": 20, "mechanical": 0, "ebike": 0,
        "is_installed": "OUI", "is_renting": "OUI", "is_returning": "OUI",
        "duedate": "2026-10-16T07:58:00+00:00",
        "coordonnees_geo": {"lon": 2.36, "lat": 48.85},
        "nom_arrondissement_communes": "Paris",
    },
    {
        "stationcode": "16107", "name": "Benjamin Godard", "capacity": 35,
        "numbikesavailable": 35, "numdocksavailable": 0, "mechanical": 30, "ebike": 5,
        "is_installed": "OUI", "is_renting": "OUI", "is_returning": "NON",
        "duedate": "2026-10-16T07:50:00+00:00", "coordonnees_geo": None,
    },
    {"stationcode": None, "name": "SANS CODE"},  # ignorée
]

TS = datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc)  # 10h à Paris


def test_snapshot_time_floors_to_hour():
    ts = snapshot_time(datetime(2026, 10, 16, 8, 47, 12, tzinfo=timezone.utc))
    assert ts == TS
    assert snapshot_id_for(ts) == "20261016T0800Z"


def test_to_silver_cleans_and_enriches():
    rows = {r["station_id"]: r for r in to_silver(RECORDS, TS).to_pylist()}

    assert set(rows) == {"11104", "16107"}
    bastille, godard = rows["11104"], rows["16107"]

    assert bastille["name"] == "BASTILLE - BISCORNET"
    assert bastille["is_empty"] and not bastille["is_full"]
    assert bastille["bike_ratio"] == 0
    assert godard["is_full"] and godard["is_returning"] is False
    assert godard["lat"] is None

    # Colonnes calendaires en heure de Paris
    assert bastille["hour"] == 10
    assert bastille["weekday"] == "Friday"
    assert bastille["date"] == "2026-10-16"


def test_to_silver_rejects_empty_payload():
    with pytest.raises(ValueError):
        to_silver([{"stationcode": None}], TS)


def test_merge_silver_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKE_ROOT", str(tmp_path))
    from pipeline.lake import merge_silver

    table = to_silver(RECORDS, TS)
    assert merge_silver(table) == 2
    assert merge_silver(table) == 0  # même snapshot rejoué

    later = to_silver(RECORDS, datetime(2026, 10, 16, 9, tzinfo=timezone.utc))
    assert merge_silver(later) == 2

    dt = DeltaTable(str(tmp_path / "silver" / "station_status"))
    assert dt.to_pyarrow_table().num_rows == 4


def test_snapshot_exists_reports_writer(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKE_ROOT", str(tmp_path))
    monkeypatch.setenv("PIPELINE_ORCHESTRATOR", "airflow")
    from pipeline.lake import merge_silver, snapshot_exists

    assert snapshot_exists("20261016T0800Z", "2026-10-16") is None
    merge_silver(to_silver(RECORDS, TS))
    assert snapshot_exists("20261016T0800Z", "2026-10-16") == "airflow"
    assert snapshot_exists("20261016T0900Z", "2026-10-16") is None


def test_fallback_ingest_skips_when_primary_already_ran(tmp_path, monkeypatch):
    """GitHub Actions (secours) ne réécrit rien si Airflow a déjà collecté l'heure."""
    import pipeline.run as run
    from pipeline.lake import merge_silver

    monkeypatch.setenv("LAKE_ROOT", str(tmp_path))
    monkeypatch.setattr(run, "snapshot_time", lambda: TS)
    monkeypatch.setattr(run, "fetch_raw", lambda: pytest.fail("ne doit pas appeler l'API"))

    monkeypatch.setenv("PIPELINE_ORCHESTRATOR", "airflow")
    merge_silver(to_silver(RECORDS, TS))

    monkeypatch.setenv("PIPELINE_ORCHESTRATOR", "github-actions")
    result = run.ingest()
    assert result == {"snapshot_id": "20261016T0800Z", "skipped": True, "already_ingested_by": "airflow"}


@pytest.mark.parametrize("question, intent", [
    ("Bastille est-elle souvent vide le matin ?", "historical"),
    ("Quelle est la tendance à Nation le week-end ?", "historical"),
    ("Y a-t-il des stations vides maintenant ?", "anomaly"),
    ("Y a-t-il des anomalies ?", "anomaly"),
    ("Combien de vélos à Bastille en ce moment ?", "realtime"),
    ("Génère un rapport PDF", "report"),
    ("Bastille sera vide dans une heure ?", "forecast"),
    ("Quel est le risque à Nation ?", "forecast"),
    ("Est-ce que je trouverai un vélo à Nation à 18h ?", "forecast"),
    ("République sera pleine dans 3 heures ?", "forecast"),
    ("Bastille est-elle souvent vide à 18h ?", "historical"),
])
def test_router_intents(question, intent):
    from src.ai.router import classify_question_rule_based
    assert classify_question_rule_based(question)["intent"] == intent


@pytest.mark.parametrize("question, expected", [
    ("Nation dans 3 h ?", {"hours_ahead": 3}),
    ("et dans deux heures ?", {"hours_ahead": 2}),
    ("Un vélo à Bastille à 18h ?", {"at_hour": 18}),
    ("vers 7 h demain", {"at_hour": 7}),
    ("Bastille est vide ?", {}),
])
def test_parse_time_target(question, expected):
    from src.ai.nlp_utils import parse_time_target
    assert parse_time_target(question) == expected


def test_search_stations_ignores_accents_order_and_punctuation():
    import pandas as pd
    from src.data.filters import search_stations

    df = pd.DataFrame({"name": [
        "ATHÈNES - CLICHY", "MAIRIE DE CLICHY", "BESSIÈRES - PORTE DE CLICHY",
        "PLACE DE LA RÉPUBLIQUE - VOLTAIRE", "BASTILLE - BISCORNET",
    ]})
    assert len(search_stations(df, "clichy")) == 3
    assert search_stations(df, "porte clichy")["name"].tolist() == ["BESSIÈRES - PORTE DE CLICHY"]
    assert search_stations(df, "republique")["name"].tolist() == ["PLACE DE LA RÉPUBLIQUE - VOLTAIRE"]
    assert search_stations(df, "athenes")["name"].tolist() == ["ATHÈNES - CLICHY"]
    assert search_stations(df, "  ").empty


class _FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.ok, self.reason, self.text = status < 400, "error", str(body)
        self.headers = {"retry-after": "0"}

    def json(self):
        return self._body


def _fake_groq(monkeypatch, status=200, body=None):
    import llm_client
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.clear()
        sent.update(json)
        return _FakeResponse(status, body)

    monkeypatch.setattr(llm_client, "GROQ_API_KEY", "test")
    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    return llm_client, sent


def test_reasoning_model_gets_low_effort_and_token_floor(monkeypatch):
    body = {"choices": [{"message": {"content": "<think>hmm</think> Bonjour", "reasoning": "..."}}], "usage": {}}
    llm, sent = _fake_groq(monkeypatch, body=body)

    data = llm.call_llm([{"role": "user", "content": "salut"}], model="openai/gpt-oss-120b", max_tokens=150)
    assert sent["reasoning_effort"] == "low"
    assert sent["max_tokens"] == 1024
    message = data["choices"][0]["message"]
    assert message["content"] == "Bonjour" and "reasoning" not in message

    llm.call_llm([{"role": "user", "content": "salut"}], model="llama-autre", max_tokens=150)
    assert "reasoning_effort" not in sent and sent["max_tokens"] == 150


def test_groq_error_message_is_explicit(monkeypatch):
    body = {"error": {"message": "The model `llama-3.3-70b-versatile` does not exist"}}
    llm, _ = _fake_groq(monkeypatch, status=404, body=body)
    with pytest.raises(RuntimeError, match="Groq 404 : The model"):
        llm.call_llm([{"role": "user", "content": "salut"}])


def test_track_usage_sums_calls_tokens_and_cost(monkeypatch):
    body = {"choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}
    llm, _ = _fake_groq(monkeypatch, body=body)

    with llm.track_usage() as usage:
        llm.call_llm([{"role": "user", "content": "a"}], model="openai/gpt-oss-120b")
        llm.call_llm([{"role": "user", "content": "b"}], model="openai/gpt-oss-120b")

    assert usage.calls == 2 and usage.total_tokens == 3000
    # 2 x (1000 x 0.15 + 500 x 0.60) / 1e6
    assert abs(usage.cost_usd - 0.0009) < 1e-9
    # mesures imbriquées : l'appel compte dans les deux
    with llm.track_usage() as outer:
        with llm.track_usage() as inner:
            llm.call_llm([{"role": "user", "content": "d"}], model="openai/gpt-oss-120b")
    assert outer.calls == inner.calls == 1
    # hors d'un bloc track_usage, rien n'est compté
    llm.call_llm([{"role": "user", "content": "c"}], model="openai/gpt-oss-120b")
    assert usage.calls == 2


def test_router_reports_rule_and_matched_keywords():
    from src.ai.router import classify_question_rule_based

    route = classify_question_rule_based("Bastille est-elle souvent vide le matin ?")
    assert route["rule"] == "habitude" and route["matched"] == ["souvent"]
    assert route["confidence"] is None  # pas de faux pourcentage pour une règle

    route = classify_question_rule_based("salut")
    assert route["intent"] == "general" and route["rule"] == "aucune" and route["matched"] == []


def test_incomplete_api_response_is_retried(monkeypatch):
    """Stations sans code ni nom (API en cours de mise à jour) : on réessaie au lieu d'écrire un snapshot vide."""
    import pipeline.extract as extract

    good = [{"stationcode": str(i), "name": f"S{i}"} for i in range(10)]
    responses = iter([[{"duedate": "x"}] * 10, good])

    class Response:
        def __init__(self, data): self.data, self.content = data, b""
        def raise_for_status(self): pass
        def json(self): return self.data

    monkeypatch.setattr(extract.requests, "get", lambda *a, **k: Response(next(responses)))
    assert extract.fetch_raw(wait_seconds=0) == good
