"""Gateway LLM : routage Groq / Claude, juge Haiku, repli en cas de refus, coût et plafond."""
from types import SimpleNamespace

import pytest

import llm_client


class _FakeMessages:
    def __init__(self, sink, text, stop_reason="end_turn"):
        self.sink, self.text, self.stop_reason = sink, text, stop_reason

    def create(self, **params):
        self.sink.append(params)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                                  cache_read_input_tokens=5000, cache_creation_input_tokens=0),
        )


@pytest.fixture
def fake_claude(monkeypatch):
    """Faux client Anthropic : enregistre les requêtes, renvoie une note de juge."""
    sent = []

    def install(text='{"score": 87, "explication": "Fondée sur les données."}', stop_reason="end_turn"):
        messages = _FakeMessages(sent, text, stop_reason)
        client = SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages))
        monkeypatch.setattr(llm_client, "ANTHROPIC_API_KEY", "test")
        monkeypatch.setattr(llm_client, "_anthropic_singleton", client)
        monkeypatch.setattr(llm_client, "_claude_spend", {})
        return sent

    return install


def test_judge_uses_claude_haiku_with_structured_output(fake_claude):
    sent = fake_claude()
    with llm_client.track_usage() as usage:
        score, why = llm_client.verify_relevance(question="q", answer="a", context="c")
    assert (score, why) == (87, "Fondée sur les données.")
    request = sent[0]
    assert request["model"] == "claude-haiku-5-5"
    assert request["output_config"]["effort"] == "low"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "fallbacks" not in request  # pas de repli serveur sur Haiku 5.5
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    # 1000 × 0,10 + 200 × 0,50 + 5000 (cache) × 0,01  par million
    assert usage.cost_usd == pytest.approx((1000 * 0.10 + 200 * 0.50 + 5000 * 0.01) / 1e6)
    assert usage.by_model["claude-haiku-5-5"]["calls"] == 1


def test_opus_enables_server_side_refusal_fallback(fake_claude):
    sent = fake_claude(text="plan")
    response = llm_client.call_claude(model="claude-opus-5-5", system="s",
                                      messages=[{"role": "user", "content": "x"}], effort="high")
    assert llm_client.claude_text(response) == "plan"
    assert sent[0]["fallbacks"] == "default"
    assert sent[0]["betas"] == ["server-side-fallback-2026-07-01"]


def test_refusal_is_raised(fake_claude):
    fake_claude(text="", stop_reason="refusal")
    with pytest.raises(llm_client.ClaudeRefusal):
        llm_client.call_claude(model="claude-opus-5-5", system="s", messages=[{"role": "user", "content": "x"}])


def test_budget_cap_falls_back_to_groq(fake_claude, monkeypatch):
    sent = fake_claude()
    month = llm_client.datetime.now(llm_client.timezone.utc).strftime("%Y-%m")
    monkeypatch.setattr(llm_client, "_claude_spend", {month: llm_client.CLAUDE_MONTHLY_BUDGET_USD})
    assert not llm_client.claude_available()

    groq_calls = []
    monkeypatch.setattr(llm_client, "call_llm_json",
                        lambda *a, **k: (groq_calls.append(1) or {"score": 70, "explication": "via Groq"}, {}))
    assert llm_client.verify_relevance(question="q", answer="a", context="c") == (70, "via Groq")
    assert sent == [] and groq_calls == [1]


def test_without_anthropic_key_judge_uses_groq(monkeypatch):
    monkeypatch.setattr(llm_client, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(llm_client, "call_llm_json", lambda *a, **k: ({"score": 55, "explication": "g"}, {}))
    assert llm_client.verify_relevance(question="q", answer="a", context="c") == (55, "g")
