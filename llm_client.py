"""
llm_client.py — Gateway LLM : un seul point d'entrée pour Groq et Claude.

Routage par tâche :
- Groq (gpt-oss-120b) : routage, HyDE, synthèses, RAG, agent simple — rapide et quasi gratuit
- Claude Haiku 5.5    : juge de pertinence (un autre modèle que celui qui répond, moins de biais)
                        et superviseur de l'agent opérateur (choix simple, peu coûteux)
- Claude Opus 5.5     : planificateur de l'agent opérateur et escalade des réponses faibles

La gateway centralise aussi la mesure (appels, tokens, latence, coût par modèle), le plafond
de dépense Claude, le prompt caching Claude et les erreurs explicites.
"""

import json
import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = os.environ.get(
    "GROQ_URL",
    "https://api.groq.com/openai/v1/chat/completions",
)
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_JUDGE_MODEL = "claude-haiku-5-5"
CLAUDE_COMPLEX_MODEL = "claude-opus-5-5"
# Plafond mensuel de dépense Claude suivi par l'application (filet de sécurité : la limite
# de dépense de la console Anthropic). Au-delà, les tâches Claude repassent sur Groq.
CLAUDE_MONTHLY_BUDGET_USD = float(os.environ.get("CLAUDE_MONTHLY_BUDGET_USD", 80))
# Repli serveur en cas de refus des classifieurs de sécurité (Opus 5.5 ; pas disponible sur Haiku 5.5)
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Les modèles qui raisonnent consomment des tokens de réflexion dans la même limite
# que la réponse : on réduit la réflexion et on garantit un minimum de tokens.
REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "low")
MIN_TOKENS_REASONING_MODEL = 1024
_THINK_TAGS = re.compile(r"<think>.*?</think>", re.DOTALL)


# Tarifs en $ par million de tokens : (entrée, sortie, lecture du cache, écriture du cache).
# Groq : relevés sur /models (rien n'est facturé sur l'offre gratuite : coût équivalent payant).
# Claude : tarifs publics ; écriture du cache ≈ 1,25 × entrée, lecture ≈ 0,1 × entrée.
PRICES_PER_MILLION = {
    "openai/gpt-oss-120b": (0.15, 0.60, 0.0, 0.0),
    "openai/gpt-oss-20b": (0.075, 0.30, 0.0, 0.0),
    "qwen/qwen3.8-27b": (0.80, 4.00, 0.0, 0.0),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}


# ---------------------------------------------------------------- mesure de la consommation

@dataclass
class UsageReport:
    """Consommation cumulée des appels LLM faits pendant un bloc `track_usage()`."""
    calls: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_seconds: float = 0.0
    cost_usd: float = 0.0
    models: set = field(default_factory=set)
    by_model: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {
            "llm_calls": self.calls,
            "llm_errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_seconds": round(self.llm_seconds, 2),
            "cost_usd": round(self.cost_usd, 6),
            "by_model": {m: {k: round(v, 6) if isinstance(v, float) else v for k, v in d.items()}
                         for m, d in self.by_model.items()},
        }


# Pile des mesures actives : un appel compte dans toutes (ex. l'assistant qui délègue à l'agent)
_active_reports: ContextVar[tuple] = ContextVar("llm_usage", default=())
# Dépense Claude de l'application, par mois (par processus : la limite de la console reste le plafond dur)
_claude_spend: dict[str, float] = {}


@contextmanager
def track_usage():
    """Mesure tous les appels LLM faits dans le bloc (y compris HyDE, routeur, juge…)."""
    report = UsageReport()
    token = _active_reports.set(_active_reports.get() + (report,))
    try:
        yield report
    finally:
        _active_reports.reset(token)


def _cost(model: str, prompt: int, completion: int, cache_read: int = 0, cache_write: int = 0) -> float:
    price_in, price_out, price_read, price_write = PRICES_PER_MILLION.get(model, (0.0, 0.0, 0.0, 0.0))
    return (prompt * price_in + completion * price_out + cache_read * price_read + cache_write * price_write) / 1_000_000


def _record_usage(model: str, prompt: int, completion: int, seconds: float, error: bool = False,
                  cache_read: int = 0, cache_write: int = 0) -> None:
    cost = _cost(model, prompt, completion, cache_read, cache_write)
    if model.startswith("claude"):
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        _claude_spend[month] = _claude_spend.get(month, 0.0) + cost
    for report in _active_reports.get():
        report.calls += 1
        report.errors += int(error)
        report.prompt_tokens += prompt + cache_read + cache_write
        report.completion_tokens += completion
        report.llm_seconds += seconds
        report.cost_usd += cost
        report.models.add(model)
        per_model = report.by_model.setdefault(model, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        per_model["calls"] += 1
        per_model["tokens"] += prompt + cache_read + cache_write + completion
        per_model["cost_usd"] += cost


def claude_spend_this_month() -> float:
    return _claude_spend.get(datetime.now(timezone.utc).strftime("%Y-%m"), 0.0)


def claude_available() -> bool:
    """Claude est utilisable : clé configurée et plafond mensuel de l'application non atteint."""
    return bool(ANTHROPIC_API_KEY) and claude_spend_this_month() < CLAUDE_MONTHLY_BUDGET_USD


# ---------------------------------------------------------------- Groq

MAX_RATE_LIMIT_RETRIES = 2


def _retry_delay(response: requests.Response) -> float:
    try:
        return min(float(response.headers.get("retry-after", 5)), 20.0)
    except (TypeError, ValueError):
        return 5.0


def _is_reasoning_model(model: str) -> bool:
    return "gpt-oss" in model.lower()


def _clean_content(content: str | None) -> str:
    """Retire une éventuelle réflexion laissée dans le texte (<think>…</think>)."""
    return _THINK_TAGS.sub("", content or "").strip()


def _raise_for_groq_error(response: requests.Response) -> None:
    """Remonte le message d'erreur de Groq (modèle inconnu, clé invalide, quota…)."""
    if response.ok:
        return
    try:
        detail = response.json().get("error", {}).get("message", "")
    except ValueError:
        detail = response.text[:300]
    raise RuntimeError(f"Groq {response.status_code} : {detail or response.reason}")


def call_llm(
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = "auto",
    max_tokens: int = 1000,
    temperature: float = 0.2,
    timeout: int = 30,
) -> dict[str, Any]:
    """Appel Groq (format OpenAI) : texte, JSON et outils. Renvoie la réponse JSON complète."""
    if not GROQ_API_KEY:
        raise RuntimeError("Clé GROQ_API_KEY non configurée.")

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    if _is_reasoning_model(model):
        payload["reasoning_effort"] = REASONING_EFFORT
        payload["max_tokens"] = max(max_tokens, MIN_TOKENS_REASONING_MODEL)

    started = time.perf_counter()
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        response = requests.post(
            GROQ_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json=payload,
            timeout=timeout,
        )
        # Limite de débit de l'offre gratuite : on attend le délai indiqué par Groq et on réessaie
        if response.status_code != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
            break
        time.sleep(_retry_delay(response))

    if not response.ok:
        _record_usage(model, 0, 0, time.perf_counter() - started, error=True)
    _raise_for_groq_error(response)
    data = response.json()
    usage = data.get("usage") or {}
    _record_usage(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                  time.perf_counter() - started)

    # Réponse normalisée : texte sans réflexion, et pas de champ « reasoning »
    # (l'agent renvoie ce message à Groq lors du second appel)
    for choice in data.get("choices", []):
        message = choice.get("message", {})
        message.pop("reasoning", None)
        message["content"] = _clean_content(message.get("content"))
    return data


def call_llm_text(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    timeout: int = 30,
) -> tuple[str, dict[str, int]]:
    """Appel simple texte -> texte (Groq). Renvoie la réponse et les tokens."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    data = call_llm(messages, model=model, max_tokens=max_tokens, temperature=temperature, timeout=timeout)
    usage = data.get("usage", {})
    return data["choices"][0]["message"].get("content", ""), {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Réponse LLM non JSON valide : {cleaned}") from e


def call_llm_json(
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 500,
    temperature: float = 0.1,
    timeout: int = 30,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Appel Groq qui doit renvoyer du JSON (les blocs Markdown ```json sont retirés)."""
    answer, tokens = call_llm_text(prompt, system_prompt=system_prompt, model=model,
                                   max_tokens=max_tokens, temperature=temperature, timeout=timeout)
    return _parse_json(answer), tokens


# ---------------------------------------------------------------- Claude

class ClaudeRefusal(RuntimeError):
    """Les classifieurs de sécurité de Claude ont décliné la requête (après repli éventuel)."""


_anthropic_singleton = None


def _anthropic_client():
    global _anthropic_singleton
    if _anthropic_singleton is None:
        import anthropic
        _anthropic_singleton = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_singleton


def call_claude(
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    effort: str | None = None,
    output_schema: dict[str, Any] | None = None,
    max_tokens: int = 16000,
):
    """
    Appel Claude via le SDK officiel. Renvoie le message de l'API.
    - prompt caching : le préfixe stable (outils + prompt système) est mis en cache
    - Opus 5.5 : repli serveur automatique en cas de refus (fallbacks="default")
    - output_schema : sortie JSON garantie conforme au schéma (structured outputs)
    """
    if not claude_available():
        raise RuntimeError("Claude indisponible : clé absente ou plafond mensuel atteint.")

    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": messages,
    }
    if tools:
        params["tools"] = tools
    output_config: dict[str, Any] = {}
    if effort:
        output_config["effort"] = effort
    if output_schema:
        output_config["format"] = {"type": "json_schema", "schema": output_schema}
    if output_config:
        params["output_config"] = output_config

    import anthropic

    client = _anthropic_client()
    started = time.perf_counter()
    try:
        if model == CLAUDE_COMPLEX_MODEL:
            response = client.beta.messages.create(**params, betas=[_FALLBACK_BETA], fallbacks="default")
        else:
            response = client.messages.create(**params)
    except anthropic.APIStatusError as e:
        _record_usage(model, 0, 0, time.perf_counter() - started, error=True)
        raise RuntimeError(f"Claude {e.status_code} : {e.message}") from e
    except anthropic.APIConnectionError as e:
        _record_usage(model, 0, 0, time.perf_counter() - started, error=True)
        raise RuntimeError(f"Claude injoignable : {e}") from e

    usage = response.usage
    _record_usage(
        model, usage.input_tokens, usage.output_tokens, time.perf_counter() - started,
        cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )
    if response.stop_reason == "refusal":
        raise ClaudeRefusal("Claude a décliné la requête.")
    return response


def claude_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text").strip()


# ---------------------------------------------------------------- juge de pertinence

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "explication": {"type": "string"},
    },
    "required": ["score", "explication"],
    "additionalProperties": False,
}

_JUDGE_SYSTEM = """Tu évalues la qualité des réponses d'un assistant sur le réseau Vélib' de Paris.
Note la réponse de 0 à 100 au regard de la question et des données fournies, et explique la note
en une ou deux phrases.
- 90-100 : précise, entièrement fondée sur les données, sans invention
- 70-89 : correcte mais incomplète
- 50-69 : partiellement correcte
- 0-49 : incorrecte, ou affirme des faits absents des données"""


def verify_relevance(
    *,
    question: str,
    answer: str,
    context: str,
    evaluator_name: str = "système RAG",
) -> tuple[int, str]:
    """
    Note la pertinence d'une réponse (0-100) et l'explique.
    Juge : Claude Haiku 5.5 si disponible (un autre modèle que celui qui répond), sinon Groq.
    """
    content = (f"Système évalué : {evaluator_name}\n\nQuestion :\n{question}\n\n"
               f"Données utilisées :\n{context[:2000]}\n\nRéponse générée :\n{answer[:1500]}")
    try:
        if claude_available():
            response = call_claude(model=CLAUDE_JUDGE_MODEL, system=_JUDGE_SYSTEM,
                                   messages=[{"role": "user", "content": content}],
                                   effort="low", output_schema=_JUDGE_SCHEMA, max_tokens=2000)
            data = json.loads(claude_text(response))
        else:
            data, _ = call_llm_json(
                f"{_JUDGE_SYSTEM}\n\n{content}\n\nRéponds UNIQUEMENT avec un JSON valide : "
                '{"score": <entier entre 0 et 100>, "explication": "<1 ou 2 phrases>"}',
                max_tokens=200, temperature=0.1, timeout=20,
            )
        return max(0, min(100, int(data.get("score", 0)))), data.get("explication", "")
    except Exception as e:
        return 0, f"Erreur évaluation : {e}"


# ---------------------------------------------------------------- tâches complexes

def call_complex_json(*, system: str, prompt: str, schema: dict[str, Any], effort: str = "high",
                      max_tokens: int = 16000, model: str = CLAUDE_COMPLEX_MODEL) -> tuple[dict[str, Any], str]:
    """
    Tâche à sortie JSON : Claude (Opus 5.5 par défaut, sortie structurée garantie) si disponible,
    sinon Groq avec le schéma décrit dans le prompt. Renvoie (données, modèle utilisé).
    """
    if claude_available():
        response = call_claude(model=model, system=system,
                               messages=[{"role": "user", "content": prompt}],
                               effort=effort, output_schema=schema, max_tokens=max_tokens)
        return json.loads(claude_text(response)), model
    data, _ = call_llm_json(
        f"{prompt}\n\nRéponds UNIQUEMENT avec un JSON conforme à ce schéma :\n{json.dumps(schema, ensure_ascii=False)}",
        system_prompt=system, max_tokens=4000, temperature=0.1, timeout=60,
    )
    return data, DEFAULT_MODEL
