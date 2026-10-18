# Vélib' AI Platform — agents IA, lakehouse et ML

**Airflow 3 · dbt · DuckDB · Delta Lake · AWS S3 · GitHub Actions · LightGBM · LangGraph · LangChain · Claude · Groq · RAG · Streamlit**

Plateforme Data & IA sur le réseau Vélib' de Paris (~1 500 stations) :

- un **pipeline** collecte toutes les heures l'état du réseau et l'historise dans un **lakehouse S3** (bronze / silver / gold) ;
- un **dashboard Streamlit** affiche le temps réel et l'historique ;
- des **modèles LightGBM** prévoient, de 1 à 6 heures à l'avance, les stations vides, les stations pleines et le nombre de vélos ;
- un **assistant**, un **agent** et un **agent opérateur multi-agents (LangGraph)** répondent en langage naturel ou proposent des rééquilibrages, avec des **métriques de qualité et de consommation** pour chaque réponse.

> Le temps réel dit où sont les vélos ; l'historique dit pourquoi ils n'y sont pas.

- Application : [velib-data-pipeline.streamlit.app](https://velib-data-pipeline-e8bwwhyt8nevmwfcbfq5sa.streamlit.app/)
- 🎬 **Vidéo de présentation** : [Video_app_velib.mp4](Video_app_velib.mp4) (temps réel, historique, assistant, agent, agent opérateur multi-agents, métriques de qualité et de coût)

---

## Pourquoi une v2 « low-cost »

La v1 tournait entièrement sur AWS : EventBridge, Lambda, Aurora Serverless v2, API Gateway, Terraform. Aurora Serverless v2 facture au minimum 0,5 ACU en continu, même sans requête : c'était l'essentiel de la facture.

La v2 garde **S3 comme seul service AWS** et déplace le calcul vers des outils gratuits. Coût estimé : **quelques centimes par mois**.

| Rôle | v1 | v2 |
|---|---|---|
| Orchestration | EventBridge + Lambda | **Airflow 3** (principal) + **GitHub Actions** (secours) |
| Stockage | S3 CSV + Aurora PostgreSQL | **Lakehouse S3** : Delta Lake + Parquet |
| Moteur SQL | Aurora (serveur 24 h/24) | **DuckDB** embarqué |
| Transformations | pandas dans la Lambda | **dbt-duckdb** + tests qualité |
| Accès AWS | clés longues | **OIDC** (GitHub), IAM à privilèges minimaux |

Le code de la v1 reste consultable via le tag git [`v1-aws`](https://github.com/fayssalzakaria/Velib_AI_Platform/tree/v1-aws).

---

## Architecture

```text
API Open Data Vélib' (v2.1)
        │ toutes les heures
        ├── Airflow 3 (Docker, LocalExecutor)    principal : :05
        └── GitHub Actions (OIDC)                secours : :40 / :55, seulement si le snapshot manque
                │  même code : pipeline/ + dbt/
                ▼
S3  lake/
  ├── bronze/   JSON brut de l'API (rejouable)
  ├── silver/   Delta Lake : 1 ligne par station et par heure (MERGE idempotent)
  └── gold/     Parquet (dbt + DuckDB, chaque jour) + embeddings des profils de stations
                ▼
Streamlit Cloud (DuckDB en lecture seule)
  ├── ⏱️ Temps réel : snapshot de l'API, rafraîchi à la demande
  ├── 📚 Historique : statistiques sur tous les snapshots du lakehouse
  └── 🤖 IA : Assistant · Agent · Opérateur · Qualité & consommation   (Groq, Claude, RAG)
```

### Pipeline de données

| Étape | Quand | Détail |
|---|---|---|
| `ingest` | chaque heure | API → bronze (JSON gzip) → silver (Delta, `MERGE` sur `(date, snapshot_id, station_id)`) |
| `maintain` | chaque jour | `OPTIMIZE` (compaction) + `VACUUM` + checkpoint Delta |
| `dbt build` | chaque jour | modèles gold et tests qualité (unicité, plages de valeurs, valeurs acceptées) |
| `embed` | chaque jour | encode les profils de stations (multilingual-e5) → gold + Qdrant |
| `external` | chaque jour | météo horaire (Open-Meteo) et vacances scolaires (API Éducation nationale) |
| `forecast` | **à la main** | entraîne les modèles de prévision (vide, pleine, vélos) ; chacun publié seulement s'il bat les baselines |
| `monitor` | chaque jour | compare les prévisions enregistrées à la réalité : AUC par jour, calibration, dérive |

**Deux orchestrateurs, sans doublons ni pertes :**

- au début de chaque `ingest`, le pipeline vérifie si le snapshot de l'heure existe déjà : si oui, il s'arrête sans rien écrire (colonne `ingested_by` = qui a collecté) ;
- en cas d'écriture simultanée, le **contrôle de concurrence optimiste** de Delta, appuyé sur les écritures conditionnelles S3, rejette le second commit, qui est alors rejoué. Ce comportement est vérifié sur S3 par `scripts/check_concurrent_writes.py`.

### Modèles gold (dbt)

| Modèle | Contenu |
|---|---|
| `gold_station_hourly_profile` | profil par station × jour de la semaine × heure (90 jours glissants) |
| `gold_station_period_profile` | profil par station × semaine/week-end × période : source du RAG |
| `gold_station_daily` | synthèse journalière par station |
| `gold_network_snapshots` | indicateurs globaux du réseau à chaque snapshot |

---

## Application

**⏱️ Mode Temps réel** : un snapshot de l'API est pris à l'arrivée sur la page et rafraîchi avec le bouton « Récupérer un nouveau snapshot ». On y trouve les statistiques de ce snapshot, une carte interactive (rouge : aucun vélo, vert sinon, détail au survol), une recherche de station insensible aux accents, des graphiques, un PDF et un CSV. Si l'API ne répond pas, l'app affiche le dernier snapshot du lakehouse.

**📚 Mode Historique** : statistiques sur tous les snapshots de la période choisie (24 h, 7 j, 30 j ou tout) :

- évolution du réseau, avec un point par snapshot ;
- heures critiques ;
- stations les plus souvent vides ou pleines ;
- carte de la fréquence des stations vides ;
- historique d'une station ;
- export CSV.

### Prévision ML : vide, pleine, nombre de vélos, de 1 à 6 heures

Trois questions, un modèle LightGBM par question, l'horizon (1 à 6 h) étant une feature :

| Question | Type de modèle | Métriques |
|---|---|---|
| La station sera-t-elle **vide** à t + h ? | classification | AUC, précision moyenne, Brier, précision sur les 20 stations les plus à risque |
| La station sera-t-elle **pleine** à t + h ? | classification | idem |
| **Combien de vélos** à t + h ? | régression quantile (10 %, 50 %, 90 %) : valeur + intervalle de 80 % | erreur moyenne, couverture de l'intervalle |

| Étape | Choix |
|---|---|
| Features | état actuel, variations sur 1 h et 2 h, heure et jour visés, habitudes de la station à l'heure visée (taux de vide, de plein, vélos moyens, calculés uniquement sur le passé), météo de l'heure visée, jours fériés et vacances scolaires |
| Évaluation | découpage temporel entraînement < validation < test, métriques **par horizon**, comparées à deux baselines : persistance (« comme maintenant ») et habitudes de la station |
| Publication | registre de modèles versionné dans S3 (`lake/models/forecast/`) ; chaque question est publiée seulement si son modèle bat les deux baselines |
| Entraînement | lancé à la main (DAG Airflow `velib_train_forecast` ou `python -m pipeline.run forecast`), avec un historique minimal à choisir (7 jours par défaut) et une option de test qui enregistre les métriques sans servir les modèles |
| Garde-fous | mêmes features à l'entraînement et en prédiction (`src/ml/features.py`), vérifié par les tests pour chaque horizon |
| Monitoring | à chaque collecte, les prévisions des 6 horizons sont enregistrées avec leurs features ; chaque jour : AUC vide/pleine, erreur sur les vélos et couverture **par jour et par horizon**, calibration, dérive des features (PSI) |

Dans l'app (mode Temps réel) : choix de l'horizon et de la question, stations à réapprovisionner ou à délester, carte du risque, courbe des vélos attendus d'une station avec son intervalle, fiche des modèles et performance en production. L'IA dispose d'un outil `forecast_station` qui comprend « dans 3 h » ou « à 18h ».

### IA

| Onglet | Principe |
|---|---|
| 💬 **Assistant** | **workflow** : le code route la question par règles (temps réel, anomalies, historique, prévision, rapport), appelle les outils, puis le LLM rédige. Les questions non reconnues sont déléguées à l'agent |
| 🤖 **Agent** | **agent** : le LLM choisit ses outils par function calling (état actuel, anomalies, historique, prévision), sur 3 tours maximum |
| 🧭 **Opérateur** | **graphe LangGraph de type superviseur** : un superviseur choisit les spécialistes (temps réel, prévision, historique), exécutés en parallèle ; un planificateur propose des déplacements de vélos ; un vérificateur déterministe contrôle stations, stocks, bornes libres et distances, et renvoie corriger le plan (2 fois au plus). Seules les actions vérifiées sont affichées |
| 📊 **Qualité & consommation** | métriques de la session, banc d'évaluation Assistant vs Agent et banc de l'agent opérateur |

Les outils sont définis **une seule fois** comme outils LangChain ([src/ai/toolkit.py](src/ai/toolkit.py)) : leurs schémas pour Groq et pour Claude en sont générés.

**RAG hybride** sur les profils de stations, exposé comme retriever LangChain : HyDE → BM25 + similarité cosinus → fusion RRF → MMR → citations `[Source N]`. Les embeddings (`intfloat/multilingual-e5-small`) sont calculés en batch dans le pipeline : l'application n'encode que la question.

**Gateway LLM** ([llm_client.py](llm_client.py)) : chaque tâche va au modèle adapté.

| Tâche | Modèle |
|---|---|
| synthèses, agent, HyDE | Groq `openai/gpt-oss-120b` (rapide, quasi gratuit) |
| juge de pertinence, superviseur de l'opérateur | Claude Haiku 5.5 (un autre modèle que celui qui répond) |
| planificateur de l'opérateur, reprise des réponses faibles | Claude Opus 5.5 |

Sans clé Anthropic, ou une fois le plafond mensuel atteint (`CLAUDE_MONTHLY_BUDGET_USD`), tout repasse sur Groq.

**Métriques mesurées pour chaque réponse** :

- qualité : pertinence notée par le juge (0-100). Sous 60, la réponse est reprise par Claude Opus avec les mêmes données, et la meilleure des deux est gardée ;
- consommation : appels LLM par modèle, tokens (cache compris), latence, coût estimé.

Le banc d'évaluation (`python -m src.ai.evaluation`) pose des questions annotées aux systèmes et compare les outils appelés, la station citée, la pertinence, les tokens, la latence et le coût.

---

## Structure

```text
app.py                  point d'entrée Streamlit
llm_client.py           gateway LLM Groq + Claude (routage par tâche, juge, consommation, coût, plafond)
src/
  ui/                   ui.py (temps réel), history.py (historique), ai_panel.py (IA)
  ai/                   toolkit (outils LangChain), retriever, assistant, agent, operator_agent (LangGraph), router, rag, evaluation
  ml/                   features (partagées entraînement/prédiction), prédiction, registre de modèles
  data/                 lake.py (DuckDB → S3), data_loader.py (API), filters.py
pipeline/               ingest, transform, lake (Delta), embed, forecast (LightGBM), external, monitoring, backfill, control
dbt/                    modèles gold et tests
airflow/                DAGs Airflow 3 + Docker Compose
.github/workflows/      ingest-fallback, daily-fallback, ci
tests/                  pytest (pipeline, prévision, gateway LLM, outils, agent opérateur)
```

---

## Lancer en local

```bash
pip install -r requirements.txt -r pipeline/requirements.txt

# Lakehouse local, sans AWS
export LAKE_ROOT="$PWD/lake"
python -m pipeline.run ingest
cd dbt && dbt build --profiles-dir . --target local && cd ..

export GROQ_API_KEY=...
streamlit run app.py
```

Airflow (collecteur principal) :

```bash
cd airflow && cp env.example .env   # LAKE_ROOT, clés AWS, QDRANT_*
docker compose up -d --build        # http://localhost:8080 (ou AIRFLOW_PORT)
```

Entraîner les modèles de prévision, quand on le décide :

```bash
# Airflow : DAG velib_train_forecast → Trigger (paramètres min_days et publish)
python -m pipeline.run forecast                            # 7 jours d'historique minimum, publie
python -m pipeline.run forecast --min-days 2 --no-publish  # test : métriques sans servir les modèles
```

### Variables d'environnement

| Variable | Rôle |
|---|---|
| `LAKE_ROOT` | `s3://<bucket>/lake` (ou un chemin local). Sans elle : mode temps réel seul |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` | Streamlit : utilisateur IAM en lecture seule sur `lake/*` |
| `GROQ_API_KEY`, `GROQ_MODEL` | LLM principal |
| `ANTHROPIC_API_KEY`, `CLAUDE_MONTHLY_BUDGET_USD` | Claude (optionnel) : juge, agent opérateur, reprise des réponses faibles ; plafond mensuel (80 $ par défaut) |
| `QDRANT_URL`, `QDRANT_API_KEY` | recherche sémantique (optionnel) |
| `EMBEDDING_MODEL` | modèle d'embeddings (par défaut `intfloat/multilingual-e5-small`) |

GitHub Actions utilise le secret `AWS_ROLE_ARN` (OIDC) et les variables `S3_BUCKET` et `AWS_REGION`.

---

## Limites connues

- Le juge de pertinence est un LLM : sa note reste une estimation. Des annotations humaines seraient plus fiables.
- Le banc d'évaluation compte peu de questions : il sert à comparer les systèmes, pas à mesurer une qualité absolue.
- L'API ne donne que l'état courant : une heure non collectée (PC éteint et GitHub en retard) est perdue.
- Les profils historiques sont recalculés chaque jour : ils ont jusqu'à un jour de retard.
- Le modèle de prévision a besoin de plusieurs jours de collecte pour apprendre les habitudes de chaque jour de la semaine.
- L'agent opérateur estime les distances à vol d'oiseau (2 km maximum), sans itinéraire réel.
- La météo de l'heure prédite est une prévision au moment de la prédiction, mais une valeur observée à l'entraînement : un léger écart est possible.
