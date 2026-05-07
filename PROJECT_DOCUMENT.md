# Graph-based Early Network Inefficiency AI Agent

## Graph-Based Machine Learning for Airline Crew Scheduling Optimization

**Organization:** American Airlines — Crew Scheduling & Operations Research  
**Platform:** Neo4j Aura (Graph Database) + PyTorch (Deep Learning) + GENIE Agent (Azure AI Foundry)  
**Date:** May 2026

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Business Context & Impact](#2-business-context--impact)
3. [Data Sources](#3-data-sources)
4. [Solution Architecture](#4-solution-architecture)
5. [Graph Data Model](#5-graph-data-model)
6. [Feature Engineering with Neo4j GDS](#6-feature-engineering-with-neo4j-gds)
7. [Node2Vec Gaussian-Weighted Embeddings](#7-node2vec-gaussian-weighted-embeddings)
8. [Neural Network Model](#8-neural-network-model)
9. [Model Performance & Results](#9-model-performance--results)
10. [GENIE Agent & Prediction Export](#10-genie-agent--prediction-export)
11. [Project Files](#11-project-files)
12. [How to Run](#12-how-to-run)

---

## 1. Problem Statement

In airline crew scheduling, a **duty period** is a continuous block of work during which a crew member operates one or more flights before taking a required rest break. Identifying which flight in a sequence is the **last flight of a duty period** (the "duty-end flight") is fundamental to:

- Building legal and efficient crew pairings
- Ensuring compliance with FAA rest requirements (14 CFR Part 117)
- Optimizing crew utilization across the flight network
- Detecting where crews will **Remain Overnight (RON)** at outstations

**The core question:** Given a scheduled flight within a Line of Flying (LOF), can we predict whether that flight is the last flight a crew will operate before their mandatory rest period?

Traditional rule-based approaches rely on fixed contractual limits (e.g., maximum duty time of 9–14 hours depending on report time). This project explores whether **graph-based machine learning** — leveraging the structural properties of the flight network — can learn the duty-end pattern automatically and generalize to new schedules.

---

## 2. Business Context & Impact

| Dimension | Details |
|-----------|---------|
| **Schedule scope** | Proposal 2025-10 F0 (October–December 2025 schedule) |
| **Total flights** | 324,192 scheduled flight legs |
| **Airports** | 254 unique stations |
| **Fleet types** | 8 — 32T, 737, 319, 321, 777, 787, 320, 32X |
| **Body types** | NB (Narrow Body: A319, A320, A321, 737) and WB (Wide Body: 777, 787) |
| **Class balance** | 164,035 duty-end flights (50.6%) vs. 160,157 non-duty-end (49.4%) — nearly balanced |

### Why Graph ML?

Flights are not independent events — they form **sequences** (Lines of Flying) where crew members fly from base → outstation → ... → base. The graph structure captures:

- **Connectivity patterns** at stations (hub vs. spoke behavior)
- **Sequence position** within a crew trip
- **Community structure** — groups of flights that tend to be operated together
- **Centrality** — how "important" a flight is in bridging different parts of the network

A graph neural approach naturally encodes these relational signals that tabular models would miss.

---

## 3. Data Sources

| Source | Description | Format |
|--------|-------------|--------|
| `gemini_with_duty_flag.csv` | Crew scheduling data from the Gemini system with duty-end flags, LOF assignments, fleet types, station pairs, block times, MOGT, turn times | CSV |
| `CrewSeqData_2025_10_11_12.txt` | Crew sequence data for Oct–Dec 2025 | Tab-delimited |
| `CrewSeqData_2026_1.txt` | Crew sequence data for Jan 2026 | Tab-delimited |
| `airport_coordinates.csv` | Airport latitude/longitude for Airport nodes | CSV |
| `secrets.json` | Neo4j Aura credentials, Teradata credentials (encrypted) | JSON |

### Key Fields from Gemini Data

| Field | Description |
|-------|-------------|
| `LOF` | Line of Flying identifier — a multi-day sequence of flights assigned to a crew |
| `FLT_NUM` | Flight number |
| `DEP_STA` / `ARR_STA` | Departure and arrival IATA airport codes |
| `FLEET` / `EQP` / `BODY_TYPE` | Aircraft fleet, equipment type, narrow/wide body |
| `PER_SEQ` | Sequence position within the LOF |
| `SKD_BLK` | Scheduled block time (gate-to-gate) in minutes |
| `isDutyEndFlight` | Target label — True if this is the last flight before crew rest |
| `TURN_TIME` | Ground time between consecutive flights (minutes) |
| `MOGT` | Minimum Operationally required Ground Time (minutes) |
| `TIME_OVER_MOGT` | Excess ground time beyond MOGT (TURN_TIME − MOGT) |

---

## 4. Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│                                                                 │
│  gemini_with_duty_flag.csv                                      │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────┐    ┌─────────────────────────────┐        │
│  │ create_lof_graph │───▶│      Neo4j Aura DB          │        │
│  │     .py          │    │                             │        │
│  └──────────────────┘    │  Flight nodes (324K)        │        │
│                          │  Airport nodes (254)        │        │
│                          │  NEXT / DEPARTS_FROM /      │        │
│                          │  ARRIVES_AT relationships   │        │
│                          └────────────┬────────────────┘        │
│                                       │                         │
│                                       ▼                         │
│                          ┌─────────────────────────────┐        │
│                          │  GDS Feature Engineering    │        │
│                          │   (gds_feature_queries.cypher)│       │
│                          │                             │        │
│                          │  • Degree Centrality        │        │
│                          │  • PageRank                 │        │
│                          │  • Betweenness Centrality   │        │
│                          │  • ArticleRank              │        │
│                          │  • Louvain Communities      │        │
│                          │  • Label Propagation        │        │
│                          │  • Node2Vec Embeddings      │        │
│                          └────────────┬────────────────┘        │
│                                       │                         │
│                                       ▼                         │
│                          ┌─────────────────────────────┐        │
│                          │  PyTorch Neural Network     │        │
│                          │  (train_duty_end_model.py)  │        │
│                          │                             │        │
│                          │  128 → 64 → 32 hidden      │        │
│                          │  31 features                │        │
│                          │  Binary classification      │        │
│                          └────────────┬────────────────┘        │
│                                       │                         │
│                              ┌────────┴────────┐                │
│                              ▼                 ▼                │
│                   ┌──────────────┐   ┌──────────────────┐       │
│                   │  Model .pt   │   │ predictions.csv  │       │
│                   │  (saved)     │   │ (324K rows)      │       │
│                   └──────────────┘   └────────┬─────────┘       │
│                                               │                 │
│                                               ▼                 │
│                                    ┌──────────────────┐         │
│                                    │  Streamlit UI    │         │
│                                    │  + Azure OpenAI  │         │
│                                    │  Chat Agent      │         │
│                                    └──────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Graph Data Model

### Node Types

| Node | Properties | Count |
|------|-----------|-------|
| **Flight** | flight_number, lof, dep_station, arr_station, fleet, body_type, eqp, depDateTime, arr_lcl, isDutyEndFlight, sequence_index, skd_blk, turn_time, mogt, time_over_mogt, + all GDS-computed features | 324,192 |
| **Airport** | airport_code, latitude, longitude | 254 |

### Relationship Types

| Relationship | Pattern | Description |
|-------------|---------|-------------|
| **NEXT** | (Flight)→(Flight) | Sequential flights within the same LOF — the crew's flight-by-flight path |
| **DEPARTS_FROM** | (Flight)→(Airport) | Flight departs from this airport |
| **ARRIVES_AT** | (Flight)→(Airport) | Flight arrives at this airport |
| **CO_DEPARTS** | (Flight)↔(Flight) | Flights departing from the same station within 30 minutes (co-departure) |
| **CO_ARRIVES** | (Flight)↔(Flight) | Flights arriving at the same station within 30 minutes (co-arrival) |

### Graph Construction (`create_lof_graph.py`)

- Filters CSV to `SCHEDULE = 'Proposal 2025-10 F0'` and `ML_RG = 'ML'`
- Groups flights by LOF and EQP, orders by PER_SEQ
- Creates Flight nodes with all properties
- Creates NEXT relationships between consecutive flights in each LOF
- Links flights to existing Airport nodes via DEPARTS_FROM and ARRIVES_AT

---

## 6. Feature Engineering with Neo4j GDS

The Graph Data Science (GDS) library computes structural features directly on the Neo4j graph. These features capture patterns invisible to flat tabular data.

### 6.1 Graph Centrality Features

| Algorithm | Property Written | Graph Projected | Purpose |
|-----------|-----------------|-----------------|---------|
| **Degree Centrality** (departure) | `departureDegree` | CO_DEPARTS (undirected) | How many flights co-depart at the same station/time — measures station congestion |
| **Degree Centrality** (arrival) | `arrivalDegree` | CO_ARRIVES (undirected) | How many flights co-arrive — measures arrival bank density |
| **PageRank** | `sequencePageRank` | NEXT (directed) | Flight importance in the LOF chain — flights "pointed to" by many predecessors rank higher |
| **Betweenness Centrality** | `sequenceBetweenness` | NEXT (directed) | Flights that bridge different LOF segments — high betweenness may indicate transition points |
| **ArticleRank** | `articleRank` | NEXT (directed) | Variant of PageRank with less bias toward sink/source nodes — better for chain-like topologies |

### 6.2 Community Detection Features

| Algorithm | Property Written | Purpose |
|-----------|-----------------|---------|
| **Louvain** | `louvainCommunity` | Groups flights into communities based on connectivity — flights in the same community tend to share crew flow patterns |
| **Label Propagation** | `labelPropCommunity` | Faster community detection — complementary to Louvain for ensemble diversity |

### 6.3 Positional Features (LOF Sequence)

| Feature | Description |
|---------|-------------|
| `sequenceIndex` | 0-based position of the flight within its LOF |
| `lofSize` | Total number of flights in this LOF |
| `relativePosition` | sequenceIndex / max(sequenceIndex) — normalized position (0.0 = first, 1.0 = last) |
| `isLastInLof` | Binary: 1 if this is the final flight in the LOF |
| `isFirstInLof` | Binary: 1 if this is the first flight |
| `hasNextFlight` | Binary: 1 if a NEXT relationship exists (not the last in chain) |
| `hasPrevFlight` | Binary: 1 if a predecessor exists via NEXT (not the first) |

### 6.4 Temporal Features

| Feature | Description |
|---------|-------------|
| `depHour` | Departure hour (0–23) |
| `arrHour` | Arrival hour (0–23) |
| `depDayOfWeek` | 1=Monday … 7=Sunday |
| `isRedEye` | 1 if departure hour ≥ 21 (red-eye flight) |
| `isLateArrival` | 1 if arrival hour ≥ 18 (evening arrival — more likely duty-end) |
| `blockTimeMinutes` | Scheduled block time (gate-to-gate) in minutes |

### 6.5 Station Connectivity Features

| Feature | Description |
|---------|-------------|
| `depStationFlightCount` | Total flights departing from this station — measures hub size |
| `arrStationFlightCount` | Total flights arriving at this station |
| `depStationUniqueRoutes` | Number of unique destination airports from this departure station |

### 6.6 Fleet Features

| Feature | Description |
|---------|-------------|
| `fleet` | Aircraft type code (32T, 737, 319, etc.) |
| `body_type` | NB (Narrow Body) or WB (Wide Body) |
| `eqp` | Equipment subtype |
| `fleetPopularity` | Count of flights using this fleet type |

**Total features used by the model: 31** (24 numeric + 7 categorical, label-encoded)

---

## 7. Node2Vec Gaussian-Weighted Embeddings

An advanced embedding approach using **Gaussian-weighted random walks** on the flight sequence graph to capture overnight/RON (Remain Overnight) patterns.

### Concept

The NEXT relationships between flights have an associated `TURN_TIME` — the ground time between one flight's arrival and the next flight's departure. Long turn times (>8 hours) typically indicate an overnight rest.

A **Gaussian weight function** is applied to the turn time:

$$w = \exp\left(-\frac{(t - \mu)^2}{2\sigma^2}\right)$$

Where:
- $t$ = turn time in minutes
- $\mu$ = 240 minutes (4 hours) — ideal layover
- $\sigma$ = 120 minutes (2 hours) — spread

### Weight Interpretation

| Turn Time | Gaussian Weight | Interpretation |
|-----------|----------------|----------------|
| ~60 min (1 hr) | 0.32 | Tight connection — walks less likely to traverse |
| ~240 min (4 hr) | 1.00 | Ideal layover — walks prefer these edges |
| ~480 min (8 hr) | 0.32 | Long layover — potentially overnight |
| ~720 min (12 hr) | 0.04 | Almost certainly overnight — walks avoid these |

### Node2Vec Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `embeddingDimension` | 64 | Compact but expressive representation |
| `walkLength` | 10 | Captures multi-hop sequences within an LOF |
| `walksPerNode` | 20 | Sufficient coverage per flight |
| `returnFactor (p)` | 1.0 | Neutral about returning to prior node |
| `inOutFactor (q)` | 0.5 | Biased toward exploration — finds multi-hop overnight patterns |
| `relationshipWeightProperty` | gaussianWeight | Uses the Gaussian turn-time weights |

### Result

Flights with normal connections (~4 hr turns) cluster together in embedding space, while flights across overnight boundaries (long turn times) land in **different** embedding clusters. This makes the embeddings directly useful for predicting duty-end flights and RON detection.

---

## 8. Neural Network Model

### Architecture

```
Input (31 features)
    │
    ▼
Linear(31 → 128) → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Linear(64 → 32) → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Linear(32 → 1) → Sigmoid
    │
    ▼
Output: P(isDutyEndFlight)
```

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Hidden dimensions | [128, 64, 32] |
| Dropout rate | 0.3 |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss function | Weighted Binary Cross-Entropy (pos_weight = neg/pos ratio) |
| Batch size | 512 |
| Max epochs | 100 |
| Early stopping | Patience = 10 epochs (monitors validation loss) |
| Gradient clipping | max_norm = 1.0 |

### Data Splits

| Split | Size | Percentage |
|-------|------|------------|
| Training | ~220,450 | 68% |
| Validation | ~38,903 | 12% |
| Test | ~64,839 | 20% |

Stratified splitting ensures class balance is preserved in all splits.

### Preprocessing Pipeline

1. **Target encoding:** Boolean `isDutyEndFlight` → binary 0/1
2. **Numeric features:** `pd.to_numeric` with coerce, fill NaN with 0
3. **Categorical features:** `LabelEncoder` for each (dep_station, arr_station, fleet, body_type, eqp, louvainCommunity, labelPropCommunity)
4. **Infinity/overflow cleanup:** Replace inf → NaN → 0, clip extreme values to prevent float32 overflow
5. **Standard scaling:** Zero mean, unit variance via `StandardScaler`

---

## 9. Model Performance & Results

### Test Set Metrics

| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.893** |
| **F1 Score** | **0.800** |
| **Accuracy** | **81%** |
| **Optimal Threshold** | **0.412** |

### Classification Report

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Not Duty End (0) | 0.81 | 0.79 | 0.80 |
| Duty End (1) | 0.80 | 0.82 | 0.81 |
| **Weighted Avg** | **0.81** | **0.81** | **0.80** |

### Top 5 Most Important Features (Gradient-Based)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `depStationFlightCount` | Highest — hub stations have distinct duty-end patterns |
| 2 | `arrHour` | Late arrivals strongly correlate with duty-end |
| 3 | `arrStationFlightCount` | Arrival station connectivity matters |
| 4 | `depHour` | Early/late departures signal different duty structures |
| 5 | `blockTimeMinutes` | Longer flights more likely to end a duty period |

### Key Observations

- **Station connectivity features are the top predictors** — graph-derived features (station flight counts, degree centrality) outperform pure temporal features, validating the graph ML approach.
- **Arrival hour is a strong signal** — duty periods naturally end in the evening (18:00–23:00), making `arrHour` and `isLateArrival` highly predictive.
- **The model is well-calibrated** — with an optimal threshold of 0.412 (below 0.5), the model slightly favors predicting duty-end to capture more true positives.
- **Nearly balanced accuracy** — 81% for both classes, indicating no significant bias toward either class.

---

## 10. GENIE Agent & Prediction Export

**GENIE Agent** — **G**raph-based **E**arly **N**etwork **I**nefficiency **E**valuator — is the conversational AI agent built on top of the prediction pipeline.

### Prediction Export (`export_predictions.py`)

- Loads the trained `duty_end_flight_model.pt`
- Runs inference on all 324,192 flights from Neo4j
- Exports to `predictions_duty_end.csv` with 18 columns:

| Column | Description |
|--------|-------------|
| flight_number | AA flight number |
| lof | Line of Flying identifier |
| dep_station / arr_station | Origin and destination IATA codes |
| fleet / body_type / eqp | Aircraft information |
| depDateTime | Departure timestamp (ISO format) |
| arrDateTime | Arrival timestamp |
| depHour / arrHour / depDayOfWeek | Temporal features |
| blockTimeMinutes | Flight duration |
| sequenceIndex / lofSize | LOF position |
| actual_duty_end | Ground truth label |
| predicted_duty_end | Model prediction (0 or 1) |
| predicted_probability | Model confidence (0.0–1.0) |

### GENIE Agent Chat Interface

A **Streamlit-based chat UI** (`chat_ui.py`) with an Azure OpenAI GPT-4o backend allows natural language querying of the predictions:

- **"Show me duty-end predictions for DFW departures"**
- **"Which fleet type has the highest accuracy?"**
- **"What routes have the most duty-end flights?"**
- **"Give me predictions for October 15, 2025"**

The agent supports 9 tool functions: summary, by-airport, by-fleet, by-body-type, by-hour, by-route, by-date, misclassification analysis, and custom filters.

An offline fallback engine handles queries when Azure OpenAI credentials are not configured.

---

## 11. Project Files

| File | Purpose |
|------|---------|
| `create_lof_graph.py` | Loads CSV data into Neo4j as a flight sequence graph |
| `create_airport_nodes.py` | Creates Airport nodes from coordinates CSV |
| `gds_feature_queries.cypher` | 10-section GDS feature engineering (run in Neo4j Browser) |
| `node2vec_gaussian_embedding.cypher` | Gaussian-weighted Node2Vec embeddings (run in Neo4j Browser) |
| `train_duty_end_model.py` | End-to-end ML pipeline: extract → preprocess → train → evaluate → save |
| `export_predictions.py` | Run inference on all flights, export predictions to CSV |
| `chat_ui.py` | GENIE Agent Streamlit chat UI with Azure OpenAI function calling |
| `agent_chat.py` | GENIE Agent CLI chat (alternative to Streamlit UI) |
| `duty_end_flight_model.pt` | Saved trained PyTorch model |
| `predictions_duty_end.csv` | 324,192 rows of flight predictions |
| `gemini_with_duty_flag.csv` | Source crew scheduling data |
| `airport_coordinates.csv` | Airport lat/long data |
| `secrets.json` | Credentials (Neo4j, Azure OpenAI) |

---

## 12. How to Run

### Prerequisites

```bash
pip install torch pandas numpy scikit-learn neo4j streamlit openai
```

### Step-by-Step

```bash
# 1. Load flight data into Neo4j
python create_lof_graph.py

# 2. Run GDS feature queries in Neo4j Browser (copy-paste sections sequentially)
#    File: gds_feature_queries.cypher

# 3. (Optional) Run Node2Vec Gaussian embeddings in Neo4j Browser
#    File: node2vec_gaussian_embedding.cypher

# 4. Train the neural network
python train_duty_end_model.py

# 5. Export predictions to CSV
python export_predictions.py

# 6. Launch GENIE Agent chat UI
streamlit run chat_ui.py
```

### Configuration

Add Azure OpenAI credentials to `secrets.json` for full natural language chat support:

```json
{
  "azure_openai": {
    "endpoint": "https://your-resource.openai.azure.com/",
    "api_key": "your-api-key",
    "deployment": "gpt-4o",
    "api_version": "2024-12-01-preview"
  }
}
```

---

*Built with Neo4j Graph Data Science, PyTorch, and GENIE Agent (Azure AI Foundry) for the Neo4j Graphathon.*
