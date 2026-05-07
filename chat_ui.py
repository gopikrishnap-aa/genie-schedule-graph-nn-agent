"""
Streamlit Chat UI for GENIE — Graph-based Early Network Inefficiency Evaluator.

A production-quality chat interface to query model predictions using
Azure OpenAI (GPT-4o) with function calling.

Run:
  streamlit run chat_ui.py

Prerequisites:
  1. pip install streamlit openai
  2. Run export_predictions.py to generate predictions_duty_end.csv
  3. (Optional) Set Azure OpenAI credentials in secrets.json
"""

import json
import os
import re

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_PATH = os.path.join(BASE_DIR, "secrets.json")
PREDICTIONS_CSV = os.path.join(BASE_DIR, "predictions_duty_end.csv")

# ── Brand palette (light, clean, professional) ──────────────
NAVY        = "#f8f9fb"       # page background — very light grey
PANEL       = "#ffffff"       # sidebar background — white
CARD        = "#ffffff"       # card / chat bubble background — white
ACCENT      = "#0066cc"       # AA blue
ACCENT_DIM  = "rgba(0,102,204,0.08)"
GOLD        = "#e0920f"
GREEN       = "#0e8a5f"
RED         = "#d1344b"
TXT         = "#1a1a2e"       # primary text — near black
TXT2        = "#5a6577"       # secondary text — medium grey
BORDER      = "#dfe3ea"       # borders — light grey


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Graph-based Early Network Inefficiency AI Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Global CSS ───────────────────────────────────────────────
st.markdown(f"""
<style>
/* ---------- foundations ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, .stApp {{font-family:'Inter',sans-serif;}}
.stApp {{background:{NAVY};}}
.block-container {{padding:1rem 2rem 2rem 2rem; max-width:1200px;}}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{background:{PANEL};border-right:1px solid {BORDER};}}
section[data-testid="stSidebar"] [data-testid="stMetric"] {{
    background:#f1f4f8;border:1px solid {BORDER};border-radius:10px;padding:0.65rem 0.75rem;
}}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {{color:{TXT};font-size:1.15rem;}}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {{color:{TXT2};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.04em;}}
section[data-testid="stSidebar"] .stMarkdown h4 {{color:{TXT};font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;margin:1.2rem 0 0.4rem 0;}}
section[data-testid="stSidebar"] .stMarkdown p {{color:{TXT2};font-size:0.8rem;line-height:1.55;}}

/* sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {{
    background:#f1f4f8 !important;border:1px solid {BORDER} !important;color:{TXT2} !important;
    border-radius:8px !important;font-size:0.78rem !important;
    padding:0.55rem 0.7rem !important;text-align:left !important;
    transition:all 0.15s ease !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    border-color:{ACCENT} !important;color:{ACCENT} !important;background:{ACCENT_DIM} !important;
}}

/* ---------- header banner ---------- */
.hero {{
    background:linear-gradient(135deg,#003876 0%,#0055a4 60%,#003876 100%);
    border:none;border-radius:14px;
    padding:1.3rem 1.8rem;margin-bottom:1rem;
    display:flex;align-items:center;gap:1rem;
    box-shadow:0 2px 8px rgba(0,0,0,0.1);
}}
.hero-icon {{font-size:2rem;}}
.hero h1 {{color:#ffffff;font-size:1.35rem;font-weight:700;margin:0;letter-spacing:-0.01em;}}
.hero p {{color:#b8d4f0;font-size:0.82rem;margin:0.1rem 0 0;}}

/* ---------- KPI strip ---------- */
.kpi-strip {{display:grid;grid-template-columns:repeat(4,1fr);gap:0.65rem;margin-bottom:1rem;}}
.kpi {{
    background:{CARD};border:1px solid {BORDER};border-radius:11px;
    padding:0.9rem 1rem;position:relative;overflow:hidden;
    box-shadow:0 1px 3px rgba(0,0,0,0.04);
}}
.kpi::after {{content:'';position:absolute;top:0;left:0;right:0;height:3px;}}
.kpi.c1::after {{background:{ACCENT};}}
.kpi.c2::after {{background:{GOLD};}}
.kpi.c3::after {{background:{GREEN};}}
.kpi.c4::after {{background:{RED};}}
.kpi .lab {{color:{TXT2};font-size:0.65rem;text-transform:uppercase;letter-spacing:0.07em;margin:0;}}
.kpi .val {{color:{TXT};font-size:1.45rem;font-weight:700;margin:0.15rem 0 0;letter-spacing:-0.02em;}}
.kpi .sub {{color:{TXT2};font-size:0.65rem;margin:0.1rem 0 0;}}

/* ---------- welcome ---------- */
.welcome {{
    background:{CARD};border:1px solid {BORDER};border-radius:14px;
    padding:2.5rem 2rem;text-align:center;margin:1.5rem 0;
    box-shadow:0 1px 3px rgba(0,0,0,0.04);
}}
.welcome h2 {{color:{TXT};font-size:1.15rem;font-weight:600;margin:0 0 0.25rem;}}
.welcome p  {{color:{TXT2};font-size:0.85rem;margin:0 0 1.5rem;}}
.chips {{display:flex;flex-wrap:wrap;gap:0.5rem;justify-content:center;max-width:680px;margin:0 auto;}}
.chip {{
    background:{ACCENT_DIM};border:1px solid {BORDER};border-radius:20px;
    padding:0.45rem 1rem;color:{TXT2};font-size:0.78rem;white-space:nowrap;
}}

/* ---------- chat ---------- */
.stChatMessage {{
    background:{CARD} !important;border:1px solid {BORDER} !important;
    border-radius:12px !important;margin-bottom:0.4rem !important;
    box-shadow:0 1px 2px rgba(0,0,0,0.03) !important;
}}
.stChatMessage p, .stChatMessage li {{color:{TXT} !important;}}
.stChatMessage table {{color:{TXT} !important;border-collapse:collapse;width:100%;}}
.stChatMessage th {{color:{ACCENT} !important;border-bottom:2px solid {BORDER} !important;text-align:left;padding:0.4rem 0.6rem;font-size:0.8rem;font-weight:600;}}
.stChatMessage td {{border-bottom:1px solid #eef0f4 !important;padding:0.35rem 0.6rem;font-size:0.8rem;color:#2d3748 !important;}}
.stChatMessage strong {{color:{TXT} !important;}}
.stChatMessage h3 {{color:{ACCENT} !important;font-size:1rem;margin:0.6rem 0 0.4rem;}}
.stChatMessage blockquote {{border-left:3px solid {ACCENT} !important;background:#f0f6ff !important;padding:0.5rem 1rem !important;border-radius:0 8px 8px 0 !important;}}
.stChatMessage blockquote p {{color:{TXT2} !important;}}

/* ---------- chat input ---------- */
[data-testid="stChatInput"] > div {{
    border:2px solid {BORDER} !important;border-radius:12px !important;
    background:#ffffff !important;box-shadow:0 1px 4px rgba(0,0,0,0.06) !important;
}}
[data-testid="stChatInput"] > div:focus-within {{
    border-color:{ACCENT} !important;box-shadow:0 0 0 3px rgba(0,102,204,0.15) !important;
}}
[data-testid="stChatInput"] textarea {{
    color:#1a1a2e !important;font-size:0.9rem !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color:#9ca3af !important;
}}

/* ---------- misc ---------- */
hr {{border-color:{BORDER} !important;opacity:0.5;}}
::-webkit-scrollbar {{width:5px;}}
::-webkit-scrollbar-track {{background:#f8f9fb;}}
::-webkit-scrollbar-thumb {{background:#ccd1db;border-radius:3px;}}
::-webkit-scrollbar-thumb:hover {{background:{ACCENT};}}
</style>
""", unsafe_allow_html=True)


# ── Data ─────────────────────────────────────────────────────
@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Load predictions CSV."""
    if not os.path.exists(PREDICTIONS_CSV):
        st.error("Predictions file not found. Run `python export_predictions.py` first.")
        st.stop()
    return pd.read_csv(PREDICTIONS_CSV)


df = load_predictions()
N = len(df)
PRED_T = int(df["predicted_duty_end"].sum())
PRED_F = N - PRED_T
ACC = (df["predicted_duty_end"] == df["actual_duty_end"]).mean()
MISS = N - int((df["predicted_duty_end"] == df["actual_duty_end"]).sum())


# ── Query helpers ────────────────────────────────────────────
def _summary(df):
    return json.dumps({"total_flights": N, "predicted_duty_end_true": PRED_T, "predicted_duty_end_false": PRED_F, "actual_duty_end_true": int(df["actual_duty_end"].sum()), "accuracy": round(ACC, 4), "average_predicted_probability": round(df["predicted_probability"].mean(), 4)})

def _by_airport(df, col="arr_station", n=10):
    g = df.groupby(col).agg(total=("predicted_duty_end","count"), predicted=("predicted_duty_end","sum"), actual=("actual_duty_end","sum"), prob=("predicted_probability","mean")).sort_values("predicted", ascending=False).head(n)
    return g.reset_index().to_json(orient="records")

def _by_fleet(df):
    g = df.groupby("fleet").agg(total=("predicted_duty_end","count"), predicted=("predicted_duty_end","sum"), actual=("actual_duty_end","sum"), prob=("predicted_probability","mean")).sort_values("predicted", ascending=False)
    return g.reset_index().to_json(orient="records")

def _by_body(df):
    g = df.groupby("body_type").agg(total=("predicted_duty_end","count"), predicted=("predicted_duty_end","sum"), actual=("actual_duty_end","sum"), prob=("predicted_probability","mean")).sort_values("predicted", ascending=False)
    return g.reset_index().to_json(orient="records")

def _by_hour(df, col="depHour"):
    col = col if col in df.columns else "depHour"
    g = df.groupby(col).agg(total=("predicted_duty_end","count"), predicted=("predicted_duty_end","sum"), pct=("predicted_duty_end","mean")).sort_index()
    g["pct"] = (g["pct"]*100).round(1)
    return g.reset_index().to_json(orient="records")

def _by_route(df, n=15):
    dc = df.copy(); dc["route"] = dc["dep_station"]+" → "+dc["arr_station"]
    g = dc.groupby("route").agg(total=("predicted_duty_end","count"), predicted=("predicted_duty_end","sum"), pct=("predicted_duty_end","mean"), prob=("predicted_probability","mean")).sort_values("predicted", ascending=False).head(n)
    g["pct"] = (g["pct"]*100).round(1)
    return g.reset_index().to_json(orient="records")

def _misclass(df, etype="all", n=20):
    m = df[df["predicted_duty_end"] != df["actual_duty_end"]]
    if etype == "false_positive": m = m[(m["predicted_duty_end"]==1)&(m["actual_duty_end"]==0)]
    elif etype == "false_negative": m = m[(m["predicted_duty_end"]==0)&(m["actual_duty_end"]==1)]
    fp = int(((m["predicted_duty_end"]==1)&(m["actual_duty_end"]==0)).sum())
    fn = int(((m["predicted_duty_end"]==0)&(m["actual_duty_end"]==1)).sum())
    return json.dumps({"total_misclassified": len(m), "false_positives": fp, "false_negatives": fn, "sample": m.head(n).to_dict(orient="records")}, default=str)

def _filter(df, filters):
    f = df.copy()
    for c,v in filters.items():
        if c in f.columns:
            f = f[f[c].isin(v)] if isinstance(v, list) else f[f[c]==v]
    if f.empty: return json.dumps({"error": "No flights match filters"})
    return json.dumps({"filters_applied": filters, "matching_flights": len(f), "predicted_duty_end_true": int(f["predicted_duty_end"].sum()), "predicted_duty_end_false": int((f["predicted_duty_end"]==0).sum()), "actual_duty_end_true": int(f["actual_duty_end"].sum()), "accuracy": round((f["predicted_duty_end"]==f["actual_duty_end"]).mean(),4), "avg_probability": round(f["predicted_probability"].mean(),4)})

def _by_date(df, date_str, station=None, station_type=None):
    """Filter flights by date (and optionally station). date_str like '2025-10-15' or '2025-10'."""
    f = df.copy()
    if "depDateTime" not in f.columns:
        return json.dumps({"error": "depDateTime column not available. Re-run export_predictions.py."})
    f["depDateTime"] = f["depDateTime"].astype(str)
    f = f[f["depDateTime"].str.startswith(date_str)]
    if station and station_type:
        f = f[f[station_type] == station]
    if f.empty:
        return json.dumps({"error": f"No flights found for date '{date_str}'"})
    result = {
        "date_filter": date_str,
        "matching_flights": len(f),
        "predicted_duty_end_true": int(f["predicted_duty_end"].sum()),
        "predicted_duty_end_false": int((f["predicted_duty_end"]==0).sum()),
        "actual_duty_end_true": int(f["actual_duty_end"].sum()),
        "accuracy": round((f["predicted_duty_end"]==f["actual_duty_end"]).mean(), 4),
        "avg_probability": round(f["predicted_probability"].mean(), 4),
        "sample": f.head(20).to_dict(orient="records"),
    }
    if station:
        result["station_filter"] = f"{station_type}={station}"
    return json.dumps(result, default=str)


TOOLS = [
    {"type":"function","function":{"name":"get_prediction_summary","description":"Overall summary.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"get_predictions_by_airport","description":"By airport.","parameters":{"type":"object","properties":{"airport_type":{"type":"string","enum":["dep_station","arr_station"]},"top_n":{"type":"integer"}},"required":[]}}},
    {"type":"function","function":{"name":"get_predictions_by_fleet","description":"By fleet.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"get_predictions_by_body_type","description":"By body type.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"get_predictions_by_hour","description":"By hour.","parameters":{"type":"object","properties":{"hour_type":{"type":"string","enum":["depHour","arrHour"]}},"required":[]}}},
    {"type":"function","function":{"name":"get_predictions_by_route","description":"By route.","parameters":{"type":"object","properties":{"top_n":{"type":"integer"}},"required":[]}}},
    {"type":"function","function":{"name":"get_misclassified_flights","description":"Errors.","parameters":{"type":"object","properties":{"error_type":{"type":"string","enum":["all","false_positive","false_negative"]},"top_n":{"type":"integer"}},"required":[]}}},
    {"type":"function","function":{"name":"filter_and_query","description":"Custom filter.","parameters":{"type":"object","properties":{"filters":{"type":"object"}},"required":["filters"]}}},
    {"type":"function","function":{"name":"get_predictions_by_date","description":"Get predictions for a specific date or month. Use date_str like '2025-10-15' for a day or '2025-10' for a month. Optionally filter by station.","parameters":{"type":"object","properties":{"date_str":{"type":"string","description":"Date prefix to filter, e.g. '2025-10-15' or '2025-10' or '2025-11'"},"station":{"type":"string","description":"Optional 3-letter airport code"},"station_type":{"type":"string","enum":["dep_station","arr_station"],"description":"Whether station is departure or arrival"}},"required":["date_str"]}}},
]

SYSTEM_PROMPT = """You are GENIE Agent (Graph-based Early Network Inefficiency Evaluator), an AI assistant for American Airlines crew scheduling.
You answer questions about predictions from a neural network model that predicts whether a flight
is the last flight in a crew duty period (isDutyEndFlight).

## Model Details
- Architecture: PyTorch Neural Network (128 → 64 → 32 hidden layers, dropout 0.3)
- Performance: AUC-ROC 0.893, F1 0.800, Accuracy 81%, optimal threshold 0.412
- Training data: 324,192 flights from a Neo4j graph database
- 31 features: graph centrality (PageRank, Betweenness), community detection (Louvain, Label Propagation),
  temporal (hour, day of week), station connectivity, fleet/aircraft type, LOF sequence position

## Dataset Schema — predictions_duty_end.csv
Each row is one scheduled flight. Columns:

| Column | Type | Description |
|--------|------|-------------|
| flight_number | int | AA flight number (e.g., 117, 179) |
| lof | int | Line-of-Flying identifier — a sequence of flights forming a crew trip |
| dep_station | string | **Departure** airport IATA code (e.g., DFW, ORD, LAX). Also called origin. |
| arr_station | string | **Arrival** airport IATA code (e.g., MIA, CLT, JFK). Also called destination. |
| fleet | string | Fleet/aircraft type code: 32T, 737, 319, 321, 777, 787, 320, 32X |
| body_type | string | Aircraft body: NB = Narrow Body, WB = Wide Body |
| eqp | string | Equipment type code (same granularity as fleet) |
| depDateTime | string | Departure date+time in ISO format (e.g., "2025-10-02T19:23:00Z"). Use this for date-based filtering. |
| arrDateTime | string | Arrival date+time (e.g., "10/5/2025 23:11:00", format M/d/yyyy HH:mm:ss). Use this for arrival date filtering. |
| depHour | int | Departure hour (0–23, local time) |
| arrHour | int | Arrival hour (0–23, local time) |
| depDayOfWeek | int | Departure day of week (1=Monday … 7=Sunday) |
| blockTimeMinutes | int | Scheduled block time in minutes (flight duration gate-to-gate) |
| sequenceIndex | int | Position of this flight within its LOF (0-based) |
| lofSize | int | Total number of flights in this LOF |
| actual_duty_end | int | Ground truth: 1 = this flight IS the last flight in a duty period, 0 = it is not |
| predicted_duty_end | int | Model prediction: 1 = predicted duty-end, 0 = predicted non-duty-end (threshold 0.412) |
| predicted_probability | float | Model confidence (0.0–1.0). Higher = more likely to be a duty-end flight |

## Key Domain Terminology
- **Duty period**: A crew's continuous working period (multiple flights), ending with a rest
- **Duty-end flight**: The LAST flight a crew flies before their required rest period
- **LOF (Line of Flying)**: A multi-day sequence of flights assigned to a crew
- **RON (Remain Overnight)**: When crew stays overnight at a station
- **dep_station / departure / origin**: Where the flight takes off FROM
- **arr_station / arrival / destination**: Where the flight lands AT
- **NB (Narrow Body)**: Single-aisle aircraft (A319, A320, A321, 737)
- **WB (Wide Body)**: Twin-aisle aircraft (777, 787)

## Data Summary
- 324,192 total flights across 254 unique airports
- Predicted duty-end: 163,857 (50.5%) | Non-duty-end: 160,335 (49.5%)
- Actual duty-end: 164,035 | Actual non-duty-end: 160,157
- Nearly balanced classes (0.98 neg/pos ratio)

## Data Coverage — Monthly Flight Counts
The dataset covers **October, November, and December 2025 ONLY** (schedule proposal 2025-10 F0).
There is NO data for January through September 2025 or any other year.

| Month | Flights |
|-------|--------:|
| October 2025 | 101,417 |
| November 2025 | 99,151 |
| December 2025 | 106,738 |
| **Total** | **324,192** |

- depDateTime prefix for October: "2025-10"
- depDateTime prefix for November: "2025-11"
- depDateTime prefix for December: "2025-12"
- If user asks about months outside Oct–Dec 2025, inform them no data is available for those months.

## Response Guidelines
- Use the available tools to query data. Never guess numbers.
- Format responses with markdown tables when showing grouped data.
- When users say "departure", "origin", "from", or "departing" → they mean dep_station.
- When users say "arrival", "destination", "to", "landing", or "arriving" → they mean arr_station.
- When users say "aircraft", "plane", or "equipment" → check both fleet and body_type.
- Day of week: 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun.
- When users ask for a specific DATE (e.g., "Oct 15" or "2025-10-15"), filter using depDateTime which contains ISO timestamps like "2025-10-15T...". Extract the date portion (first 10 chars) for comparison.
- When users say "in October" or "October 2025", filter depDateTime strings that start with "2025-10".
- Always include context (e.g., "out of X total flights") to give perspective.
- Suggest 1-2 follow-up questions the user might want to ask next.
- Be concise but complete. Use bullet points for summaries, tables for comparisons."""

def _exec_tool(name, args, df):
    d = {"get_prediction_summary": lambda: _summary(df), "get_predictions_by_airport": lambda: _by_airport(df,args.get("airport_type","arr_station"),args.get("top_n",10)), "get_predictions_by_fleet": lambda: _by_fleet(df), "get_predictions_by_body_type": lambda: _by_body(df), "get_predictions_by_hour": lambda: _by_hour(df,args.get("hour_type","depHour")), "get_predictions_by_route": lambda: _by_route(df,args.get("top_n",15)), "get_misclassified_flights": lambda: _misclass(df,args.get("error_type","all"),args.get("top_n",20)), "filter_and_query": lambda: _filter(df,args.get("filters",{})), "get_predictions_by_date": lambda: _by_date(df,args.get("date_str",""),args.get("station"),args.get("station_type"))}
    return d.get(name, lambda: '{"error":"unknown"}')()


@st.cache_resource
def _azure_client():
    try:
        with open(SECRETS_PATH) as f: cfg = json.load(f).get("azure_openai",{})
    except Exception: return None, "gpt-4o"
    ep = cfg.get("endpoint", os.environ.get("AZURE_OPENAI_ENDPOINT",""))
    key = cfg.get("api_key", os.environ.get("AZURE_OPENAI_API_KEY",""))
    dep = cfg.get("deployment","gpt-4o")
    if not ep or not key: return None, dep
    from openai import AzureOpenAI
    return AzureOpenAI(azure_endpoint=ep, api_key=key, api_version=cfg.get("api_version","2024-12-01-preview")), dep


# ── Offline response engine ──────────────────────────────────
def offline_reply(prompt, df):
    pl = prompt.lower()
    if any(k in pl for k in ["summary","how many","total","count","overview"]):
        d = json.loads(_summary(df))
        return f"### Prediction Summary\n\n| Metric | Value |\n|:---|---:|\n| Total Flights | **{d['total_flights']:,}** |\n| Predicted Duty-End | **{d['predicted_duty_end_true']:,}** |\n| Predicted Non-Duty-End | **{d['predicted_duty_end_false']:,}** |\n| Actual Duty-End | **{d['actual_duty_end_true']:,}** |\n| Accuracy | **{d['accuracy']:.1%}** |\n| Avg Probability | **{d['average_predicted_probability']:.3f}** |"
    if any(k in pl for k in ["airport","destination","arrival","station","landing"]):
        at = "dep_station" if any(k in pl for k in ["departure","dep","origin","takeoff"]) else "arr_station"
        data = json.loads(_by_airport(df,at,10)); lb = "Departure" if at=="dep_station" else "Arrival"
        t = f"### Top 10 {lb} Airports\n\n| Station | Flights | Predicted | Actual | Avg Prob |\n|:---|---:|---:|---:|---:|\n"
        for r in data: t += f"| **{r[at]}** | {int(r['total']):,} | {int(r['predicted']):,} | {int(r['actual']):,} | {r['prob']:.3f} |\n"
        return t
    if "fleet" in pl:
        data = json.loads(_by_fleet(df))
        t = "### Predictions by Fleet\n\n| Fleet | Flights | Predicted | Actual | Avg Prob |\n|:---|---:|---:|---:|---:|\n"
        for r in data: t += f"| **{r['fleet']}** | {int(r['total']):,} | {int(r['predicted']):,} | {int(r['actual']):,} | {r['prob']:.3f} |\n"
        return t
    if any(k in pl for k in ["body","narrow","wide"]):
        data = json.loads(_by_body(df))
        t = "### Predictions by Body Type\n\n| Type | Flights | Predicted | Actual | Avg Prob |\n|:---|---:|---:|---:|---:|\n"
        for r in data: t += f"| **{r['body_type']}** | {int(r['total']):,} | {int(r['predicted']):,} | {int(r['actual']):,} | {r['prob']:.3f} |\n"
        return t
    if any(k in pl for k in ["route","pair","od "]):
        data = json.loads(_by_route(df,12))
        t = "### Top Routes\n\n| Route | Flights | Predicted | Rate | Avg Prob |\n|:---|---:|---:|---:|---:|\n"
        for r in data: t += f"| **{r['route']}** | {int(r['total']):,} | {int(r['predicted']):,} | {r['pct']}% | {r['prob']:.3f} |\n"
        return t
    if any(k in pl for k in ["hour","time","when","clock"]):
        ht = "arrHour" if "arr" in pl else "depHour"
        data = json.loads(_by_hour(df,ht)); lb = "Arrival" if ht=="arrHour" else "Departure"
        t = f"### Predictions by {lb} Hour\n\n| Hour | Flights | Predicted | Rate |\n|:---|---:|---:|---:|\n"
        for r in data: t += f"| **{int(r[ht]):02d}:00** | {int(r['total']):,} | {int(r['predicted']):,} | {r['pct']}% |\n"
        return t
    if any(k in pl for k in ["error","wrong","mistake","misclass","false"]):
        et = "false_positive" if "false pos" in pl else ("false_negative" if "false neg" in pl else "all")
        d = json.loads(_misclass(df,et))
        return f"### Misclassification Analysis\n\n| Metric | Count |\n|:---|---:|\n| Total Errors | **{d['total_misclassified']:,}** |\n| False Positives | **{d['false_positives']:,}** |\n| False Negatives | **{d['false_negatives']:,}** |\n\n> *FP = predicted duty-end but wasn't &bull; FN = missed a real duty-end*"
    m = re.search(r'\b([A-Z]{3})\b', prompt)
    if m:
        s = m.group(1)
        for c in ["dep_station","arr_station"]:
            if s in df[c].values:
                r = json.loads(_filter(df,{c:s}))
                if "error" not in r:
                    lb = "departing" if c=="dep_station" else "arriving at"
                    return f"### Flights {lb} {s}\n\n| Metric | Value |\n|:---|---:|\n| Matching | **{r['matching_flights']:,}** |\n| Predicted Duty-End | **{r['predicted_duty_end_true']:,}** |\n| Non-Duty-End | **{r['predicted_duty_end_false']:,}** |\n| Accuracy | **{r['accuracy']:.1%}** |\n| Avg Probability | **{r['avg_probability']:.3f}** |"
    return "I can answer questions about:\n\n- **Counts** — *How many flights are predicted duty-end?*\n- **Airports** — *Which airport has the most duty-end flights?*\n- **Fleet** — *Show predictions by fleet type*\n- **Routes** — *Top routes by duty-end rate*\n- **Time** — *Duty-end flights by hour*\n- **Errors** — *Where is the model wrong?*\n- **Stations** — Just type a 3-letter code like *DFW*\n\n> Add Azure OpenAI credentials to `secrets.json` for full natural language support."


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:0.8rem 0 0.3rem;">
        <span style="font-size:2rem;">✈️</span><br/>
        <span style="color:{TXT};font-weight:700;font-size:0.95rem;">GENIE Agent</span><br/>
        <span style="color:{TXT2};font-size:0.72rem;">Graph-based Early Network Inefficiency Evaluator</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Model Metrics")
    c1, c2 = st.columns(2)
    c1.metric("AUC-ROC", "0.893")
    c2.metric("F1 Score", "0.800")
    c3, c4 = st.columns(2)
    c3.metric("Accuracy", f"{ACC:.1%}")
    c4.metric("Threshold", "0.412")

    st.markdown("---")
    st.markdown("#### Quick Questions")
    qs = [
        ("📊","How many flights are predicted as duty-end?"),
        ("🛬","Which arrival airport has the most duty-end predictions?"),
        ("🔧","Show accuracy breakdown by fleet type"),
        ("🗺️","Top routes with highest duty-end rate"),
        ("🕐","What departure hour has the most duty-ends?"),
        ("✈️","Show predictions for DFW departures"),
        ("❌","Where is the model making mistakes?"),
        ("📈","Compare narrow-body vs wide-body"),
    ]
    for ico, q in qs:
        if st.button(f"{ico}  {q}", key=f"sb_{hash(q)}", use_container_width=True):
            st.session_state["pending_q"] = q

    st.markdown("---")
    st.markdown("#### Architecture")
    st.markdown(f"""<div style="color:{TXT2};font-size:0.76rem;line-height:1.7;">
    <b style="color:{TXT}">Neural Net</b> 128 → 64 → 32<br/>
    <b style="color:{TXT}">Features</b> 31 (graph + temporal)<br/>
    <b style="color:{TXT}">Data</b> 324K flights · Neo4j<br/>
    <b style="color:{TXT}">Agent</b> GENIE Agent · Azure AI Foundry
    </div>""", unsafe_allow_html=True)

    st.markdown("")
    if st.button("🗑  Clear conversation", key="clr", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════

# Hero
st.markdown(f"""
<div class="hero">
    <div class="hero-icon">✈️</div>
    <div>
        <h1>Graph-based Early Network Inefficiency AI Agent</h1>
        <p>GENIE Agent &bull; Neo4j Graph ML &amp; Azure AI Foundry</p>
    </div>
</div>""", unsafe_allow_html=True)

# KPI
st.markdown(f"""
<div class="kpi-strip" style="grid-template-columns:repeat(3,1fr);">
    <div class="kpi c1"><p class="lab">Total Flights</p><p class="val">{N:,}</p><p class="sub">analyzed by the model</p></div>
    <div class="kpi c2"><p class="lab">Predicted Duty-End</p><p class="val">{PRED_T:,}</p><p class="sub">{PRED_T/N*100:.1f}% of all flights</p></div>
    <div class="kpi c3"><p class="lab">Model Accuracy</p><p class="val">{ACC:.1%}</p><p class="sub">test evaluation</p></div>
</div>""", unsafe_allow_html=True)

# Init
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Welcome
if not st.session_state["messages"]:
    st.markdown(f"""
    <div class="welcome">
        <h2>Welcome! Ask me anything about the predictions.</h2>
        <p>I've analyzed {N:,} flight predictions. Try a question below or type your own.</p>
        <div class="chips">
            <div class="chip">📊 Prediction summary</div>
            <div class="chip">🛬 Top duty-end airports</div>
            <div class="chip">🗺️ Highest duty-end routes</div>
            <div class="chip">🕐 Patterns by time of day</div>
            <div class="chip">🔧 Breakdown by fleet</div>
            <div class="chip">❌ Misclassification analysis</div>
        </div>
    </div>""", unsafe_allow_html=True)

# History
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar="✈️" if msg["role"]=="assistant" else "👤"):
        st.markdown(msg["content"])

# Pending sidebar click
if "pending_q" in st.session_state:
    st.session_state["messages"].append({"role":"user","content": st.session_state.pop("pending_q")})
    st.rerun()

# Input
if prompt := st.chat_input("Ask about flight duty-end predictions..."):
    st.session_state["messages"].append({"role":"user","content":prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✈️"):
        with st.spinner("Analyzing..."):
            client, dep = _azure_client()
            if client is None:
                reply = offline_reply(prompt, df)
            else:
                msgs = [{"role":"system","content":SYSTEM_PROMPT}] + [{"role":m["role"],"content":m["content"]} for m in st.session_state["messages"]]
                resp = client.chat.completions.create(model=dep, messages=msgs, tools=TOOLS, tool_choice="auto", temperature=0.1)
                am = resp.choices[0].message
                while am.tool_calls:
                    msgs.append(am)
                    for tc in am.tool_calls:
                        msgs.append({"role":"tool","tool_call_id":tc.id,"content":_exec_tool(tc.function.name, json.loads(tc.function.arguments), df)})
                    resp = client.chat.completions.create(model=dep, messages=msgs, tools=TOOLS, tool_choice="auto", temperature=0.1)
                    am = resp.choices[0].message
                reply = am.content

            st.markdown(reply)
            st.session_state["messages"].append({"role":"assistant","content":reply})
