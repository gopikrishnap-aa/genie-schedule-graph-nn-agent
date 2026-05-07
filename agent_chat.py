"""
GENIE (Graph-based Early Network Inefficiency Evaluator) — CLI Chat Interface.

Provides a conversational interface to query model predictions using
Azure OpenAI (GPT-4o) with function calling. The agent can answer
questions like:
  - "How many flights are predicted as duty-end?"
  - "Which destination airport has the most predicted duty-end flights?"
  - "Show me the accuracy by fleet type"
  - "What's the average predicted probability for red-eye flights?"

Prerequisites:
  1. Run export_predictions.py first to generate predictions_duty_end.csv
  2. Set Azure OpenAI credentials in secrets.json under "azure_openai" key

Usage:
  python agent_chat.py
"""

import json
import os
import logging
from typing import Optional

import pandas as pd
from openai import AzureOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_PATH = os.path.join(BASE_DIR, "secrets.json")
PREDICTIONS_CSV = os.path.join(BASE_DIR, "predictions_duty_end.csv")


# ==========================================
# DATA LAYER - Query functions for the agent
# ==========================================


def load_predictions() -> pd.DataFrame:
    """Load the predictions CSV into a DataFrame."""
    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_CSV}\n"
            "Run 'python export_predictions.py' first."
        )
    return pd.read_csv(PREDICTIONS_CSV)


def get_prediction_summary(df: pd.DataFrame) -> str:
    """Get overall prediction summary statistics."""
    total = len(df)
    pred_true = df["predicted_duty_end"].sum()
    pred_false = total - pred_true
    actual_true = df["actual_duty_end"].sum()
    accuracy = (df["predicted_duty_end"] == df["actual_duty_end"]).mean()
    avg_prob = df["predicted_probability"].mean()

    return json.dumps({
        "total_flights": int(total),
        "predicted_duty_end_true": int(pred_true),
        "predicted_duty_end_false": int(pred_false),
        "actual_duty_end_true": int(actual_true),
        "accuracy": round(accuracy, 4),
        "average_predicted_probability": round(avg_prob, 4),
    })


def get_predictions_by_airport(df: pd.DataFrame, airport_type: str = "arr_station", top_n: int = 10) -> str:
    """Get prediction counts grouped by airport (departure or arrival)."""
    col = airport_type if airport_type in df.columns else "arr_station"
    grouped = df.groupby(col).agg(
        total_flights=("predicted_duty_end", "count"),
        predicted_duty_end=("predicted_duty_end", "sum"),
        actual_duty_end=("actual_duty_end", "sum"),
        avg_probability=("predicted_probability", "mean"),
    ).sort_values("predicted_duty_end", ascending=False).head(top_n)

    grouped["accuracy"] = ((df.groupby(col).apply(
        lambda x: (x["predicted_duty_end"] == x["actual_duty_end"]).mean()
    )).reindex(grouped.index)).round(4)

    return grouped.reset_index().to_json(orient="records")


def get_predictions_by_fleet(df: pd.DataFrame) -> str:
    """Get prediction statistics grouped by fleet type."""
    grouped = df.groupby("fleet").agg(
        total_flights=("predicted_duty_end", "count"),
        predicted_duty_end=("predicted_duty_end", "sum"),
        actual_duty_end=("actual_duty_end", "sum"),
        avg_probability=("predicted_probability", "mean"),
    ).sort_values("predicted_duty_end", ascending=False)

    return grouped.reset_index().to_json(orient="records")


def get_predictions_by_body_type(df: pd.DataFrame) -> str:
    """Get prediction statistics grouped by body/aircraft type."""
    grouped = df.groupby("body_type").agg(
        total_flights=("predicted_duty_end", "count"),
        predicted_duty_end=("predicted_duty_end", "sum"),
        actual_duty_end=("actual_duty_end", "sum"),
        avg_probability=("predicted_probability", "mean"),
    ).sort_values("predicted_duty_end", ascending=False)

    return grouped.reset_index().to_json(orient="records")


def get_predictions_by_hour(df: pd.DataFrame, hour_type: str = "depHour") -> str:
    """Get prediction statistics grouped by hour of day."""
    col = hour_type if hour_type in df.columns else "depHour"
    grouped = df.groupby(col).agg(
        total_flights=("predicted_duty_end", "count"),
        predicted_duty_end=("predicted_duty_end", "sum"),
        pct_duty_end=("predicted_duty_end", "mean"),
    ).sort_index()

    grouped["pct_duty_end"] = (grouped["pct_duty_end"] * 100).round(1)
    return grouped.reset_index().to_json(orient="records")


def get_predictions_by_route(df: pd.DataFrame, top_n: int = 15) -> str:
    """Get prediction statistics by route (dep_station -> arr_station)."""
    df_copy = df.copy()
    df_copy["route"] = df_copy["dep_station"] + " → " + df_copy["arr_station"]

    grouped = df_copy.groupby("route").agg(
        total_flights=("predicted_duty_end", "count"),
        predicted_duty_end=("predicted_duty_end", "sum"),
        pct_duty_end=("predicted_duty_end", "mean"),
        avg_probability=("predicted_probability", "mean"),
    ).sort_values("predicted_duty_end", ascending=False).head(top_n)

    grouped["pct_duty_end"] = (grouped["pct_duty_end"] * 100).round(1)
    return grouped.reset_index().to_json(orient="records")


def get_misclassified_flights(df: pd.DataFrame, error_type: str = "all", top_n: int = 20) -> str:
    """Get flights where prediction differs from actual."""
    misclassified = df[df["predicted_duty_end"] != df["actual_duty_end"]]

    if error_type == "false_positive":
        misclassified = misclassified[
            (misclassified["predicted_duty_end"] == 1) & (misclassified["actual_duty_end"] == 0)
        ]
    elif error_type == "false_negative":
        misclassified = misclassified[
            (misclassified["predicted_duty_end"] == 0) & (misclassified["actual_duty_end"] == 1)
        ]

    summary = {
        "total_misclassified": len(misclassified),
        "false_positives": int(((misclassified["predicted_duty_end"] == 1) & (misclassified["actual_duty_end"] == 0)).sum()),
        "false_negatives": int(((misclassified["predicted_duty_end"] == 0) & (misclassified["actual_duty_end"] == 1)).sum()),
        "sample": misclassified.head(top_n).to_dict(orient="records"),
    }
    return json.dumps(summary, default=str)


def filter_and_query(df: pd.DataFrame, filters: dict) -> str:
    """Apply arbitrary filters and return summary."""
    filtered = df.copy()

    for col, value in filters.items():
        if col in filtered.columns:
            if isinstance(value, list):
                filtered = filtered[filtered[col].isin(value)]
            else:
                filtered = filtered[filtered[col] == value]

    if filtered.empty:
        return json.dumps({"error": "No flights match the given filters", "filters_applied": filters})

    result = {
        "filters_applied": filters,
        "matching_flights": len(filtered),
        "predicted_duty_end_true": int(filtered["predicted_duty_end"].sum()),
        "predicted_duty_end_false": int((filtered["predicted_duty_end"] == 0).sum()),
        "actual_duty_end_true": int(filtered["actual_duty_end"].sum()),
        "accuracy": round((filtered["predicted_duty_end"] == filtered["actual_duty_end"]).mean(), 4),
        "avg_probability": round(filtered["predicted_probability"].mean(), 4),
    }
    return json.dumps(result)


# ==========================================
# TOOL DEFINITIONS for Azure OpenAI Function Calling
# ==========================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_prediction_summary",
            "description": "Get overall summary statistics of the model predictions including total flights, predicted duty-end counts, accuracy, and average probability.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_predictions_by_airport",
            "description": "Get prediction statistics grouped by airport. Can show top airports by number of predicted duty-end flights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "airport_type": {
                        "type": "string",
                        "enum": ["dep_station", "arr_station"],
                        "description": "Whether to group by departure or arrival airport.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top airports to return (default 10).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_predictions_by_fleet",
            "description": "Get prediction statistics grouped by fleet type (e.g., 737, A320, etc).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_predictions_by_body_type",
            "description": "Get prediction statistics grouped by aircraft body type (NB=narrow body, WB=wide body).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_predictions_by_hour",
            "description": "Get prediction statistics grouped by hour of day (0-23). Shows how duty-end predictions vary by time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour_type": {
                        "type": "string",
                        "enum": ["depHour", "arrHour"],
                        "description": "Group by departure hour or arrival hour.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_predictions_by_route",
            "description": "Get prediction statistics by route (departure → arrival station pair). Shows routes with most duty-end predictions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top routes to return (default 15).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_misclassified_flights",
            "description": "Get information about misclassified flights (where prediction != actual). Can filter by false positives or false negatives.",
            "parameters": {
                "type": "object",
                "properties": {
                    "error_type": {
                        "type": "string",
                        "enum": ["all", "false_positive", "false_negative"],
                        "description": "Type of errors to show: all, false_positive (predicted duty-end but wasn't), or false_negative (missed actual duty-end).",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of sample flights to return.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_and_query",
            "description": "Apply custom filters to predictions and get statistics. Filter by any column: dep_station, arr_station, fleet, body_type, eqp, depHour, arrHour, predicted_duty_end, actual_duty_end, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "description": "Key-value pairs where key is column name and value is the filter value. Example: {\"dep_station\": \"DFW\", \"predicted_duty_end\": 1}",
                    },
                },
                "required": ["filters"],
            },
        },
    },
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
- Always include context (e.g., "out of X total flights") to give perspective.
- Suggest 1-2 follow-up questions the user might want to ask next.
- Be concise but complete. Use bullet points for summaries, tables for comparisons.
"""


# ==========================================
# AGENT EXECUTION
# ==========================================


def execute_tool_call(function_name: str, arguments: dict, df: pd.DataFrame) -> str:
    """Execute a tool function and return the result."""
    if function_name == "get_prediction_summary":
        return get_prediction_summary(df)
    elif function_name == "get_predictions_by_airport":
        return get_predictions_by_airport(
            df,
            airport_type=arguments.get("airport_type", "arr_station"),
            top_n=arguments.get("top_n", 10),
        )
    elif function_name == "get_predictions_by_fleet":
        return get_predictions_by_fleet(df)
    elif function_name == "get_predictions_by_body_type":
        return get_predictions_by_body_type(df)
    elif function_name == "get_predictions_by_hour":
        return get_predictions_by_hour(df, hour_type=arguments.get("hour_type", "depHour"))
    elif function_name == "get_predictions_by_route":
        return get_predictions_by_route(df, top_n=arguments.get("top_n", 15))
    elif function_name == "get_misclassified_flights":
        return get_misclassified_flights(
            df,
            error_type=arguments.get("error_type", "all"),
            top_n=arguments.get("top_n", 20),
        )
    elif function_name == "filter_and_query":
        return filter_and_query(df, filters=arguments.get("filters", {}))
    else:
        return json.dumps({"error": f"Unknown function: {function_name}"})


def create_azure_client() -> AzureOpenAI:
    """Create Azure OpenAI client from secrets.json."""
    with open(SECRETS_PATH, "r") as fh:
        secrets = json.load(fh)

    azure_config = secrets.get("azure_openai", {})

    endpoint = azure_config.get("endpoint", os.environ.get("AZURE_OPENAI_ENDPOINT", ""))
    api_key = azure_config.get("api_key", os.environ.get("AZURE_OPENAI_API_KEY", ""))
    api_version = azure_config.get("api_version", "2024-12-01-preview")

    if not endpoint or not api_key:
        raise ValueError(
            "Azure OpenAI credentials not found.\n"
            "Add to secrets.json:\n"
            '  "azure_openai": {\n'
            '    "endpoint": "https://your-resource.openai.azure.com/",\n'
            '    "api_key": "your-key-here",\n'
            '    "deployment": "gpt-4o",\n'
            '    "api_version": "2024-12-01-preview"\n'
            "  }\n"
            "Or set environment variables: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY"
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def chat_loop():
    """Main interactive chat loop with Azure OpenAI function calling."""
    print("\n" + "=" * 60)
    print("  GENIE Agent — Graph-based Early Network Inefficiency Evaluator")
    print("  Powered by Azure AI Foundry (GPT-4o)")
    print("=" * 60)
    print("\nLoading predictions data...")

    df = load_predictions()
    print(f"  Loaded {len(df):,} flight predictions")
    print(f"  Predicted duty-end: {df['predicted_duty_end'].sum():,}")
    print(f"  Predicted non-duty-end: {(df['predicted_duty_end'] == 0).sum():,}")

    print("\nConnecting to Azure OpenAI...")
    client = create_azure_client()

    with open(SECRETS_PATH, "r") as fh:
        deployment = json.load(fh).get("azure_openai", {}).get("deployment", "gpt-4o")

    print(f"  Using deployment: {deployment}")
    print("\n" + "-" * 60)
    print("Ask questions about flight predictions. Type 'quit' to exit.")
    print("Examples:")
    print("  - How many flights are predicted as duty-end?")
    print("  - Which arrival airport has the most predicted duty-end flights?")
    print("  - Show accuracy by fleet type")
    print("  - What routes have the highest duty-end prediction rate?")
    print("  - Show me flights departing DFW that are predicted duty-end")
    print("-" * 60 + "\n")

    # Conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Call Azure OpenAI with tools
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )

            assistant_message = response.choices[0].message

            # Handle tool calls (may be multiple)
            while assistant_message.tool_calls:
                messages.append(assistant_message)

                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    logger.debug(f"  Tool call: {function_name}({arguments})")
                    result = execute_tool_call(function_name, arguments, df)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                # Get final response after tool execution
                response = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                )
                assistant_message = response.choices[0].message

            # Print the response
            reply = assistant_message.content
            messages.append({"role": "assistant", "content": reply})
            print(f"\nGENIE: {reply}")

        except Exception as e:
            print(f"\nError: {e}")
            # Remove the failed user message to keep conversation clean
            messages.pop()


# ==========================================
# OFFLINE MODE (no Azure, uses pandas directly)
# ==========================================


def offline_chat():
    """Simple offline mode that answers questions using pandas directly (no LLM)."""
    print("\n" + "=" * 60)
    print("  GENIE Agent — Offline Query Mode")
    print("=" * 60)
    print("\nLoading predictions data...")

    df = load_predictions()
    print(f"  Loaded {len(df):,} flight predictions\n")

    commands = {
        "summary": "Overall prediction summary",
        "airports": "Top airports by predicted duty-end count",
        "fleet": "Predictions by fleet type",
        "body": "Predictions by body type",
        "hours": "Predictions by departure hour",
        "routes": "Top routes by duty-end predictions",
        "errors": "Misclassification analysis",
        "help": "Show available commands",
        "quit": "Exit",
    }

    print("Available commands:")
    for cmd, desc in commands.items():
        print(f"  {cmd:<12} - {desc}")
    print()

    while True:
        try:
            user_input = input("Query> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input == "summary":
            print(json.dumps(json.loads(get_prediction_summary(df)), indent=2))
        elif user_input == "airports":
            data = json.loads(get_predictions_by_airport(df))
            print(pd.DataFrame(data).to_string(index=False))
        elif user_input == "fleet":
            data = json.loads(get_predictions_by_fleet(df))
            print(pd.DataFrame(data).to_string(index=False))
        elif user_input == "body":
            data = json.loads(get_predictions_by_body_type(df))
            print(pd.DataFrame(data).to_string(index=False))
        elif user_input == "hours":
            data = json.loads(get_predictions_by_hour(df))
            print(pd.DataFrame(data).to_string(index=False))
        elif user_input == "routes":
            data = json.loads(get_predictions_by_route(df))
            print(pd.DataFrame(data).to_string(index=False))
        elif user_input == "errors":
            data = json.loads(get_misclassified_flights(df))
            print(f"Total misclassified: {data['total_misclassified']}")
            print(f"False positives: {data['false_positives']}")
            print(f"False negatives: {data['false_negatives']}")
        elif user_input == "help":
            for cmd, desc in commands.items():
                print(f"  {cmd:<12} - {desc}")
        else:
            # Try as a station filter
            parts = user_input.split()
            if len(parts) >= 1 and len(parts[0]) == 3 and parts[0].isalpha():
                station = parts[0].upper()
                result = json.loads(filter_and_query(df, {"dep_station": station}))
                print(json.dumps(result, indent=2))
            else:
                print("Unknown command. Type 'help' for available commands.")
        print()


# ==========================================
# ENTRY POINT
# ==========================================


if __name__ == "__main__":
    import sys

    if "--offline" in sys.argv:
        offline_chat()
    else:
        try:
            chat_loop()
        except ValueError as e:
            print(f"\n{e}")
            print("\nFalling back to offline mode...\n")
            offline_chat()
