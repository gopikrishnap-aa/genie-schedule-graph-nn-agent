"""
Export model predictions to CSV for downstream querying.

Loads the trained duty_end_flight_model.pt, runs inference on all flights
from Neo4j, and saves predictions alongside flight metadata to a CSV file.

Output: predictions_duty_end.csv
"""

import json
import os
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_PATH = os.path.join(BASE_DIR, "secrets.json")
MODEL_PATH = os.path.join(BASE_DIR, "duty_end_flight_model.pt")
OUTPUT_CSV = os.path.join(BASE_DIR, "predictions_duty_end.csv")

# Must match train_duty_end_model.py
FEATURE_QUERY = """
MATCH (f:Flight)
WHERE f.isDutyEndFlight IS NOT NULL
RETURN
    id(f)                             AS nodeId,
    f.flight_number                   AS flight_number,
    f.lof                             AS lof,
    f.dep_station                     AS dep_station,
    f.arr_station                     AS arr_station,
    f.isDutyEndFlight                 AS isDutyEndFlight,
    coalesce(f.departureDegree, 0)    AS departureDegree,
    coalesce(f.arrivalDegree, 0)      AS arrivalDegree,
    coalesce(f.sequencePageRank, 0)   AS sequencePageRank,
    coalesce(f.sequenceBetweenness, 0) AS sequenceBetweenness,
    coalesce(f.articleRank, 0)        AS articleRank,
    coalesce(f.louvainCommunity, -1)  AS louvainCommunity,
    coalesce(f.labelPropCommunity, -1) AS labelPropCommunity,
    coalesce(f.triangleCount, 0)      AS triangleCount,
    coalesce(f.clusteringCoefficient, 0) AS clusteringCoefficient,
    coalesce(f.sequence_index, 0)     AS sequenceIndex,
    coalesce(f.lofSize, 1)           AS lofSize,
    coalesce(f.relativePosition, 0)   AS relativePosition,
    coalesce(f.isLastInLof, 0)       AS isLastInLof,
    coalesce(f.isFirstInLof, 0)      AS isFirstInLof,
    coalesce(f.hasNextFlight, 1)     AS hasNextFlight,
    coalesce(f.hasPrevFlight, 0)     AS hasPrevFlight,
    coalesce(f.depHour, 12)          AS depHour,
    coalesce(f.arrHour, 12)          AS arrHour,
    coalesce(f.depDayOfWeek, 1)      AS depDayOfWeek,
    coalesce(f.isRedEye, 0)          AS isRedEye,
    coalesce(f.isLateArrival, 0)     AS isLateArrival,
    coalesce(f.blockTimeMinutes, 0)  AS blockTimeMinutes,
    coalesce(f.depStationFlightCount, 0)  AS depStationFlightCount,
    coalesce(f.arrStationFlightCount, 0)  AS arrStationFlightCount,
    coalesce(f.depStationUniqueRoutes, 0) AS depStationUniqueRoutes,
    f.fleet                          AS fleet,
    f.body_type                      AS body_type,
    f.eqp                            AS eqp,
    coalesce(f.fleetPopularity, 0)   AS fleetPopularity,
    toString(f.depDateTime)           AS depDateTime,
    f.arr_lcl                         AS arrDateTime
"""

NUMERIC_FEATURES = [
    "departureDegree", "arrivalDegree", "sequencePageRank",
    "sequenceBetweenness", "articleRank", "triangleCount",
    "clusteringCoefficient", "sequenceIndex", "lofSize",
    "relativePosition", "isLastInLof", "isFirstInLof",
    "hasNextFlight", "hasPrevFlight", "depHour", "arrHour",
    "depDayOfWeek", "isRedEye", "isLateArrival", "blockTimeMinutes",
    "depStationFlightCount", "arrStationFlightCount",
    "depStationUniqueRoutes", "fleetPopularity",
]

CATEGORICAL_FEATURES = [
    "dep_station", "arr_station", "fleet",
    "body_type", "eqp", "louvainCommunity", "labelPropCommunity",
]


def main():
    """Run inference and export predictions to CSV."""
    # Load credentials and connect
    with open(SECRETS_PATH, "r") as fh:
        creds = json.load(fh)["crew_neo4j"]

    driver = GraphDatabase.driver(
        creds["connection_string"],
        auth=(creds["user"], creds["password"]),
    )

    # Extract data
    logger.info("Extracting flight data from Neo4j...")
    with driver.session() as session:
        result = session.run(FEATURE_QUERY)
        records = [dict(r) for r in result]
    driver.close()

    df = pd.DataFrame(records)
    logger.info(f"  Extracted {len(df):,} flights")

    # Preprocess (same as training)
    df_export = df.copy()

    # Encode target
    df["target"] = df["isDutyEndFlight"].apply(
        lambda v: 1 if v is True or str(v).strip().lower() in ("true", "1") else 0
    )

    # Numeric
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # Categorical
    encoded_cat_cols = []
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("UNKNOWN")
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col])
            encoded_cat_cols.append(f"{col}_encoded")

    # Build feature matrix
    all_feature_cols = NUMERIC_FEATURES + encoded_cat_cols
    X_raw = df[all_feature_cols].values.astype(np.float64)
    X_raw = np.where(np.isinf(X_raw), np.nan, X_raw)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = np.finfo(np.float32).max / 10
    X_raw = np.clip(X_raw, -max_val, max_val).astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Load model
    logger.info(f"Loading model from {MODEL_PATH}...")
    from train_duty_end_model import DutyEndFlightPredictor

    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model = DutyEndFlightPredictor(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Run inference
    logger.info("Running inference...")
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        predictions = model(X_t).numpy()

    # Build output DataFrame
    OPTIMAL_THRESHOLD = 0.412  # From training results
    df_export["predicted_probability"] = predictions
    df_export["predicted_duty_end"] = (predictions >= OPTIMAL_THRESHOLD).astype(int)
    df_export["actual_duty_end"] = df["target"].values

    # Select columns for export
    export_cols = [
        "flight_number", "lof", "dep_station", "arr_station",
        "fleet", "body_type", "eqp",
        "depDateTime", "arrDateTime",
        "depHour", "arrHour", "depDayOfWeek",
        "blockTimeMinutes", "sequenceIndex", "lofSize",
        "actual_duty_end", "predicted_duty_end", "predicted_probability",
    ]
    df_out = df_export[export_cols].copy()

    # Save
    df_out.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Predictions saved to {OUTPUT_CSV}")
    logger.info(f"  Total flights: {len(df_out):,}")
    logger.info(f"  Predicted duty-end (True): {df_out['predicted_duty_end'].sum():,}")
    logger.info(f"  Predicted non-duty-end (False): {(df_out['predicted_duty_end'] == 0).sum():,}")
    logger.info(f"  Accuracy: {(df_out['predicted_duty_end'] == df_out['actual_duty_end']).mean():.4f}")


if __name__ == "__main__":
    main()
