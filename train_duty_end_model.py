"""
Neural Network model to predict isDutyEndFlight (boolean)
using graph features extracted from Neo4j GDS.

Flight node properties:
  - isDutyEndFlight: boolean (true/false) — target variable
  - depDateTime: native Neo4j datetime (e.g. 2025-10-02T19:23:00Z)
  - dep_lcl: string (legacy, e.g. "10/2/2025 19:23:00")

Pipeline:
  1. Extract enriched Flight node features from Neo4j
  2. Preprocess & encode features
  3. Train a PyTorch neural network for binary classification
  4. Evaluate with metrics (accuracy, precision, recall, F1, AUC-ROC)

Connection: crew_neo4j from secrets.json
"""

import json
import os
import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)
from neo4j import GraphDatabase

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------- Configuration ---------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_PATH = os.path.join(BASE_DIR, "secrets.json")

# Hyperparameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15  # of remaining after test split
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 10  # Early stopping patience
HIDDEN_DIMS = [128, 64, 32]
DROPOUT_RATE = 0.3

# Features to extract from Neo4j (graph-computed + raw)
FEATURE_QUERY = """
MATCH (f:Flight)
WHERE f.isDutyEndFlight IS NOT NULL
RETURN
    // Identifiers
    id(f)                             AS nodeId,
    f.flight_number                   AS flight_number,
    f.lof                             AS lof,
    f.dep_station                     AS dep_station,
    f.arr_station                     AS arr_station,

    // Target variable (boolean: true/false)
    f.isDutyEndFlight                 AS isDutyEndFlight,

    // --- GDS Graph Features ---
    coalesce(f.departureDegree, 0)    AS departureDegree,
    coalesce(f.arrivalDegree, 0)      AS arrivalDegree,
    coalesce(f.sequencePageRank, 0)   AS sequencePageRank,
    coalesce(f.sequenceBetweenness, 0) AS sequenceBetweenness,
    coalesce(f.articleRank, 0)        AS articleRank,
    coalesce(f.louvainCommunity, -1)  AS louvainCommunity,
    coalesce(f.labelPropCommunity, -1) AS labelPropCommunity,
    coalesce(f.triangleCount, 0)      AS triangleCount,
    coalesce(f.clusteringCoefficient, 0) AS clusteringCoefficient,

    // --- Positional Features ---
    coalesce(f.sequence_index, 0)     AS sequenceIndex,
    coalesce(f.lofSize, 1)           AS lofSize,
    coalesce(f.relativePosition, 0)   AS relativePosition,
    coalesce(f.isLastInLof, 0)       AS isLastInLof,
    coalesce(f.isFirstInLof, 0)      AS isFirstInLof,
    coalesce(f.hasNextFlight, 1)     AS hasNextFlight,
    coalesce(f.hasPrevFlight, 0)     AS hasPrevFlight,

    // --- Temporal Features ---
    coalesce(f.depHour, 12)          AS depHour,
    coalesce(f.arrHour, 12)          AS arrHour,
    coalesce(f.depDayOfWeek, 1)      AS depDayOfWeek,
    coalesce(f.isRedEye, 0)          AS isRedEye,
    coalesce(f.isLateArrival, 0)     AS isLateArrival,
    coalesce(f.blockTimeMinutes, 0)  AS blockTimeMinutes,

    // --- Station Features ---
    coalesce(f.depStationFlightCount, 0)  AS depStationFlightCount,
    coalesce(f.arrStationFlightCount, 0)  AS arrStationFlightCount,
    coalesce(f.depStationUniqueRoutes, 0) AS depStationUniqueRoutes,

    // --- Fleet Features ---
    f.fleet                          AS fleet,
    f.body_type                      AS body_type,
    f.eqp                            AS eqp,
    coalesce(f.fleetPopularity, 0)   AS fleetPopularity
"""

# Numerical feature columns for the model
NUMERIC_FEATURES = [
    "departureDegree",
    "arrivalDegree",
    "sequencePageRank",
    "sequenceBetweenness",
    "articleRank",
    "triangleCount",
    "clusteringCoefficient",
    "sequenceIndex",
    "lofSize",
    "relativePosition",
    "isLastInLof",
    "isFirstInLof",
    "hasNextFlight",
    "hasPrevFlight",
    "depHour",
    "arrHour",
    "depDayOfWeek",
    "isRedEye",
    "isLateArrival",
    "blockTimeMinutes",
    "depStationFlightCount",
    "arrStationFlightCount",
    "depStationUniqueRoutes",
    "fleetPopularity",
]

# Categorical feature columns (will be label-encoded)
CATEGORICAL_FEATURES = [
    "dep_station",
    "arr_station",
    "fleet",
    "body_type",
    "eqp",
    "louvainCommunity",
    "labelPropCommunity",
]


# ==========================================
# 1. DATA EXTRACTION
# ==========================================


def load_credentials() -> dict:
    """Load Neo4j credentials from secrets.json using crew_neo4j key."""
    with open(SECRETS_PATH, "r") as fh:
        secrets = json.load(fh)
    return secrets["crew_neo4j"]


def create_driver(creds: dict):
    """Create a Neo4j driver instance."""
    return GraphDatabase.driver(
        creds["connection_string"],
        auth=(creds["user"], creds["password"]),
        max_connection_lifetime=5 * 60,
        max_connection_pool_size=5,
        connection_acquisition_timeout=60,
    )


def extract_features_from_neo4j(driver) -> pd.DataFrame:
    """
    Extract all flight features from Neo4j into a pandas DataFrame.

    Returns:
        DataFrame with graph features, positional features,
        temporal features, and the target label.
    """
    logger.info("Extracting features from Neo4j...")
    with driver.session() as session:
        result = session.run(FEATURE_QUERY)
        records = [dict(record) for record in result]

    df = pd.DataFrame(records)
    logger.info(f"  Extracted {len(df):,} flight records with {len(df.columns)} columns")

    # Log class balance
    if "isDutyEndFlight" in df.columns:
        counts = df["isDutyEndFlight"].value_counts()
        logger.info(f"  Class distribution:\n{counts.to_string()}")

    return df


# ==========================================
# 2. PREPROCESSING
# ==========================================


def preprocess_data(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Preprocess extracted data for neural network training.

    Steps:
        - Encode target label (boolean true -> 1, false -> 0)
        - Fill missing numerics with 0
        - Label-encode categoricals
        - StandardScale all features

    Args:
        df: Raw DataFrame from Neo4j extraction.

    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array)
        metadata: Dict with encoders, scaler, feature names for inference
    """
    logger.info("Preprocessing data...")
    df = df.copy()

    # --- Encode target (isDutyEndFlight is boolean: true/false) ---
    df["target"] = df["isDutyEndFlight"].apply(
        lambda v: 1 if v is True or str(v).strip().lower() in ("true", "1", "y") else 0
    )
    y = df["target"].values

    # --- Numeric features ---
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            logger.warning(f"  Missing numeric feature: {col}, filling with 0")
            df[col] = 0

    # --- Categorical features (label encode) ---
    label_encoders = {}
    encoded_cat_cols = []
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("UNKNOWN")
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col])
            label_encoders[col] = le
            encoded_cat_cols.append(f"{col}_encoded")
        else:
            logger.warning(f"  Missing categorical feature: {col}, skipping")

    # Combine numeric + encoded categorical
    all_feature_cols = NUMERIC_FEATURES + encoded_cat_cols
    X_raw = df[all_feature_cols].values.astype(np.float64)

    # --- Clean inf/large values ---
    inf_count = np.isinf(X_raw).sum()
    if inf_count > 0:
        logger.warning(f"  Found {inf_count} infinity values — replacing with NaN then 0")
    X_raw = np.where(np.isinf(X_raw), np.nan, X_raw)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip extreme values to prevent float32 overflow
    max_val = np.finfo(np.float32).max / 10
    clipped = np.abs(X_raw) > max_val
    if clipped.any():
        logger.warning(f"  Clipping {clipped.sum()} extreme values")
    X_raw = np.clip(X_raw, -max_val, max_val).astype(np.float32)

    # --- Standard scaling ---
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    logger.info(f"  Final feature matrix shape: {X.shape}")
    logger.info(f"  Target: {np.sum(y == 1):,} positive / {np.sum(y == 0):,} negative")

    metadata = {
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_columns": all_feature_cols,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "encoded_cat_cols": encoded_cat_cols,
        "node_ids": df["nodeId"].values if "nodeId" in df.columns else None,
    }

    return X, y, metadata


# ==========================================
# 3. NEURAL NETWORK DEFINITION
# ==========================================


class DutyEndFlightPredictor(nn.Module):
    """
    Multi-layer feed-forward neural network for binary classification.

    Architecture:
        Input -> [Hidden + BatchNorm + ReLU + Dropout] x N -> Sigmoid output

    Attributes:
        layers: Sequential stack of linear/BN/activation/dropout blocks.
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        """
        Initialize the predictor network.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer sizes, e.g. [128, 64, 32].
            dropout: Dropout probability.
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim

        # Output layer - single neuron for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x).squeeze(-1)


# ==========================================
# 4. TRAINING LOOP
# ==========================================


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Compute positive class weight to handle class imbalance.

    Args:
        y: Binary target array.

    Returns:
        pos_weight tensor for BCEWithLogitsLoss.
    """
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0:
        return torch.tensor(1.0)
    weight = n_neg / n_pos
    logger.info(f"  Class weight (neg/pos ratio): {weight:.2f}")
    return torch.tensor(weight, dtype=torch.float32)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
) -> Tuple[DutyEndFlightPredictor, dict]:
    """
    Train the neural network with early stopping.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        input_dim: Number of input features.

    Returns:
        model: Trained PyTorch model.
        history: Dict with training/validation losses and metrics per epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = DutyEndFlightPredictor(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT_RATE,
    ).to(device)
    logger.info(f"Model architecture:\n{model}")

    # Loss with class weighting
    pos_weight = compute_class_weights(y_train).to(device)
    criterion = nn.BCELoss(weight=None)  # We'll use weighted sampling instead

    # For imbalanced data, use pos_weight with BCEWithLogitsLoss
    # But since our model outputs sigmoid, we manually weight:
    def weighted_bce_loss(pred, target):
        """BCE loss with positive class weighting."""
        weight = torch.where(target == 1, pos_weight, torch.tensor(1.0).to(device))
        bce = -target * torch.log(pred + 1e-7) - (1 - target) * torch.log(1 - pred + 1e-7)
        return (weight * bce).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop with early stopping
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # --- Training ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = weighted_bce_loss(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = weighted_bce_loss(val_pred, y_val_t).item()

            val_pred_np = val_pred.cpu().numpy()
            val_labels = (val_pred_np >= 0.5).astype(int)
            val_f1 = f1_score(y_val, val_labels, zero_division=0)

            try:
                val_auc = roc_auc_score(y_val, val_pred_np)
            except ValueError:
                val_auc = 0.0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{EPOCHS} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Val AUC: {val_auc:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"  Restored best model (val_loss={best_val_loss:.4f})")

    return model, history


# ==========================================
# 5. EVALUATION
# ==========================================


def evaluate_model(
    model: DutyEndFlightPredictor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate trained model on test set.

    Args:
        model: Trained PyTorch model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary with evaluation metrics.
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(X_test_t).cpu().numpy()

    pred_labels = (predictions >= 0.5).astype(int)

    # Metrics
    report = classification_report(y_test, pred_labels, target_names=["Not Duty End", "Duty End"])
    cm = confusion_matrix(y_test, pred_labels)

    try:
        auc_roc = roc_auc_score(y_test, predictions)
    except ValueError:
        auc_roc = 0.0

    f1 = f1_score(y_test, pred_labels, zero_division=0)

    logger.info(f"\n{'='*60}")
    logger.info(f"TEST SET EVALUATION")
    logger.info(f"{'='*60}")
    logger.info(f"\n{report}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Find optimal threshold using precision-recall
    precisions, recalls, thresholds = precision_recall_curve(y_test, predictions)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-7)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    logger.info(f"Optimal threshold: {optimal_threshold:.3f} (F1={f1_scores[optimal_idx]:.4f})")

    return {
        "auc_roc": auc_roc,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": predictions,
        "pred_labels": pred_labels,
        "optimal_threshold": optimal_threshold,
    }


# ==========================================
# 6. FEATURE IMPORTANCE (gradient-based)
# ==========================================


def compute_feature_importance(
    model: DutyEndFlightPredictor,
    X: np.ndarray,
    feature_names: list,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Compute gradient-based feature importance.

    Args:
        model: Trained model.
        X: Feature matrix.
        feature_names: List of feature column names.
        top_k: Number of top features to display.

    Returns:
        DataFrame with feature importance scores.
    """
    device = next(model.parameters()).device
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
    output = model(X_t)
    output.sum().backward()

    # Average absolute gradient as importance
    importance = X_t.grad.abs().mean(dim=0).cpu().numpy()

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    logger.info(f"\nTop {top_k} Most Important Features:")
    logger.info(f"{'Feature':<35} {'Importance':>12}")
    logger.info("-" * 50)
    for _, row in importance_df.head(top_k).iterrows():
        logger.info(f"  {row['feature']:<33} {row['importance']:>12.6f}")

    return importance_df


# ==========================================
# 7. WRITE PREDICTIONS BACK TO NEO4J
# ==========================================


def write_predictions_to_neo4j(
    driver,
    node_ids: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.5,
    batch_size: int = 5000,
):
    """
    Write predicted isDutyEndFlight probabilities back to Neo4j Flight nodes.

    Args:
        driver: Neo4j driver.
        node_ids: Array of Neo4j internal node IDs.
        predictions: Predicted probabilities.
        threshold: Classification threshold.
        batch_size: Write batch size.
    """
    logger.info(f"Writing {len(predictions):,} predictions back to Neo4j...")

    total_written = 0
    for i in range(0, len(node_ids), batch_size):
        batch_ids = node_ids[i : i + batch_size].tolist()
        batch_preds = predictions[i : i + batch_size].tolist()
        batch_labels = ["Y" if p >= threshold else "N" for p in batch_preds]

        records = [
            {"nodeId": int(nid), "prob": float(prob), "label": label}
            for nid, prob, label in zip(batch_ids, batch_preds, batch_labels)
        ]

        with driver.session() as session:
            session.run(
                """
                UNWIND $records AS rec
                MATCH (f:Flight) WHERE id(f) = rec.nodeId
                SET f.predicted_duty_end_prob = rec.prob,
                    f.predicted_duty_end = rec.label
                """,
                records=records,
            )

        total_written += len(records)
        if total_written % 20000 == 0:
            logger.info(f"  Written {total_written:,} / {len(predictions):,}")

    logger.info(f"  Completed: {total_written:,} predictions written to Neo4j")


# ==========================================
# 8. SAVE / LOAD MODEL
# ==========================================


def save_model(model, metadata, filepath: str):
    """Save trained model and metadata."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": model.network[0].in_features,
            "hidden_dims": HIDDEN_DIMS,
            "dropout": DROPOUT_RATE,
            "feature_columns": metadata["feature_columns"],
        },
        filepath,
    )
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> DutyEndFlightPredictor:
    """Load a trained model from file."""
    checkpoint = torch.load(filepath, weights_only=False)
    model = DutyEndFlightPredictor(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Model loaded from {filepath}")
    return model


# ==========================================
# MAIN PIPELINE
# ==========================================


def main():
    """End-to-end pipeline: extract -> preprocess -> train -> evaluate -> save model."""
    logger.info("=" * 60)
    logger.info("isDutyEndFlight Prediction Pipeline")
    logger.info("=" * 60)

    # Step 1: Connect to Neo4j
    creds = load_credentials()
    driver = create_driver(creds)
    logger.info(f"Connected to {creds['connection_string']}")

    try:
        # Step 2: Extract features
        df = extract_features_from_neo4j(driver)
        if df.empty:
            logger.error("No data extracted from Neo4j. Run GDS feature queries first.")
            return

        # Step 3: Preprocess
        X, y, metadata = preprocess_data(df)

        # Step 4: Train/Val/Test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_temp
        )

        logger.info(f"\nData splits:")
        logger.info(f"  Train: {X_train.shape[0]:,} samples")
        logger.info(f"  Val:   {X_val.shape[0]:,} samples")
        logger.info(f"  Test:  {X_test.shape[0]:,} samples")

        # Step 5: Train
        logger.info("\n--- Training Neural Network ---")
        model, history = train_model(X_train, y_train, X_val, y_val, X.shape[1])

        # Step 6: Evaluate
        eval_results = evaluate_model(model, X_test, y_test)

        # Step 7: Feature importance
        importance_df = compute_feature_importance(
            model, X_test, metadata["feature_columns"]
        )

        # Step 8: Save model
        model_path = os.path.join(BASE_DIR, "duty_end_flight_model.pt")
        save_model(model, metadata, model_path)

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Test AUC-ROC: {eval_results['auc_roc']:.4f}")
        logger.info(f"  Test F1:      {eval_results['f1']:.4f}")
        logger.info(f"  Threshold:    {eval_results['optimal_threshold']:.3f}")
        logger.info(f"  Features:     {X.shape[1]}")

    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        raise
    finally:
        driver.close()
        logger.info("Neo4j connection closed.")


if __name__ == "__main__":
    main()
