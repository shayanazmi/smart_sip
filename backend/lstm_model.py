"""
LSTM Model Training & Persistence
======================================================
Builds the 3-layer stacked LSTM architecture from the project spec,
trains it on Nifty 50 sequences from Phase 1, validates using
walk-forward testing, and exports the model as .h5 for the
FastAPI backend.

Architecture (per spec):
    • 3-Layer Stacked LSTM (50 units each) with Dropout
    • Dense output layer with Linear activation
    • Optimizer : Adam (lr=0.001)
    • Loss      : Mean Squared Error (MSE)

Usage:
    python lstm_model.py              # train from scratch
    python lstm_model.py --load       # skip training, just run evaluation
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")               # non-interactive backend (safe for VS)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import (EarlyStopping,
                                        ModelCheckpoint,
                                        ReduceLROnPlateau)
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Import pipeline ──────────────────────────────────────────────────
from data_pipeline import load_artifacts, build_dataset, ARTIFACTS_DIR, FEATURE_COLUMNS

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
LSTM_UNITS      = 50        # units per LSTM layer (per spec)
DROPOUT_RATE    = 0.15       # dropout between layers (prevents overfitting)
LEARNING_RATE   = 0.001     # Adam lr (per spec)
EPOCHS          = 100       # max epochs (EarlyStopping will cut this short)
BATCH_SIZE      = 32
PATIENCE        = 15        # EarlyStopping patience
MODEL_PATH      = "artifacts/smartsip_lstm.h5"
PLOTS_DIR       = "artifacts/plots"

# Walk-Forward validation: how many test folds to run
WF_FOLDS        = 5


# ─────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────
def build_model(window_size: int, n_features: int) -> tf.keras.Model:
    """
    3-Layer Stacked LSTM with Dropout — exactly as specified in the document.

    Input shape : (window_size, n_features)  →  (60, 14)
    Output      : single scalar (next-day scaled Close price)
    Activation  : Linear (regression output — NOT sigmoid/softmax)
    """
    model = Sequential([
        Input(shape=(window_size, n_features)),

        # ── Layer 1 ────────────────────────────────────────────────────────
        LSTM(LSTM_UNITS, return_sequences=True),   # return_sequences=True
        Dropout(DROPOUT_RATE),                     # passes full sequence to L2

        # ── Layer 2 ────────────────────────────────────────────────────────
        LSTM(LSTM_UNITS, return_sequences=True),
        Dropout(DROPOUT_RATE),

        # ── Layer 3 ────────────────────────────────────────────────────────
        LSTM(LSTM_UNITS, return_sequences=False),  # only last timestep out
        Dropout(DROPOUT_RATE),

        # ── Output ─────────────────────────────────────────────────────────
        Dense(1, activation="linear"),             # linear activation (regression)
    ], name="SmartSIP_LSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",                 # MSE (per spec)
        metrics=["mae"]
    )

    return model


# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks() -> list:
    """
    Three callbacks:
    1. EarlyStopping    — stops when val_loss stops improving (patience=10)
    2. ModelCheckpoint  — saves the best weights automatically mid-training
    3. ReduceLROnPlateau— halves LR when stuck (helps escape plateaus)
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    return [early_stop, checkpoint, reduce_lr]


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train_model(model: tf.keras.Model,
                X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray,  y_test: np.ndarray
                ) -> tf.keras.callbacks.History:
    """
    Trains the LSTM on the 80% training split.
    Validation is done on the held-out 20% test split.
    """
    print(f"\n[2/4] Training LSTM ...")
    print(f"      X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"      Epochs: {EPOCHS} (EarlyStopping patience={PATIENCE})")
    print(f"      Batch size: {BATCH_SIZE}\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=1
    )

    print(f"\n    ✓ Training complete — model saved to '{MODEL_PATH}'")
    return history


# ─────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────
def walk_forward_validation(X: np.ndarray, y: np.ndarray,
                             window_size: int, n_features: int,
                             folds: int = WF_FOLDS) -> pd.DataFrame:
    """
    Walk-Forward Validation — the correct way to validate time-series models.

    Simulates real-world deployment: the model is retrained each fold
    on an expanding window and tested on the next unseen chunk.
    This proves the model isn't 'memorising' history.

    Returns a DataFrame of fold-by-fold RMSE and MAE scores.
    """
    print(f"\n[3/4] Walk-Forward Validation ({folds} folds) ...")

    total      = len(X)
    fold_size  = total // (folds + 1)   # +1 so there's always a training block
    results    = []

    for fold in range(folds):
        train_end  = fold_size * (fold + 1)
        test_start = train_end
        test_end   = test_start + fold_size

        if test_end > total:
            break

        X_tr, y_tr = X[:train_end],        y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        # Fresh model per fold (clean walk-forward)
        fold_model = build_model(window_size, n_features)

        fold_model.fit(
            X_tr, y_tr,
            epochs=30,          # fewer epochs per fold (speed)
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[EarlyStopping(monitor="loss", patience=5,
                                     restore_best_weights=True)]
        )

        preds = fold_model.predict(X_te, verbose=0).flatten()
        rmse  = np.sqrt(mean_squared_error(y_te, preds))
        mae   = mean_absolute_error(y_te, preds)

        results.append({"fold": fold + 1,
                         "train_samples": train_end,
                         "test_samples":  fold_size,
                         "RMSE":          round(rmse, 6),
                         "MAE":           round(mae, 6)})

        print(f"      Fold {fold+1}/{folds} — RMSE: {rmse:.6f} | MAE: {mae:.6f}")

    df_results = pd.DataFrame(results)
    print(f"\n    Walk-Forward Summary:")
    print(df_results.to_string(index=False))
    print(f"\n    Mean RMSE: {df_results['RMSE'].mean():.6f}  "
          f"Std: {df_results['RMSE'].std():.6f}")

    return df_results


# ─────────────────────────────────────────────
# EVALUATE FINAL MODEL
# ─────────────────────────────────────────────
def evaluate_model(model: tf.keras.Model,
                   X_test: np.ndarray, y_test: np.ndarray,
                   scaler,
                   df_raw: pd.DataFrame) -> dict:
    """
    Runs the trained model on the test set and computes:
      - MSE / RMSE / MAE on scaled values
      - RMSE in actual INR (inverse-transformed)
    Also derives the market regime classification:
      Oversold / Neutral / Overbought based on predicted price vs EMA 200.
    """
    print("\n[4/4] Evaluating final model on test set ...")

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    mse  = mean_squared_error(y_test, y_pred_scaled)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred_scaled)

    # ── Inverse transform to real INR prices ─────────────────────────────────
    # Scaler was fit on ALL features; Close is column 0.
    # We reconstruct a dummy array with the correct number of columns.
    n_features = scaler.n_features_in_

    def inverse_close(scaled_col: np.ndarray) -> np.ndarray:
        dummy = np.zeros((len(scaled_col), n_features))
        dummy[:, 0] = scaled_col
        return scaler.inverse_transform(dummy)[:, 0]

    y_test_inr = inverse_close(y_test)
    y_pred_inr = inverse_close(y_pred_scaled)

    rmse_inr = np.sqrt(mean_squared_error(y_test_inr, y_pred_inr))
    mae_inr  = mean_absolute_error(y_test_inr, y_pred_inr)

    # ── Market Regime ─────────────────────────────────────────────────────────
    # Use the last EMA 200 value from the enriched DataFrame as the reference
    ema200_last = df_raw["ema_200"].iloc[-1]
    last_pred   = y_pred_inr[-1]

    if last_pred < ema200_last * 0.97:
        regime = "Oversold"
    elif last_pred > ema200_last * 1.03:
        regime = "Overbought"
    else:
        regime = "Neutral"

    metrics = {
        "mse_scaled":  round(mse, 8),
        "rmse_scaled": round(rmse, 8),
        "mae_scaled":  round(mae, 8),
        "rmse_inr":    round(rmse_inr, 2),
        "mae_inr":     round(mae_inr, 2),
        "regime":      regime,
        "last_pred_inr": round(last_pred, 2),
        "ema200_inr":    round(ema200_last, 2),
        "y_test_inr":  y_test_inr,
        "y_pred_inr":  y_pred_inr,
    }

    print(f"    ✓ RMSE (scaled) : {rmse:.8f}")
    print(f"    ✓ MAE  (scaled) : {mae:.8f}")
    print(f"    ✓ RMSE (INR)    : ₹{rmse_inr:,.2f}")
    print(f"    ✓ MAE  (INR)    : ₹{mae_inr:,.2f}")
    print(f"    ✓ Last predicted: ₹{last_pred:,.2f}")
    print(f"    ✓ EMA 200 ref   : ₹{ema200_last:,.2f}")
    print(f"    ✓ Market Regime : {regime}")

    return metrics


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
def save_plots(history, metrics: dict, wf_results: pd.DataFrame) -> None:
    """
    Saves three plots to artifacts/plots/:
      1. Training & Validation Loss curve
      2. Predicted vs Actual Close price (INR)
      3. Walk-Forward RMSE per fold
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Plot 1: Loss Curve ────────────────────────────────────────────────────
    if history is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(history.history["loss"],     label="Train Loss (MSE)")
        plt.plot(history.history["val_loss"], label="Val Loss (MSE)")
        plt.title("SmartSIP LSTM — Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/loss_curve.png", dpi=120)
        plt.close()
        print(f"    ✓ Saved: {PLOTS_DIR}/loss_curve.png")

    # ── Plot 2: Prediction vs Actual ─────────────────────────────────────────
    plt.figure(figsize=(14, 5))
    plt.plot(metrics["y_test_inr"], label="Actual Close (INR)",    color="steelblue")
    plt.plot(metrics["y_pred_inr"], label="Predicted Close (INR)", color="tomato",
             linestyle="--", alpha=0.8)
    plt.title("SmartSIP LSTM — Predicted vs Actual Nifty 50 Close")
    plt.xlabel("Test Day Index")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/prediction_vs_actual.png", dpi=120)
    plt.close()
    print(f"    ✓ Saved: {PLOTS_DIR}/prediction_vs_actual.png")

    # ── Plot 3: Walk-Forward RMSE ─────────────────────────────────────────────
    plt.figure(figsize=(7, 4))
    plt.bar(wf_results["fold"].astype(str), wf_results["RMSE"], color="mediumseagreen")
    plt.axhline(wf_results["RMSE"].mean(), color="red", linestyle="--",
                label=f"Mean RMSE = {wf_results['RMSE'].mean():.5f}")
    plt.title("Walk-Forward Validation — RMSE per Fold")
    plt.xlabel("Fold")
    plt.ylabel("RMSE (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/walk_forward_rmse.png", dpi=120)
    plt.close()
    print(f"    ✓ Saved: {PLOTS_DIR}/walk_forward_rmse.png")


# ─────────────────────────────────────────────
# TECHNICAL SCORE
# ─────────────────────────────────────────────
def compute_technical_score(model: tf.keras.Model,
                             scaler,
                             df_raw: pd.DataFrame,
                             window_size: int = 60) -> float:
    """
    Computes the normalised Technical Score [0.0 – 1.0] for TODAY.
    This is the value that flows into the multiplier formula:
        Final = (Technical_Score * 0.6) + (Sentiment_Score * 0.4)

    Logic:
        1. Take the most recent 60-day window from nifty_enriched.csv
        2. Run LSTM inference → get predicted next-day Close
        3. Compare prediction against EMA 200 and current RSI
        4. Map the combined signal to [0, 1] where:
               0.0 = strong sell (Overbought)
               0.5 = neutral
               1.0 = strong buy  (Oversold)
    """
    from data_pipeline import FEATURE_COLUMNS

    # Build live feature window from the last `window_size` rows
    feature_data = df_raw[FEATURE_COLUMNS].values[-window_size:]

    # Re-scale using the saved scaler
    feature_scaled = scaler.transform(feature_data)

    # Shape → (1, 60, 14) for LSTM
    X_live = feature_scaled.reshape(1, window_size, len(FEATURE_COLUMNS))

    # Predict
    pred_scaled = model.predict(X_live, verbose=0)[0][0]

    # Inverse-transform just the Close column
    n_features = scaler.n_features_in_
    dummy = np.zeros((1, n_features))
    dummy[0, 0] = pred_scaled
    pred_inr = scaler.inverse_transform(dummy)[0][0]

    # ── Scoring logic ─────────────────────────────────────────────────────────
    ema200   = df_raw["ema_200"].iloc[-1]
    rsi_now  = df_raw["rsi"].iloc[-1]

    # Price deviation from EMA 200 (normalised)
    price_deviation = (pred_inr - ema200) / ema200   # negative = undervalued

    # RSI score: oversold (<30) → 1.0, overbought (>70) → 0.0
    rsi_score = 1.0 - (rsi_now / 100.0)

    # Combine: 60% price-based, 40% RSI-based
    raw_score = (-price_deviation * 5 * 0.6) + (rsi_score * 0.4)

    # Clip to [0, 1]
    technical_score = float(np.clip(raw_score, 0.0, 1.0))

    return technical_score


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(skip_training: bool = False):
    print("=" * 55)
    print("  SmartSIP — Phase 2: LSTM Training & Validation")
    print("=" * 55)

    # ── Load Phase 1 artifacts ────────────────────────────────────────────────
    print("\n[1/4] Loading Phase 1 artifacts ...")
    try:
        data = load_artifacts()
        print("    ✓ Artifacts loaded from disk (no re-download needed)")
    except FileNotFoundError:
        print("    ⚠  Artifacts not found — running Phase 1 pipeline first ...")
        data = build_dataset(save=True)

    X_train      = data["X_train"]
    X_test       = data["X_test"]
    y_train      = data["y_train"]
    y_test       = data["y_test"]
    scaler       = data["scaler"]
    df_raw       = data["df"]
    window_size  = data["window_size"]
    n_features   = len(data["feature_cols"])

    print(f"    X_train : {X_train.shape}")
    print(f"    X_test  : {X_test.shape}")
    print(f"    Features: {n_features}  |  Window: {window_size} days")

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(window_size, n_features)
    model.summary()

    history = None

    if not skip_training:
        # ── Train ─────────────────────────────────────────────────────────────
        history = train_model(model, X_train, y_train, X_test, y_test)
    else:
        # ── Load best saved weights ───────────────────────────────────────────
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            print(f"\n    ✓ Loaded saved model from '{MODEL_PATH}'")
        else:
            print(f"\n    ✗ No saved model found at '{MODEL_PATH}'. Run without --load first.")
            sys.exit(1)

    # ── Walk-Forward Validation ───────────────────────────────────────────────
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    wf_results = walk_forward_validation(X_all, y_all, window_size, n_features)

    # ── Final Evaluation ──────────────────────────────────────────────────────
    metrics = evaluate_model(model, X_test, y_test, scaler, df_raw)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n    Saving plots ...")
    save_plots(history, metrics, wf_results)

    # ── Live Technical Score (smoke test) ────────────────────────────────────
    print("\n─── Live Technical Score (today) ─────────────────")
    score = compute_technical_score(model, scaler, df_raw, window_size)
    print(f"    Technical Score = {score:.4f}  "
          f"({'Bullish / Top-Up' if score > 0.5 else 'Bearish / Hold'})")

    print("\n Model complete!")
    print(f"   Model saved : {MODEL_PATH}")
    print(f"   Plots saved : {PLOTS_DIR}/")
    print(f"\n   Pass this into FastAPI:")
    print(f"     model  = load_model('{MODEL_PATH}')")
    print(f"     scaler = pickle.load(open('artifacts/scaler.pkl', 'rb'))")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartSIP Phase 2 — LSTM Training")
    parser.add_argument("--load", action="store_true",
                        help="Skip training and load existing saved model")
    args = parser.parse_args()
    main(skip_training=args.load)