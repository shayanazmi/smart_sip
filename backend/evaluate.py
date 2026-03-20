"""
SmartSIP - Model Evaluation Report
====================================
Run this after lstm_model.py to get a complete accuracy breakdown:
  - RMSE / MAE in INR
  - MAPE  (Mean Absolute Percentage Error)  ← the most intuitive metric
  - Directional Accuracy                    ← did the model predict UP/DOWN correctly?
  - Regime Classification Accuracy          ← Oversold / Neutral / Overbought

Usage:
    python evaluate.py
"""

import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_pipeline import ARTIFACTS_DIR, FEATURE_COLUMNS, WINDOW_SIZE

# ── Load artifacts ────────────────────────────────────────────────────────────
model  = load_model(f"{ARTIFACTS_DIR}/smartsip_lstm.h5")
scaler = pickle.load(open(f"{ARTIFACTS_DIR}/scaler.pkl", "rb"))
X_test = np.load(f"{ARTIFACTS_DIR}/X_test.npy")
y_test = np.load(f"{ARTIFACTS_DIR}/y_test.npy")
df     = pd.read_csv(f"{ARTIFACTS_DIR}/nifty_enriched.csv")

# ── Predict ───────────────────────────────────────────────────────────────────
y_pred_scaled = model.predict(X_test, verbose=0).flatten()

# ── Inverse transform to INR ──────────────────────────────────────────────────
n = scaler.n_features_in_

def to_inr(col):
    dummy = np.zeros((len(col), n))
    dummy[:, 0] = col
    return scaler.inverse_transform(dummy)[:, 0]

y_true_inr = to_inr(y_test)
y_pred_inr = to_inr(y_pred_scaled)

# ── Core metrics ──────────────────────────────────────────────────────────────
rmse = np.sqrt(mean_squared_error(y_true_inr, y_pred_inr))
mae  = mean_absolute_error(y_true_inr, y_pred_inr)
mape = np.mean(np.abs((y_true_inr - y_pred_inr) / y_true_inr)) * 100

# ── Directional accuracy ──────────────────────────────────────────────────────
# Did the model correctly predict whether the market moved UP or DOWN?
true_dir = np.diff(y_true_inr)   # +ve = up day, -ve = down day
pred_dir = np.diff(y_pred_inr)
dir_acc  = np.mean(np.sign(true_dir) == np.sign(pred_dir)) * 100

# ── Regime classification accuracy ───────────────────────────────────────────
# Compare predicted regime vs actual regime using EMA 200 as reference
test_start = len(df) - len(y_test)
ema200_test = df["ema_200"].values[test_start : test_start + len(y_test)]

def classify_regime(price, ema200):
    deviation = (price - ema200) / ema200
    if deviation < -0.03:  return "Oversold"
    elif deviation > 0.03: return "Overbought"
    else:                  return "Neutral"

true_regimes = [classify_regime(p, e) for p, e in zip(y_true_inr, ema200_test)]
pred_regimes = [classify_regime(p, e) for p, e in zip(y_pred_inr, ema200_test)]
regime_acc   = np.mean([t == p for t, p in zip(true_regimes, pred_regimes)]) * 100

# ── Print report ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  SmartSIP — Model Evaluation Report")
print("=" * 50)

print(f"\n  Test set size     : {len(y_test)} trading days")
print(f"  Index range (true): ₹{y_true_inr.min():,.0f} – ₹{y_true_inr.max():,.0f}")

print(f"\n── Price Prediction Accuracy ─────────────────")
print(f"  RMSE              : ₹{rmse:,.2f}")
print(f"  MAE               : ₹{mae:,.2f}")
print(f"  MAPE              : {mape:.2f}%   ← % error on average")

print(f"\n── Directional Accuracy ──────────────────────")
print(f"  Up/Down correct   : {dir_acc:.1f}%  ← did it predict the right direction?")
print(f"  (Random baseline  : 50.0%)")

print(f"\n── Regime Classification ─────────────────────")
print(f"  Regime accuracy   : {regime_acc:.1f}%  ← Oversold/Neutral/Overbought")

print(f"\n── Interpretation ────────────────────────────")
print(f"  MAPE < 3%  → {'✓ Excellent' if mape < 3 else '~ Acceptable' if mape < 5 else '✗ Needs work'}")
print(f"  Dir  > 55% → {'✓ Useful signal' if dir_acc > 55 else '~ Marginal' if dir_acc > 50 else '✗ No better than random'}")
print(f"  Regime > 70% → {'✓ Reliable' if regime_acc > 70 else '~ Acceptable' if regime_acc > 55 else '✗ Needs work'}")
print("=" * 50)