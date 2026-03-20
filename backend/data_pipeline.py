"""
Data Engineering Pipeline
=======================================================
Fetches Nifty 50 data, computes all technical indicators,
scales features, and builds the 60-day sliding window
sequences required by the LSTM model.

Library note: Uses 'ta' (pip install ta) instead of pandas-ta
which is broken on Python 3.10 PyPI.

Usage:
    python data_pipeline.py                  # runs full pipeline & saves artifacts
    from data_pipeline import build_dataset  # import in lstm_model.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ta library — pip install ta
import ta
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TICKER          = "^NSEI"          # Nifty 50
PERIOD_YEARS    = 10               # years of historical data to fetch
WINDOW_SIZE     = 60               # sliding window (days) for LSTM input
TRAIN_SPLIT     = 0.80             # 80% train / 20% test
ARTIFACTS_DIR   = "artifacts"      # folder where scaler + sequences are saved

# Technical indicator parameters
RSI_PERIOD      = 14
BB_PERIOD       = 20
BB_STD          = 2
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
STD_PERIOD      = 20
EMA_SHORT       = 50
EMA_LONG        = 200


# ─────────────────────────────────────────────
# DATA INGESTION
# ─────────────────────────────────────────────
def fetch_data(ticker: str = TICKER, years: int = PERIOD_YEARS) -> pd.DataFrame:
    """
    Downloads historical OHLCV data from Yahoo Finance.
    Returns a clean DataFrame indexed by date.
    """
    print(f"[1/5] Fetching {years} years of data for {ticker} ...")

    end_date   = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Flatten multi-level columns if present (yfinance >= 0.2)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    print(f"    ✓ Downloaded {len(df)} trading days  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all indicators specified in the project document:
      • Log Returns        — ensures stationarity for the LSTM
      • RSI (14)           — momentum / exhaustion detector
      • Bollinger Bands    — volatility-based price channels
      • MACD (12/26/9)     — trend strength
      • Std Dev (20-day)   — raw risk/dispersion measure
      • EMA 50 / EMA 200   — Golden Cross / Death Cross signals
    """
    print("[2/5] Computing technical indicators ...")

    close = df["Close"].squeeze()   # ensure 1-D Series for ta library

    # ── Log Returns ──────────────────────────────────────────────────────────
    # ln(Pt / Pt-1) — makes the series stationary for the neural network
    df["log_return"] = np.log(close / close.shift(1))

    # ── RSI (14-period) ───────────────────────────────────────────────────────
    rsi_ind = RSIIndicator(close=close, window=RSI_PERIOD)
    df["rsi"] = rsi_ind.rsi()

    # ── Bollinger Bands (20-day SMA ± 2 SD) ──────────────────────────────────
    bb_ind = BollingerBands(close=close, window=BB_PERIOD, window_dev=BB_STD)
    df["bb_upper"] = bb_ind.bollinger_hband()
    df["bb_mid"]   = bb_ind.bollinger_mavg()
    df["bb_lower"] = bb_ind.bollinger_lband()
    # Bandwidth = (upper - lower) / mid — measures squeeze strength
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # ── MACD (12 / 26 / 9) ───────────────────────────────────────────────────
    macd_ind = MACD(close=close,
                    window_fast=MACD_FAST,
                    window_slow=MACD_SLOW,
                    window_sign=MACD_SIGNAL)
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"]   = macd_ind.macd_diff()

    # ── 20-day Rolling Standard Deviation ────────────────────────────────────
    df["std_20"] = close.rolling(window=STD_PERIOD).std()

    # ── EMA 50 & EMA 200 ─────────────────────────────────────────────────────
    df["ema_50"]  = EMAIndicator(close=close, window=EMA_SHORT).ema_indicator()
    df["ema_200"] = EMAIndicator(close=close, window=EMA_LONG).ema_indicator()

    # Golden / Death Cross flag: +1 = golden cross, -1 = death cross, 0 = neutral
    df["cross_signal"] = np.where(df["ema_50"] > df["ema_200"],  1,
                         np.where(df["ema_50"] < df["ema_200"], -1, 0))

    # ── Drop rows with NaN (EMA 200 needs ~200 days to warm up) ──────────────
    df.dropna(inplace=True)
    df.reset_index(drop=False, inplace=True)   # keep 'Date' as a plain column

    print(f"    ✓ Features computed. Dataset shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────
# These are the columns fed into the LSTM (input vector)
FEATURE_COLUMNS = [
    "Close",
    "log_return",
    "rsi",
    "bb_upper", "bb_lower", "bb_width",
    "macd", "macd_signal", "macd_hist",
    "std_20",
    "ema_50", "ema_200",
    "cross_signal",
    "Volume"
]

TARGET_COLUMN = "Close"   # what we want to predict


# ─────────────────────────────────────────────
# SCALING
# ─────────────────────────────────────────────
def scale_features(df: pd.DataFrame,
                   feature_cols: list = FEATURE_COLUMNS
                   ) -> tuple:
    """
    Applies MinMaxScaler to normalise all feature columns to [0, 1].
    Returns:
        scaled_array  — numpy array of shape (n_rows, n_features)
        scaler        — fitted MinMaxScaler (must be saved for inference)
    """
    print("[3/5] Scaling features with MinMaxScaler ...")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[feature_cols].values)

    print(f"    ✓ Scaled {len(feature_cols)} features over {scaled.shape[0]} rows")
    return scaled, scaler


# ─────────────────────────────────────────────
# SLIDING WINDOW (60-day sequences)
# ─────────────────────────────────────────────
def create_sequences(scaled_data: np.ndarray,
                     window: int = WINDOW_SIZE
                     ) -> tuple:
    """
    Builds the (X, y) pairs used by the LSTM.

    For each position i:
        X[i] = scaled_data[i : i+window]        shape → (window, n_features)
        y[i] = scaled_data[i+window, 0]         close price (index 0) at t+1

    Returns:
        X — shape (n_samples, window, n_features)
        y — shape (n_samples,)
    """
    print("[4/5] Building 60-day sliding window sequences ...")

    X, y = [], []
    for i in range(len(scaled_data) - window):
        X.append(scaled_data[i : i + window])          # 60-day feature window
        y.append(scaled_data[i + window, 0])            # next Close (col 0)

    X, y = np.array(X), np.array(y)
    print(f"    ✓ Created {len(X)} sequences — X: {X.shape}, y: {y.shape}")
    return X, y


# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
def split_data(X: np.ndarray, y: np.ndarray,
               train_ratio: float = TRAIN_SPLIT
               ) -> tuple:
    """
    Chronological split — NO shuffling (time-series integrity).
    Returns: X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * train_ratio)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"    ✓ Train: {len(X_train)} sequences | "
          f"Test: {len(X_test)} sequences  (split @ {train_ratio:.0%})")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# PERSIST ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(scaler: MinMaxScaler,
                   X_train: np.ndarray, X_test: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray,
                   df: pd.DataFrame,
                   out_dir: str = ARTIFACTS_DIR) -> None:
    """
    Saves the scaler and processed arrays to disk so that
    Phase 2 (lstm_model.py) can load them without re-running the pipeline.
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save(f"{out_dir}/X_train.npy", X_train)
    np.save(f"{out_dir}/X_test.npy",  X_test)
    np.save(f"{out_dir}/y_train.npy", y_train)
    np.save(f"{out_dir}/y_test.npy",  y_test)

    # Save the raw enriched DataFrame for backtesting (Phase 6 frontend)
    df.to_csv(f"{out_dir}/nifty_enriched.csv", index=False)

    print(f"[5/5] Artifacts saved to '{out_dir}/'")
    print(f"      scaler.pkl | X_train.npy | X_test.npy | "
          f"y_train.npy | y_test.npy | nifty_enriched.csv")


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT  (used by lstm_model.py)
# ─────────────────────────────────────────────
def build_dataset(save: bool = True) -> dict:
    """
    Runs the full pipeline end-to-end.

    Returns a dict with keys:
        X_train, X_test, y_train, y_test  — numpy arrays
        scaler                            — fitted MinMaxScaler
        df                                — enriched DataFrame
        feature_cols                      — list of feature names
        window_size                       — int
    """
    print("=" * 55)
    print("  SmartSIP — Phase 1: Data Pipeline")
    print("=" * 55)

    df             = fetch_data()
    df             = add_technical_indicators(df)
    scaled, scaler = scale_features(df)
    X, y           = create_sequences(scaled)
    X_train, X_test, y_train, y_test = split_data(X, y)

    if save:
        save_artifacts(scaler, X_train, X_test, y_train, y_test, df)

    print("\n Pipeline complete!\n")

    return {
        "X_train":      X_train,
        "X_test":       X_test,
        "y_train":      y_train,
        "y_test":       y_test,
        "scaler":       scaler,
        "df":           df,
        "feature_cols": FEATURE_COLUMNS,
        "window_size":  WINDOW_SIZE,
    }


# ─────────────────────────────────────────────
# HELPER — load saved artifacts
# ─────────────────────────────────────────────
def load_artifacts(out_dir: str = ARTIFACTS_DIR) -> dict:
    """
    Loads previously saved artifacts so it doesn't
    need to re-download/re-process data.
    """
    with open(f"{out_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return {
        "X_train":      np.load(f"{out_dir}/X_train.npy"),
        "X_test":       np.load(f"{out_dir}/X_test.npy"),
        "y_train":      np.load(f"{out_dir}/y_train.npy"),
        "y_test":       np.load(f"{out_dir}/y_test.npy"),
        "scaler":       scaler,
        "df":           pd.read_csv(f"{out_dir}/nifty_enriched.csv"),
        "feature_cols": FEATURE_COLUMNS,
        "window_size":  WINDOW_SIZE,
    }


# ─────────────────────────────────────────────
# RUN DIRECTLY
# ─────────────────────────────────────────────
if __name__ == "__main__":
    data = build_dataset(save=True)

    print("\n─── Quick Sanity Check ───────────────────────────")
    print(f"X_train shape : {data['X_train'].shape}")
    print(f"X_test  shape : {data['X_test'].shape}")
    print(f"y_train shape : {data['y_train'].shape}")
    print(f"y_test  shape : {data['y_test'].shape}")

    df = data["df"]
    print(f"\nLast 3 rows of enriched DataFrame:")
    print(df[["Date", "Close", "rsi", "macd", "ema_50", "ema_200",
              "cross_signal", "bb_width"]].tail(3).to_string(index=False))