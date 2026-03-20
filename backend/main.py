"""
FastAPI Backend
====================================
Orchestrates the full AI pipeline and exposes a single
REST endpoint that the React frontend (Phase 5) calls.

Endpoint:
    GET  /get-recommendation          → full SmartSIP recommendation
    GET  /health                      → server health check
    GET  /sentiment-trendline         → 30-day sentiment sparkline data

On startup the server:
    1. Loads the trained LSTM model  (artifacts/smartsip_lstm.h5)
    2. Loads the fitted scaler       (artifacts/scaler.pkl)
    3. Loads the enriched DataFrame  (artifacts/nifty_enriched.csv)
    4. Connects to Ollama            (localhost:11434)

Run:
    uvicorn main:app --reload --port 8000

Then open:
    http://localhost:8000/docs        → Swagger UI (auto-generated)
    http://localhost:8000/get-recommendation
"""

import os
import pickle
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# ── data_pipeline & lstm_model imports ──────────────────────────────────────────────────────
from data_pipeline import ARTIFACTS_DIR, FEATURE_COLUMNS, WINDOW_SIZE
from lstm_model    import compute_technical_score

# ── sentiment imports ──────────────────────────────────────────────────────────
from sentiment import (get_sentiment_score,
                       get_llm_explanation,
                       get_sentiment_trendline)

# ─────────────────────────────────────────────
# GLOBAL STATE  (loaded once on startup)
# ─────────────────────────────────────────────
class AppState:
    model    = None
    scaler   = None
    df       = None

state = AppState()


# ─────────────────────────────────────────────
# LIFESPAN — load models on startup
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads all heavy artifacts once when the server starts."""
    print("\n SmartSIP API starting up ...")

    model_path  = f"{ARTIFACTS_DIR}/smartsip_lstm.h5"
    scaler_path = f"{ARTIFACTS_DIR}/scaler.pkl"
    df_path     = f"{ARTIFACTS_DIR}/nifty_enriched.csv"

    # Validate files exist
    for path in [model_path, scaler_path, df_path]:
        if not os.path.exists(path):
            raise RuntimeError(
                f"Required artifact not found: {path}\n"
                f"Run data_pipeline.py and lstm_model.py first."
            )

    state.model  = load_model(model_path)
    print(f"   ✓ LSTM model loaded       ({model_path})")

    with open(scaler_path, "rb") as f:
        state.scaler = pickle.load(f)
    print(f"   ✓ Scaler loaded           ({scaler_path})")

    state.df = pd.read_csv(df_path)
    print(f"   ✓ Enriched DataFrame loaded — {len(state.df)} rows")

    print("   ✓ SmartSIP API ready!\n")
    yield

    # Shutdown
    print("👋 SmartSIP API shutting down.")


# ─────────────────────────────────────────────
# APP INITIALISATION
# ─────────────────────────────────────────────
app = FastAPI(
    title       = "SmartSIP API",
    description = "AI-Driven Dynamic SIP Accelerator — Nifty 50",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# Allow React dev server (port 3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────────
class IndicatorSnapshot(BaseModel):
    rsi:           float
    macd:          float
    macd_signal:   float
    bb_upper:      float
    bb_lower:      float
    bb_width:      float
    ema_50:        float
    ema_200:       float
    cross_signal:  int
    std_20:        float
    close:         float
    log_return:    float


class RecommendationResponse(BaseModel):
    # Core recommendation
    base_sip_amount:    int
    final_multiplier:   float
    topup_amount:       int
    total_investment:   int

    # AI scores
    technical_score:    float
    sentiment_score:    float

    # Market context
    regime:             str          # Oversold / Neutral / Overbought
    sentiment_label:    str          # Fearful / Neutral / Greedy
    predicted_close:    float        # LSTM next-day prediction (INR)
    current_close:      float        # today's actual close (INR)
    ema_200:            float

    # Logic breakdown (for radar chart in frontend)
    technical_weight:   float        # always 0.6
    sentiment_weight:   float        # always 0.4

    # Deep-dive indicators (for mini-charts)
    indicators:         IndicatorSnapshot

    # LLM explanation
    explanation:        str

    # News
    headlines:          list[str]
    article_count:      int


class TrendlinePoint(BaseModel):
    date:            str
    sentiment_score: float


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _get_latest_indicators() -> dict:
    """Extracts the most recent row of technical indicators from the DataFrame."""
    row = state.df.iloc[-1]
    return {
        "rsi":          float(row["rsi"]),
        "macd":         float(row["macd"]),
        "macd_signal":  float(row["macd_signal"]),
        "bb_upper":     float(row["bb_upper"]),
        "bb_lower":     float(row["bb_lower"]),
        "bb_width":     float(row["bb_width"]),
        "ema_50":       float(row["ema_50"]),
        "ema_200":      float(row["ema_200"]),
        "cross_signal": int(row["cross_signal"]),
        "std_20":       float(row["std_20"]),
        "close":        float(row["Close"]),
        "log_return":   float(row["log_return"]),
    }


def _determine_regime(predicted_close: float, ema_200: float, rsi: float) -> str:
    """
    Classifies the market into one of three regimes:
        Oversold    → good time to top-up (buy the dip)
        Neutral     → invest normally
        Overbought  → caution, reduce top-up
    """
    price_vs_ema = (predicted_close - ema_200) / ema_200   # % deviation

    if rsi < 35 or price_vs_ema < -0.03:
        return "Oversold"
    elif rsi > 65 or price_vs_ema > 0.03:
        return "Overbought"
    else:
        return "Neutral"


def _predict_next_close(scaler, model, df: pd.DataFrame) -> float:
    """Runs LSTM inference on the latest 60-day window → returns INR price."""
    feature_data   = df[FEATURE_COLUMNS].values[-WINDOW_SIZE:]
    feature_scaled = scaler.transform(feature_data)
    X_live         = feature_scaled.reshape(1, WINDOW_SIZE, len(FEATURE_COLUMNS))

    pred_scaled = model.predict(X_live, verbose=0)[0][0]

    n_features  = scaler.n_features_in_
    dummy       = np.zeros((1, n_features))
    dummy[0, 0] = pred_scaled
    pred_inr    = scaler.inverse_transform(dummy)[0][0]

    return round(float(pred_inr), 2)


def _compute_multiplier(technical_score: float,
                        sentiment_score: float) -> float:
    """
    Core formula from the project spec:
        multiplier = (Technical_Score * 0.6) + (Sentiment_Score * 0.4)

    Clamped to [0.25, 2.0]:
        0.25 — minimum: always invest at least 25% of base SIP
        2.0  — maximum: never invest more than 2× base SIP
    """
    raw = (technical_score * 0.6) + (sentiment_score * 0.4)
    return round(float(np.clip(raw, 0.25, 2.0)), 4)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """Quick liveness check — the frontend polls this on load."""
    return {
        "status":    "ok",
        "model":     "loaded" if state.model  else "not loaded",
        "scaler":    "loaded" if state.scaler else "not loaded",
        "dataframe": f"{len(state.df)} rows" if state.df is not None else "not loaded",
    }


@app.get("/get-recommendation", response_model=RecommendationResponse)
async def get_recommendation(base_sip: int = 5000):
    """
    Main endpoint — orchestrates the full SmartSIP pipeline:

    1. LSTM inference           → predicted next-day Close + Technical Score
    2. Marketaux sentiment      → Sentiment Score + headlines
    3. Multiplier formula       → (Tech * 0.6) + (Sentiment * 0.4)
    4. Llama 3.2 explanation    → human-readable reasoning
    5. Returns structured JSON  → consumed by React dashboard
    """
    try:
        # ── Step 1: Technical signals ─────────────────────────────────────────
        indicators      = _get_latest_indicators()
        predicted_close = _predict_next_close(state.scaler, state.model, state.df)
        technical_score = compute_technical_score(
            state.model, state.scaler, state.df, WINDOW_SIZE
        )
        regime = _determine_regime(
            predicted_close, indicators["ema_200"], indicators["rsi"]
        )

        # ── Step 2: Sentiment ─────────────────────────────────────────────────
        sentiment_data  = get_sentiment_score()
        sentiment_score = sentiment_data["sentiment_score"]
        headlines       = sentiment_data["headlines"]

        # ── Step 3: Multiplier ────────────────────────────────────────────────
        multiplier       = _compute_multiplier(technical_score, sentiment_score)
        topup_amount     = int(base_sip * multiplier)
        total_investment = base_sip + topup_amount

        # ── Step 4: LLM explanation ───────────────────────────────────────────
        explanation = get_llm_explanation(
            technical_score  = technical_score,
            sentiment_score  = sentiment_score,
            final_multiplier = multiplier,
            regime           = regime,
            rsi              = indicators["rsi"],
            macd             = indicators["macd"],
            headlines        = headlines,
            base_sip         = base_sip,
        )

        # ── Step 5: Return ────────────────────────────────────────────────────
        return RecommendationResponse(
            base_sip_amount   = base_sip,
            final_multiplier  = multiplier,
            topup_amount      = topup_amount,
            total_investment  = total_investment,
            technical_score   = round(technical_score, 4),
            sentiment_score   = round(sentiment_score, 4),
            regime            = regime,
            sentiment_label   = sentiment_data["label"],
            predicted_close   = predicted_close,
            current_close     = indicators["close"],
            ema_200           = round(indicators["ema_200"], 2),
            technical_weight  = 0.6,
            sentiment_weight  = 0.4,
            indicators        = IndicatorSnapshot(**indicators),
            explanation       = explanation,
            headlines         = headlines,
            article_count     = sentiment_data["article_count"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sentiment-trendline", response_model=list[TrendlinePoint])
async def sentiment_trendline(days: int = 30):
    """
    Returns 30-day sentiment history for the frontend sparkline chart.
    """
    try:
        trendline = get_sentiment_trendline(days=days)
        return trendline
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))