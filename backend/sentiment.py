"""
Sentiment & LLM Integration
================================================
Two responsibilities:
  A) Marketaux API  — fetches today's financial headlines for Nifty/India
                      and extracts a normalised sentiment score [0.0 – 1.0]

  B) Ollama Bridge  — sends technical signals + headlines to Llama 3.2 1B
                      and receives a human-readable investment explanation

The sentiment score feeds directly into the multiplier formula (Phase 5):
    Final_Multiplier = (Technical_Score * 0.6) + (Sentiment_Score * 0.4)

Setup required:
  1. Get a free Marketaux API key → https://marketaux.com  (free tier: 100 req/day)
  2. Create a .env file in your project root:
         MARKETAUX_API_KEY=your_key_here
  3. Install & run Ollama → https://ollama.com
         ollama pull llama3.2:1b
         ollama serve          (runs on localhost:11434 by default)

Usage:
    python sentiment.py               # run full sentiment pipeline
    from sentiment import get_sentiment_score, get_llm_explanation
"""

import os
import re
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()   # loads MARKETAUX_API_KEY from .env

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MARKETAUX_API_KEY   = os.getenv("MARKETAUX_API_KEY", "")
MARKETAUX_BASE_URL  = "https://api.marketaux.com/v1/news/all"

# News search terms for Nifty 50 / Indian market sentiment
NEWS_SYMBOLS        = "NSEI"
NEWS_KEYWORDS       = "Nifty,India market,BSE,RBI,FII,Sensex"
NEWS_LANGUAGE       = "en"
NEWS_LIMIT          = 10          # headlines per call (free tier limit)
NEWS_LOOKBACK_DAYS  = 1           # how many days back to pull headlines

# Ollama settings
OLLAMA_BASE_URL     = "http://localhost:11434"
OLLAMA_MODEL        = "llama3.2:1b"
OLLAMA_TIMEOUT      = 120         # seconds


# ─────────────────────────────────────────────
# MARKETAUX SENTIMENT
# ─────────────────────────────────────────────

def fetch_news(api_key: str = MARKETAUX_API_KEY) -> list[dict]:
    """
    Fetches recent financial headlines from Marketaux.
    Returns a list of article dicts, each containing:
        title, description, sentiment_score, published_at, url
    """
    if not api_key:
        print("    ⚠  MARKETAUX_API_KEY not set — using mock headlines for testing.")
        return _mock_headlines()

    published_after = (datetime.utcnow() - timedelta(days=NEWS_LOOKBACK_DAYS)
                       ).strftime("%Y-%m-%dT%H:%M")

    params = {
        "api_token":       api_key,
        "symbols":         NEWS_SYMBOLS,
        "language":        NEWS_LANGUAGE,
        "limit":           NEWS_LIMIT,
        "published_after": published_after,
        "sort":            "published_at",
    }

    try:
        resp = requests.get(MARKETAUX_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for item in data.get("data", []):
            # Marketaux provides pre-calculated sentiment per entity
            sentiment_scores = [
                e.get("sentiment_score", 0.0)
                for e in item.get("entities", [])
                if e.get("sentiment_score") is not None
            ]
            avg_sentiment = (sum(sentiment_scores) / len(sentiment_scores)
                             if sentiment_scores else 0.0)

            articles.append({
                "title":           item.get("title", ""),
                "description":     item.get("description", ""),
                "sentiment_score": round(avg_sentiment, 4),   # raw: -1 to +1
                "published_at":    item.get("published_at", ""),
                "url":             item.get("url", ""),
            })

        return articles

    except requests.exceptions.RequestException as e:
        print(f"    ⚠  Marketaux API error: {e} — falling back to mock data.")
        return _mock_headlines()


def _mock_headlines() -> list[dict]:
    """
    Fallback mock headlines used when API key is missing or API is down.
    Reflects the current bearish market regime seen in Phase 1/2 output.
    Sentiment scores range: -1.0 (very negative) to +1.0 (very positive).
    """
    return [
        {"title": "FII selling continues as global uncertainty weighs on Nifty",
         "description": "Foreign institutional investors pulled out ₹4,200 crore amid global risk-off sentiment.",
         "sentiment_score": -0.62, "published_at": datetime.utcnow().isoformat(), "url": ""},

        {"title": "RBI holds rates steady, signals data-dependent approach",
         "description": "The Reserve Bank of India kept repo rate unchanged at 6.5% as inflation stays within target.",
         "sentiment_score": 0.15, "published_at": datetime.utcnow().isoformat(), "url": ""},

        {"title": "Nifty slips below 23,500 on weak global cues",
         "description": "Bears dominated Dalal Street as US tariff concerns and weak earnings weighed on sentiment.",
         "sentiment_score": -0.55, "published_at": datetime.utcnow().isoformat(), "url": ""},

        {"title": "IT sector shows resilience amid market selloff",
         "description": "Infosys and TCS bucked the trend as rupee weakness boosted export earnings outlook.",
         "sentiment_score": 0.30, "published_at": datetime.utcnow().isoformat(), "url": ""},

        {"title": "India GDP growth forecast revised to 6.8% for FY26",
         "description": "IMF raises India's growth outlook, citing robust domestic consumption and capex spending.",
         "sentiment_score": 0.48, "published_at": datetime.utcnow().isoformat(), "url": ""},
    ]


def compute_sentiment_score(articles: list[dict]) -> dict:
    """
    Aggregates article-level Marketaux sentiment scores into a single
    normalised Sentiment Score in [0.0, 1.0].

    Marketaux raw scores range from -1.0 (very negative) to +1.0 (very positive).
    Normalisation: score = (raw + 1) / 2
        -1.0  →  0.0  (maximum fear)
         0.0  →  0.5  (neutral)
        +1.0  →  1.0  (maximum greed)

    Returns a dict with:
        sentiment_score    — normalised float [0, 1]
        raw_avg            — mean of raw Marketaux scores
        label              — "Fearful" / "Neutral" / "Greedy"
        article_count      — number of articles used
        headlines          — list of headline strings (for LLM)
    """
    if not articles:
        return {
            "sentiment_score": 0.5,
            "raw_avg": 0.0,
            "label": "Neutral",
            "article_count": 0,
            "headlines": [],
        }

    raw_scores = [a["sentiment_score"] for a in articles]
    raw_avg    = sum(raw_scores) / len(raw_scores)

    # Normalise from [-1, 1] → [0, 1]
    normalised = round((raw_avg + 1.0) / 2.0, 4)
    normalised = max(0.0, min(1.0, normalised))   # safety clip

    # Label
    if normalised < 0.40:
        label = "Fearful"
    elif normalised > 0.60:
        label = "Greedy"
    else:
        label = "Neutral"

    headlines = [a["title"] for a in articles if a["title"]]

    return {
        "sentiment_score": normalised,
        "raw_avg":         round(raw_avg, 4),
        "label":           label,
        "article_count":   len(articles),
        "headlines":       headlines,
    }


def get_sentiment_score() -> dict:
    """
    Public entry point for Phase 5 (FastAPI).
    Returns the full sentiment dict including normalised score + headlines.
    """
    articles = fetch_news()
    result   = compute_sentiment_score(articles)
    return result


# ─────────────────────────────────────────────
# OLLAMA / LLAMA 3.2 BRIDGE
# ─────────────────────────────────────────────

def _check_ollama_running() -> bool:
    """Pings the Ollama server to confirm it's up."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _build_prompt(technical_score: float,
                  sentiment_score: float,
                  final_multiplier: float,
                  regime: str,
                  rsi: float,
                  macd: float,
                  headlines: list[str],
                  base_sip: int = 5000) -> str:
    """
    Constructs the structured prompt sent to Llama 3.2.
    The system prompt matches the project spec exactly:
        "You are a Senior Quantitative Investment Advisor.
         Explain financial decisions briefly and logically to retail investors."
    """
    top_up_amount = int(base_sip * final_multiplier)
    headline_text = "\n".join(f"  - {h}" for h in headlines[:5]) or "  - No headlines available."

    prompt = f"""You are a Senior Quantitative Investment Advisor. \
Explain financial decisions briefly and logically to retail investors.

Today's SmartSIP analysis for Nifty 50:

TECHNICAL INDICATORS:
  - Market Regime  : {regime}
  - RSI (14-period): {rsi:.1f} {"(Oversold zone)" if rsi < 35 else "(Overbought zone)" if rsi > 65 else "(Neutral zone)"}
  - MACD           : {macd:.2f} {"(Bearish)" if macd < 0 else "(Bullish)"}
  - Technical Score: {technical_score:.2f} / 1.00

NEWS SENTIMENT:
  - Sentiment Score: {sentiment_score:.2f} / 1.00
  - Top Headlines  :
{headline_text}

RECOMMENDATION:
  - Base SIP Amount   : ₹{base_sip:,}
  - Final Multiplier  : {final_multiplier:.2f}x
  - Recommended Top-Up: ₹{top_up_amount:,}

In 3-4 sentences, explain to a retail investor WHY this top-up amount \
was recommended today, referencing the specific indicators above. \
Be clear, logical, and avoid excessive jargon. End with one actionable insight."""

    return prompt


def get_llm_explanation(technical_score: float,
                        sentiment_score: float,
                        final_multiplier: float,
                        regime: str,
                        rsi: float,
                        macd: float,
                        headlines: list[str],
                        base_sip: int = 5000) -> str:
    """
    Sends the structured prompt to Llama 3.2 1B via Ollama's REST API
    and returns the natural language explanation string.

    Falls back to a template explanation if Ollama is not running.
    """
    if not _check_ollama_running():
        print("    ⚠  Ollama not running — returning template explanation.")
        return _fallback_explanation(technical_score, sentiment_score,
                                     final_multiplier, regime, rsi, base_sip)

    prompt = _build_prompt(technical_score, sentiment_score, final_multiplier,
                           regime, rsi, macd, headlines, base_sip)

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,    # low temperature = consistent, factual output
            "num_predict": 200,    # max tokens in response (~3-4 sentences)
            "top_p": 0.9,
        }
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )
        resp.raise_for_status()
        result = resp.json()
        explanation = result.get("response", "").strip()
        return explanation

    except requests.exceptions.RequestException as e:
        print(f"    ⚠  Ollama API error: {e} — returning template explanation.")
        return _fallback_explanation(technical_score, sentiment_score,
                                     final_multiplier, regime, rsi, base_sip)


def _fallback_explanation(technical_score: float,
                           sentiment_score: float,
                           final_multiplier: float,
                           regime: str,
                           rsi: float,
                           base_sip: int) -> str:
    """
    Rule-based explanation used when Ollama is unavailable.
    Mirrors the logic Llama would produce, ensuring the app never breaks.
    """
    top_up = int(base_sip * final_multiplier)
    rsi_desc = ("oversold" if rsi < 35 else "overbought" if rsi > 65 else "neutral")

    return (
        f"The Nifty 50 is currently in a {regime.lower()} regime with RSI at {rsi:.0f}, "
        f"indicating {rsi_desc} conditions — historically a favourable entry point. "
        f"News sentiment scored {sentiment_score:.2f}/1.00, reflecting mixed market mood. "
        f"Combining technical signals ({technical_score:.2f}) and sentiment ({sentiment_score:.2f}), "
        f"SmartSIP recommends a {final_multiplier:.2f}x top-up of ₹{top_up:,} today. "
        f"Consider this a disciplined opportunity to accumulate at lower prices."
    )


# ─────────────────────────────────────────────
# 30-DAY SENTIMENT TRENDLINE (for frontend)
# ─────────────────────────────────────────────

def get_sentiment_trendline(days: int = 30) -> list[dict]:
    """
    Returns a 30-day sentiment history list for the frontend sparkline chart.

    In production this would query a database of stored daily scores.
    Here we generate a realistic synthetic trendline seeded from today's score
    so the frontend always has data to render.

    Each entry: { "date": "YYYY-MM-DD", "sentiment_score": float }
    """
    import random
    import math

    today_result = get_sentiment_score()
    today_score  = today_result["sentiment_score"]

    trendline = []
    score = today_score

    for i in range(days - 1, -1, -1):
        date_str = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        # Add realistic mean-reverting noise
        noise = random.gauss(0, 0.04)
        score  = float(np.clip(score + noise, 0.1, 0.9)) if i > 0 else today_score
        trendline.append({
            "date":            date_str,
            "sentiment_score": round(score, 4)
        })

    # Ensure the last entry is exactly today's real score
    trendline[-1]["sentiment_score"] = today_score
    return trendline


# ─────────────────────────────────────────────
# MAIN — smoke test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    print("=" * 55)
    print("  SmartSIP — Phase 3: Sentiment Pipeline")
    print("=" * 55)

    # ── Part A: Sentiment Score ───────────────────────────────────────────────
    print("\n[1/3] Fetching news & computing sentiment ...")
    result = get_sentiment_score()

    print(f"\n    Articles fetched  : {result['article_count']}")
    print(f"    Raw avg sentiment : {result['raw_avg']}")
    print(f"    Normalised score  : {result['sentiment_score']}")
    print(f"    Sentiment label   : {result['label']}")
    print(f"\n    Headlines:")
    for h in result["headlines"]:
        print(f"      • {h}")

    # ── Simulate values from lstm_model output ──────────────────────────────────
    technical_score  = 0.36       # from lstm_model live score
    sentiment_score  = result["sentiment_score"]
    final_multiplier = round((technical_score * 0.6) + (sentiment_score * 0.4), 4)
    regime           = "Oversold"
    rsi              = 36.8
    macd             = -545.87

    print(f"\n[2/3] Computing final multiplier ...")
    print(f"    Technical Score  : {technical_score:.4f}  × 0.6 = {technical_score*0.6:.4f}")
    print(f"    Sentiment Score  : {sentiment_score:.4f}  × 0.4 = {sentiment_score*0.4:.4f}")
    print(f"    ─────────────────────────────────────────")
    print(f"    Final Multiplier : {final_multiplier:.4f}x")
    print(f"    Top-Up Amount    : ₹{int(5000 * final_multiplier):,}  (base: ₹5,000)")

    # ── Part B: LLM Explanation ───────────────────────────────────────────────
    print(f"\n[3/3] Generating LLM explanation (Llama 3.2 via Ollama) ...")
    ollama_up = _check_ollama_running()
    print(f"    Ollama status: {'✓ Running' if ollama_up else '⚠  Not running (using fallback)'}")

    explanation = get_llm_explanation(
        technical_score  = technical_score,
        sentiment_score  = sentiment_score,
        final_multiplier = final_multiplier,
        regime           = regime,
        rsi              = rsi,
        macd             = macd,
        headlines        = result["headlines"],
        base_sip         = 5000
    )

    print(f"\n    ── Llama 3.2 Explanation ──────────────────────")
    print(f"    {explanation}")

    print("\n Sentiment & LLM Integration complete!")
    print("\n    Ready for FastAPI. Import these two functions:")
    print("      from sentiment import get_sentiment_score, get_llm_explanation")