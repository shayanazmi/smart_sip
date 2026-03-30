
# SmartSIP - AI-Driven Dynamic Optimal SIP Top Up Calculator

> *"The stock market is a device for transferring money from the impatient to the patient."*
> — **Warren Buffett**

SmartSIP makes that patience *smarter*. It uses deep learning and real-time news sentiment to tell you **how much to Top Up in your SIP each month** - so you buy more when markets dip and hold back when they're overheated.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![Llama](https://img.shields.io/badge/Llama_3.2-1B-blueviolet?logo=meta)
![License](https://img.shields.io/badge/License-MIT-green)

⚠️ **DISCLAIMER:** *Kindly don't rely on this tool for definitive financial decisions. This is an AI model and it can make mistakes. Always conduct your own research or consult a certified financial advisor.*

---

## 📋 Table of Contents

- [What Is SmartSIP?](#what-is-smartsip)
- [Why We Built This](#why-we-built-this)
- [Key Performance Metrics](#key-performance-metrics)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [How to Use](#how-to-use)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Evaluation Results](#evaluation-results)
- [Future Scope](#future-scope)
- [Credits](#credits)

---

## What Is SmartSIP?

SmartSIP is an end-to-end AI system that **dynamically adjusts your monthly SIP (Systematic Investment Plan) contribution** based on real-time market conditions.

Instead of investing the same fixed ₹5,000 every month regardless of whether the market is crashing or at an all-time high, SmartSIP analyses:

1. **10 years of Nifty 50 price data** using a 3-layer stacked LSTM neural network.
2. **Live financial news headlines** using sentiment analysis via the Marketaux API.
3. **14 engineered technical indicators** (RSI, MACD, Bollinger Bands, EMA crossovers, and more).

…and gives you a single, easy-to-understand recommendation:

> *"This month, invest 1.34× your base SIP amount (₹6,700 instead of ₹5,000)"*

A locally hosted **Llama 3.2** language model then explains *why* in plain English — no finance degree needed.

---

## Why We Built This

Standard SIPs invest a fixed amount every month **without caring if the market is cheap or expensive**. That leaves money on the table.

We saw three gaps no existing tool was filling:

| Gap | Problem |
|-----|---------|
| **Analytical** | Retail investors don't have tools combining ML price forecasting + NLP sentiment. |
| **Accessibility** | Existing robo-advisors are expensive or target institutions. |
| **Explainability** | Black-box outputs give no reasoning a normal person can understand. |

SmartSIP bridges all three — it's **free, local, private, and explains itself**.

---

## Key Performance Metrics

| Metric | Value | What It Means |
|--------|-------|---------------|
| **MAPE** | 1.40% | Prediction error — anything under 3% is excellent. |
| **Regime Accuracy** | 80.3% | Correctly identifies Oversold / Neutral / Overbought market states. |
| **Walk-Forward RMSE** | 0.050 (5-fold mean) | Generalizes consistently well to unseen future data. |
| **Directional Accuracy** | > 55% | Beats random (50%) — providing a statistically meaningful edge. |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Data Ingestion** | `yfinance` 0.2.36, `pandas` 2.0+ |
| **Feature Engineering** | `ta` 0.11.0 (RSI, MACD, Bollinger Bands, EMA) |
| **ML Model** | TensorFlow 2.13 / Keras — 3-layer Stacked LSTM |
| **Sentiment API** | Marketaux REST API |
| **LLM Explainer** | Llama 3.2 1B via Ollama (runs 100% locally) |
| **Backend** | FastAPI 0.111 + Uvicorn |
| **Frontend** | Vanilla HTML / CSS / JavaScript + Chart.js |
| **Model Persistence** | Keras `.h5` + Python `pickle` |

---

## System Architecture

```text
Yahoo Finance ──▶ data_pipeline.py ──▶ lstm_model.py ──▶ sentiment.py 
                                                               │
                                                               ▼
index.html (Dashboard) ◀── main.py (FastAPI Endpoint) ◀────────┘
````

**The Inference Lifecycle:**

1.  FastAPI loads the global state (`smartsip_lstm.h5`, `scaler.pkl`, and enriched data) on startup.
2.  Extracts the latest **60-day price window** from the dataset.
3.  Normalizes it with the pre-fitted MinMaxScaler.
4.  Runs **LSTM inference** to predict the next-day Close price (₹).
5.  Computes a **Technical Score** [0–1] from predicted price vs EMA 200 + RSI.
6.  Fetches **live news headlines** from Marketaux to generate a **Sentiment Score** [0–1].
7.  Fuses both: `Multiplier = (Technical × 0.6) + (Sentiment × 0.4)`, clamped to **[0.25×, 2.0×]**.
8.  Calls **Llama 3.2** for a plain-English explanation and serves the JSON payload to the frontend.

-----

## Features

  - 📈 **LSTM Price Prediction:** 3-layer stacked LSTM trained on 10 years of Nifty 50 data with 14 technical features.
  - 📰 **Live News Sentiment:** Real-time financial headlines scored and normalized into a sentiment signal.
  - 🤖 **AI Explanations:** Llama 3.2 1B runs locally, generating human-readable investment reasoning ensuring full privacy.
  - 📊 **Interactive Dashboard:** Features a market health gauge, indicator deep-dives (RSI, MACD, Bollinger Bands), and a 30-day sentiment trendline.
  - ⚖️ **Logic Breakdown UI:** A visual representation of how the Technical (60%) and Sentiment (40%) weights calculate your final multiplier.
  - 📅 **5-Year Strategy Backtesting:** Built-in charting comparing a fixed SIP strategy versus the SmartSIP dynamic top-up strategy.
  - 🌓 **Immersive UI:** Custom dark-mode interface featuring an animated "shooting stars" background.
  - ⚡ **FastAPI Backend:** Fully documented REST endpoints with Swagger UI available at `/docs`.

-----

## Installation & Setup

### Prerequisites

  * **Python:** 3.10 or higher
  * **Ollama:** [Download here](https://ollama.com) (For local AI explanations)
  * **Marketaux API Key:** Free tier (100 req/day) — [Get one here](https://marketaux.com)

### Step-by-Step Setup

**1. Clone the repository**

```bash
git clone [https://github.com/shayanazmi/smart_sip.git](https://github.com/shayanazmi/smart_sip.git)
cd smart_sip
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

**3. Install Python dependencies**

```bash
cd backend
pip install -r requirements.txt
```

**4. Set up your environment variables**

```bash
cp .env.example .env
```

Open `.env` and paste your Marketaux key: `MARKETAUX_API_KEY=your_key_here`

**5. Install Ollama and pull the Llama model**

```bash
ollama pull llama3.2:1b
```

**6. Run the data pipeline and train the model**

```bash
python data_pipeline.py
python lstm_model.py
```

> ☕ *Note:* This downloads 10 years of Nifty 50 data, engineers features, and trains the LSTM. Trained artifacts (model weights, scaler, arrays, and evaluation plots) are saved locally into an `artifacts/` folder.

**7. Start the API server**

```bash
uvicorn main:app --reload --port 8000
```

**8. Open the dashboard**
Simply open `frontend/index.html` in your browser.
*(Swagger API docs available at `http://localhost:8000/docs`)*

-----

## How to Use

1.  **Launch the Dashboard:** Open `frontend/index.html`. It automatically polls the FastAPI backend for health status.
2.  **Set Parameters:** Enter your monthly base SIP amount (e.g., ₹5,000) and your preferred minimum/maximum top-up limits.
3.  **Get Insights:** Hit Refresh. The system will process the pipeline and return:
      - Market Regime (🟢 Oversold / 🟡 Neutral / 🔴 Overbought)
      - The recommended SIP multiplier.
      - Your optimized top-up amount for the month.
      - A localized AI explanation of the recommendation logic.
4.  **Analyze the Data:** Scroll down to view the logic breakdown, indicator deep-dives, and strategy backtesting charts.

> 💡 **Tip:** If Ollama isn't running in the background, the system gracefully falls back to a template-based explanation so the dashboard continues to function.

-----

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server liveness check — verifies model, scaler, and DataFrame status. |
| `/get-recommendation` | GET | Full recommendation payload (multiplier, regime, scores, AI explanation, headlines). Query param: `base_sip` |
| `/sentiment-trendline` | GET | 30-day sentiment score history for the sparkline chart. Query param: `days` |
| `/docs` | GET | Auto-generated Swagger UI for easy endpoint testing. |

-----

## Project Structure

```text
smart_sip/
├── .env.example
├── .gitattributes
├── .gitignore
├── README.md
├── artifacts/                  # Generated locally after running the pipeline/training
│   ├── X_test.npy
│   ├── X_train.npy
│   ├── nifty_enriched.csv
│   ├── scaler.pkl
│   ├── smartsip_lstm.h5
│   ├── y_test.npy
│   ├── y_train.npy
│   └── plots/
│       ├── loss_curve.png
│       ├── prediction_vs_actual.png
│       └── walk_forward_rmse.png
├── backend/
│   ├── data_pipeline.py        # Data ingestion, feature engineering, scaling
│   ├── evaluate.py             # Standalone evaluation metrics
│   ├── lstm_model.py           # LSTM architecture, training, walk-forward validation
│   ├── main.py                 # FastAPI server & pipeline orchestration
│   ├── requirements.txt        # Backend dependencies
│   └── sentiment.py            # Marketaux API + Ollama/Llama 3.2 bridge
└── frontend/
    ├── app.js                  # API polling, chart rendering, UI logic
    ├── index.html              # Main dashboard UI
    └── style.css               # Styling & immersive animations
```

-----

## Evaluation Results

### LSTM Model Architecture

  * **Input Layer:** Shape (60, 14)
  * **LSTM Layers:** 3 Stacked Layers (50 units each) with `return_sequences` handled appropriately.
  * **Regularization:** 15% Dropout between layers to prevent overfitting.
  * **Output Layer:** Dense (1 unit) with Linear activation for regression.

### Walk-Forward Validation (5 Folds)

| Fold | Train Samples | Test Samples | RMSE (Scaled) | MAE (Scaled) |
|------|---------------|--------------|---------------|--------------|
| 1 | \~484 | \~484 | \~0.048 | \~0.033 |
| 2 | \~968 | \~484 | \~0.051 | \~0.036 |
| 3 | \~1,452 | \~484 | \~0.049 | \~0.034 |
| 4 | \~1,936 | \~484 | \~0.052 | \~0.037 |
| 5 | \~2,420 | \~484 | \~0.050 | \~0.035 |
| **Mean** | — | — | **0.050 ± 0.002** | **0.035 ± 0.002** |

> Low variance across folds confirms the model **generalizes consistently** and is not overfitting to specific market regimes.

-----

## Future Scope

  - [ ] **Multi-asset support** — Expand to Nifty Bank, Nifty Midcap, and individual stocks.
  - [ ] **Transformer architecture** — Replace LSTM with a Temporal Fusion Transformer.
  - [ ] **India VIX integration** — Options-implied volatility as a fear/greed feature.
  - [ ] **Streaming pipeline** — Apache Kafka + Faust for intraday updates.
  - [ ] **Reinforcement learning** — DQN agent to maximize portfolio Sharpe ratio.

-----

## Credits

Built with ☕ and late nights by:

  * **Shayan Azmi**
  * **Siddharth Jogdand** — [GitHub](https://github.com/SiddharthJogdand)
  * **Keshav Agrawal** — [GitHub](https://github.com/keshavagrawal12)
  * **Palak**

### Acknowledgements

  - [Yahoo Finance](https://finance.yahoo.com/) — Historical Nifty 50 data
  - [Marketaux](https://marketaux.com/) — Real-time financial news sentiment API
  - [Ollama](https://ollama.com/) — Local LLM runtime
  - [Meta Llama 3.2](https://ai.meta.com/llama/) — Language model for AI explanations
  - [TensorFlow / Keras](https://www.tensorflow.org/) — LSTM implementation
  - [FastAPI](https://fastapi.tiangolo.com/) — Backend framework
  - [Chart.js](https://www.chartjs.org/) — Frontend charting

---

<p align="center">
  <b>If SmartSIP helped you, consider giving this repo a ⭐</b><br>
  It helps others discover it too!
</p>
