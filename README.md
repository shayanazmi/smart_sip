---

# SmartSIP — AI-Driven Dynamic SIP Accelerator

> *"The stock market is a device for transferring money from the impatient to the patient."*
> — **Warren Buffett**

SmartSIP makes that patience *smarter*. It uses deep learning and real-time news sentiment to tell you **how much extra to invest** in your SIP each month — so you buy more when markets dip and hold back when they're overheated.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![Llama](https://img.shields.io/badge/Llama_3.2-1B-blueviolet?logo=meta)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🏷️ Topics

`sip` `systematic-investment-plan` `lstm` `deep-learning` `nifty50` `stock-market` `sentiment-analysis` `llama` `ollama` `fastapi` `fintech` `ai` `machine-learning` `nlp` `technical-analysis` `india` `mutual-funds` `personal-finance` `tensorflow` `keras`

> 💡 **To add these on GitHub:** Go to your repo → ⚙️ Settings (gear icon next to "About" on the right sidebar) → paste the tags above into the **Topics** field.

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
- [Future Roadmap](#future-roadmap)
- [Credits](#credits)
- [License](#license)

---

## What Is SmartSIP?

SmartSIP is an end-to-end AI system that **dynamically adjusts your monthly SIP (Systematic Investment Plan) contribution** based on real-time market conditions.

Instead of investing the same fixed ₹5,000 every month regardless of whether the market is crashing or at an all-time high, SmartSIP analyses:

1. **10 years of Nifty 50 price data** using a 3-layer stacked LSTM neural network
2. **Live financial news headlines** using sentiment analysis via the Marketaux API
3. **14 engineered technical indicators** (RSI, MACD, Bollinger Bands, EMA crossovers, and more)

…and gives you a single, easy-to-understand recommendation:

> "This month, invest 1.34× your base SIP amount (₹6,700 instead of ₹5,000)"

A locally hosted **Llama 3.2** language model then explains *why* in plain English — no finance degree needed.

---

## Why We Built This

Standard SIPs invest a fixed amount every month **without caring if the market is cheap or expensive**. That leaves money on the table.

We saw three gaps no existing tool was filling:

| Gap | Problem |
|-----|---------|
| **Analytical** | Retail investors don't have tools combining ML price forecasting + NLP sentiment |
| **Accessibility** | Existing robo-advisors are expensive or target institutions |
| **Explainability** | Black-box outputs give no reasoning a normal person can understand |

SmartSIP bridges all three — it's **free, local, private, and explains itself**.

---

## Key Performance Metrics

| Metric | Value | What It Means |
|--------|-------|---------------|
| **MAPE** | 1.40% | Prediction error — anything under 3% is excellent |
| **Regime Accuracy** | 80.3% | Correctly identifies Oversold / Neutral / Overbought |
| **Walk-Forward RMSE** | 0.050 (5-fold mean) | Generalises well to unseen future data |
| **Directional Accuracy** | > 55% | Beats random (50%) — statistically meaningful edge |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Ingestion | `yfinance` 0.2.36, `pandas` 2.0+ |
| Feature Engineering | `ta` 0.11.0 (RSI, MACD, Bollinger Bands, EMA) |
| ML Model | TensorFlow 2.13 / Keras — 3-layer Stacked LSTM |
| Sentiment API | Marketaux REST API |
| LLM Explainer | Llama 3.2 1B via Ollama (runs 100% locally) |
| Backend | FastAPI 0.111 + Uvicorn |
| Frontend | Vanilla HTML / CSS / JavaScript + Chart.js |
| Model Persistence | Keras `.h5` + Python `pickle` |

---

## System Architecture

```
Yahoo Finance ──▶ data_pipeline.py ──▶ lstm_model.py ──▶ sentiment.py ──▶ main.py (FastAPI) ──▶ index.html (Dashboard)
```

**How it works when you hit "Refresh":**

1. Extracts the latest **60-day price window** from the enriched dataset
2. Normalises it with the pre-fitted MinMaxScaler
3. Runs **LSTM inference** → predicted next-day Close price (₹)
4. Computes a **Technical Score** [0–1] from predicted price vs EMA 200 + RSI
5. Fetches **live news headlines** from Marketaux → **Sentiment Score** [0–1]
6. Fuses both: `Multiplier = (Technical × 0.6) + (Sentiment × 0.4)`, clamped to **[0.25×, 2.0×]**
7. Calls **Llama 3.2** for a plain-English explanation
8. Returns everything as a clean JSON response

---

## Features

- 📈 **LSTM Price Prediction** — 3-layer stacked LSTM trained on 10 years of Nifty 50 data with 14 technical features
- 📰 **Live News Sentiment** — Real-time financial headlines scored and normalised into a sentiment signal
- 🤖 **AI Explanations** — Llama 3.2 1B runs locally, generating human-readable investment reasoning (no cloud API, no cost, full privacy)
- 📊 **Interactive Dashboard** — Live charts, market health gauge, 30-day sentiment trendline, technical indicator cards
- 🌓 **Dark / Light Mode** — Because your eyes matter at midnight
- ⚡ **FastAPI Backend** — Three documented REST endpoints with Swagger UI at `/docs`
- 🔒 **Privacy First** — All ML inference and LLM generation happen on your machine. Nothing leaves localhost.

---

## Installation & Setup

### Prerequisites

| Requirement | Version / Details |
|---|---|
| Python | 3.10 or higher |
| Ollama | Latest — [download here](https://ollama.com) |
| Llama 3.2 Model | Pulled via Ollama (see step 5 below) |
| Marketaux API Key | Free tier — [get one here](https://marketaux.com) (100 requests/day) |
| Git | For cloning the repo |

### Step-by-Step Setup

**1. Clone the repo**

```bash
git clone https://github.com/shayanazmi/smart_sip.git
cd smart_sip
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**3. Install Python dependencies**

```bash
cd backend
pip install -r requirements.txt
```

**4. Set up your API key**

```bash
cp .env.example .env
```

Now open `.env` in any text editor and paste your Marketaux key:

```
MARKETAUX_API_KEY=your_key_here
```

> ⚠️ Never commit your `.env` file. It's already in `.gitignore`.

**5. Install Ollama and pull the Llama model**

Download Ollama from [https://ollama.com](https://ollama.com), then:

```bash
ollama pull llama3.2:1b
```

**6. Run the data pipeline and train the model**

```bash
python data_pipeline.py
python lstm_model.py
```

> ☕ This may take a few minutes. It downloads 10 years of Nifty 50 data, engineers 14 features, and trains the LSTM. Trained artifacts (model weights, scaler, enriched CSV, train/test arrays, and evaluation plots) are saved locally into the `artifacts/` folder — this folder is git-ignored and will only exist on your machine after training.

**7. Start the API server**

```bash
uvicorn main:app --reload --port 8000
```

**8. Open the dashboard**

Open `frontend/index.html` in your browser.

Or visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger API docs.

✅ **You're all set!**

---

## How to Use

1. **Open the dashboard** (`frontend/index.html`) — it automatically connects to the FastAPI backend
2. **Enter your base SIP amount** (default ₹5,000), along with your preferred min/max top-up limits
3. **Hit Refresh** — the system runs the full AI pipeline and returns:
   - Today's market regime (🟢 Oversold / 🟡 Neutral / 🔴 Overbought)
   - The recommended multiplier (e.g., 1.34×)
   - Your optimal top-up amount and total investment for the month
   - An AI-generated explanation of *why*
4. **Review the supporting data:**
   - Price prediction chart (LSTM forecast vs actual vs EMA 200)
   - 30-day sentiment trendline
   - Live technical indicator values (RSI, MACD, Bollinger Width, etc.)
   - Latest news headlines with sentiment polarity labels

> 💡 **Tip:** If Ollama isn't running, the system gracefully falls back to a template-based explanation — everything else still works.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server liveness check — returns model, scaler, and DataFrame status |
| `/get-recommendation` | GET | Full recommendation payload (multiplier, regime, scores, AI explanation, headlines). Optional query param: `base_sip` (default: 5000) |
| `/sentiment-trendline` | GET | 30-day sentiment score history for the sparkline chart. Optional query param: `days` (default: 30) |
| `/docs` | GET | Auto-generated Swagger UI (built into FastAPI) |

---

## Project Structure

```
smart_sip/
├── backend/
│   ├── main.py                 # FastAPI server & pipeline orchestration
│   ├── data_pipeline.py        # Data ingestion, feature engineering, scaling
│   ├── lstm_model.py           # LSTM architecture, training, walk-forward validation
│   ├── sentiment.py            # Marketaux API + Ollama/Llama 3.2 bridge
│   ├── evaluate.py             # Standalone evaluation (MAPE, directional acc, regime acc)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── index.html              # Main dashboard
│   ├── app.js                  # API polling, chart rendering, UI logic
│   └── style.css               # Styling + dark/light mode
└── README.md
```

> **Note:** Running `data_pipeline.py` and `lstm_model.py` generates an `artifacts/` folder locally containing:
> - `smartsip_lstm.h5` — Trained LSTM model weights
> - `scaler.pkl` — Fitted MinMaxScaler
> - `nifty_enriched.csv` — Enriched DataFrame for live inference
> - `X_train.npy` / `X_test.npy` / `y_train.npy` / `y_test.npy` — Train/test arrays
> - `plots/` — `loss_curve.png`, `prediction_vs_actual.png`, `walk_forward_rmse.png`
>
> This folder is **git-ignored** and will not appear in the repository. It is created on your machine after training.

---

## Evaluation Results

### LSTM Model Architecture

| Layer | Type | Units | Config |
|-------|------|-------|--------|
| Input | Input Layer | — | shape = (60, 14) |
| LSTM 1 | LSTM | 50 | return_sequences=True |
| Dropout 1 | Dropout | — | rate = 0.15 |
| LSTM 2 | LSTM | 50 | return_sequences=True |
| Dropout 2 | Dropout | — | rate = 0.15 |
| LSTM 3 | LSTM | 50 | return_sequences=False |
| Dropout 3 | Dropout | — | rate = 0.15 |
| Output | Dense | 1 | activation='linear' |

### Engineered Features (14 total)

| Feature | Description |
|---------|-------------|
| `Close` | Raw closing price |
| `log_return` | ln(Pₜ / Pₜ₋₁) — ensures stationarity |
| `rsi` | 14-period RSI |
| `bb_upper` / `bb_lower` | Bollinger Bands (20d, 2σ) |
| `bb_width` | Bollinger squeeze strength |
| `macd` / `macd_signal` / `macd_hist` | MACD line, signal, histogram |
| `std_20` | 20-day rolling standard deviation |
| `ema_50` / `ema_200` | 50-day and 200-day EMAs |
| `cross_signal` | Golden / Death Cross flag (+1 / 0 / −1) |
| `Volume` | Daily trading volume |

### Walk-Forward Validation (5 Folds)

| Fold | Train Samples | Test Samples | RMSE (Scaled) | MAE (Scaled) |
|------|---------------|--------------|---------------|--------------|
| 1 | ~484 | ~484 | ~0.048 | ~0.033 |
| 2 | ~968 | ~484 | ~0.051 | ~0.036 |
| 3 | ~1,452 | ~484 | ~0.049 | ~0.034 |
| 4 | ~1,936 | ~484 | ~0.052 | ~0.037 |
| 5 | ~2,420 | ~484 | ~0.050 | ~0.035 |
| **Mean ± Std** | — | — | **0.050 ± 0.002** | **0.035 ± 0.002** |

> Low variance across folds confirms the model **generalises consistently** and is not overfitting to any specific market regime.

### Multiplier Formula

```
Technical Score = clip((-price_deviation × 5 × 0.6) + (rsi_score × 0.4), 0, 1)

Final Multiplier = (Technical_Score × 0.6) + (Sentiment_Score × 0.4)
                    clamped to [0.25, 2.0]

Your Top-Up = Base SIP × Multiplier
```

- 🟢 **Multiplier > 1.0** → Market looks undervalued — invest more
- 🟡 **Multiplier ≈ 1.0** → Market is fairly valued — stay the course
- 🔴 **Multiplier < 1.0** → Market looks overheated — invest less (but never zero)

---

## Future Scope

- [ ] **Multi-asset support** — Expand to Nifty Bank, Nifty Midcap, and individual stocks
- [ ] **Transformer architecture** — Replace LSTM with a Temporal Fusion Transformer
- [ ] **India VIX integration** — Options-implied volatility as a fear/greed feature
- [ ] **Streaming pipeline** — Apache Kafka + Faust for intraday updates
- [ ] **Reinforcement learning** — DQN agent to maximise portfolio Sharpe ratio
- [ ] **Mobile app** — React Native with push notifications for daily recommendations
- [ ] **Backtesting engine** — Compare SmartSIP vs fixed-SIP returns over the full 10-year dataset
- [ ] **LLM upgrade** — Mistral 7B or Llama 3 8B for richer explanations

---

## Credits

Built with ☕ and late nights by:

#Team Members:

Shayan Azmi
Siddharth Jogdand — https://github.com/SiddharthJogdand
Keshav Agrawal — https://github.com/keshavagrawal12
Palak


### References & Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com/) (via `yfinance`) — Historical Nifty 50 data
- [Marketaux](https://marketaux.com/) — Real-time financial news sentiment API
- [Ollama](https://ollama.com/) — Local LLM runtime
- [Meta Llama 3.2](https://ai.meta.com/llama/) — Language model for AI explanations
- [TensorFlow / Keras](https://www.tensorflow.org/) — LSTM implementation
- [FastAPI](https://fastapi.tiangolo.com/) — Backend framework
- [Chart.js](https://www.chartjs.org/) — Frontend charting
- [Shields.io](https://shields.io/) — README badges

---

## License

This project is licensed under the [MIT License](LICENSE).


---

<p align="center">
  <b>If SmartSIP helped you, consider giving this repo a ⭐</b><br>
  It helps others discover it too!
</p>


---

