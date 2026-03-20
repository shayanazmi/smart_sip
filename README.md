# SmartSIP — AI-Driven Dynamic Investment Accelerator

An end-to-end AI system that optimises Systematic Investment Plans (SIPs) 
using LSTM price prediction and news sentiment analysis for Nifty 50.

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/shayanazmi/smart_sip.git
cd smart_sip

### 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

### 3. Install dependencies
cd backend
pip install -r requirements.txt

### 4. Set up your API key
Copy the .env.example file and rename it to .env:

cp .env.example .env

Then open .env and replace the placeholder with your real Marketaux API key.
Get a free key at: https://marketaux.com

### 5. Set up Ollama (for AI explanations)
Download and install Ollama from https://ollama.com
Then pull the Llama 3.2 model:

ollama pull llama3.2:1b

Ollama runs automatically in the background after installation.

### 6. Run the data pipeline and train the model
python data_pipeline.py
python lstm_model.py

This will create an artifacts/ folder with the trained model and data files.
Training takes approximately 5-15 minutes on CPU.

### 7. Start the backend server
uvicorn main:app --reload --port 8000

The API will be available at http://localhost:8000
Interactive docs at http://localhost:8000/docs

## Project Structure
```
backend/
├── data_pipeline.py   # Phase 1 — data ingestion and feature engineering
├── lstm_model.py      # Phase 2 — LSTM training and validation
├── sentiment.py       # Phase 3 — Marketaux sentiment + Llama 3.2 bridge
├── main.py            # Phase 4 — FastAPI backend
├── evaluate.py        # Model evaluation and accuracy report
├── requirements.txt   # Python dependencies
└── .env.example       # API key template — copy to .env and fill in your key
```

## Model Performance
- MAPE: 1.40% (price prediction error)
- Regime Accuracy: 80.3% (Oversold / Neutral / Overbought classification)
- Walk-Forward Validation: Mean RMSE 0.050 across 5 folds

## API Endpoints
- GET /get-recommendation — full SmartSIP recommendation
- GET /health — server health check  
- GET /sentiment-trendline — 30-day sentiment history