# AI Investment Assistant PoC (Product-Focused)

A Streamlit-based Proof of Concept to help beginner investors in Sri Lanka understand stock performance using clean metrics, recent market news, and a beginner-friendly AI explanation. Optional ML models provide price predictions and performance views.

## Quick Start (Windows)

```powershell
# From project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional: enable AI explanations via OpenAI/DeepSeek
pip install openai

# Optional: create .env with your keys
# FINNHUB_API_KEY=your_key
# NEWSAPI_API_KEY=your_key
# DEEPSEEK_API_KEY=your_key   # or OPENAI_API_KEY=your_key

# Launch the app
streamlit run app.py
```

## Interfaces (3)

- Stock Data
  - Enter a symbol (e.g., HNB) → click "Analyze Stock".
  - Shows: Current Price, Day High, Day Low, Volume, Change %, Company, Sector, Last Updated, Data Source.
  - Fallback: Uses sample JSON if APIs or keys are unavailable.

- Market News
  - Shows 2–5 recent articles relevant to the symbol.
  - Displays: Title, Source, Published Time, Summary, and "Read Full Article" link.
  - Fallback: Uses curated `sample_data/sample_news.json` when NewsAPI is unavailable.

- AI Explanation
  - Beginner-friendly narrative tying metrics and news with 3–5 key points and a disclaimer.
  - Optional: ML predicted price vs current price if models are available.
  - Fallback: Deterministic explanation when AI APIs are unavailable.

## Architecture Overview

- UI: `app.py` renders tabs (Stock Data, Market News, AI Explanation, Visualizations, Predictions, Model Performance).
- Services: `services/stocks.py` (Finnhub), `services/news.py` (NewsAPI), `services/ai_agent.py` (OpenAI/DeepSeek).
- Models: `models/best_model.pkl`, `models/scaler.pkl`, `models/metadata.json` for predictions and metrics.
- Data: `Dataset/trade-summary-merged.csv` for visualizations and statistics; sample JSON ensures resilience.
- Diagram source: see [visualizations/architecture.mmd](visualizations/architecture.mmd) (Mermaid).

## Evidence Checklist (for PoC Report)

- Screenshots
  - Stock Data: "Stock Data for HNB — price, high/low, volume, change %, company, sector, source, timestamp."
  - Market News: "Market News for HNB — latest articles with source, published time, summary, link."
  - AI Explanation: "AI Analysis for HNB — guidance, key points, disclaimer; prediction metrics when model is loaded."
- Video: Short demo navigating Stock Data → Market News → AI Explanation (optionally Predictions/Visualizations).
- Repo: URL + commit hash used in demo.
- Optional Metrics: R², MAE, RMSE table from Model Performance.

## Training (Optional)

- Script: `notebooks/train_model.py` trains 4 models (Linear Regression, Random Forest, Gradient Boosting, SVR) and saves artifacts in `models/`.
- Visualizations saved to `visualizations/`: `model_comparison.png`, `best_model_performance.png`, `feature_importance.png` (if tree-based), `distribution_analysis.png`.

## Notes

- If external API keys are missing, the app uses sample data so evaluators can still use the PoC.
- To enable AI explanations via API, ensure `openai` is installed and set either `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` in `.env`.
