# Mutual Funds Historical Analysis

This project contains a Streamlit-based mutual fund analytics dashboard with local data files and live data integrations.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- The app uses local CSV/XLSX data plus live third-party endpoints.
- Some endpoints may be rate-limited or temporarily unavailable.
- The app is designed to degrade gracefully when external data is unavailable.
