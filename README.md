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
- Moneycontrol responses are persisted in `data/moneycontrol_cache.sqlite3`, so normal app sessions reuse saved values instead of calling the API again.
- `.github/workflows/daily-data-refresh.yml` runs at 21:00 IST (`15:30 UTC`) and commits the refreshed cache plus organized files under `data/daily_snapshots/`.
- Run `python refresh_data.py` manually to perform the same refresh. The current Moneycontrol graph endpoint returns a selected history duration, so the scheduled job makes one request per fund and endpoint but does not yet use a provider-supported one-day delta endpoint.
