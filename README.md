# Top40Monitor

Live FTSE/JSE Top 40 entry and exit probabilities, built on the FTSE/JSE Ground Rules v9.3 (Sep 2026).

## Run

```
pip install -r requirements.txt
python -m streamlit run app.py          # opens http://localhost:8501
python snapshot.py                      # dated Excel + JSON in outputs/ (feeds the deck)
```

## Data

- **Prices and volumes**: JSE closes via Yahoo Finance (`.JO`), cached for 15 minutes. Use **Refresh prices now** in the sidebar to force an update.
- **Constituents, shares, IWFs**: `data/universe.csv`, seeded from the FactSet *FTSE JSE 40* report dated 25 Sep 2026.
- **Non-constituents**: shares come from Yahoo. IWFs are estimated in one of two ways:
  - 1 − strategic holdings (Yahoo insiders), or
  - a manual SA-register estimate for foreign-primary listings (South32, Mondi, Quilter, etc.).

  The app marks these rows *estimated*.
- **To remove the estimates**, export the FTSE/JSE All Share (J203) constituents from FactSet with *Symbol*, *Unadj. Shares* and *Invest Factor*, then upload the file in the sidebar.

## Model

1. Investable market cap = price × shares × IWF. Lines of the same company (e.g. Investec plc and Ltd) are added together.
2. Correlated zero-drift lognormal paths to the next ranking cut-off. The covariance comes from daily returns.
3. Each path is re-ranked and run through rules 5.3.2–5.3.6:
   - a stock enters at 35th or better,
   - a stock exits at 46th or worse,
   - the index is then restored to 40.
4. Outputs per company: P(rank ≤ 35), P(rank ≥ 46), P(added) and P(deleted), plus a closed-form distance-to-threshold check.

## Review calendar

Taken from the FTSE/JSE notice of 11 Aug 2026. Add future reviews to `REVIEWS` in `top40/engine.py` when they are published.
