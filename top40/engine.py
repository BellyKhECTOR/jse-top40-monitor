"""FTSE/JSE Top 40 entry/exit probability engine.

Rules implemented (FTSE/JSE Africa Index Series Ground Rules v9.3, Sep 2026):
  * Top 40 = 40 companies ranked by investable market cap within the All Share (4.5.1).
    Multiple lines / dual-listed structures are aggregated per company (4.2).
  * Non-constituent inserted if it ranks 35th or higher; constituent deleted if it ranks
    46th or lower (5.3.2, 5.3.3). The count is then restored to 40 (5.3.6).
  * Ranking snapshot = close on the Monday four weeks before the effective date (5.1.1).
  * Liquidity (tested Mar/Sep): turnover >= 0.5% of free-float shares per month in 10 of
    12 months for new entrants; constituents fail if they miss in more than 4 months (4.4.3).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA / "cache"

ENTRY_RANK, EXIT_RANK, N_CONST = 35, 46, 40
LIQ_HURDLE = 0.005

# Official FTSE/JSE review timetable (FTSE/JSE notice, 11 Aug 2026)
REVIEWS = pd.DataFrame([
    ("Sep 2026", date(2026, 8, 24), date(2026, 9, 2), date(2026, 9, 21)),
    ("Dec 2026", date(2026, 11, 23), date(2026, 12, 2), date(2026, 12, 21)),
    ("Mar 2027", date(2027, 2, 22), date(2027, 3, 3), date(2027, 3, 22)),
    ("Jun 2027", date(2027, 5, 24), date(2027, 6, 2), date(2027, 6, 21)),
], columns=["review", "cutoff", "announce", "effective"])

SA_HOLIDAYS = [
    "2026-01-01", "2026-04-03", "2026-04-06", "2026-04-27", "2026-05-01", "2026-06-16",
    "2026-08-10", "2026-09-24", "2026-12-16", "2026-12-25",
    "2027-01-01", "2027-03-22", "2027-03-26", "2027-03-29", "2027-04-27", "2027-05-01",
    "2027-06-16", "2027-08-09", "2027-09-24", "2027-12-16", "2027-12-27",
]


def next_review(today: date) -> pd.Series:
    future = REVIEWS[REVIEWS["cutoff"] >= today]
    return future.iloc[0] if len(future) else REVIEWS.iloc[-1]


def trading_days(start: date, end: date) -> int:
    """JSE trading days after `start` up to and including `end`."""
    return int(np.busday_count(np.datetime64(start) + 1, np.datetime64(end) + 1,
                               holidays=SA_HOLIDAYS))


def norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ----------------------------------------------------------------------------- data
def load_universe(path: Path | None = None) -> pd.DataFrame:
    u = pd.read_csv(path or DATA / "universe.csv")
    u["constituent"] = u["constituent"].astype(bool)
    u["foreign_primary"] = u["foreign_primary"].astype(bool)
    return u


def apply_factset_export(u: pd.DataFrame, export: pd.DataFrame) -> pd.DataFrame:
    """Overwrite shares / IWF from a FactSet constituents export (e.g. J203 All Share).

    Expects columns 'Symbol' (e.g. 'MRP-ZA'), 'Unadj. Shares' and 'Invest Factor'.
    """
    ex = export.copy()
    ex.columns = [c.strip() for c in ex.columns]
    ex = ex[ex["Symbol"].astype(str).str.endswith("-ZA")]
    ex["ticker"] = ex["Symbol"].str.replace("-ZA", "", regex=False)
    ex = ex.set_index("ticker")
    u = u.copy()
    for i, r in u.iterrows():
        if r["ticker"] in ex.index:
            row = ex.loc[r["ticker"]]
            u.at[i, "shares_m"] = float(str(row["Unadj. Shares"]).replace(",", ""))
            u.at[i, "iwf"] = float(row["Invest Factor"])
            u.at[i, "iwf_source"] = "FactSet upload"
            u.at[i, "shares_source"] = "FactSet upload"
    return u


def _cache_ok(p: Path, max_age_min: float) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) < max_age_min * 60


def fetch_market(tickers: list[str], max_age_min: float = 15, meta_tickers: list[str] | None = None) -> dict:
    """Daily close (ZAR) and volume for ~15 months; shares and insider % for `meta_tickers`."""
    meta_tickers = tickers if meta_tickers is None else meta_tickers
    import yfinance as yf

    CACHE.mkdir(parents=True, exist_ok=True)
    pc, mc = CACHE / "prices.pkl", CACHE / "meta.pkl"
    hist = meta = None
    if _cache_ok(pc, max_age_min):
        hist = pd.read_pickle(pc)
        if not set(tickers) <= set(hist["close"].columns):
            hist = None
    if _cache_ok(mc, 24 * 60):
        meta = pd.read_pickle(mc)
        if not set(meta_tickers) <= set(meta.index):
            meta = None
    if hist is not None and meta is not None:
        return {"hist": hist, "meta": meta}

    if hist is None:
        sym = [t + ".JO" for t in tickers]
        raw = yf.download(sym, period="15mo", interval="1d", auto_adjust=False,
                          progress=False, group_by="column", threads=True)
        close = raw["Close"].rename(columns=lambda c: c.replace(".JO", "")) / 100.0  # ZAc -> ZAR
        vol = raw["Volume"].rename(columns=lambda c: c.replace(".JO", ""))
        hist = {"close": close.sort_index(), "volume": vol.sort_index(),
                "fetched": pd.Timestamp.now()}
        pd.to_pickle(hist, pc)
    if meta is not None:
        return {"hist": hist, "meta": meta}

    rows = []
    for t in meta_tickers:
        tk = yf.Ticker(t + ".JO")
        fi, info = tk.fast_info, {}
        try:
            info = tk.info
        except Exception:
            pass
        last = fi.get("lastPrice")
        rows.append({"ticker": t,
                     "live_price": last / 100.0 if last else np.nan,
                     "y_shares_m": (fi.get("shares") or np.nan) / 1e6,
                     "insiders": info.get("heldPercentInsiders", np.nan)})
    meta = pd.DataFrame(rows, columns=["ticker", "live_price", "y_shares_m", "insiders"]).set_index("ticker")
    meta["fetched"] = pd.Timestamp.now()
    pd.to_pickle(meta, mc)
    return {"hist": hist, "meta": meta}


def complete_universe(u: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Fill shares from Yahoo and IWF = 1 - strategic/insider holdings where not supplied."""
    u = u.copy()
    for i, r in u.iterrows():
        m = meta.loc[r["ticker"]] if r["ticker"] in meta.index else None
        if pd.isna(r["shares_m"]) and m is not None:
            u.at[i, "shares_m"] = m["y_shares_m"]
        if pd.isna(r["iwf"]):
            ins = m["insiders"] if m is not None else np.nan
            u.at[i, "iwf"] = round(min(1.0, max(0.05, 1 - (0 if pd.isna(ins) else ins))), 2)
            u.at[i, "iwf_source"] = "Estimate: 1 - strategic holdings (Yahoo)"
    u["iwf_estimated"] = ~u["iwf_source"].str.startswith("FactSet")
    return u


# ----------------------------------------------------------------------------- ranking
def line_values(u: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    d = u.copy()
    d["price"] = d["ticker"].map(prices)
    d["full_mcap"] = d["price"] * d["shares_m"]            # R m
    d["inv_mcap"] = d["full_mcap"] * d["iwf"]
    return d


def company_table(lines: pd.DataFrame) -> pd.DataFrame:
    g = lines.groupby("company").agg(
        name=("name", lambda s: s.iloc[0] if len(s) == 1 else " / ".join(s)),
        tickers=("ticker", lambda s: "+".join(s)),
        constituent=("constituent", "max"),
        full_mcap=("full_mcap", "sum"),
        inv_mcap=("inv_mcap", "sum"),
        iwf_estimated=("iwf_estimated", "max"),
    )
    g["name"] = g["name"].str.replace("Investec PLC / Investec Ltd", "Investec (plc + Ltd)")
    g = g.dropna(subset=["inv_mcap"]).sort_values("inv_mcap", ascending=False)
    g["rank"] = np.arange(1, len(g) + 1)
    return g


def apply_rules(ranked: pd.DataFrame) -> pd.DataFrame:
    """Deterministic FTSE/JSE selection on one ranking (rules 5.3.2, 5.3.3, 5.3.6)."""
    r = ranked.copy()
    ins = r[(~r["constituent"]) & (r["rank"] <= ENTRY_RANK)].index.tolist()
    dels = r[(r["constituent"]) & (r["rank"] >= EXIT_RANK)].index.tolist()
    if len(ins) > len(dels):
        stay = r[r["constituent"] & ~r.index.isin(dels)].sort_values("rank", ascending=False)
        dels += stay.index[: len(ins) - len(dels)].tolist()
    elif len(dels) > len(ins):
        cand = r[~r["constituent"] & ~r.index.isin(ins)].sort_values("rank")
        ins += cand.index[: len(dels) - len(ins)].tolist()
    r["action"] = ""
    r.loc[ins, "action"] = "ADD"
    r.loc[dels, "action"] = "DELETE"
    r["in_index_after"] = (r["constituent"] & (r["action"] != "DELETE")) | (r["action"] == "ADD")
    return r


# ----------------------------------------------------------------------------- simulation
@dataclass
class SimResult:
    table: pd.DataFrame
    horizon_days: int
    n_paths: int
    cov_window: int
    rank_dist: pd.DataFrame


def _company_returns(lines: pd.DataFrame, close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Daily log returns per company (multi-line companies: inv-mcap weighted)."""
    lr = np.log(close).diff().iloc[-window:]
    out = {}
    for comp, grp in lines.groupby("company"):
        cols = [t for t in grp["ticker"] if t in lr.columns]
        if not cols:
            continue
        w = grp.set_index("ticker").loc[cols, "inv_mcap"]
        w = w / w.sum()
        out[comp] = (lr[cols].fillna(0) * w.values).sum(axis=1)
    return pd.DataFrame(out).fillna(0.0)


def simulate(lines: pd.DataFrame, close: pd.DataFrame, horizon: int,
             n_paths: int = 20000, window: int = 252, seed: int = 7) -> SimResult:
    comp = company_table(lines)
    rets = _company_returns(lines, close, window)[comp.index]
    cov = rets.cov().values * horizon
    cov += np.eye(len(cov)) * 1e-10
    L = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, len(comp))) @ L.T
    sims = comp["inv_mcap"].values * np.exp(z - 0.5 * np.diag(cov))  # zero expected drift

    order = np.argsort(-sims, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(n_paths)[:, None], order] = np.arange(1, len(comp) + 1)
    is_c = comp["constituent"].values.astype(bool)

    added = np.zeros(len(comp))
    deleted = np.zeros(len(comp))
    for p in range(n_paths):
        rk = ranks[p]
        ins = np.where(~is_c & (rk <= ENTRY_RANK))[0]
        dels = np.where(is_c & (rk >= EXIT_RANK))[0]
        if len(ins) > len(dels):
            stay = np.where(is_c & ~np.isin(np.arange(len(rk)), dels))[0]
            extra = stay[np.argsort(-rk[stay])][: len(ins) - len(dels)]
            dels = np.concatenate([dels, extra])
        elif len(dels) > len(ins):
            cand = np.where(~is_c & ~np.isin(np.arange(len(rk)), ins))[0]
            extra = cand[np.argsort(rk[cand])][: len(dels) - len(ins)]
            ins = np.concatenate([ins, extra])
        added[ins] += 1
        deleted[dels] += 1

    t = comp.copy()
    t["vol_ann"] = rets.std().values * math.sqrt(252)
    t["p_rank_le35"] = (ranks <= ENTRY_RANK).mean(axis=0)
    t["p_rank_ge46"] = (ranks >= EXIT_RANK).mean(axis=0)
    t["p_in_top40"] = (ranks <= N_CONST).mean(axis=0)
    t["median_rank"] = np.median(ranks, axis=0)
    t["rank_p10"] = np.percentile(ranks, 10, axis=0)
    t["rank_p90"] = np.percentile(ranks, 90, axis=0)
    t["p_add"] = added / n_paths
    t["p_delete"] = deleted / n_paths
    dist = pd.DataFrame({c: np.bincount(ranks[:, j], minlength=len(comp) + 1)[1:] / n_paths
                         for j, c in enumerate(comp.index)}, index=np.arange(1, len(comp) + 1))
    return SimResult(t, horizon, n_paths, window, dist)


def rank_history(u: pd.DataFrame, close: pd.DataFrame, days: int = 260) -> pd.DataFrame:
    """Daily company rank by investable market cap, holding today's shares and IWFs fixed."""
    c = close.ffill().iloc[-days:]
    w = u.set_index("ticker")
    cols = [t for t in w.index if t in c.columns]
    inv = c[cols] * (w.loc[cols, "shares_m"] * w.loc[cols, "iwf"]).values
    comp = inv.T.groupby(w.loc[cols, "company"]).sum(min_count=1).T
    return comp.rank(axis=1, ascending=False, method="first")


def signal(row, p_star: float) -> str:
    if row["constituent"]:
        if row["p_delete"] >= 0.5:
            return "REVIEW HOLDING"
        return "AT RISK" if row["p_delete"] >= 0.2 else ("WATCH" if row["p_delete"] >= 0.02 else "")
    if row["p_add"] >= 0.7:
        return "BUY - FULL"
    if row["p_add"] >= p_star + 0.10:
        return "BUY - STARTER"
    return "WATCH" if row["p_add"] >= 0.02 else ""


# ----------------------------------------------------------------------------- closed form
def closed_form(lines: pd.DataFrame, close: pd.DataFrame, company: str, horizon: int,
                window: int = 252) -> dict:
    """Distance-to-threshold probability vs the company currently at the relevant rank."""
    comp = company_table(lines)
    rets = _company_returns(lines, close, window)
    row = comp.loc[company]
    entry_side = not row["constituent"]
    thr_rank = ENTRY_RANK if entry_side else EXIT_RANK
    others = comp.drop(index=company)
    others = others.assign(rank=np.arange(1, len(others) + 1))
    # the company whose value we must beat (entry) or stay above (exit) once `company` is excluded
    thr_comp = others.index[(thr_rank - 1) if entry_side else (thr_rank - 2)]
    thr = comp.loc[thr_comp]
    g = math.log(row["inv_mcap"] / thr["inv_mcap"])
    s_i = rets[company].std() * math.sqrt(252)
    s_t = rets[thr_comp].std() * math.sqrt(252)
    rho = rets[company].corr(rets[thr_comp])
    s_rel = math.sqrt(max(s_i ** 2 + s_t ** 2 - 2 * rho * s_i * s_t, 1e-12))
    T = horizon / 252
    z = g / (s_rel * math.sqrt(T))
    p_above = norm_cdf(z)
    return dict(company=company, name=row["name"], entry_side=entry_side, rank=int(row["rank"]),
                inv_mcap=row["inv_mcap"], thr_company=thr_comp, thr_name=thr["name"],
                thr_rank=thr_rank, thr_inv_mcap=thr["inv_mcap"], gap_pct=math.exp(g) - 1, g=g,
                sigma_i=s_i, sigma_t=s_t, rho=rho, sigma_rel=s_rel, horizon=horizon, T=T,
                spread=s_rel * math.sqrt(T), z=z,
                probability=p_above if entry_side else 1 - p_above,
                required_move=-g)


# ----------------------------------------------------------------------------- liquidity
def liquidity_test(lines: pd.DataFrame, volume: pd.DataFrame, review_month_end: str) -> pd.DataFrame:
    """Monthly JSE turnover as % of free-float shares for the 12 months to `review_month_end`."""
    v = volume.copy()
    v.index = pd.to_datetime(v.index)
    end = pd.Period(review_month_end, "M")
    months = pd.period_range(end - 11, end, freq="M")
    mv = v.groupby(v.index.to_period("M")).agg(["sum", "count"])
    out = []
    for _, r in lines.iterrows():
        t = r["ticker"]
        if t not in volume.columns:
            continue
        ff_shares = r["shares_m"] * 1e6 * r["iwf"]
        passes, tested, worst = 0, 0, np.inf
        for m in months:
            if m not in mv.index or mv.loc[m, (t, "count")] < 5:
                continue
            tested += 1
            ratio = mv.loc[m, (t, "sum")] / ff_shares
            worst = min(worst, ratio)
            passes += ratio >= LIQ_HURDLE
        fails = tested - passes
        out.append(dict(ticker=t, company=r["company"], months_tested=tested,
                        months_passed=passes, min_monthly_turnover_pct=worst * 100,
                        new_entrant_pass=passes >= min(10, tested) if tested else False,
                        constituent_pass=fails <= 4))
    return pd.DataFrame(out).set_index("ticker")


# ----------------------------------------------------------------------------- review audit
def price_on(close: pd.DataFrame, d: date) -> pd.Series:
    c = close.loc[:pd.Timestamp(d)]
    return c.ffill().iloc[-1]


def audit_review(u: pd.DataFrame, close: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Re-run the rules at a past ranking cut-off and test the published constituents against them."""
    comp = company_table(line_values(u, price_on(close, cutoff)))
    def verdict(r):
        if r["constituent"] and r["rank"] >= EXIT_RANK:
            return "BREACH: constituent ranked 46th or lower"
        if not r["constituent"] and r["rank"] <= ENTRY_RANK:
            return "BREACH: non-constituent ranked 35th or higher"
        if r["constituent"] and r["rank"] > N_CONST:
            return "OK - held by the 46 buffer"
        if not r["constituent"] and r["rank"] <= N_CONST:
            return "OK - kept out by the 35 buffer"
        return "OK"
    comp["verdict"] = comp.apply(verdict, axis=1)
    return comp
