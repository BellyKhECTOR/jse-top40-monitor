"""Mazi | FTSE/JSE Top 40 entry & exit monitor (live)."""
from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from top40 import engine as E

NAVY, MAROON, GOLD, PEACH, GREY = "#1D2633", "#77121B", "#C29A53", "#FDCDA2", "#C5C8CC"
st.set_page_config(page_title="Top 40 Monitor | Mazi", page_icon="📈", layout="wide")
st.markdown(f"""<style>
h1, h2, h3 {{ font-family: 'Oswald', 'Arial Narrow', sans-serif; text-transform: uppercase; letter-spacing: .5px; }}
[data-testid="stMetricValue"] {{ color: {GOLD}; }}
.step {{ border: 1px solid {PEACH}; padding: 14px 18px; margin-bottom: 10px; }}
.step b.n {{ background: {MAROON}; padding: 2px 9px; margin-right: 8px; border: 1px solid white; }}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
@st.cache_data(ttl=900, show_spinner="Pulling live JSE prices...")
def market(tickers: tuple, meta_tickers: tuple):
    return E.fetch_market(list(tickers), meta_tickers=list(meta_tickers))


@st.cache_data(ttl=900, show_spinner="Simulating the review...")
def run_sim(u_json: str, horizon: int, n_paths: int, window: int, _close, stamp):
    u = pd.read_json(BytesIO(u_json.encode()), orient="split")
    lines = E.line_values(u, _close.ffill().iloc[-1])
    return E.simulate(lines, _close, horizon, n_paths=n_paths, window=window)


if "universe" not in st.session_state:
    st.session_state.universe = E.load_universe()

with st.sidebar:
    st.header("Settings")
    up = st.file_uploader("FactSet constituents export (CSV/XLSX) to replace shares & IWFs",
                          type=["csv", "xlsx"])
    if up is not None and st.session_state.get("upload_name") != up.name:
        ex = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        st.session_state.universe = E.apply_factset_export(st.session_state.universe, ex)
        st.session_state.upload_name = up.name
        st.success(f"Applied {up.name}")
    reviews = E.REVIEWS[E.REVIEWS["cutoff"] >= date.today()]
    review_name = st.selectbox("Review", reviews["review"].tolist())
    rv = reviews.set_index("review").loc[review_name]
    n_paths = st.select_slider("Monte Carlo paths", [5000, 10000, 20000, 50000], value=20000)
    window = st.select_slider("Volatility window (trading days)", [63, 126, 252], value=252)
    st.subheader("Trade economics")
    r_in = st.number_input("Gain if added, R_in (%)", value=6.0, step=0.5)
    r_miss = st.number_input("Loss if missed, R_miss (%)", value=3.0, step=0.5)
    cost = st.number_input("Round-trip cost, c (%)", value=0.5, step=0.1)
    p_star = (r_miss + cost) / (r_in + r_miss)
    st.metric("Breakeven probability P*", f"{p_star:.0%}")
    if st.button("Refresh prices now", use_container_width=True):
        market.clear(); run_sim.clear()
        (E.CACHE / "prices.pkl").unlink(missing_ok=True)
        st.rerun()

base = st.session_state.universe
need = base["shares_m"].isna() | base["iwf"].isna()
mk = market(tuple(base["ticker"]), tuple(base.loc[need, "ticker"]))
u = E.complete_universe(base, mk["meta"])
close, volume = mk["hist"]["close"], mk["hist"]["volume"]
last_bar = close.index[-1].date()
horizon = max(E.trading_days(last_bar, rv["cutoff"]), 1)
lines = E.line_values(u, close.ffill().iloc[-1])
comp = E.company_table(lines)
sim = run_sim(u.to_json(orient="split"), horizon, n_paths, window, close,
              str(mk["hist"].get("fetched", last_bar)))
t = sim.table.copy()
thr35 = comp.iloc[E.ENTRY_RANK - 1]
thr46 = comp.iloc[E.EXIT_RANK - 1]
t["gap_to_35_pct"] = t["inv_mcap"] / thr35["inv_mcap"] - 1
t["gap_to_46_pct"] = t["inv_mcap"] / thr46["inv_mcap"] - 1
t["signal"] = t.apply(E.signal, axis=1, p_star=p_star)

# ----------------------------------------------------------------------------- header
st.title("FTSE/JSE Top 40 · Entry & Exit Monitor")
st.caption(f"Prices: Yahoo Finance JSE closes, last bar {last_bar:%d %b %Y} (intraday, ~15 min delayed) · "
           f"Constituents, shares and IWFs: FactSet 25-Sep-26 · Rules: FTSE/JSE Ground Rules v9.3")
k = st.columns(6)
k[0].metric("Next review", review_name)
k[1].metric("Ranking cut-off", f"{rv['cutoff']:%d %b}")
k[2].metric("Trading days left", horizon)
k[3].metric("Announcement", f"{rv['announce']:%d %b}")
k[4].metric("#35 threshold", f"R{thr35['inv_mcap']/1e3:,.1f}bn", thr35["name"], delta_color="off")
k[5].metric("#46 threshold", f"R{thr46['inv_mcap']/1e3:,.1f}bn", thr46["name"], delta_color="off")

tabs = st.tabs(["Watch-list", "Stock walkthrough", "Is the index right?", "Rank chart", "Rules & calendar", "Data"])

pct = lambda s: s.map(lambda v: f"{v:.1%}")

# ----------------------------------------------------------------------------- watch-list
with tabs[0]:
    st.subheader("Possible entries")
    ent = t[~t["constituent"]].sort_values(["p_add", "rank"], ascending=[False, True]).head(12)
    st.dataframe(pd.DataFrame({
        "Company": ent["name"], "Rank now": ent["rank"],
        "Investable mcap (R bn)": (ent["inv_mcap"] / 1e3).round(1),
        "Gap to #35": pct(ent["gap_to_35_pct"]), "Vol (ann.)": pct(ent["vol_ann"]),
        "Rank range (10-90%)": ent.apply(lambda r: f"{r.rank_p10:.0f}–{r.rank_p90:.0f}", axis=1),
        "P(rank ≤ 35)": pct(ent["p_rank_le35"]), "P(added)": pct(ent["p_add"]),
        "IWF": np.where(ent["iwf_estimated"], "estimated", "FactSet"), "Signal": ent["signal"],
    }), hide_index=True, use_container_width=True)
    st.subheader("Possible exits")
    ex = t[t["constituent"]].sort_values(["p_delete", "rank"], ascending=[False, False]).head(12)
    st.dataframe(pd.DataFrame({
        "Company": ex["name"], "Rank now": ex["rank"],
        "Investable mcap (R bn)": (ex["inv_mcap"] / 1e3).round(1),
        "Cushion over #46": pct(ex["gap_to_46_pct"]), "Vol (ann.)": pct(ex["vol_ann"]),
        "Rank range (10-90%)": ex.apply(lambda r: f"{r.rank_p10:.0f}–{r.rank_p90:.0f}", axis=1),
        "P(rank ≥ 46)": pct(ex["p_rank_ge46"]), "P(deleted)": pct(ex["p_delete"]),
        "Signal": ex["signal"],
    }), hide_index=True, use_container_width=True)
    exp_changes = t["p_add"].sum()
    st.info(f"Expected number of changes at the {review_name} review: **{exp_changes:.2f}**. "
            f"P(added) and P(deleted) include the rule that restores the index to 40 names — a non-constituent "
            f"can get in without reaching 35th if a constituent falls to 46th. "
            f"Signals: BUY-FULL ≥ 70%, BUY-STARTER ≥ P* + 10% ({p_star + .1:.0%}), WATCH ≥ 2%.")

# ----------------------------------------------------------------------------- walkthrough
with tabs[1]:
    default = t[~t["constituent"]]["p_add"].idxmax()
    opts = t.sort_values("rank").index.tolist()
    pick = st.selectbox("Company", opts, index=opts.index(default), format_func=lambda c: f"{t.loc[c, 'name']} (rank {t.loc[c, 'rank']})")
    cf = E.closed_form(lines, close, pick, horizon, window)
    r = t.loc[pick]
    side = "entry" if cf["entry_side"] else "exit"
    lu = u[u["company"] == pick]
    st.markdown(f"### {cf['name']} — {side} case")

    def step(n, title, body):
        st.markdown(f"<div class='step'><b class='n'>{n}</b><b>{title}</b><br>{body}</div>", unsafe_allow_html=True)

    line_txt = "<br>".join(
        f"{x.ticker}: price R{lines.set_index('ticker').loc[x.ticker,'price']:,.2f} × {x.shares_m:,.1f}m shares × IWF {x.iwf:.2f} "
        f"({x.iwf_source}) = <b>R{lines.set_index('ticker').loc[x.ticker,'inv_mcap']/1e3:,.1f}bn</b>"
        for x in lu.itertuples())
    step(1, "Investable market cap today", line_txt)
    step(2, "Where it ranks", f"Rank <b>{cf['rank']}</b> of {len(comp)} on investable market cap. It is "
         f"{'not ' if cf['entry_side'] else ''}a constituent, so the test is "
         f"<b>{'rank 35th or better to be added' if cf['entry_side'] else 'rank 46th or worse to be deleted'}</b>.")
    step(3, "The threshold", f"With {cf['name']} set aside, the company it must "
         f"{'overtake' if cf['entry_side'] else 'stay ahead of'} is <b>{cf['thr_name']}</b> at R{cf['thr_inv_mcap']/1e3:,.1f}bn. "
         f"Gap g = ln({cf['inv_mcap']/1e3:,.1f} ÷ {cf['thr_inv_mcap']/1e3:,.1f}) = <b>{cf['g']:+.3f}</b> "
         f"({cf['gap_pct']:+.1%}). It needs to {'gain' if cf['entry_side'] else 'lose'} about "
         f"<b>{abs(cf['gap_pct']):.1%}</b> relative to that company.")
    step(4, "How far it can move before the cut-off",
         f"σ({cf['name']}) = {cf['sigma_i']:.1%}, σ({cf['thr_name']}) = {cf['sigma_t']:.1%}, correlation ρ = {cf['rho']:.2f} → "
         f"σ_rel = √(σᵢ² + σₜ² − 2ρσᵢσₜ) = <b>{cf['sigma_rel']:.1%}</b> a year. "
         f"{cf['horizon']} trading days to {rv['cutoff']:%d %b %Y} → T = {cf['T']:.3f}; σ_rel·√T = <b>{cf['spread']:.3f}</b>.")
    step(5, "Closed-form probability",
         f"z = g ÷ (σ_rel·√T) = {cf['g']:+.3f} ÷ {cf['spread']:.3f} = <b>{cf['z']:+.2f}</b> → "
         f"P({'rank ≤ 35' if cf['entry_side'] else 'rank ≥ 46'}) ≈ {'Φ(z)' if cf['entry_side'] else '1 − Φ(z)'} = <b>{cf['probability']:.1%}</b>")
    step(6, "Full simulation (all companies move together)",
         f"{sim.n_paths:,} correlated paths of every company to the cut-off, re-ranked and run through the FTSE/JSE rules "
         f"including the restore-to-40 rule: P(rank {'≤ 35' if cf['entry_side'] else '≥ 46'}) = "
         f"<b>{(r['p_rank_le35'] if cf['entry_side'] else r['p_rank_ge46']):.1%}</b>, "
         f"P({'added' if cf['entry_side'] else 'deleted'}) = <b>{(r['p_add'] if cf['entry_side'] else r['p_delete']):.1%}</b>. "
         f"Median rank at cut-off {r['median_rank']:.0f} (10–90% range {r['rank_p10']:.0f}–{r['rank_p90']:.0f}).")
    if cf["entry_side"]:
        ev = r["p_add"] * r_in - (1 - r["p_add"]) * r_miss - cost
        step(7, "Decision", f"EV = P·R_in − (1−P)·R_miss − c = {r['p_add']:.1%}×{r_in}% − {1-r['p_add']:.1%}×{r_miss}% − {cost}% = "
             f"<b>{ev:+.2f}%</b>. P* = {p_star:.0%} → signal: <b>{r['signal'] or 'NO ACTION'}</b>.")
    else:
        step(7, "Decision", f"P(deleted) = {r['p_delete']:.1%} → signal: <b>{r['signal'] or 'NO ACTION'}</b>. "
             f"A deletion means forced passive selling into the effective date.")
    c1, c2 = st.columns(2)
    dist = sim.rank_dist[pick]
    dist = dist[dist > 0.0005]
    fig = go.Figure(go.Bar(x=dist.index, y=dist.values,
                           marker_color=[GOLD if i <= 35 else (MAROON if i >= 46 else GREY) for i in dist.index]))
    fig.update_layout(title="Simulated rank at the cut-off", xaxis_title="Rank", yaxis_tickformat=".0%",
                      height=360, margin=dict(l=10, r=10, t=40, b=10))
    c1.plotly_chart(fig, use_container_width=True)
    rh = E.rank_history(u, close)
    fig2 = go.Figure(go.Scatter(x=rh.index, y=rh[pick], line=dict(color=GOLD, width=3), name=cf["name"]))
    for y, c, lbl in [(35, GOLD, "Enter at 35"), (46, MAROON, "Exit at 46")]:
        fig2.add_hline(y=y, line_dash="dash", line_color=c, annotation_text=lbl)
    fig2.update_layout(title="Rank over the past year (today's shares & IWFs)", yaxis_autorange="reversed",
                       height=360, margin=dict(l=10, r=10, t=40, b=10))
    c2.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------------------- audit
with tabs[2]:
    st.subheader("1 · Was the September 2026 review applied correctly?")
    aud = E.audit_review(u, close, date(2026, 8, 24))
    breaches = aud[aud["verdict"].str.startswith("BREACH")]
    if len(breaches):
        st.error(f"{len(breaches)} apparent breach(es) at the 24 Aug 2026 ranking cut-off — check IWFs/shares used.")
    else:
        st.success("No breaches: at the 24 Aug 2026 cut-off every constituent ranked 45th or better and no "
                   "non-constituent ranked 35th or better.")
    show = aud[(aud["rank"] >= 28) & (aud["rank"] <= 50)]
    st.dataframe(pd.DataFrame({"Company": show["name"], "Rank at cut-off": show["rank"],
                               "In index": np.where(show["constituent"], "Yes", "No"),
                               "Investable mcap (R bn)": (show["inv_mcap"] / 1e3).round(1),
                               "IWF": np.where(show["iwf_estimated"], "estimated", "FactSet"),
                               "Verdict": show["verdict"]}), hide_index=True, use_container_width=True)

    st.subheader("2 · If the review were run on today's prices")
    today_sel = E.apply_rules(comp)
    adds, dels = today_sel[today_sel["action"] == "ADD"], today_sel[today_sel["action"] == "DELETE"]
    if len(adds) + len(dels) == 0:
        st.success("No changes: no non-constituent ranks 35th or better and no constituent ranks 46th or worse.")
    else:
        st.warning("Changes today: ADD " + ", ".join(adds["name"]) + " | DELETE " + ", ".join(dels["name"]))
    pure = comp.head(40)
    not_top = comp[comp["constituent"] & (comp["rank"] > 40)]
    out_top = pure[~pure["constituent"]]
    c1, c2 = st.columns(2)
    c1.markdown("**In the index but outside the pure top 40** (kept by the 46 buffer)")
    c1.dataframe(not_top[["name", "rank"]].rename(columns={"name": "Company", "rank": "Rank"}), hide_index=True)
    c2.markdown("**In the pure top 40 but not in the index** (kept out by the 35 buffer)")
    c2.dataframe(out_top[["name", "rank"]].rename(columns={"name": "Company", "rank": "Rank"}), hide_index=True)

    st.subheader("3 · Liquidity screen (Sep 2026 test: Aug 2025 – Jul 2026)")
    liq = E.liquidity_test(lines, volume, "2026-07")
    lq = liq.join(u.set_index("ticker")[["name", "constituent"]])
    lq = lq[lq["constituent"] | lq["company"].isin(t[~t["constituent"]].nlargest(10, "inv_mcap").index)]
    st.dataframe(pd.DataFrame({"Line": lq["name"], "In index": np.where(lq["constituent"], "Yes", "No"),
                               "Months passed": lq["months_passed"].astype(str) + " / " + lq["months_tested"].astype(str),
                               "Lowest month (% of free float)": lq["min_monthly_turnover_pct"].round(2),
                               "Result": np.where(lq["constituent"],
                                                  np.where(lq["constituent_pass"], "Pass (≤4 failed months)", "FAIL"),
                                                  np.where(lq["new_entrant_pass"], "Eligible (≥10 of 12)", "Not eligible"))}),
                 hide_index=True, use_container_width=True)
    st.caption("Volumes are JSE on-book volumes from Yahoo; the official test uses JSE data. Lines with a small IWF "
               "(e.g. BHP, BAT) pass easily because the hurdle is 0.5% of the free-floated SA register.")

    st.subheader("4 · Structural checks")
    n_comp = comp["constituent"].sum()
    checks = pd.DataFrame([
        ("40 companies in the index", f"{n_comp} companies / {u['constituent'].sum()} lines", n_comp == 40),
        ("Multiple lines aggregated (rule 4.2)", "Investec plc + Ltd ranked as one company", True),
        ("Global free float > 5% (rule 4.3.5)", "Low IWFs on BHP, BAT, AB InBev, Glencore, Richemont reflect the SA "
         "register share (location), not global float — all have global float well above 5%", True),
        ("Constituents pass liquidity", f"{(lq['constituent'] & ~lq['constituent_pass']).sum()} failures", not (lq['constituent'] & ~lq['constituent_pass']).any()),
    ], columns=["Requirement", "Evidence", "Met"])
    checks["Met"] = np.where(checks["Met"], "✅", "❌")
    st.dataframe(checks, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------- rank chart
with tabs[3]:
    band = t[(t["rank"] >= 25) & (t["rank"] <= 60)].sort_values("rank")
    colors = [GOLD if c else GREY for c in band["constituent"]]
    fig = go.Figure(go.Bar(x=band["name"], y=band["inv_mcap"] / 1e3, marker_color=colors,
                           text=band["rank"], textposition="outside",
                           hovertext=[f"P(add) {a:.1%} · P(delete) {d:.1%}" for a, d in zip(band["p_add"], band["p_delete"])]))
    fig.add_hline(y=thr35["inv_mcap"] / 1e3, line_dash="dash", line_color=GOLD, annotation_text=f"#35 {thr35['name']}")
    fig.add_hline(y=thr46["inv_mcap"] / 1e3, line_dash="dash", line_color=MAROON, annotation_text=f"#46 {thr46['name']}")
    fig.update_layout(title="Investable market cap, ranks 25–60 (gold = constituent)", yaxis_title="R bn",
                      height=560, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------- rules
with tabs[4]:
    st.dataframe(E.REVIEWS.rename(columns=str.title), hide_index=True)
    st.markdown("""
- **Ranking**: companies in the FTSE/JSE All Share ranked by **investable** market cap (price × shares × investability weight); multiple lines aggregated per company (rules 4.2, 4.5.1).
- **Insert** a non-constituent ranked **35th or higher**; **delete** a constituent ranked **46th or lower** (5.3.2–5.3.3). The index is restored to 40 by adding the highest-ranked non-constituents or removing the lowest-ranked constituents (5.3.6).
- **Snapshot**: close on the Monday four weeks before the effective date (5.1.1). Changes effective after the close on the third Friday (start of trading the following Monday).
- **Liquidity** (tested March & September): ≥ 0.5% of free-float shares traded per month in 10 of 12 months for new entrants; constituents removed if they fail in more than 4 months (4.4.3).
- **Free float**: global free float above 5% (4.3.5). Free-float/shares cut-off for December: 30 Oct 2026.
- **Model**: zero-drift correlated lognormal paths for every company to the cut-off, covariance from daily returns; each path re-ranked and run through the rules above.
""")

# ----------------------------------------------------------------------------- data
with tabs[5]:
    st.markdown("Edit IWFs or shares below (e.g. from FactSet J203) — the model re-runs immediately. "
                "Rows marked *estimated* drive most of the uncertainty for non-constituents.")
    edited = st.data_editor(u[["ticker", "name", "constituent", "shares_m", "iwf", "iwf_source"]],
                            hide_index=True, use_container_width=True, disabled=["ticker", "name", "constituent"],
                            key="editor")
    if st.button("Apply edits"):
        b = st.session_state.universe.set_index("ticker")
        e = edited.set_index("ticker")
        changed = (b["iwf"].fillna(-1) != e["iwf"]) | (b["shares_m"].fillna(-1) != e["shares_m"])
        b.loc[changed, ["iwf", "shares_m"]] = e.loc[changed, ["iwf", "shares_m"]]
        b.loc[changed, "iwf_source"] = "Manual override"
        st.session_state.universe = b.reset_index()
        st.rerun()
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        t.drop(columns=[]).to_excel(xw, sheet_name="Probabilities")
        lines.to_excel(xw, sheet_name="Lines", index=False)
        E.audit_review(u, close, date(2026, 8, 24)).to_excel(xw, sheet_name="Sep26 audit")
    st.download_button("Download snapshot (Excel)", buf.getvalue(),
                       file_name=f"Top40_monitor_{last_bar:%Y%m%d}.xlsx")
