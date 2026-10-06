"""Write a dated snapshot of the monitor (Excel + JSON) for decks and audit trail.

Usage:  python snapshot.py            -> outputs/Top40_snapshot_YYYYMMDD.xlsx / .json
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd

from top40 import engine as E

OUT = Path(__file__).resolve().parent / "outputs"


def main(n_paths: int = 20000) -> dict:
    u0 = E.load_universe()
    need = u0["shares_m"].isna() | u0["iwf"].isna()
    mk = E.fetch_market(u0["ticker"].tolist(), meta_tickers=u0.loc[need, "ticker"].tolist())
    u = E.complete_universe(u0, mk["meta"])
    close, volume = mk["hist"]["close"], mk["hist"]["volume"]
    last = close.index[-1].date()
    rv = E.next_review(date.today())
    h = E.trading_days(last, rv["cutoff"])
    lines = E.line_values(u, close.ffill().iloc[-1])
    comp = E.company_table(lines)
    sim = E.simulate(lines, close, h, n_paths=n_paths)
    t = sim.table
    t["gap_to_35_pct"] = t["inv_mcap"] / comp.iloc[34]["inv_mcap"] - 1
    t["gap_to_46_pct"] = t["inv_mcap"] / comp.iloc[45]["inv_mcap"] - 1
    p_star = (3.0 + 0.5) / (6.0 + 3.0)
    t["signal"] = t.apply(E.signal, axis=1, p_star=p_star)
    aud = E.audit_review(u, close, date(2026, 8, 24))
    liq = E.liquidity_test(lines, volume, "2026-07").join(u.set_index("ticker")[["name", "constituent"]])
    today_sel = E.apply_rules(comp)

    ent = t[~t["constituent"]].sort_values(["p_add", "rank"], ascending=[False, True])
    ex = t[t["constituent"]].sort_values(["p_delete", "rank"], ascending=[False, False])
    walk_entry = E.closed_form(lines, close, ent.index[0], h)
    walk_exit = E.closed_form(lines, close, ex.index[0], h)

    OUT.mkdir(exist_ok=True)
    stem = OUT / f"Top40_snapshot_{last:%Y%m%d}"
    with pd.ExcelWriter(f"{stem}.xlsx") as xw:
        ent.to_excel(xw, sheet_name="Entry candidates")
        ex.to_excel(xw, sheet_name="Exit candidates")
        aud.to_excel(xw, sheet_name="Sep26 audit")
        liq.to_excel(xw, sheet_name="Liquidity Sep26")
        lines.to_excel(xw, sheet_name="Lines", index=False)
    cols = ["name", "rank", "inv_mcap", "vol_ann", "p_rank_le35", "p_rank_ge46", "p_add", "p_delete",
            "rank_p10", "rank_p90", "gap_to_35_pct", "gap_to_46_pct", "iwf_estimated", "signal"]
    summary = dict(
        as_of=str(last), review=rv["review"], cutoff=str(rv["cutoff"]), announce=str(rv["announce"]),
        effective=str(rv["effective"]), horizon=h, n_paths=n_paths, p_star=p_star,
        thr35=dict(name=comp.iloc[34]["name"], inv=comp.iloc[34]["inv_mcap"]),
        thr46=dict(name=comp.iloc[45]["name"], inv=comp.iloc[45]["inv_mcap"]),
        entries=ent.head(8)[cols].reset_index().to_dict("records"),
        exits=ex.head(8)[cols].reset_index().to_dict("records"),
        walk_entry=walk_entry, walk_exit=walk_exit,
        walk_entry_sim=t.loc[ent.index[0], cols].to_dict(), walk_exit_sim=t.loc[ex.index[0], cols].to_dict(),
        rank_dist_entry=sim.rank_dist[ent.index[0]].to_dict(),
        audit_breaches=aud[aud["verdict"].str.startswith("BREACH")]["name"].tolist(),
        audit_band=aud[(aud["rank"] >= 30) & (aud["rank"] <= 48)][["name", "rank", "constituent", "inv_mcap", "verdict"]].reset_index().to_dict("records"),
        today_adds=today_sel[today_sel["action"] == "ADD"]["name"].tolist(),
        today_dels=today_sel[today_sel["action"] == "DELETE"]["name"].tolist(),
        buffer_held=comp[comp["constituent"] & (comp["rank"] > 40)]["name"].tolist(),
        buffer_out=comp.head(40)[~comp.head(40)["constituent"]]["name"].tolist(),
        n_companies=int(comp["constituent"].sum()), n_lines=int(u["constituent"].sum()),
        liq_constituent_fails=liq[liq["constituent"] & ~liq["constituent_pass"]]["name"].tolist(),
        liq_min=liq[liq["constituent"]].sort_values("min_monthly_turnover_pct").head(5)[["name", "months_passed", "months_tested", "min_monthly_turnover_pct"]].reset_index().to_dict("records"),
        expected_changes=float(t["p_add"].sum()),
        estimated_iwf_names=u[u["iwf_estimated"]]["name"].tolist(),
        band=t[(t["rank"] >= 28) & (t["rank"] <= 50)].sort_values("rank")[cols + ["constituent"]].reset_index().to_dict("records"),
        walk_entry_lines=lines[lines["company"] == ent.index[0]][["ticker", "price", "shares_m", "iwf", "iwf_source", "inv_mcap"]].to_dict("records"),
    )
    with open(f"{stem}.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"wrote {stem}.xlsx / .json")
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps({k: s[k] for k in ["as_of", "review", "horizon", "thr35", "thr46", "audit_breaches",
                                        "today_adds", "today_dels", "buffer_held", "buffer_out",
                                        "liq_constituent_fails", "liq_min", "expected_changes"]}, indent=1, default=str))
    for k in ("entries", "exits"):
        print(k)
        for r in s[k]:
            print(f"  {r['name']:<28} rank {r['rank']:>3}  inv R{r['inv_mcap']/1e3:6.1f}bn  vol {r['vol_ann']:.0%}  "
                  f"P35 {r['p_rank_le35']:.1%}  P46 {r['p_rank_ge46']:.1%}  add {r['p_add']:.1%}  del {r['p_delete']:.1%}  {r['signal']}")
