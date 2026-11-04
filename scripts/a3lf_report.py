#!/usr/bin/env python3
"""A3LF report: tables and tests of docs/A3LF_PREREG.md Section 4 -> results/a3lf/report.json + report.md.

Comparators are read with `git show` from the committed results of other sessions:
  B1a, B1b  origin/art/arms           results/baselines/<setting>/s<seed>_<k>_n<npm>/<config>.json
  A3        origin/art/grid-{1,2,3}   results/grid/<setting>/s<seed>/n<npm>/A3/result.json
  B6        origin/art/confirm-E4     explore/E4/results/<setting>/s<seed>_n<npm>/a=track60k-w=1-st=20000.json
            (fallback: results/b6/<setting>/s<seed>/n<npm>/B6.json on this branch)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "a3lf"
MAIN = [(s, seed, n) for s in ("k4", "alt") for seed in range(5, 10) for n in (4, 16)]
EXTRA = [(s, seed, 4) for s in ("k1", "contig", "b3p1") for seed in range(5, 10)]
B6_SPEC = "a=track60k-w=1-st=20000"


def _git_json(ref, path):
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def b1b_config(setting):
    from toyrepair.arms.b1b import best_configs
    return best_configs(setting, 1)[0]


def baseline(arm, setting, seed, npm):
    k = setting if setting in ("k1", "k2", "k4") else "k4"
    cfg = "R_hopB_bsup_neg_AMdist" if arm == "B1a" else b1b_config(setting)
    d = _git_json("origin/art/arms", f"results/baselines/{setting}/s{seed}_{k}_n{npm}/{cfg}.json")
    if not d:
        return None
    m = d.get("metrics", d)
    return {"D": m["tv_to_ideal"], "removed": 1 - m["neg_mass_ratio"], "collateral": m["retain_tv"],
            "mass_off": m.get("mass_off")}


def a3(setting, seed, npm):
    for g in ("grid-1", "grid-2", "grid-3"):
        d = _git_json(f"origin/art/{g}", f"results/grid/{setting}/s{seed}/n{npm}/A3/result.json")
        if d:
            if (d.get("output") or {}).get("invalid"):
                return {"invalid": d["output"]["invalid"]}
            f = d["final"]
            return {"D": f["D"], "removed": f["removed"], "collateral": f["collateral"], "mass_off": f.get("mass_off"),
                    "cost_usd": (d.get("output") or {}).get("cost_usd"), "src": g}
    return None


def b6(setting, seed, npm):
    d = _git_json("origin/art/confirm-E4", f"explore/E4/results/{setting}/s{seed}_n{npm}/{B6_SPEC}.json")
    if d:
        return {"D": d["D"], "removed": d["Removed"], "collateral": d["Collateral"], "mass_off": d.get("mass_off"),
                "src": "E4"}
    p = REPO / "results" / "b6" / setting / f"s{seed}" / f"n{npm}" / "B6.json"
    if p.exists():
        f = json.loads(p.read_text())["final"]
        return {"D": f["D"], "removed": f["removed"], "collateral": f["collateral"], "src": "results/b6"}
    return None


def a3lf(setting, seed, npm):
    p = OUT / setting / f"s{seed}" / f"n{npm}" / "result.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    o = d.get("output") or {}
    if o.get("invalid"):
        return {"invalid": o["invalid"]}
    f = d["final"]
    sess = d.get("sessions") or []
    return {"D": f["D"], "removed": f["removed"], "collateral": f["collateral"], "mass_off": f.get("mass_off"),
            "cost_usd": o.get("cost_usd"), "sessions": o.get("sessions"), "gate_checks": o.get("gate_checks"),
            "accepted": len(o.get("accepted_chain") or []), "stop": o.get("stop"), "shadow": d.get("shadow"),
            "surgeon_cost": float(sum(s.get("cost_usd") or 0 for s in sess)),
            "triage_cost": float(sum(t.get("cost_usd") or 0 for t in d.get("triage") or [])),
            "calls": [s.get("call") for s in o.get("accepted_chain") or []]}


def _ok(x):
    return x is not None and "D" in x


def paired(rows, comp, key="D", alt="less", unit="cell"):
    """Paired t of A3-LF - comp. unit 'cell' pairs every (setting, seed, npm); 'seed' averages over npm."""
    pairs = [(r["A3-LF"], r[comp], r) for r in rows if _ok(r.get("A3-LF")) and _ok(r.get(comp))]
    if unit == "seed":
        g = {}
        for a, b, r in pairs:
            g.setdefault((r["setting"], r["seed"]), []).append((a, b))
        # a (setting, seed) enters only when both arms have every npm the A3-LF arm ran there
        want = {}
        for r in rows:
            if _ok(r.get("A3-LF")):
                want.setdefault((r["setting"], r["seed"]), 0)
                want[(r["setting"], r["seed"])] += 1
        pairs = [({key: np.mean([a[key] for a, _ in v]), "collateral": np.mean([a["collateral"] for a, _ in v])},
                  {key: np.mean([b[key] for _, b in v]), "collateral": np.mean([b["collateral"] for _, b in v])}, k)
                 for k, v in g.items() if len(v) == want.get(k)]
    n = len(pairs)
    if n < 2:
        return {"n": n}
    d = np.array([a[key] - b[key] for a, b, _ in pairs])
    dc = np.array([a["collateral"] - b["collateral"] for a, b, _ in pairs])
    t = stats.ttest_1samp(d, 0.0, alternative=alt)
    se = d.std(ddof=1) / np.sqrt(n)
    tq = stats.t.ppf(0.975, n - 1)
    return {"n": n, "mean_diff": float(d.mean()), "se": float(se), "ci95": [float(d.mean() - tq * se), float(d.mean() + tq * se)],
            "t": float(t.statistic), "p": float(t.pvalue), "lower_in": int((d < 0).sum()),
            "mean_collateral_diff": float(dc.mean())}


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j]))
        adj[j] = run
    return adj.tolist()


def main():
    rows = []
    for (s, seed, n) in MAIN + EXTRA:
        r = {"setting": s, "seed": seed, "npm": n, "main": (s, seed, n) in MAIN,
             "A3-LF": a3lf(s, seed, n), "B1a": baseline("B1a", s, seed, n), "B1b": baseline("B1b", s, seed, n),
             "A3": a3(s, seed, n), "B6": b6(s, seed, n)}
        rows.append(r)
    main_rows = [r for r in rows if r["main"]]
    tests = {}
    tests["L1"] = paired(main_rows, "B1a")
    tests["L2"] = paired(main_rows, "B1b")
    tests["L3"] = paired(main_rows, "A3", alt="two-sided")
    ps = [tests[k].get("p", np.nan) for k in ("L1", "L2")]
    if all(np.isfinite(ps)):
        for k, a in zip(("L1", "L2"), holm(ps)):
            tests[k]["p_holm"] = a
            tests[k]["reject"] = a < 0.05
            tests[k]["supported"] = bool(a < 0.05 and tests[k]["mean_collateral_diff"] <= 0.02)
    tests["L3"]["reject"] = tests["L3"].get("p", 1) < 0.05
    rob = {f"{c}_{u}_{scope}": paired(rr, c, alt=("two-sided" if c == "A3" else "less"), unit=u)
           for c in ("B1a", "B1b", "A3", "B6") for u in ("cell", "seed")
           for scope, rr in (("main", main_rows), ("all", rows))}
    # per setting summary
    per = {}
    for s in ("k4", "alt", "k1", "contig", "b3p1"):
        rr = [r for r in rows if r["setting"] == s]
        if not any(_ok(r["A3-LF"]) for r in rr):
            continue
        e = {}
        for arm in ("A3-LF", "B1a", "B1b", "A3", "B6"):
            v = [r[arm] for r in rr if _ok(r[arm]) and _ok(r["A3-LF"])]
            if v:
                e[arm] = {k: (float(np.mean([x[k] for x in v])), float(np.std([x[k] for x in v], ddof=1) / np.sqrt(len(v)))
                              if len(v) > 1 else 0.0) for k in ("D", "removed", "collateral")}
                e[arm]["n"] = len(v)
            if arm != "A3-LF":
                pr = [(r["A3-LF"]["D"], r[arm]["D"]) for r in rr if _ok(r["A3-LF"]) and _ok(r[arm])]
                if pr:
                    d = np.array([a - b for a, b in pr])
                    e[arm]["lower_in"] = f"{int((d < 0).sum())} of {len(d)}"
                    e[arm]["diff"] = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0)
        per[s] = e
    # gate agreement
    agree = {"n": 0, "agree": 0, "lf_acc_or_acc": 0, "lf_acc_or_rej": 0, "lf_rej_or_acc": 0, "lf_rej_or_rej": 0,
             "per_test": {}}
    for p in sorted(OUT.glob("*/s*/n*/ledger.jsonl")):
        for line in open(p):
            rec = json.loads(line)
            if rec.get("kind") != "note" or (rec.get("data") or {}).get("event") != "gate_shadow":
                continue
            d = rec["data"]
            agree["n"] += 1
            agree["agree"] += int(d["agree"])
            la, oa = d["lf_accepted"], d["oracle_accepted"]
            agree[f"lf_{'acc' if la else 'rej'}_or_{'acc' if oa else 'rej'}"] += 1
            for t, v in d["lf_passed"].items():
                pt = agree["per_test"].setdefault(t, [0, 0])
                pt[0] += int(bool(v) == bool(d["oracle_passed"].get(t)))
                pt[1] += 1
    cost = [r["A3-LF"] for r in rows if _ok(r["A3-LF"])]
    costs = {"runs": len(cost), "mean_cost_usd": float(np.mean([c["cost_usd"] or 0 for c in cost])) if cost else None,
             "sd_cost_usd": float(np.std([c["cost_usd"] or 0 for c in cost], ddof=1)) if len(cost) > 1 else None,
             "mean_sessions": float(np.mean([c["sessions"] or 0 for c in cost])) if cost else None,
             "mean_accepted": float(np.mean([c["accepted"] for c in cost])) if cost else None,
             "mean_gate_checks": float(np.mean([c["gate_checks"] or 0 for c in cost])) if cost else None,
             "total_cost_usd": float(np.sum([c["cost_usd"] or 0 for c in cost])) if cost else None,
             "stops": {k: sum(1 for c in cost if c["stop"] == k) for k in {c["stop"] for c in cost}}}
    rep = {"tests": tests, "robustness": rob, "per_setting": per, "agreement": agree, "cost": costs, "rows": rows}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"tests": tests, "agreement": agree, "cost": costs}, indent=1, default=float))
    for s, e in per.items():
        print(s, {a: (round(v["D"][0], 3), v.get("lower_in")) for a, v in e.items()})


if __name__ == "__main__":
    main()
