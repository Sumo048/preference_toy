"""W4 analysis: Phase 7 statistics and tables (PLAN 9.5, 9.6; PREREG Sections 1, 6, Amendment 1).

Reads
  results/grid/<setting>/s<seed>/n<npm>/<arm>/result.json   (W3 shares; A3_rep1 = second A3 replicate)
  results/baselines/<setting>/s<seed>_k<nf>_n<npm>/*.json    (B0, B1a, B1b; W1-C, repo code)
  results/b6/<setting>/s<seed>/n<npm>/B6.json                (B6; PREREG Amendment 1)
Writes
  results/w4/rows.csv, results/w4/stats.json, results/w4/tables.md, results/method_map_rows.json

Usage: python scripts/w4_analyze.py [--root .]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats

SETTINGS = ["k1", "k2", "k4", "contig", "alt", "pairs", "b3p1"]
NF = {"k1": 1, "k2": 2, "k4": 4, "contig": 4, "alt": 4, "pairs": 4, "b3p1": 4}
SEEDS = [5, 6, 7, 8, 9]
B1A = "R_hopB_bsup_neg_AMdist"
MAIN_ARMS = ["B0", "B1a", "B1b", "B5", "B6", "B4", "R0", "D1", "A1", "A2", "A3"]
EXPL_ARMS = ["A3-nobranch", "A3-sonnet", "A3-nogate"]
# (name, arm, comparator, sided, claim kind)
CLAIMS = [("T1a", "A3", "B1a", "less", "T1"), ("T1b", "A3", "B5", "less", "T1"),
          ("T1c", "A3", "B6", "less", "T1"), ("T2", "A3", "D1", "less", "T2"),
          ("T3", "A2", "A1", "two-sided", "T3"), ("T4", "A2", "R0", "two-sided", "T4")]
MARGIN = 0.02


def b1b_config(root, setting):
    mm = json.load(open(os.path.join(root, "reference", "method_map.json")))["conditional"]["values"][setting]
    cand = sorted((v["tv"], k) for k, v in mm.items() if k != "B0_frozen" and not k.startswith("SDPO_"))
    return cand[0][1]


def _row(arm, setting, seed, npm, D, removed, collateral, mass_off, src, valid=True, reason="", extra=None):
    r = dict(arm=arm, setting=setting, seed=seed, npm=npm, D=D, removed=removed, collateral=collateral,
             mass_off=mass_off, src=src, valid=valid, reason=reason)
    r.update(extra or {})
    return r


def _invalid(d, path):
    """A W3 run is invalid when its JSON says so or an INVALID marker sits beside it."""
    out = d.get("output") or {}
    for k in ("invalid", "invalid_reason"):
        v = d.get(k) if d.get(k) is not None else out.get(k)
        if v:
            return str(v)
    for m in ("INVALID", "INVALID.md", "invalid.txt", "invalid.json"):
        p = os.path.join(os.path.dirname(path), m)
        if os.path.exists(p):
            return open(p).read().strip()[:200] or m
    return ""


def load_grid(root):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "results/grid/*/s*/n*/*/result.json"))):
        parts = p.split(os.sep)
        arm, npm, seed, setting = parts[-2], int(parts[-3][1:]), int(parts[-4][1:]), parts[-5]
        try:
            d = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            rows.append(_row(arm, setting, seed, npm, None, None, None, None, p, False, f"unreadable: {e}"))
            continue
        f = d.get("final") or {}
        out = d.get("output") or {}
        reason = _invalid(d, p)
        if f.get("D") is None:
            reason = reason or "no final evaluation"
        sess = d.get("sessions") or []
        extra = dict(
            cost=_cost(d), sessions=out.get("sessions", len(sess) or None),
            gate_checks=out.get("gate_checks"), decodes=out.get("decodes"),
            accepted=len(out.get("accepted_chain") or out.get("accepted") or []),
            wall_s=d.get("wall_s_total") or d.get("wall_s"), stop=out.get("stop"),
            tokens=_tokens(sess))
        rows.append(_row(arm, setting, seed, npm, f.get("D"), f.get("removed"), f.get("collateral"),
                         f.get("mass_off"), p, not reason, reason, extra))
    return rows


def _cost(d):
    out = d.get("output") or {}
    if out.get("cost_usd") is not None:
        return float(out["cost_usd"])
    c = sum(float(s.get("cost_usd") or 0) for s in d.get("sessions") or [])
    c += sum(float(t.get("cost_usd") or 0) for t in d.get("triage") or [])
    return c or None


def _tokens(sess):
    tot = 0
    seen = False
    for s in sess:
        u = s.get("usage") or {}
        for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            if u.get(k) is not None:
                tot += u[k]
                seen = True
    return tot if seen else None


def load_baselines(root):
    rows = []
    for st in SETTINGS:
        b1b = b1b_config(root, st)
        for seed in SEEDS:
            for npm in (1, 4, 16, 64):
                d0 = os.path.join(root, "results/baselines", st, f"s{seed}_k{NF[st]}_n{npm}")
                for arm, fn in (("B0", "B0_frozen"), ("B1a", B1A), ("B1b", b1b)):
                    p = os.path.join(d0, fn + ".json")
                    if not os.path.exists(p):
                        continue
                    d = json.load(open(p))
                    rows.append(_row(arm, st, seed, npm, d["tv_to_ideal"], 1 - d["neg_mass_ratio"],
                                     d["retain_tv"], d["mass_off"], p, extra=dict(config=fn)))
    for p in sorted(glob.glob(os.path.join(root, "results/b6/*/s*/n*/B6.json"))):
        parts = p.split(os.sep)
        st, seed, npm = parts[-4], int(parts[-3][1:]), int(parts[-2][1:])
        d = json.load(open(p))
        f = d.get("final") or d
        D = f.get("D", f.get("tv_to_ideal"))
        rem = f.get("removed", 1 - f["neg_mass_ratio"] if "neg_mass_ratio" in f else None)
        col = f.get("collateral", f.get("retain_tv"))
        rows.append(_row("B6", st, seed, npm, D, rem, col, f.get("mass_off"), p))
    # Fallback for B6 cells that results/b6 lacks: session E4 ran the identical E3 spec on seeds 5-9
    # (explore/E4/results/<setting>/s<seed>_n<npm>/a=track60k-w=1-st=20000.json). PREREG Amendment 1 shows the
    # B6 port reproduces E3 bit for bit, so these are the same numbers. Each row records its source.
    have = {(r["setting"], r["seed"], r["npm"]) for r in rows if r["arm"] == "B6"}
    for p in sorted(glob.glob(os.path.join(root, "explore/E4/results/*/s*_n*/a=track60k-w=1-st=20000.json"))):
        parts = p.split(os.sep)
        st = parts[-3]
        sd, n = parts[-2].split("_")
        seed, npm = int(sd[1:]), int(n[1:])
        if (st, seed, npm) in have:
            continue
        d = json.load(open(p))
        rows.append(_row("B6", st, seed, npm, d["D"], d["Removed"], d["Collateral"], d.get("mass_off"), p,
                         extra=dict(config="E4 a=track60k-w=1-st=20000")))
    return rows


def index(rows):
    """{(arm, setting, seed, npm): row}; grid rows win over baseline rows of the same key."""
    ix = {}
    for r in rows:
        ix[(r["arm"], r["setting"], r["seed"], r["npm"])] = r
    return ix


def a3_cells(ix, arm="A3"):
    return sorted({(s, sd, n) for (a, s, sd, n), r in ix.items() if a == arm and r["valid"]})


def paired_units(ix, arm, comp, metric="D", arm_sub=None):
    """(setting, seed) units with D averaged over the npm values ``arm`` ran on in that setting;
    a unit counts only if both arms are valid on every such npm (PREREG Section 1)."""
    arm_src = arm_sub or arm
    by = defaultdict(set)
    for (a, s, sd, n), r in ix.items():
        if a == arm_src and r["valid"]:
            by[(s, sd)].add(n)
    if arm == "A2" or comp in ("A1",):
        pass
    units = []
    for (s, sd), npms in sorted(by.items()):
        xs, ys, ok = [], [], True
        for n in sorted(npms):
            ra, rc = ix.get((arm_src, s, sd, n)), ix.get((comp, s, sd, n))
            if not (ra and rc and ra["valid"] and rc["valid"] and ra[metric] is not None and rc[metric] is not None):
                ok = False
                break
            xs.append(ra[metric])
            ys.append(rc[metric])
        if ok and xs:
            units.append(dict(setting=s, seed=sd, npms=sorted(npms), a=float(np.mean(xs)), c=float(np.mean(ys))))
    return units


def ttest(diff, sided):
    n = len(diff)
    if n < 2:
        return dict(n=n, mean=float(np.mean(diff)) if n else None, se=None, t=None, p=None, ci=None)
    m, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    se = sd / math.sqrt(n)
    tq = stats.t.ppf(0.975, n - 1)
    if se == 0:
        t, p = (-math.inf if m < 0 else math.inf if m > 0 else 0.0), (0.0 if m != 0 else 1.0)
        if sided == "less":
            p = 0.0 if m < 0 else 1.0
    else:
        t = m / se
        p = float(stats.t.cdf(t, n - 1)) if sided == "less" else float(2 * stats.t.sf(abs(t), n - 1))
    return dict(n=n, mean=m, sd=sd, se=se, t=t, p=p, ci=[m - tq * se, m + tq * se])


def holm(ps):
    """Holm step-down over a dict {name: p}; returns {name: (adjusted p, reject at 0.05)}."""
    items = sorted((p, k) for k, p in ps.items() if p is not None)
    m = len(items)
    out, running, stop = {}, 0.0, False
    for i, (p, k) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        rej = (not stop) and p <= 0.05 / (m - i)
        if not rej:
            stop = True
        out[k] = (adj, rej)
    for k, p in ps.items():
        if p is None:
            out[k] = (None, False)
    return out


def label(kind, res, rej, coll_gap):
    if res["n"] is None or res["n"] < 2 or res["ci"] is None:
        return "inconclusive (too few pairs)"
    lo, hi = res["ci"]
    if kind in ("T1", "T2"):
        if rej and res["mean"] < 0 and (coll_gap is None or coll_gap <= MARGIN):
            return "supported"
        if lo > -MARGIN:
            return "unsupported"
        return "inconclusive"
    if rej:
        return "supported (" + ("arm lower" if res["mean"] < 0 else "arm higher") + ")"
    if lo > -MARGIN and hi < MARGIN:
        return "unsupported"
    return "inconclusive"


def mean_coll(ix, arm, units, arm_src=None):
    v = [ix[(arm_src or arm, u["setting"], u["seed"], n)]["collateral"] for u in units for n in u["npms"]]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def claims(ix, arm_sub=None):
    res, ps = {}, {}
    for name, arm, comp, sided, kind in CLAIMS:
        src = arm_sub if (arm_sub and arm == "A3") else None
        u = paired_units(ix, arm, comp, arm_sub=src)
        diff = np.array([x["a"] - x["c"] for x in u])
        r = ttest(diff, sided)
        ca = mean_coll(ix, arm, u, src)
        cc = mean_coll(ix, comp, u)
        r.update(arm=arm, comp=comp, sided=sided, kind=kind, arm_D=float(np.mean([x["a"] for x in u])) if u else None,
                 comp_D=float(np.mean([x["c"] for x in u])) if u else None, arm_coll=ca, comp_coll=cc,
                 lower_in=int((diff < 0).sum()), units=u)
        res[name] = r
        ps[name] = r["p"]
    h = holm(ps)
    for name, (adj, rej) in h.items():
        r = res[name]
        r["p_holm"], r["reject"] = adj, rej
        gap = (r["arm_coll"] - r["comp_coll"]) if (r["arm_coll"] is not None and r["comp_coll"] is not None) else None
        r["label"] = label(r["kind"], r, rej, gap)
    t1 = [res[k]["label"] for k in ("T1a", "T1b", "T1c")]
    t1_label = ("supported" if all(x == "supported" for x in t1) else
                "unsupported" if any(x == "unsupported" for x in t1) else "inconclusive")
    return res, t1_label


def seed_level(ix, arm, comp, sided):
    """Robustness: D averaged over every (setting, npm) cell where both arms are valid, per seed (n = 5)."""
    out = []
    for sd in SEEDS:
        a, c = [], []
        for (ar, s, sd2, n), r in ix.items():
            if ar == arm and sd2 == sd and r["valid"]:
                rc = ix.get((comp, s, sd, n))
                if rc and rc["valid"] and rc["D"] is not None and r["D"] is not None:
                    a.append(r["D"])
                    c.append(rc["D"])
        if a:
            out.append((sd, float(np.mean(a)), float(np.mean(c)), len(a)))
    r = ttest(np.array([x[1] - x[2] for x in out]), sided)
    r["per_seed"] = out
    return r


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def per_setting(ix, arm, metric, cells=None):
    """mean ± SE over seeds and npm on the given cells."""
    out = {}
    for st in SETTINGS:
        v = [r[metric] for (a, s, sd, n), r in ix.items() if a == arm and s == st and r["valid"]
             and r[metric] is not None and (cells is None or (s, sd, n) in cells)]
        if v:
            out[st] = (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0, len(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = args.root
    rows = load_baselines(root) + load_grid(root)
    ix = index(rows)
    os.makedirs(os.path.join(root, "results/w4"), exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(root, "results/w4/rows.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    res, t1 = claims(ix)
    rep_res, rep_t1 = claims(ix, arm_sub="A3_rep1")
    seed = {name: seed_level(ix, arm, comp, sided) for name, arm, comp, sided, _ in CLAIMS}
    expl = {}
    for e in EXPL_ARMS:
        u = paired_units(ix, "A3", e)
        r = ttest(np.array([x["a"] - x["c"] for x in u]), "two-sided")
        r.update(A3_D=float(np.mean([x["a"] for x in u])) if u else None,
                 arm_D=float(np.mean([x["c"] for x in u])) if u else None, units=u)
        expl[e] = r

    a3c = set(a3_cells(ix))
    L = []
    L.append("## Claims (PREREG Section 1; Holm over six tests)\n")
    L.append("| claim | comparison | pairs | arm D | comp D | mean diff | 95% CI | p | p (Holm) | lower in | arm Coll | comp Coll | label |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, *_ in CLAIMS:
        r = res[name]
        ci = f"[{fmt(r['ci'][0])}, {fmt(r['ci'][1])}]" if r.get("ci") else "—"
        L.append(f"| {name} | {r['arm']} vs {r['comp']} ({'one-sided' if r['sided']=='less' else 'two-sided'}) | {r['n']} | "
                 f"{fmt(r['arm_D'])} | {fmt(r['comp_D'])} | {fmt(r['mean'])} | {ci} | {fmt(r['p'],4)} | "
                 f"{fmt(r['p_holm'],4)} | {r['lower_in']}/{r['n']} | {fmt(r['arm_coll'])} | {fmt(r['comp_coll'])} | {r['label']} |")
    L.append(f"\nT1 (all of T1a, T1b, T1c): **{t1}**.\n")
    L.append("### Robustness: seed level (D averaged over settings and npm, n = seeds)\n")
    L.append("| claim | n | mean diff | 95% CI | p (unadjusted) |")
    L.append("|---|---|---|---|---|")
    for name, *_ in CLAIMS:
        r = seed[name]
        ci = f"[{fmt(r['ci'][0])}, {fmt(r['ci'][1])}]" if r.get("ci") else "—"
        L.append(f"| {name} | {r['n']} | {fmt(r['mean'])} | {ci} | {fmt(r['p'],4)} |")
    L.append("\n### Robustness: A3 second replicate (A3_rep1) in place of A3\n")
    L.append("| claim | pairs | mean diff | 95% CI | p (Holm) | label |")
    L.append("|---|---|---|---|---|---|")
    for name in ("T1a", "T1b", "T1c", "T2"):
        r = rep_res[name]
        ci = f"[{fmt(r['ci'][0])}, {fmt(r['ci'][1])}]" if r.get("ci") else "—"
        L.append(f"| {name} | {r['n']} | {fmt(r['mean'])} | {ci} | {fmt(r['p_holm'],4)} | {r['label']} |")
    L.append(f"\nT1 with the replicate: {rep_t1}.\n")

    for metric, title in (("D", "D"), ("removed", "Removed"), ("collateral", "Collateral")):
        L.append(f"## Table 1{'abc'[['D','removed','collateral'].index(metric)]}: {title} per arm and setting (seeds 5-9, all cells each arm ran; mean ± SE, n cells)\n")
        L.append("| arm | " + " | ".join(SETTINGS) + " |")
        L.append("|---|" + "---|" * len(SETTINGS))
        for arm in MAIN_ARMS:
            ps = per_setting(ix, arm, metric)
            if not ps:
                continue
            L.append(f"| {arm} | " + " | ".join(f"{fmt(ps[s][0])} ± {fmt(ps[s][1])} ({ps[s][2]})" if s in ps else "—"
                                             for s in SETTINGS) + " |")
        L.append("")
    L.append("## Table 1d: D on A3's cells only (paired view; mean ± SE, n cells)\n")
    L.append("| arm | " + " | ".join(SETTINGS) + " |")
    L.append("|---|" + "---|" * len(SETTINGS))
    for arm in MAIN_ARMS + ["A3_rep1"]:
        ps = per_setting(ix, arm, "D", a3c)
        if ps:
            L.append(f"| {arm} | " + " | ".join(f"{fmt(ps[s][0])} ± {fmt(ps[s][1])} ({ps[s][2]})" if s in ps else "—"
                                                 for s in SETTINGS) + " |")
    L.append("\n## Per-setting paired differences, method-map format (A3 minus comparator, mean ± SE over seeds and npm, lower in n of N)\n")
    L.append("| comparator | " + " | ".join(SETTINGS) + " |")
    L.append("|---|" + "---|" * len(SETTINGS))
    for comp in ["B0", "B1a", "B1b", "B5", "B6", "B4", "R0", "D1", "A1", "A2"]:
        cellsd = []
        for st in SETTINGS:
            d = [ix[("A3", s, sd, n)]["D"] - ix[(comp, s, sd, n)]["D"] for (s, sd, n) in a3c if s == st
                 and (comp, s, sd, n) in ix and ix[(comp, s, sd, n)]["valid"] and ix[(comp, s, sd, n)]["D"] is not None]
            cellsd.append(f"{fmt(np.mean(d))} ± {fmt(np.std(d, ddof=1)/math.sqrt(len(d)) if len(d)>1 else 0)} ({sum(x<0 for x in d)}/{len(d)})" if d else "—")
        L.append(f"| {comp} | " + " | ".join(cellsd) + " |")
    L.append("\n## Table 3: exploratory arms (k4, npm 16, seeds 5-9; A3 minus arm)\n")
    L.append("| arm | pairs | arm D | A3 D | mean diff | 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for e, r in expl.items():
        ci = f"[{fmt(r['ci'][0])}, {fmt(r['ci'][1])}]" if r.get("ci") else "—"
        L.append(f"| {e} | {r['n']} | {fmt(r['arm_D'])} | {fmt(r['A3_D'])} | {fmt(r['mean'])} | {ci} |")
    L.append("\n## Table 5: cost, decodes, wall time per arm (grid runs)\n")
    L.append("| arm | runs (valid/all) | sessions/run | cost/run $ | total $ | accepted/run | $/accepted edit | gate checks/run | decodes/run | wall h/run |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ["B4", "R0", "D1", "B5", "A1", "A2", "A3", "A3_rep1"] + EXPL_ARMS:
        rr = [r for r in rows if r["arm"] == arm and r["src"].endswith("result.json")]
        if not rr:
            continue
        def m(k):
            v = [r.get(k) for r in rr if r.get(k) is not None]
            return float(np.mean(v)) if v else None
        tot = sum(r.get("cost") or 0 for r in rr)
        acc = sum(r.get("accepted") or 0 for r in rr)
        L.append(f"| {arm} | {sum(r['valid'] for r in rr)}/{len(rr)} | {fmt(m('sessions'),1)} | {fmt(m('cost'),2)} | {tot:.2f} | "
                 f"{fmt(m('accepted'),2)} | {fmt(tot/acc if acc else None,2)} | {fmt(m('gate_checks'),1)} | "
                 f"{fmt(m('decodes'),0)} | {fmt((m('wall_s') or 0)/3600 if m('wall_s') else None,2)} |")
    # Table 4: accepted A3 edits (from result.json gates; per-edit gate D and final gain need the run ledgers,
    # which the W3 sessions did not commit)
    L.append("\n## Table 4: accepted A3 edits (grid, seeds 5-9)\n")
    L.append("| cell | version | cluster | edit call | gate target fall | 95% CI | run final D (B0 D) |")
    L.append("|---|---|---|---|---|---|---|")
    calls = defaultdict(int)
    tried = defaultdict(int)
    n_edits = 0
    for p in sorted(glob.glob(os.path.join(root, "results/grid/*/s*/n*/A3/result.json"))):
        d = json.load(open(p))
        b0 = ix.get(("B0", d["setting"], d["seed"], d["npm"]))
        for g in d.get("gates") or []:
            tried[g.get("call")] += 1
            v = g.get("view") or {}
            if not v.get("accepted"):
                continue
            n_edits += 1
            calls[g.get("call")] += 1
            ci = v.get("target_fall_ci") or [None, None]
            L.append(f"| {d['setting']} s{d['seed']} n{d['npm']} | {g.get('version')} | {g.get('cluster')} | {g.get('call')} | "
                     f"{fmt(v.get('target_fall'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt((d.get('final') or {}).get('D'))} "
                     f"({fmt(b0['D'] if b0 else None)}) |")
    L.append(f"\nAccepted A3 edits: {n_edits}; by call: {dict(calls)}; gate submissions by call: {dict(tried)}.")
    inval = [r for r in rows if not r["valid"]]
    L.append(f"\nInvalid or unevaluated runs: {len(inval)}.")
    for r in inval:
        L.append(f"- {r['arm']} {r['setting']} s{r['seed']} n{r['npm']}: {r['reason']}")
    open(os.path.join(root, "results/w4/tables.md"), "w").write("\n".join(L) + "\n")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
            return None
        return o
    json.dump(clean(dict(claims=res, T1=t1, replicate=rep_res, T1_replicate=rep_t1, seed_level=seed,
                         exploratory=expl)), open(os.path.join(root, "results/w4/stats.json"), "w"), indent=1)

    # method_map_rows.json: the toyrepair arms in the method map's DATA layout (configs + confirm.values)
    fam = {"A1": "Agent: targeted_train only", "A2": "Agent: weight writes only", "A3": "Agent: all edit calls",
           "A3_rep1": "Agent: all edit calls (replicate)", "A3-nobranch": "Agent ablation", "A3-sonnet": "Agent ablation",
           "A3-nogate": "Agent ablation", "B4": "Iterative find-then-fine-tune", "R0": "Sampling-time controller",
           "D1": "Branch-and-distill", "B5": "Gated recipe selection", "B6": "Track-masked BC"}
    mm = {"configs": [], "confirm": {"values": {s: {} for s in SETTINGS}, "n_runs": {}, "npm": {}}}
    for arm in fam:
        have = False
        for st in SETTINGS:
            cells = [r for (a, s, sd, n), r in ix.items() if a == arm and s == st and r["valid"] and r["D"] is not None]
            if not cells:
                continue
            have = True
            def ms(k):
                v = [r[k] for r in cells if r[k] is not None]
                return (round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1) / math.sqrt(len(v))), 4) if len(v) > 1 else 0.0)
            rm, rs = ms("removed")
            cm, cs = ms("collateral")
            tv, ts = ms("D")
            mm["confirm"]["values"][st][f"TR_{arm}"] = dict(removed=rm, removed_sem=rs, collateral=cm, collateral_sem=cs,
                                                            tv=tv, tv_sem=ts, n=len(cells))
            mm["confirm"]["npm"].setdefault(f"TR_{arm}", {})[st] = sorted({r["npm"] for r in cells})
        if have:
            mm["configs"].append(dict(arm=f"TR_{arm}", family=fam[arm], data="neg", unlabelled=True))
    mm["note"] = ("toyrepair arms on seeds 5-9 (confirm), each cell's D/Removed/Collateral from toypref compute on the "
                  "final-evaluation bank. Unlike the map's own rows, which average npm 1,4,16,64, these average the npm "
                  "values each arm ran on (see npm). Prefix TR_ marks toyrepair arms.")
    json.dump(mm, open(os.path.join(root, "results/method_map_rows.json"), "w"), indent=1)
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
