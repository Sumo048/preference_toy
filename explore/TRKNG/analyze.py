"""TRKNG analysis. python explore/TRKNG/analyze.py dev|confirm  -> prints markdown tables.

Per cell: builder mask quality (oracle precision/recall/kept-bad, evaluation only), label-free checks
(stability ARI and flag Jaccard between two fit draws, kept-set balance, agreement = mean flag Jaccard
with the other builders' frozen masks), and D/Removed/Collateral after the frozen B6 training step.
B6 and B1a per cell: dev from explore/E3/results (k4, alt) or explore/TRKNG/results (contig, pairs);
confirm from explore/E4/results (checked out from origin/art/confirm-E4 after the prereg).
Tests (paired over the cells of a setting; SE = sd/sqrt(n)):
  (a) best builder vs B6: EQUIVALENT if mean(D_b - D_B6) +- 2 SE lies inside [-0.01, +0.01]
  (b) best builder vs B1a: PASS if mean(D_b - D_B1a) < -2 SE
  (c) best builder mask precision: PASS if mean(prec) - 2 SE >= 0.95
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(HERE, "results")
B6SPEC = "a=track60k-w=1-st=20000"


def frozen():
    return json.load(open(os.path.join(HERE, "frozen.json")))


def baseline(stage, st, s, n, arm):
    cand = []
    if stage == "confirm":
        cand.append(os.path.join(ROOT, "explore", "E4", "results", st, f"s{s}_n{n}", (B6SPEC if arm == "B6" else "B1a") + ".json"))
    else:
        cand.append(os.path.join(ROOT, "explore", "E3", "results", st, f"s{s}_n{n}",
                                 (B6SPEC if arm == "B6" else "B1a") + ".json"))
        cand.append(os.path.join(RES, st, f"s{s}_n{n}", f"{arm}.json"))
    for f in cand:
        if os.path.exists(f):
            return json.load(open(f))
    return None


def unpack(z, key):
    return np.unpackbits(z[key])[:60_000].astype(bool)


def load(stage, names, settings, seeds, npms):
    rows = []
    for st in settings:
        for s in seeds:
            for n in npms:
                d = os.path.join(RES, st, f"s{s}_n{n}")
                flags, recs = {}, {}
                for b, name in names.items():
                    N = name.rsplit("-N", 1)[1]
                    fj = os.path.join(d, f"masks_{b}_N{N}.json")
                    if not os.path.exists(fj):
                        continue
                    m = json.load(open(fj))["masks"].get(name)
                    if m is None:
                        continue
                    flags[b] = unpack(np.load(fj[:-5] + ".npz"), f"flag|{name}")
                    tr = os.path.join(d, f"{name}.json")
                    t = json.load(open(tr)) if os.path.exists(tr) else {}
                    recs[b] = {"setting": st, "seed": s, "npm": n, "arm": b, "name": name,
                               "prec": m["prec"], "rec": m["rec"], "kept_bad": m["keep_frac_bad"],
                               "unassigned": m["unassigned"], "stab_ari": m.get("stab_ari"),
                               "stab_jacc": m.get("stab_flag_jacc"), "balance": m["balance"],
                               "D": t.get("D"), "Removed": t.get("Removed"), "Collateral": t.get("Collateral")}
                for b, r in recs.items():
                    oth = [o for o in flags if o != b]
                    if oth:
                        r["agree"] = float(np.mean([(flags[b] & flags[o]).sum() / max((flags[b] | flags[o]).sum(), 1)
                                                    for o in oth]))
                    rows.append(r)
                for arm in ("B6", "B1a"):
                    r = baseline(stage, st, s, n, arm)
                    if r is not None:
                        rows.append({"setting": st, "seed": s, "npm": n, "arm": arm, "D": r["D"],
                                     "Removed": r["Removed"], "Collateral": r["Collateral"],
                                     "prec": r.get("diag_track_flag_precision"), "rec": r.get("diag_track_flag_recall"),
                                     "kept_bad": r.get("diag_track_keep_frac_bad")})
    return pd.DataFrame(rows)


def paired(df, arm, other):
    a = df[df.arm == arm].set_index(["seed", "npm"])["D"]
    b = df[df.arm == other].set_index(["seed", "npm"])["D"]
    j = a.index.intersection(b.index)
    x = (a[j] - b[j]).dropna()
    if len(x) < 2:
        return np.nan, np.nan, 0, len(x)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), int((x < 0).sum()), len(x)


def f3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def table(df, best, arms):
    out = ["| Setting | Arm | Mask prec | Mask rec | Kept bad | D | Removed | Collateral | ΔD vs B6 ± SE | lower than B6 | ΔD vs B1a ± SE | lower than B1a | Tests |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tests = {}
    for st in df.setting.unique():
        ds = df[df.setting == st]
        for arm in arms:
            da = ds[ds.arm == arm]
            if da.empty:
                continue
            row = [st, arm, f3(da.prec.mean()), f3(da.rec.mean()), f3(da.kept_bad.mean()),
                   f3(da.D.mean()), f3(da.Removed.mean()), f3(da.Collateral.mean())]
            t = ""
            if arm not in ("B6", "B1a"):
                m6, se6, l6, n6 = paired(ds, arm, "B6")
                m1, se1, l1, n1 = paired(ds, arm, "B1a")
                row += [f"{m6:+.3f} ± {se6:.3f}", f"{l6} of {n6}", f"{m1:+.3f} ± {se1:.3f}", f"{l1} of {n1}"]
                if arm == best:
                    p = da.prec.dropna()
                    pse = p.std(ddof=1) / np.sqrt(len(p)) if len(p) > 1 else np.nan
                    ta = (m6 - 2 * se6 >= -0.01) and (m6 + 2 * se6 <= 0.01)
                    tb = m1 < -2 * se1
                    tc = p.mean() - 2 * pse >= 0.95
                    tests[st] = (ta, tb, tc)
                    t = f"(a) {'EQUIV' if ta else 'not equiv'}; (b) {'PASS' if tb else 'FAIL'}; (c) {'PASS' if tc else 'FAIL'}"
            elif arm == "B6":
                m1, se1, l1, n1 = paired(ds, arm, "B1a")
                row += ["—", "—", f"{m1:+.3f} ± {se1:.3f}", f"{l1} of {n1}"]
            else:
                row += ["—"] * 4
            out.append("| " + " | ".join(row + [t]) + " |")
    return "\n".join(out), tests


def checks_table(df):
    d = df[~df.arm.isin(["B6", "B1a"])].dropna(subset=["prec"])
    out = ["| Check | pooled ρ (n) | " + " | ".join(f"{b} ρ" for b in sorted(d.arm.unique())) + " |",
           "|---|---|" + "---|" * d.arm.nunique()]
    for ch in ("stab_ari", "stab_jacc", "balance", "agree"):
        if ch not in d:
            continue
        x = d.dropna(subset=[ch])
        rho = spearmanr(x[ch], x.prec).statistic if len(x) > 2 else np.nan
        cells = [f"{rho:+.2f} ({len(x)})"]
        for b in sorted(d.arm.unique()):
            xb = x[x.arm == b]
            r = spearmanr(xb[ch], xb.prec).statistic if len(xb) > 2 and xb[ch].nunique() > 1 and xb.prec.nunique() > 1 else np.nan
            cells.append(f"{r:+.2f}" if not np.isnan(r) else "—")
        out.append(f"| {ch} | " + " | ".join(cells) + " |")
    means = d.groupby("arm")[["prec", "stab_ari", "stab_jacc", "balance", "agree"]].mean().round(3)
    return "\n".join(out), means


def main():
    stage = sys.argv[1]
    if stage == "dev":
        names = json.loads(sys.argv[2]) if len(sys.argv) > 2 else frozen()["names"]
        df = load("dev", names, ["k4", "alt", "contig", "pairs"], [0, 1, 2], [4])
        best = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        fz = frozen()
        names, best = fz["names"], fz["best"]
        df = load("confirm", names, ["k1", "k2", "k4", "contig", "alt", "pairs", "b3p1"], [5, 6, 7, 8, 9], [1, 4])
    df.to_csv(os.path.join(HERE, f"cells_{stage}.csv"), index=False)
    arms = ["B6", "B1a"] + list(names)
    t, tests = table(df, best, arms)
    print(t)
    print()
    c, means = checks_table(df)
    print(c)
    print()
    print(means.to_markdown())
    if stage == "confirm":
        print()
        print("| npm | " + " | ".join(arms) + " |")
        print("|---|" + "---|" * len(arms))
        for n in (1, 4):
            print(f"| {n} | " + " | ".join(f3(df[(df.npm == n) & (df.arm == a)].D.mean()) for a in arms) + " |")


if __name__ == "__main__":
    main()
