"""Round rw: apply the pre-registered decision rule (docs/REROUTE_WINS_PREREG.md).

For each regime and setting: the best eligible arm, the best rerouting arm, the best
non-rerouting arm, the seed-paired difference (rerouting - non-rerouting) on tv_to_ideal
over (seed, cell), a dominance check, and PASS/FAIL of criteria 1-3.

Data: regime dirs under $RUNS/rw/<tag>/cells. Arms a regime does not rerun are borrowed
from the run that is identical for them: in the base regime every pre-existing arm comes
from the main sweep ($RUNS/cells); in cover/contam regimes the arms that never read T_p
come from the base regime.

usage: analyze_rw.py [--seeds 0,1,2,3,4] [--out results/rw/exploration.md]
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from toypref.arms import ARMS, depends_on_tp
from toypref.config import RUNS

FIELDS = ["tv_to_ideal", "neg_mass_ratio", "retain_tv", "mass_off", "retain_noise_flip",
          "neg_noise_to_retain", "rows_target_frac_off", "rows_target_frac_in_forget"]
RW_ROOT = os.path.join(RUNS, "rw")


def load(cells_root, regime, seeds, grid):
    rows = []
    for p in glob.glob(os.path.join(cells_root, "*", "*.json")):
        cell = os.path.basename(os.path.dirname(p))
        s, k, n = (int(x[1:]) for x in cell.split("_"))
        if s not in seeds or (k, n) not in grid:
            continue
        with open(p) as f:
            r = json.load(f)
        rows.append({"regime": regime, "seed": s, "n_forget": k, "n_neg": n, "arm": r["arm"],
                     **{c: r.get(c, np.nan) for c in FIELDS}})
    return pd.DataFrame(rows, columns=["regime", "seed", "n_forget", "n_neg", "arm"] + FIELDS)


def setting_of(arm):
    sp = ARMS[arm]
    if sp.kind == "frozen" or sp.privileged:
        return "reference"
    if sp.extra_label_budget:
        return "extra_labels"
    if arm == "B1_filtered":
        return "POS"
    if sp.negonly:
        return "NEG+U" if (sp.anchor or sp.kind == "anchor_only") else "NEG"
    return "POS+NEG"


def is_reroute(arm):
    return ARMS[arm].kind in ("reroute", "online")


SETTINGS = {  # setting -> which arm settings are eligible (data available)
    "NEG": {"NEG"},
    "NEG+U": {"NEG", "NEG+U"},
    "POS+NEG": {"POS", "POS+NEG"},
    "ALL (POS+NEG+U)": {"NEG", "NEG+U", "POS", "POS+NEG"},
}


def paired(df, a, b):
    x = df[df.arm == a].set_index(["seed", "n_forget", "n_neg"]).tv_to_ideal
    y = df[df.arm == b].set_index(["seed", "n_forget", "n_neg"]).tv_to_ideal
    d = (x - y).dropna()
    return d.mean(), d.sem(), len(d), int((d < 0).sum())


def dominated(means, arm, pool):
    m = means.loc[arm, ["neg_mass_ratio", "retain_tv", "mass_off"]]
    for o in pool:
        if o == arm:
            continue
        q = means.loc[o, ["neg_mass_ratio", "retain_tv", "mass_off"]]
        if (q <= m).all() and (q < m).any():
            return o
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n_forget", default="1,2,4")
    ap.add_argument("--n_neg", default="1,4,16")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "rw", "exploration.md"))
    args = ap.parse_args()
    seeds = {int(x) for x in args.seeds.split(",")}
    grid = {(int(k), int(n)) for k in args.n_forget.split(",") for n in args.n_neg.split(",")}

    main_sweep = load(os.path.join(RUNS, "cells"), "base", seeds, grid)
    regimes = {}
    base_new = load(os.path.join(RW_ROOT, "base", "cells"), "base", seeds, grid)
    base = pd.concat([main_sweep[~main_sweep.arm.isin(set(base_new.arm))], base_new])
    base = base[base.arm.isin(ARMS)]
    regimes["base"] = base
    for tag in sorted(os.listdir(RW_ROOT)) if os.path.isdir(RW_ROOT) else []:
        if tag == "base":
            continue
        df = load(os.path.join(RW_ROOT, tag, "cells"), tag, seeds, grid)
        if df.empty:
            continue
        if tag.startswith("cover") or tag.startswith("contam"):
            borrow = base[~base.arm.isin(set(df.arm)) & ~base.arm.map(lambda a: depends_on_tp(ARMS[a]))]
            df = pd.concat([df, borrow.assign(regime=tag)])
        regimes[tag] = df

    out = ["# Round rw: exploration (decision rule of docs/REROUTE_WINS_PREREG.md)", "",
           f"seeds {sorted(seeds)}, cells n_forget x n_neg = {sorted(grid)}", ""]
    body, verdict = evaluate(regimes)
    out += body
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out[-(len(verdict) + 3):]))
    print("->", args.out)


def evaluate(regimes):
    out, verdict = [], []
    for tag, df in regimes.items():
        df = df.copy()
        df["setting"] = df.arm.map(setting_of)
        means = df.groupby("arm")[FIELDS].mean()
        counts = df.groupby("arm").size()
        full = counts.max()
        out.append(f"## {tag}  ({df.arm.nunique()} arms, {full} runs per complete arm)\n")
        incomplete = counts[counts < full]
        if len(incomplete):
            out.append(f"incomplete arms (excluded): {', '.join(incomplete.index)}\n")
        for st, ok in SETTINGS.items():
            pool = [a for a in means.index if setting_of(a) in ok and counts[a] == full]
            if not pool:
                continue
            m = means.loc[pool].sort_values("tv_to_ideal")
            rr = [a for a in m.index if is_reroute(a)]
            nr = [a for a in m.index if not is_reroute(a)]
            best = m.index[0]
            line = f"**{st}** best: `{best}` tv {m.tv_to_ideal.iloc[0]:.3f} " \
                   f"(forget {m.neg_mass_ratio.iloc[0]:.3f}, retain TV {m.retain_tv.iloc[0]:.3f}, off {m.mass_off.iloc[0]:.3f})"
            out.append(line + "\n")
            if rr and nr:
                mu, se, n, wins = paired(df, rr[0], nr[0])
                dom = dominated(means, rr[0], pool)
                c1 = is_reroute(best)
                c2 = mu < -2 * se
                c3 = dom is None
                ok_all = c1 and c2 and c3
                out.append(f"- best rerouting `{rr[0]}` {m.loc[rr[0], 'tv_to_ideal']:.3f} vs best non-rerouting "
                           f"`{nr[0]}` {m.loc[nr[0], 'tv_to_ideal']:.3f}: paired diff {mu:+.4f} +- {se:.4f} "
                           f"(n={n}, rerouting lower in {wins}); dominated by: {dom or '-'}")
                out.append(f"- criteria 1/2/3: {'Y' if c1 else 'N'}/{'Y' if c2 else 'N'}/{'Y' if c3 else 'N'} -> "
                           f"**{'PASS (go to confirmation)' if ok_all else 'rerouting not best'}**\n")
                verdict.append((tag, st, rr[0], nr[0], mu, se, ok_all))
            top = m.head(8)
            out.append("| arm | tv_to_ideal | forget | retain TV | off | target off | target in forget |")
            out.append("|---|---|---|---|---|---|---|")
            for a, r in top.iterrows():
                tag_r = " (R)" if is_reroute(a) else ""
                out.append(f"| {a}{tag_r} | {r.tv_to_ideal:.3f} | {r.neg_mass_ratio:.3f} | {r.retain_tv:.3f} | "
                           f"{r.mass_off:.3f} | {r.rows_target_frac_off:.2f} | {r.rows_target_frac_in_forget:.2f} |")
            out.append("")
    out.append("## Verdict table\n")
    out.append("| regime | setting | best rerouting | best other | paired diff (R - other) | rerouting best? |")
    out.append("|---|---|---|---|---|---|")
    for tag, st, r_, o_, mu, se, ok in verdict:
        out.append(f"| {tag} | {st} | {r_} | {o_} | {mu:+.4f} +- {se:.4f} | {'**YES**' if ok else 'no'} |")
    return out, verdict


if __name__ == "__main__":
    main()
