"""Where are OUR practical algorithms best or tied-best? (user question, 2026-10-10)

Practical = usable in the robotics setting (sparse whole-trajectory negatives per setup, no behaviour identity
across setups/conditions, no extra labels): per-negative, same-state methods, plus pi_ref samples.
Excluded: anything that needs many negatives at one condition (near_nf, nf filters/anchors, pseudo-BC),
cross-condition behaviour identity (veto, vote, vote-masked anchor, mining, transplant, TRACK/JOINT/RTRACK/REACH),
PU classifiers, oracle mode width, positives, extra labels, controls.

Per setting: rank the practical arms; an arm is "tied-best" if its seed-paired difference to the best arm is
within 2 sem. Writes results/practical_best.md.
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd

RUNS = "/grogu/datasets/rollout/soumojib/toypref/runs"
out = ["# Where our practical algorithms are best (seeds 0-4; robo seeds with a working base)", ""]


def load(roots, seeds, field_map, arm_ok):
    rows = []
    for root in roots:
        for p in glob.glob(os.path.join(root, "*", "*.json")):
            a = os.path.basename(p)[:-5]
            if not arm_ok(a):
                continue
            r = json.load(open(p))
            if r.get("seed") not in seeds:
                continue
            rows.append({"arm": a, "seed": r["seed"], "n_forget": r.get("n_forget"), "n_neg": r.get("n_neg"),
                         **{k: r.get(v, np.nan) for k, v in field_map.items()}})
    return pd.DataFrame(rows)


def rank(d, keys, score, higher_better, ours, title, extra_cols=()):
    cnt = d.groupby("arm").size()
    d = d[d.arm.isin(cnt[cnt == cnt.max()].index)]
    m = d.groupby("arm")[[score] + list(extra_cols)].mean()
    m = m.sort_values(score, ascending=not higher_better)
    best = m.index[0]
    piv = d.pivot_table(index=keys, columns="arm", values=score)
    rows = []
    for a in m.index:
        z = (piv[a] - piv[best]).dropna()
        if higher_better:
            z = -z
        tied = a == best or z.mean() <= 2 * z.sem()
        rows.append((a, m.loc[a], tied))
    our_best = [a for a, _, t in rows if t and ours(a)]
    out.append(f"### {title}\n")
    out.append(f"best: `{best}` ({m.loc[best, score]:.3f}); tied-best (within 2 sem): "
               + ", ".join(f"`{a}`" for a, _, t in rows if t))
    out.append(f"**ours among the best: {', '.join(our_best) if our_best else 'NONE'}**\n")
    out.append("| rank | arm | ours? | " + " | ".join([score] + list(extra_cols)) + " | tied-best |")
    out.append("|---|---|---|" + "---|" * (len(extra_cols) + 1) + "---|")
    for i, (a, v, t) in enumerate(rows[:10]):
        out.append(f"| {i + 1} | {a} | {'yes' if ours(a) else ''} | " + " | ".join(f"{v[c]:.3f}" for c in [score] + list(extra_cols))
                   + f" | {'yes' if t else ''} |")
    out.append("")
    return best, our_best


summary = []

# ---------------- unconditional toy
U_EXCL = re.compile(r"(nf|PSEUDO|kl\d|_both|online|filt|ORACLE|B1_|DPO_cpl|RM_|B4_|_b0\.1|npo_b1$|kto_b1$|^B3_npo_b10$|^B3_npo_b100$|^B5_kto_b10$|^B5_kto_b100$)")


def u_ok(a):
    if a == "B0_frozen":
        return True
    if U_EXCL.search(a):
        return False
    return a.startswith(("R_", "SDPO_", "B3_npo", "B5_kto"))


def u_ours(a):
    return a.startswith(("R_", "SDPO_"))


fm = {"tv": "tv_to_ideal", "forget": "neg_mass_ratio", "collateral_retainTV": "retain_tv", "off": "mass_off"}
ub = load([f"{RUNS}/cells", f"{RUNS}/rw/base/cells"], set(range(5)), fm, u_ok)
ub = ub[ub.n_neg.isin([1, 4, 16])]
for k in (1, 2, 4):
    b, o = rank(ub[ub.n_forget == k], ["seed", "n_neg"], "tv", False, u_ours,
                f"Unconditional toy, forget {k} of 8 (negatives only)", ("forget", "collateral_retainTV", "off"))
    summary.append(("unconditional", f"forget {k}/8", b, o))
for tag in ("std0.45", "std0.6"):
    d = load([f"{RUNS}/rw/{tag}/cells"], set(range(5)), fm, u_ok)
    b, o = rank(d, ["seed", "n_forget", "n_neg"], "tv", False, u_ours, f"Unconditional toy, overlapping modes {tag}",
                ("forget", "collateral_retainTV", "off"))
    summary.append(("unconditional", f"overlap {tag}", b, o))

# ---------------- conditional toy
C_EXCL = re.compile(r"(hopB|hopV|_AM|mine|_tx|_pu|own|AML|TRACK|JOINT|ATYP|PRA|REACH|CTRL|PSEUDO)")


def c_ok(a):
    if a == "B0_frozen":
        return True
    if C_EXCL.search(a):
        return False
    return a.startswith(("R_", "SDPO_", "NPO_", "KTO_"))


def c_ours(a):
    return a.startswith(("R_", "SDPO_"))


for L, ks in (("random", (1, 2, 4)), ("contig", (4,)), ("alt", (4,)), ("pairs", (4,)), ("b3p1", (4,))):
    d = load([f"{RUNS}/cond/cells" if L == "random" else f"{RUNS}/cond/cells_{L}"], set(range(5)), fm, c_ok)
    for k in ks:
        dd = d[d.n_forget == k]
        name = f"forget {k}/8" + ("" if L == "random" else f", {L}")
        b, o = rank(dd, ["seed", "n_neg"], "tv", False, c_ours, f"Conditional toy, {name}", ("forget", "collateral_retainTV", "off"))
        summary.append(("conditional", name, b, o))

# ---------------- robo toy (all arms are practical)
R = f"{RUNS}/robo/cells/full_st5000_m64"
for grp, lab in (("neg", "setups with negatives"), ("noneg", "setups without negatives"), ("held", "held-out setups")):
    fmr = {"clean_success": f"{grp}_success_clean", "bad": f"{grp}_bad", "collision": f"{grp}_collision",
           "jerk": f"{grp}_jerk", "behaviour_change": f"{grp}_hom_tv"}
    d = load([R], set(range(5)), fmr, lambda a: True)
    if d.empty:
        continue
    # precondition: a seed whose frozen base does not work (clean success < 0.5) is not a valid testbed
    # (s3's base diverged at step ~10k, loss 0.09 -> 4.5, B0 clean success 0.0); drop such seeds, say so.
    b0 = d[d.arm == "B0_frozen"].set_index("seed").clean_success
    bad_seeds = sorted(b0[b0 < 0.5].index)
    if bad_seeds:
        out.append(f"_robo: dropped seed(s) {[int(x) for x in bad_seeds]}: frozen base failed (B0 clean success {b0[bad_seeds].round(3).tolist()})_\n")
        d = d[~d.seed.isin(bad_seeds)]
    d["n_forget"], d["n_neg"] = 0, 0
    b, o = rank(d, ["seed"], "clean_success", True, c_ours, f"Robo toy, {lab} (seeds {sorted(int(x) for x in d.seed.unique())})",
                ("bad", "collision", "jerk", "behaviour_change"))
    summary.append(("robo", lab, b, o))

out.insert(2, "## Summary\n\n| toy | setting | best practical arm | our arms tied-best |\n|---|---|---|---|\n" + "\n".join(
    f"| {t} | {s} | {b} | {', '.join(o) if o else '—'} |" for t, s, b, o in summary) + "\n")
p = os.path.join(os.path.dirname(__file__), "..", "results", "practical_best.md")
open(p, "w").write("\n".join(out) + "\n")
print(out[2])
