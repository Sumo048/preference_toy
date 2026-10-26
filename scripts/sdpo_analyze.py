"""Self-paired DPO (user idea) vs its relatives and the best existing methods, in every previous setting.

Seed-paired comparisons over (seed, n_neg[, n_forget]); tv_to_ideal (lower is better).
Writes results/sdpo_results.md.
"""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd

RUNS = "/grogu/datasets/rollout/soumojib/toypref/runs"
F = ["tv_to_ideal", "neg_mass_ratio", "retain_tv", "mass_off", "retain_noise_flip", "sdpo_pos_frac_in_forget"]


def load_dirs(roots, seeds, arms=None, extra=None):
    rows = []
    for root in roots:
        for p in glob.glob(os.path.join(root, "*", "*.json")):
            name = os.path.basename(p)[:-5]
            if arms is not None and name not in arms:
                continue
            r = json.load(open(p))
            if r["seed"] not in seeds:
                continue
            row = {k: r.get(k, np.nan) for k in ["seed", "n_forget", "n_neg", "arm"] + F}
            if extra:
                row.update(extra)
            rows.append(row)
    return pd.DataFrame(rows)


def paired(d, a, b, keys):
    x = d[d.arm == a].set_index(keys).tv_to_ideal
    y = d[d.arm == b].set_index(keys).tv_to_ideal
    z = (x - y).dropna()
    return z.mean(), z.sem(), int((z < 0).sum()), len(z)


def table(d, arms, keys, title, out, refs):
    m = d.groupby("arm")[F].mean()
    arms = [a for a in arms if a in m.index]
    out.append(f"### {title}\n")
    out.append(m.loc[arms].round(3).to_string() + "\n")
    sd = [a for a in arms if a.startswith("SDPO")]
    if sd:
        best = m.loc[sd, "tv_to_ideal"].idxmin()
        out.append(f"best SDPO: `{best}` ({m.loc[best, 'tv_to_ideal']:.3f})")
        for r in refs:
            if r in m.index:
                mu, se, lo, n = paired(d, best, r, keys)
                tag = "better" if mu < -2 * se else ("worse" if mu > 2 * se else "tie")
                out.append(f"- vs `{r}` ({m.loc[r, 'tv_to_ideal']:.3f}): {mu:+.4f} +- {se:.4f}, lower in {lo}/{n} -> **{tag}**")
        out.append("")
    return m


out = ["# Self-paired DPO: results in every previous toy setting", ""]

# ---------------- unconditional toy
U_SD = ["SDPO_fresh_b1", "SDPO_pin_b1", "SDPO_fresh_b25", "SDPO_pin_b25", "SDPO_nearnf_pin_b1"]
U_REF = ["B3_npo_b100_negonly", "B3_npo_b10_negonly", "B5_kto_b100_negonly", "B5_kto_b10_negonly", "R_ours_neg",
         "R_near_nf_nhood_neg", "R_near_nf_log_neg", "B5_kto_b10_negonly_Anf_dist", "PSEUDO_nf_dist", "B0_frozen"]
out.append("## Unconditional toy\n")
main = load_dirs([f"{RUNS}/cells", f"{RUNS}/rw/base/cells"], set(range(5)), set(U_SD + U_REF))
for k in (1, 2, 3, 4):
    d = main[main.n_forget == k]
    table(d, U_SD + U_REF, ["seed", "n_neg"], f"main sweep, n_forget={k} (seeds 0-4, n_neg 1-32; RW arms only on n_neg 1/4/16)", out,
          ["B3_npo_b100_negonly", "R_ours_neg", "R_near_nf_nhood_neg", "B5_kto_b10_negonly_Anf_dist"])
conf = load_dirs([f"{RUNS}/cells", f"{RUNS}/rw/base/cells"], set(range(5, 10)), set(U_SD + U_REF))
for k in (1, 2, 4):
    d = conf[conf.n_forget == k]
    table(d, U_SD + U_REF, ["seed", "n_neg"], f"seeds 5-9, n_forget={k} (n_neg 1/4/16)", out,
          ["B3_npo_b100_negonly", "R_near_nf_nhood_neg", "B5_kto_b10_negonly_Anf_dist"])
for tag in ("std0.45", "std0.6"):
    d = load_dirs([f"{RUNS}/rw/{tag}/cells"], set(range(10)), set(U_SD + U_REF))
    table(d, U_SD + U_REF, ["seed", "n_forget", "n_neg"], f"overlapping modes {tag} (seeds 0-9, n_forget 1/2/4 pooled)", out,
          ["B3_npo_b100_negonly", "R_ours_neg", "R_near_nf_nhood_neg", "B5_kto_b10_negonly_Anf_dist"])

# ---------------- conditional toy
C_SD = ["SDPO_fresh_b1", "SDPO_fresh_b10", "SDPO_fresh_b25", "SDPO_fresh_b100", "SDPO_pin_b1", "SDPO_pin_b25", "SDPO_hop_pin_b1", "SDPO_hop_pin_b25", "SDPO_hop_pin_bH1", "SDPO_hop_pin_bH3", "SDPO_hop_pin_bH10", "CTRL_hop_GA", "CTRL_synB1_hop"]
C_REF = ["R_hopB_bsup_neg_AMdist", "R_hop_bsup_neg", "R_hop_bsup_neg_AL", "R_ours_neg", "NPO_b100_neg", "NPO_b10_neg",
         "KTO_b10_neg", "NPO_b10_neg_AMdist", "PSEUDO_vote_dist", "B0_frozen"]
out.append("## Conditional toy\n")
for seeds, lab in ((set(range(5)), "seeds 0-4"), (set(range(5, 10)), "seeds 5-9"), (set(range(10, 15)), "seeds 10-14")):
    for L, k in (("random", 1), ("random", 2), ("random", 4), ("contig", 4), ("alt", 4), ("pairs", 4), ("b3p1", 4)):
        root = f"{RUNS}/cond/cells" if L == "random" else f"{RUNS}/cond/cells_{L}"
        d = load_dirs([root], seeds, set(C_SD + C_REF))
        d = d[d.n_forget == k]
        if d.empty:
            continue
        table(d, C_SD + C_REF, ["seed", "n_neg"], f"{lab}: {L}, n_forget={k}", out,
              ["NPO_b100_neg", "R_ours_neg", "R_hop_bsup_neg", "R_hopB_bsup_neg_AMdist"])

p = os.path.join(os.path.dirname(__file__), "..", "results", "sdpo_results.md")
open(p, "w").write("\n".join(out) + "\n")
print("\n".join(l for l in out if l.startswith("#") or l.startswith("best") or l.startswith("- vs")))
