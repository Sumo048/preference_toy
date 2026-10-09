"""Build the data for the 'settings x methods' overview artifact.

Rules (from the user's request):
- per grid, the settings share ONE method list;
- per method family and setting, the best config = lowest seed-mean tv_to_ideal; the union of
  those best configs is reported in every setting of the grid;
- candidate configs = configs run (complete) in EVERY setting of the grid;
- metrics: removed = 1 - neg_mass_ratio; collateral = TV between the arm's and pi_ref's
  distributions over (retained modes + off bucket), each renormalised (pi_ref = B0 of the same cell).
Seeds 0-4 everywhere (the grids where every arm ran).
"""
import glob, json, os, sys
import numpy as np, pandas as pd

RUNS = "/grogu/datasets/rollout/soumojib/toypref/runs"
M = 8
MASS = [f"mass_m{j}" for j in range(M)] + ["mass_off"]


def load_json_dir(root, seeds, grid=None):
    rows = []
    for p in glob.glob(os.path.join(root, "*", "*.json")):
        r = json.load(open(p))
        if r["seed"] not in seeds or (grid and (r["n_forget"], r["n_neg"]) not in grid):
            continue
        f = r["forget"]
        rows.append({"arm": r["arm"], "seed": r["seed"], "n_forget": r["n_forget"], "n_neg": r["n_neg"],
                     "forget": "-".join(map(str, f)), "neg_mass_ratio": r["neg_mass_ratio"],
                     "tv_to_ideal": r["tv_to_ideal"], **{m: r.get(m, np.nan) for m in MASS}})
    return pd.DataFrame(rows)


def collateral(df):
    """Per run: TV over retained modes + off, renormalised, vs B0 of the same cell."""
    key = ["seed", "n_forget", "n_neg"]
    b0 = df[df.arm == "B0_frozen"].set_index(key)[MASS]
    out = []
    for _, r in df.iterrows():
        F = [int(x) for x in str(r["forget"]).split("-")]
        R = [j for j in range(M) if j not in F]
        cols = [f"mass_m{j}" for j in R] + ["mass_off"]
        q = b0.loc[(r.seed, r.n_forget, r.n_neg), cols].values.astype(float)
        p = r[cols].values.astype(float)
        out.append(0.5 * np.abs(p / max(p.sum(), 1e-12) - q / max(q.sum(), 1e-12)).sum())
    return np.array(out)


def summarise(df):
    df = df.copy()
    df["removed"] = 1 - df.neg_mass_ratio
    df["collateral"] = collateral(df)
    g = df.groupby("arm")
    return pd.DataFrame({"removed": g.removed.mean(), "removed_sem": g.removed.sem(),
                         "collateral": g.collateral.mean(), "collateral_sem": g.collateral.sem(),
                         "tv": g.tv_to_ideal.mean(), "n": g.size()})


def pick(settings, family_of, data_of):
    """settings: {sid: summary df}. Returns config list + per-setting values."""
    full = None
    for sid, s in settings.items():
        ok = set(s.index[s.n == s.n.max()])
        full = ok if full is None else full & ok
    fams = {}
    for a in full:
        f = family_of(a)
        if f:
            fams.setdefault(f, []).append(a)
    chosen, best_for = {}, {}
    for sid, s in settings.items():
        for f, arms in fams.items():
            b = s.loc[arms, "tv"].idxmin()
            chosen.setdefault(f, set()).add(b)
            best_for.setdefault(sid, {})[f] = b
    configs = sorted({a for v in chosen.values() for a in v}, key=lambda a: (list(fams).index(family_of(a)), a))
    vals = {sid: {a: {k: float(s.loc[a, k]) for k in ("removed", "removed_sem", "collateral", "collateral_sem", "tv")}
                  for a in configs + ["B0_frozen"] if a in s.index} for sid, s in settings.items()}
    overall = {sid: min(configs, key=lambda a: settings[sid].loc[a, "tv"]) for sid in settings}
    return {"configs": [{"arm": a, "family": family_of(a), **data_of(a)} for a in configs],
            "values": vals, "best_for": best_for, "best_overall": overall,
            "n_runs": {sid: int(s.loc[configs, "n"].max()) for sid, s in settings.items()}}


# ------------------------------------------------------------------ unconditional
sys.path.insert(0, "/home/soumojib/repos/toypref-rwin/src")
from toypref.arms import ARMS as UA, depends_on_tp  # noqa: E402

seeds = {0, 1, 2, 3, 4}
grid = {(k, n) for k in (1, 2, 4) for n in (1, 4, 16)}
main = pd.read_csv("/home/soumojib/repos/toypref/results/all_runs.csv", low_memory=False)
main = main[main.seed.isin(seeds) & [(k, n) in grid for k, n in zip(main.n_forget, main.n_neg)]]
main = main[["arm", "seed", "n_forget", "n_neg", "forget", "neg_mass_ratio", "tv_to_ideal"] + MASS]
rwb = load_json_dir(f"{RUNS}/rw/base/cells", seeds, grid)
base = pd.concat([main[~main.arm.isin(set(rwb.arm))], rwb])
base = base[base.arm.isin(UA)]
nonTP = base[~base.arm.map(lambda a: depends_on_tp(UA[a]))]

nxt = pd.read_csv("/home/soumojib/repos/toypref/results/next_steps/all_runs.csv", low_memory=False)
U = {}
for k in (1, 2, 4):
    U[f"base_k{k}"] = summarise(base[base.n_forget == k])
for npos in (16, 4):
    d = nxt[(nxt.n_pos == npos) & nxt.seed.isin(seeds) & [(k, n) in grid for k, n in zip(nxt.n_forget, nxt.n_neg)]]
    d = d[["arm", "seed", "n_forget", "n_neg", "forget", "neg_mass_ratio", "tv_to_ideal"] + MASS]
    d = pd.concat([d, load_json_dir(f"{RUNS}/rw/npos{npos}/cells", seeds, grid)]).drop_duplicates(["arm", "seed", "n_forget", "n_neg"])
    U[f"npos{npos}"] = summarise(pd.concat([d, nonTP[~nonTP.arm.isin(set(d.arm))]]))
for tag in ("cover0.5", "cover0.25", "contam0.2"):
    d = load_json_dir(f"{RUNS}/rw/{tag}/cells", seeds, grid)
    U[tag] = summarise(pd.concat([d, nonTP[~nonTP.arm.isin(set(d.arm))]]))
for tag in ("std0.45", "std0.6"):
    U[tag] = summarise(load_json_dir(f"{RUNS}/rw/{tag}/cells", seeds, grid))


def ufam(a):
    s = UA[a]
    if s.kind == "frozen" or s.privileged or s.extra_label_budget or s.negative_control:
        return None
    if a == "B1_filtered":
        return "BC on positives"
    if s.kind == "anchor_only":
        return "Pseudo-positive BC"
    if s.kind == "npo":
        return "NPO, negatives only" if s.negonly else "NPO"
    if s.kind == "kto":
        return "KTO, negatives only" if s.negonly else "KTO"
    if s.kind == "dpo":
        return "Flow-DPO"
    if s.kind == "rerank":
        return "Reward model + best-of-8"
    if s.kind in ("reroute", "online"):
        return "Rerouting, negatives only" if s.negonly else "Rerouting with positives"
    return None


def udata(a):
    s = UA[a]
    if a == "B1_filtered":
        d = "pos"
    elif s.negonly or s.kind == "anchor_only":
        d = "neg"
    else:
        d = "both"
    return {"data": d, "unlabelled": bool(s.anchor) or s.kind == "anchor_only"}


out = {"unconditional": pick(U, ufam, udata)}
out["unconditional"]["oracle"] = {k: float(v.loc["ORACLE_retain_bc", "tv"]) for k, v in U.items() if "ORACLE_retain_bc" in v.index}

# ------------------------------------------------------------------ conditional
sys.path = [p for p in sys.path if "toypref-rwin" not in p]
for m in [m for m in sys.modules if m.startswith("toypref")]:
    del sys.modules[m]
sys.path.insert(0, "/home/soumojib/repos/toypref-cond/src")
from toypref.cond.experiment import ARMS as CA  # noqa: E402

C = {}
for k in (1, 2, 4):
    d = load_json_dir(f"{RUNS}/cond/cells", seeds)
    C[f"k{k}"] = summarise(d[d.n_forget == k])
for L in ("contig", "alt", "pairs", "b3p1"):
    C[L] = summarise(load_json_dir(f"{RUNS}/cond/cells_{L}", seeds))


def cfam(a):
    s = CA[a]
    if s.kind == "frozen" or not s.practical:
        return None
    return {"npo": "NPO, negatives only", "kto": "KTO, negatives only", "pseudo": "Pseudo-positive BC",
            "reroute": "Rerouting, negatives only"}[s.kind]


def cdata(a):
    s = CA[a]
    return {"data": "neg", "unlabelled": bool(s.anchor) or bool(s.amask) or s.mine or s.kind == "pseudo"}


out["conditional"] = pick(C, cfam, cdata)
json.dump(out, open("/home/soumojib/.claude/jobs/707e3708/tmp/overview/data.json", "w"), indent=1)
for g in ("unconditional", "conditional"):
    print(g, [(c["family"], c["arm"]) for c in out[g]["configs"]])
    print(" best overall:", out[g]["best_overall"])
