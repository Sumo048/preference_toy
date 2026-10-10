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


def values_for(settings, configs):
    """Same frozen config list on another seed set; best markers recomputed there."""
    arms = [c["arm"] for c in configs]
    vals, best_for, overall, n_runs, missing = {}, {}, {}, {}, {}
    for sid, s in settings.items():
        have = [a for a in arms if a in s.index]
        missing[sid] = [a for a in arms if a not in s.index]
        vals[sid] = {a: {k: float(s.loc[a, k]) for k in ("removed", "removed_sem", "collateral", "collateral_sem", "tv")}
                     for a in have + (["B0_frozen"] if "B0_frozen" in s.index else [])}
        fams = {}
        for c in configs:
            if c["arm"] in have:
                fams.setdefault(c["family"], []).append(c["arm"])
        best_for[sid] = {f: s.loc[a_, "tv"].idxmin() for f, a_ in fams.items()}
        overall[sid] = s.loc[have, "tv"].idxmin() if have else None
        n_runs[sid] = int(s.loc[have, "n"].max()) if have else 0
    return {"values": vals, "best_for": best_for, "best_overall": overall, "n_runs": n_runs, "missing": missing}


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
sys.path.insert(0, "/home/soumojib/repos/toypref-sdpo/src")
from toypref.arms import ARMS as UA, depends_on_tp  # noqa: E402

grid = {(k, n) for k in (1, 2, 4) for n in (1, 4, 16)}
SEEDSETS = {"explore": {0, 1, 2, 3, 4}, "confirm": {5, 6, 7, 8, 9}}
MAIN_CSV = pd.read_csv("/home/soumojib/repos/toypref/results/all_runs.csv", low_memory=False)
NEXT_CSV = pd.read_csv("/home/soumojib/repos/toypref/results/next_steps/all_runs.csv", low_memory=False)


def load_unconditional(seeds):
  main = MAIN_CSV
  main = main[main.seed.isin(seeds) & [(k, n) in grid for k, n in zip(main.n_forget, main.n_neg)]]
  main = main[["arm", "seed", "n_forget", "n_neg", "forget", "neg_mass_ratio", "tv_to_ideal"] + MASS]
  rwb = load_json_dir(f"{RUNS}/rw/base/cells", seeds, grid)
  if not len(main):
      main = load_json_dir(f"{RUNS}/cells", seeds, grid)
  base = pd.concat([main[~main.arm.isin(set(rwb.arm))], rwb])
  base = base[base.arm.isin(UA)]
  nonTP = base[~base.arm.map(lambda a: depends_on_tp(UA[a]))]

  nxt = NEXT_CSV
  U = {}
  for k in (1, 2, 4):
      U[f"base_k{k}"] = summarise(base[base.n_forget == k])
  for npos in (16, 4):
      d = nxt[(nxt.n_pos == npos) & nxt.seed.isin(seeds) & [(k, n) in grid for k, n in zip(nxt.n_forget, nxt.n_neg)]]
      d = d[["arm", "seed", "n_forget", "n_neg", "forget", "neg_mass_ratio", "tv_to_ideal"] + MASS]
      if not len(d):
          d = load_json_dir(f"{RUNS}/cells_npos{npos}", seeds, grid)
      d = pd.concat([d, load_json_dir(f"{RUNS}/rw/npos{npos}/cells", seeds, grid)]).drop_duplicates(["arm", "seed", "n_forget", "n_neg"])
      U[f"npos{npos}"] = summarise(pd.concat([d, nonTP[~nonTP.arm.isin(set(d.arm))]]))
  for tag in ("cover0.5", "cover0.25", "contam0.2"):
      d = load_json_dir(f"{RUNS}/rw/{tag}/cells", seeds, grid)
      U[tag] = summarise(pd.concat([d, nonTP[~nonTP.arm.isin(set(d.arm))]]))
  for tag in ("std0.45", "std0.6"):
      U[tag] = summarise(load_json_dir(f"{RUNS}/rw/{tag}/cells", seeds, grid))


  return U


def ufam(a):
    s = UA[a]
    if s.kind == "frozen" or s.privileged or s.extra_label_budget or s.negative_control:
        return None
    if a == "B1_filtered":
        return "BC on positives"
    if s.kind == "anchor_only":
        return "Pseudo-positive BC"
    if s.kind == "sdpo":
        return "Self-paired DPO"
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
    elif s.negonly or s.kind in ("anchor_only", "sdpo"):
        d = "neg"
    else:
        d = "both"
    return {"data": d, "unlabelled": bool(s.anchor) or s.kind == "anchor_only"}


U = load_unconditional(SEEDSETS["explore"])
out = {"unconditional": pick(U, ufam, udata)}
out["unconditional"]["oracle"] = {k: float(v.loc["ORACLE_retain_bc", "tv"]) for k, v in U.items() if "ORACLE_retain_bc" in v.index}
U9 = load_unconditional(SEEDSETS["confirm"])
out["unconditional"]["confirm"] = values_for(U9, out["unconditional"]["configs"])
out["unconditional"]["oracle_confirm"] = {k: float(v.loc["ORACLE_retain_bc", "tv"]) for k, v in U9.items() if "ORACLE_retain_bc" in v.index}

# ------------------------------------------------------------------ conditional
sys.path = [p for p in sys.path if "toypref-sdpo" not in p]
for m in [m for m in sys.modules if m.startswith("toypref")]:
    del sys.modules[m]
sys.path.insert(0, "/home/soumojib/repos/toypref-sdpo/src")
from toypref.cond.experiment import ARMS as CA  # noqa: E402

def load_conditional(seeds):
    C = {}
    d = load_json_dir(f"{RUNS}/cond/cells", seeds)
    for k in (1, 2, 4):
        C[f"k{k}"] = summarise(d[d.n_forget == k])
    for L in ("contig", "alt", "pairs", "b3p1"):
        C[L] = summarise(load_json_dir(f"{RUNS}/cond/cells_{L}", seeds))
    return C


C = load_conditional(SEEDSETS["explore"])


def cfam(a):
    s = CA[a]
    if s.kind == "frozen" or not s.practical or s.track in ("ungated", "synbc", "reach") or s.kind in ("maskbc", "pra"):
        return None
    return {"npo": "NPO, negatives only", "kto": "KTO, negatives only", "pseudo": "Pseudo-positive BC",
            "reroute": "Rerouting, negatives only", "sdpo": "Self-paired DPO"}.get(s.kind)


def cdata(a):
    s = CA[a]
    return {"data": "neg", "unlabelled": bool(s.anchor) or bool(s.amask) or s.mine or s.kind == "pseudo"}


# ---------------- self-DPO section data (seeds 0-9 pooled; per setting; controls + relatives)
SD_C = ["SDPO_fresh_b1", "SDPO_fresh_b10", "SDPO_fresh_b25", "SDPO_fresh_b100", "SDPO_pin_b1", "SDPO_pin_b25",
        "SDPO_hop_pin_b1", "SDPO_hop_pin_b25", "SDPO_hop_pin_bH1", "SDPO_hop_pin_bH3", "SDPO_hop_pin_bH10",
        "CTRL_hop_GA", "CTRL_synB1_hop", "R_hop_bsup_neg", "R_ours_neg", "NPO_b100_neg", "R_hopB_bsup_neg_AMdist", "B0_frozen"]
DIAG = ["sdpo_pos_frac_in_forget", "sdpo_gate_contam", "sdpo_gate_clean", "sdpo_frac_rw_neg", "sdpo_logged_end_in_forget",
        "sdpo_logged_end_off", "mass_off"]


def sd_load(root, seeds):
    rows = []
    for p in glob.glob(os.path.join(root, "*", "*.json")):
        a = os.path.basename(p)[:-5]
        if a not in SD_C:
            continue
        r = json.load(open(p))
        if r["seed"] in seeds:
            rows.append({"arm": a, "seed": r["seed"], "n_forget": r["n_forget"], "n_neg": r["n_neg"],
                         "tv": r["tv_to_ideal"], "removed": 1 - r["neg_mass_ratio"], **{k: r.get(k, np.nan) for k in DIAG}})
    return pd.DataFrame(rows)


sd = {}
for L, ks in (("random", (1, 2, 4)), ("contig", (4,)), ("alt", (4,)), ("pairs", (4,)), ("b3p1", (4,))):
    d = sd_load(f"{RUNS}/cond/cells" if L == "random" else f"{RUNS}/cond/cells_{L}", set(range(10)))
    for k in ks:
        dd = d[d.n_forget == k]
        sid = f"k{k}" if L == "random" else L
        g = dd.groupby("arm")
        sd[sid] = {a: {"tv": float(x.tv.mean()), "tv_sem": float(x.tv.sem()), "removed": float(x.removed.mean()),
                       "n": int(len(x)), **{k2: (None if np.isnan(x[k2].mean()) else float(x[k2].mean())) for k2 in DIAG}}
                   for a, x in g}
out_sd = sd


out["conditional"] = pick(C, cfam, cdata)
out["conditional"]["confirm"] = values_for(load_conditional(SEEDSETS["confirm"]), out["conditional"]["configs"])
out["sdpo_section"] = out_sd
json.dump(out, open("/home/soumojib/.claude/jobs/707e3708/tmp/overview/data.json", "w"), indent=1)
for g in ("unconditional", "conditional"):
    print(g, [(c["family"], c["arm"]) for c in out[g]["configs"]])
    print(" best overall:", out[g]["best_overall"])
    print(" confirm best overall:", out[g]["confirm"]["best_overall"])
    print(" confirm missing:", {k: v for k, v in out[g]["confirm"]["missing"].items() if v})
    print(" confirm n_runs:", out[g]["confirm"]["n_runs"])
