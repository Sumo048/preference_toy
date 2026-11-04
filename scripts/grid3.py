#!/usr/bin/env python3
"""Share 3 driver (W3-3; PREREG.md Section 6, share 3; Phase 6, seeds 5-9).

    python scripts/grid3.py manifest            # code arms B4, R0, D1, B5 -> results/grid/manifest_share3.jsonl
    python scripts/grid3.py warm <setting> <seed> <npm>   # build the cell + track caches of one cell
    python scripts/grid3.py init [--rep]        # create the LLM runs (phase-4 tools on), in share order
    python scripts/grid3.py queue [--rep]       # LLM runs in share order with their state
    python scripts/grid3.py eval --run <id>     # final evaluation of one LLM run
    python scripts/grid3.py report              # results/grid/share3_table.md

No method, threshold or prompt changes: LLM runs go through ``toyrepair.orchestrate`` exactly as in
Phase 4 (``scripts/phase4.py``), code arms through ``scripts/run_arm.py`` (full loop, no round cap).
Results: ``results/grid/<setting>/s<seed>/n<npm>/<arm>/result.json`` (replicates: ``A3_rep1/``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import toyrepair  # noqa: E402,F401

OUT = REPO / "results" / "grid"
SEEDS = (5, 6, 7, 8, 9)


def _cells(settings, npms):
    return [(s, seed, n) for s in settings for seed in SEEDS for n in npms]


# PREREG.md Section 6, share 3, rows in order
ROWS = [
    ("exploratory", _cells(["k4"], [16]), ("A3-nobranch", "A3-sonnet", "A3-nogate"), ()),
    ("item 5b", _cells(["b3p1"], [4, 16]), ("A3",), ("B4", "R0", "D1", "B5")),
    ("item 6", _cells(["k4"], [1, 64]) + _cells(["alt"], [1, 64]), ("A3",), ("B4", "R0", "D1", "B5")),
    ("item 7", _cells(["k2"], [4, 16]), ("A3",), ("B4", "R0", "D1", "B5")),
]
# replicate order: the share's A3 cells in share order
REP_CELLS = [c for _, cells, arms, _ in ROWS if "A3" in arms for c in cells]


def key_of(setting, seed, npm):
    return f"cond_{setting}_s{seed}_n{npm}"


def run_id(setting, seed, npm, armdir):
    return f"g3_{key_of(setting, seed, npm)}_{armdir}"


def res_path(setting, seed, npm, armdir):
    return OUT / setting / f"s{seed}" / f"n{npm}" / armdir / "result.json"


def llm_runs(rep=False):
    """[(run id, arm, armdir, setting, seed, npm)] in share order."""
    if rep:
        return [(run_id(s, sd, n, "A3_rep1"), "A3", "A3_rep1", s, sd, n) for s, sd, n in REP_CELLS]
    out = []
    for _, cells, arms, _ in ROWS:
        for s, sd, n in cells:
            for a in arms:
                out.append((run_id(s, sd, n, a), a, a, s, sd, n))
    return out


def manifest():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for _, cells, _, code in ROWS:
        for s, sd, n in cells:
            for arm in code:
                final = res_path(s, sd, n, arm)
                raw = REPO / "results" / "grid_raw" / s / f"s{sd}" / f"n{n}" / f"{arm}_g3.json"
                cmd = (f"python scripts/run_arm.py --arm {arm} --setting {s} --seed {sd} --npm {n} "
                       f"--out results/grid_raw --tag _g3 && mkdir -p {final.parent.relative_to(REPO)} && "
                       f"cp {raw.relative_to(REPO)} {final.relative_to(REPO)}")
                lines.append({"id": f"g3_{key_of(s, sd, n)}_{arm}", "cmd": cmd,
                              "out": str(final.relative_to(REPO))})
    p = OUT / "manifest_share3.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    print(p, len(lines), "units")


def warm(setting, seed, npm):
    import torch
    torch.set_num_threads(1)
    from types import SimpleNamespace
    from toyrepair.cli import cached_cell
    from toyrepair import tracks
    cell = cached_cell(SimpleNamespace(cell_args=("cond", setting, int(seed), int(npm))))
    tracks.get(cell)
    print("warm", cell.key)


def init(rep=False):
    root = Path(os.environ["TOYPREF_RUNS"]) / "repair"
    for rid, arm, _, s, sd, n in llm_runs(rep):
        if (root / rid / "run.json").exists():
            continue
        subprocess.run([sys.executable, "-m", "toyrepair.orchestrate", "init", "--run", rid, "--arm", arm,
                        "--grid", "cond", "--setting", s, "--seed", str(sd), "--npm", str(n), "--phase4"],
                       check=True)


def queue(rep=False):
    root = Path(os.environ["TOYPREF_RUNS"]) / "repair"
    for rid, arm, armdir, s, sd, n in llm_runs(rep):
        st = "noinit"
        rp = root / rid / "run.json"
        if rp.exists():
            r = json.loads(rp.read_text())
            st = r.get("stop") or ("invalid" if r.get("invalid") else ("pending" if r.get("pending") else "running"))
        if res_path(s, sd, n, armdir).exists():
            st += "+eval"
        print(rid, st)


def eval_run(rid):
    import phase4
    from toyrepair import orchestrate as orc
    run = orc.load_run(rid)
    armdir = "A3_rep1" if rid.endswith("_A3_rep1") else run.arm
    phase4.out_path = lambda arm, s, sd, n: res_path(s, sd, n, armdir)
    phase4.eval_run(rid)


def report():
    import numpy as np
    arms = ["B0", "B1a", "B1b", "B4", "R0", "D1", "B5", "A3", "A3_rep1", "A3-nobranch", "A3-sonnet", "A3-nogate"]
    cells = [c for _, cs, _, _ in ROWS for c in cs]
    seen, rows = set(), []
    for c in cells:
        if c in seen:
            continue
        seen.add(c)
        s, sd, n = c
        row = {"cell": f"{s} s{sd} n{n}"}
        for a in arms:
            d = None
            if a in ("B0", "B1a", "B1b"):
                d = _baseline(a, s, sd, n)
            else:
                p = res_path(s, sd, n, a)
                if p.exists():
                    f = json.loads(p.read_text())["final"]
                    d = {"D": f["D"], "removed": f["removed"], "collateral": f["collateral"]}
            row[a] = d
        rows.append(row)
    L = ["# Share 3 (W3-3): final D per cell and arm (seeds 5-9)", "",
         "D = toypref `tv_to_ideal` on the final-evaluation bank. B0/B1a/B1b: W1-C `results/baselines/`.", "",
         "| cell | " + " | ".join(arms) + " |", "|" + "---|" * (len(arms) + 1)]
    for r in rows:
        L.append(f"| {r['cell']} | " + " | ".join(f"{r[a]['D']:.3f}" if r.get(a) else "-" for a in arms) + " |")
    L += ["", "| arm | cells | mean D | Removed | Collateral |", "|---|---|---|---|---|"]
    for a in arms:
        v = [r[a] for r in rows if r.get(a)]
        if v:
            D = np.array([x["D"] for x in v])
            se = D.std(ddof=1) / np.sqrt(len(D)) if len(D) > 1 else 0.0
            rem = np.nanmean([x["removed"] if x["removed"] is not None else np.nan for x in v])
            col = np.nanmean([x["collateral"] if x["collateral"] is not None else np.nan for x in v])
            L.append(f"| {a} | {len(v)} | {D.mean():.3f} ± {se:.3f} | {rem:.3f} | {col:.3f} |")
    (OUT / "share3_table.md").write_text("\n".join(L) + "\n")
    (OUT / "share3_table.json").write_text(json.dumps(rows, indent=1, default=float))
    print("\n".join(L))


def _baseline(arm, s, sd, n):
    """W1-C seed 5-9 baselines (repo code). File names follow results/baselines/<setting>/..."""
    base = REPO / "results" / "baselines" / s
    pats = {"B0": ["B0*.json"], "B1a": ["R_hopB_bsup_neg_AMdist.json", "B1a*.json"], "B1b": ["B1b*.json"]}
    for d in (base / f"s{sd}_{s}_n{n}", base / f"s{sd}" / f"n{n}"):
        if not d.is_dir():
            continue
        for pat in pats[arm]:
            for p in sorted(d.glob(pat)):
                j = json.loads(p.read_text())
                m = j.get("metrics", j.get("final", j))
                D = m.get("tv_to_ideal", m.get("D"))
                if D is None:
                    continue
                rem = 1 - m["neg_mass_ratio"] if "neg_mass_ratio" in m else m.get("removed")
                return {"D": D, "removed": rem, "collateral": m.get("retain_tv", m.get("collateral"))}
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    w = sub.add_parser("warm")
    w.add_argument("setting"); w.add_argument("seed"); w.add_argument("npm")
    sub.add_parser("init").add_argument("--rep", action="store_true")
    sub.add_parser("queue").add_argument("--rep", action="store_true")
    sub.add_parser("eval").add_argument("--run", required=True)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        manifest()
    elif a.cmd == "warm":
        warm(a.setting, a.seed, a.npm)
    elif a.cmd == "init":
        init(a.rep)
    elif a.cmd == "queue":
        queue(a.rep)
    elif a.cmd == "eval":
        eval_run(a.run)
    elif a.cmd == "report":
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
