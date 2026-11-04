#!/usr/bin/env python3
"""Phase 1 reproduction and the seed 5-9 baselines with the repo's own code (scripts/cond_cell.py).

Owner: W1-C. Units run through scripts/jobs.py (one unit = one cell, all of its arms).

    python scripts/phase1_repro.py manifest phase1      # -> results/phase1/manifest.jsonl
    python scripts/phase1_repro.py manifest baselines   # -> results/baselines/manifest.jsonl
    python scripts/jobs.py run results/phase1/manifest.jsonl
    python scripts/phase1_repro.py report phase1        # -> results/phase1/repro.md (+ .json)
    python scripts/phase1_repro.py report baselines     # -> results/baselines/summary.md (+ .json)

Arms per cell: B0 = ``B0_frozen``; B1a = ``R_hopB_bsup_neg_AMdist`` (the frozen recipe); B1b = the
method-map config with the lowest D on seeds 0-4 for the setting (``toyrepair.arms.b1b``).
Cell outputs: ``results/<grid>/<setting>/s<seed>_k<k>_n<npm>/<toypref arm>.json`` (cond_cell format).

Reference (``reference/method_map.json``, ``values`` = seeds 0-4, mean over seeds and npm):
the phase-1 comparison averages D over seeds 0-4 and npm 1/4/16/64 and flags |miss| > 0.025.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SETTINGS = ("k1", "k2", "k4", "contig", "alt", "pairs", "b3p1")
NPMS = (1, 4, 16, 64)
SPEC = {"k1": (1, "random"), "k2": (2, "random"), "k4": (4, "random"),
        "contig": (4, "contig"), "alt": (4, "alt"), "pairs": (4, "pairs"), "b3p1": (4, "b3p1")}
B0, B1A = "B0_frozen", "R_hopB_bsup_neg_AMdist"
TOL = 0.025
GRIDS = {"phase1": {"settings": ("k4", "alt"), "seeds": range(0, 5)},
         "baselines": {"settings": SETTINGS, "seeds": range(5, 10)}}


def b1b_config(setting: str) -> str:
    from toyrepair.arms.b1b import best_configs
    return best_configs(setting, 1)[0]


def arms_for(setting: str):
    """[(arm label, toypref config)] for one setting."""
    return [("B0", B0), ("B1a", B1A), ("B1b", b1b_config(setting))]


def cell_dir(grid: str, setting: str, seed: int, npm: int) -> Path:
    k, _ = SPEC[setting]
    return REPO / "results" / grid / setting / f"s{seed}_k{k}_n{npm}"


def manifest(grid: str) -> Path:
    g = GRIDS[grid]
    out = REPO / "results" / grid / "manifest.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    # npm 64 cells are the slowest (more rows); start them first so the tail is short
    for npm in sorted(NPMS, reverse=True):
        for setting in g["settings"]:
            k, layout = SPEC[setting]
            cfgs = list(dict.fromkeys(c for _, c in arms_for(setting)))
            for seed in g["seeds"]:
                cmd = ["python", "scripts/cond_cell.py", "--seed", str(seed), "--n_forget", str(k),
                       "--n_neg", str(npm), "--layout", layout, "--arms", ",".join(cfgs),
                       "--out", str((REPO / "results" / grid / setting).relative_to(REPO))]
                lines.append({"id": f"{setting}_s{seed}_n{npm}", "cmd": cmd})
    out.write_text("".join(json.dumps(u) + "\n" for u in lines))
    print(f"wrote {len(lines)} units to {out.relative_to(REPO)}")
    return out


def load_rows(grid: str):
    g = GRIDS[grid]
    rows = []
    for setting in g["settings"]:
        for label, cfg in arms_for(setting):
            for seed in g["seeds"]:
                for npm in NPMS:
                    f = cell_dir(grid, setting, seed, npm) / f"{cfg}.json"
                    if not f.exists():
                        continue
                    r = json.load(open(f))
                    rows.append({"setting": setting, "arm": label, "config": cfg, "seed": seed, "npm": npm,
                                 "D": r["tv_to_ideal"], "removed": 1.0 - r["neg_mass_ratio"],
                                 "collateral": r["retain_tv"], "mass_off": r["mass_off"],
                                 "train_s": r.get("train_s", 0.0)})
    return rows


def summarize(rows, setting, label):
    sel = [r for r in rows if r["setting"] == setting and r["arm"] == label]
    if not sel:
        return None
    D = np.array([r["D"] for r in sel])
    # SEM over seeds of the per-seed npm-mean (method-map style)
    seeds = sorted({r["seed"] for r in sel})
    per_seed = np.array([np.mean([r["D"] for r in sel if r["seed"] == s]) for s in seeds])
    sem = float(per_seed.std(ddof=1) / np.sqrt(len(per_seed))) if len(per_seed) > 1 else float("nan")
    return {"n": len(sel), "D": float(D.mean()), "D_sem": sem,
            "removed": float(np.mean([r["removed"] for r in sel])),
            "collateral": float(np.mean([r["collateral"] for r in sel])),
            "mass_off": float(np.mean([r["mass_off"] for r in sel])),
            "train_s": float(np.mean([r["train_s"] for r in sel])), "config": sel[0]["config"],
            "by_npm": {str(n): float(np.mean([r["D"] for r in sel if r["npm"] == n]))
                       for n in NPMS if any(r["npm"] == n for r in sel)}}


def reference(setting, cfg, key="values"):
    ref = json.load(open(REPO / "reference" / "method_map.json"))["conditional"][key]
    v = ref.get(setting, {}).get(cfg)
    return None if v is None else v["tv"]


def report(grid: str):
    g = GRIDS[grid]
    rows = load_rows(grid)
    expected = sum(len(arms_for(s)) for s in g["settings"]) * len(g["seeds"]) * len(NPMS)
    refkey = "values" if grid == "phase1" else "confirm"
    out, lines = {"grid": grid, "n_rows": len(rows), "expected_rows": expected, "rows": rows, "summary": {}}, []
    lines.append(f"# {'Phase 1 reproduction' if grid == 'phase1' else 'Baselines, seeds 5-9'} "
                 f"(repo code, scripts/cond_cell.py)\n")
    lines.append(f"Rows: {len(rows)} of {expected}. D = `tv_to_ideal`, Removed = 1 - `neg_mass_ratio`, "
                 f"Collateral = `retain_tv`; means over seeds {g['seeds'].start}-{g['seeds'].stop - 1} and "
                 f"npm 1/4/16/64; SEM over seeds of the npm-mean. Reference: `reference/method_map.json` "
                 f"`{refkey}`.\n")
    lines.append("| Setting | Arm | toypref config | n | D | ± SEM | Reference D | Miss | Removed | Collateral | "
                 "mass_off | D by npm (1/4/16/64) | train s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    issues = []
    for setting in g["settings"]:
        for label, cfg in arms_for(setting):
            s = summarize(rows, setting, label)
            if s is None:
                continue
            refD = reference(setting, cfg, refkey)
            miss = None if refD is None else s["D"] - refD
            s.update({"reference_D": refD, "miss": miss})
            out["summary"][f"{setting}/{label}"] = s
            flag = ""
            if miss is not None and abs(miss) > TOL:
                flag = " **!**"
                issues.append(f"{setting} {label} ({cfg}): D {s['D']:.3f} vs {refD:.3f} (miss {miss:+.3f})")
            npm_s = "/".join(f"{s['by_npm'].get(str(n), float('nan')):.3f}" for n in NPMS)
            lines.append(f"| {setting} | {label} | `{cfg}` | {s['n']} | {s['D']:.3f} | {s['D_sem']:.3f} | "
                         f"{'—' if refD is None else f'{refD:.3f}'} | "
                         f"{'—' if miss is None else f'{miss:+.3f}'}{flag} | {s['removed']:.3f} | "
                         f"{s['collateral']:.3f} | {s['mass_off']:.3f} | {npm_s} | {s['train_s']:.0f} |")
    out["issues"] = issues
    lines.append("")
    lines.append(f"Misses above {TOL}: " + ("none" if not issues else "; ".join(issues)) + ".")
    name = "repro" if grid == "phase1" else "summary"
    (REPO / "results" / grid / f"{name}.json").write_text(json.dumps(out, indent=1))
    (REPO / "results" / grid / f"{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "report"])
    ap.add_argument("grid", choices=list(GRIDS))
    a = ap.parse_args(argv)
    (manifest if a.cmd == "manifest" else report)(a.grid)


if __name__ == "__main__":
    main()
