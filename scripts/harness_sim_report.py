#!/usr/bin/env python3
"""W1-A1 report: miner + branch tool on development cells (seeds 0-4 only).

    python scripts/harness_sim_report.py cell --grid cond --setting k4 --seed 0 --npm 4
        -> results/harness_sim/<cell key>.json  (+ results/harness_sim/purity.csv rows)
    python scripts/harness_sim_report.py manifest   -> results/harness_sim/jobs.jsonl (k4, alt x s0-4 x npm 4, 16)
    python scripts/harness_sim_report.py summarize  -> results/harness_sim/SUMMARY.md
    W1A1_CLUSTER_RULE=agglomerative ...             -> the same under the fallback rule only, in results/harness_sim/agglomerative/

Per cell: the pi_ref atlas (clusters, open clusters, method), then the branch tool on the
largest cluster (most members; ties: larger r_ref): k_commit, failure class, best primitive,
p(k) curve, decodes and CPU seconds. Purity (oracle) goes only to purity.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "harness_sim")
RULE = os.environ.get("W1A1_CLUSTER_RULE", "hdbscan")
if RULE != "hdbscan":
    OUT = os.path.join(OUT, RULE)


def run_cell(grid, setting, seed, npm):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import toyrepair  # noqa: F401
    from toyrepair import flowstate, miner, policy
    from toyrepair.cell import load_cell
    miner.CLUSTER_RULE = RULE
    assert seed <= 4, "development seeds only"
    t0 = time.process_time()
    cell = load_cell(grid, setting, seed, npm)
    t_load = time.process_time() - t0
    v0 = policy.base_version(cell)
    policy.counter.reset()
    at = miner.mine(cell, v0)
    t_mine = time.process_time() - t0 - t_load
    miner.log_purity(cell, at, os.path.join(OUT, "purity.csv"))
    big = max(at.clusters, key=lambda c: (len(c.members), float(at.residual[c.members].mean())))
    t1 = time.process_time()
    bt = flowstate.branch(cell, v0, big)
    t_branch = time.process_time() - t1
    pur = {int(c.id): miner.purity_of(cell, c.uids) for c in at.clusters}
    rec = {
        "cell": cell.key, "grid": grid, "setting": setting, "seed": seed, "npm": npm,
        "n_neg": len(cell.neg), "n_clusters": len(at.clusters), "n_open": len(at.cards),
        "cluster_method": at.meta["cluster_method"],
        "cluster_sizes": sorted([len(c.members) for c in at.clusters], reverse=True),
        "r_ref_mean": float(at.residual.mean()),
        "largest": {"id": int(big.id), "n": len(big.members), "r_ref": float(at.residual[big.members].mean()),
                    "k_commit": bt.k_commit, "class": bt.failure_class, "best_rate": bt.meta["best_rate"],
                    "best": bt.meta["best_name"], "p": [r.p for r in bt.rows], "steps": [r.k for r in bt.rows],
                    "n_branch": [r.n for r in bt.rows], "branch_decodes": bt.n_decodes},
        # oracle-side diagnostics: report only, never shown to an LLM role
        "purity_largest": pur[int(big.id)],
        "purity_mean_weighted": float(sum(pur[c.id] * len(c.members) for c in at.clusters) / len(cell.neg)),
        "atlas_tokens": miner.est_tokens(at.render()), "table": bt.render(), "atlas": at.render(),
        "cpu_s": {"load": t_load, "mine": t_mine, "branch": t_branch}, "decodes_total": policy.counter.total,
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{cell.key}.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(path + ".tmp", path)
    print(json.dumps({k: rec[k] for k in ("cell", "n_clusters", "n_open")} | {"largest": rec["largest"]["n"],
                      "k_commit": bt.k_commit, "class": bt.failure_class, "cpu": rec["cpu_s"]}))


def manifest(cells):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "jobs.jsonl")
    with open(path, "w") as fh:
        for grid, setting, seed, npm in cells:
            key = f"{grid}_{setting}_s{seed}_n{npm}"
            cmd = ["python", "scripts/harness_sim_report.py", "cell", "--grid", grid,
                   "--setting", setting, "--seed", str(seed), "--npm", str(npm)]
            if RULE != "hdbscan":
                cmd = ["env", f"W1A1_CLUSTER_RULE={RULE}"] + cmd
            fh.write(json.dumps({"id": key, "cmd": cmd, "out": os.path.relpath(os.path.join(OUT, f"{key}.json"), REPO)}) + "\n")
    print(path)


def summarize():
    rows = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith("cond_") and f.endswith(".json"):
            rows.append(json.load(open(os.path.join(OUT, f))))
    rows.sort(key=lambda r: (r["setting"], r["npm"], r["seed"]))
    lines = ["| cell | N neg | clusters | open | method | largest n | k_commit | class | best rescue | best primitive | purity (largest) | CPU s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L = r["largest"]
        cpu = sum(r["cpu_s"].values())
        lines.append(f"| {r['cell']} | {r['n_neg']} | {r['n_clusters']} | {r['n_open']} | {r['cluster_method']} | {L['n']} | "
                     f"{L['k_commit']} | {L['class']} | {L['best_rate']:.2f} | `{L['best']}` | {r['purity_largest']:.2f} | {cpu:.0f} |")
    with open(os.path.join(OUT, "SUMMARY.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--grid", default="cond")
    c.add_argument("--setting", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--npm", type=int, required=True)
    sub.add_parser("manifest")
    sub.add_parser("summarize")
    a = ap.parse_args(argv)
    if a.cmd == "cell":
        run_cell(a.grid, a.setting, a.seed, a.npm)
    elif a.cmd == "manifest":
        manifest([("cond", s, seed, n) for s in ("k4", "alt") for n in (4, 16) for seed in range(5)])
    else:
        summarize()


if __name__ == "__main__":
    sys.exit(main())
