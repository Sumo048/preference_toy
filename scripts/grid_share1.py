#!/usr/bin/env python3
"""W3-1 driver for share 1 of PREREG.md (Phase 6, seeds 5-9). No method changes: it only calls the
frozen harness (`scripts/run_arm.py`, `toyrepair.orchestrate`, `toyrepair.eval_final`).

    python scripts/grid_share1.py caches            # jobs manifest: cell + track caches per cell
    python scripts/grid_share1.py manifest          # jobs manifest: code arms B4, R0, D1, B5 per cell
    python scripts/grid_share1.py code --arm B4 --setting k4 --seed 5 --npm 4   # one code-arm unit
    python scripts/grid_share1.py tick [--slots 8] [--rep]   # advance the LLM runs; prints actions
    python scripts/grid_share1.py eval --run <run id>        # final evaluation of one LLM run
    python scripts/grid_share1.py table             # results/grid/share1_table.md

Results: results/grid/<setting>/s<seed>/n<npm>/<arm>/result.json (LLM runs also copy their
ledger.jsonl, versions.json and run.json there). Replicates: arm dir `A3_rep1`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import toyrepair  # noqa: E402,F401  (sets TOYPREF_RUNS)
sys.path.insert(0, str(REPO / "scripts"))

RUNS = Path(os.environ["TOYPREF_RUNS"])
OUT = REPO / "results" / "grid"
DRV = OUT / "_share1"
# PREREG Section 6, share 1: rows in order, seeds 5..9, npm in listed order
CELLS = [(s, seed, n) for s in ("k4", "alt") for seed in range(5, 10) for n in (4, 16)]
LLM_ARMS = ("A3", "A2", "A1")
CODE_ARMS = ("B4", "R0", "D1", "B5")


def key_of(setting, seed, npm):
    return f"cond_{setting}_s{seed}_n{npm}"


def run_id(setting, seed, npm, arm, rep=False):
    return f"grid_{key_of(setting, seed, npm)}_{arm}" + ("_rep1" if rep else "")


def res_dir(setting, seed, npm, arm_dir):
    return OUT / setting / f"s{seed}" / f"n{npm}" / arm_dir


def _write_manifest(name, lines):
    DRV.mkdir(parents=True, exist_ok=True)
    p = DRV / f"{name}.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    print(p, len(lines), "units")


def caches():
    _write_manifest("caches", [
        {"id": f"cache_{key_of(*c)}", "cmd": ["python", "scripts/grid_share1.py", "cache", "--setting", c[0],
                                             "--seed", str(c[1]), "--npm", str(c[2])]} for c in CELLS])


def cache(setting, seed, npm):
    import torch
    torch.set_num_threads(1)
    from run_arm import cached_cell_for  # noqa
    from toyrepair import tracks
    cell = cached_cell_for("cond", setting, seed, npm)
    tracks.get(cell)
    print("cached", cell.key)


def manifest():
    lines = []
    for c in CELLS:
        for arm in ("D1", "B4", "B5", "R0"):       # slowest first inside a cell
            lines.append({"id": f"{key_of(*c)}_{arm}",
                          "cmd": ["python", "scripts/grid_share1.py", "code", "--arm", arm, "--setting", c[0],
                                  "--seed", str(c[1]), "--npm", str(c[2])],
                          "out": str((res_dir(*c, arm) / "result.json").relative_to(REPO))})
    _write_manifest("code", lines)


def code(arm, setting, seed, npm):
    raw = RUNS / "grid_code"
    rc = subprocess.run([sys.executable, "scripts/run_arm.py", "--arm", arm, "--grid", "cond", "--setting", setting,
                         "--seed", str(seed), "--npm", str(npm), "--out", str(raw), "--tag", "_grid"], cwd=REPO).returncode
    if rc:
        sys.exit(rc)
    src = raw / setting / f"s{seed}" / f"n{npm}" / f"{arm}_grid.json"
    dst = res_dir(setting, seed, npm, arm)
    dst.mkdir(parents=True, exist_ok=True)
    tmp = dst / "result.json.tmp"
    shutil.copy(src, tmp)
    os.replace(tmp, dst / "result.json")


# ----------------------------------------------------------------------------- LLM runs

def _llm_order(rep):
    if rep:
        return [(c, "A3") for c in CELLS]
    return [(c, a) for c in CELLS for a in LLM_ARMS]


def _state():
    p = DRV / "llm_state.json"
    return json.loads(p.read_text()) if p.exists() else {"inflight": {}, "done": [], "eval_started": []}


def _save_state(st):
    DRV.mkdir(parents=True, exist_ok=True)
    p = DRV / "llm_state.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, p)


def _init(rid, arm, setting, seed, npm):
    if (RUNS / "repair" / rid / "run.json").exists():
        return
    subprocess.run([sys.executable, "-m", "toyrepair.orchestrate", "init", "--run", rid, "--arm", arm, "--grid",
                    "cond", "--setting", setting, "--seed", str(seed), "--npm", str(npm), "--phase4"],
                   check=True, cwd=REPO, stdout=subprocess.DEVNULL)


def tick(slots=8, rep=False):
    """Keep up to `slots` runs active, in share order. For every active run without an in-flight
    token, call `orchestrate next` (in parallel) and print new SESSION/TRIAGE actions; start the
    final evaluation of finished runs in the background."""
    st = _state()
    order = _llm_order(rep)
    active = []
    for (c, arm) in order:
        rid = run_id(*c, arm, rep=rep)
        if rid in st["done"]:
            continue
        active.append((rid, arm, c))
        if len(active) >= slots:
            break
    procs = {}
    for rid, arm, c in active:
        if rid in st["inflight"]:
            continue
        _init(rid, arm, *c)
        procs[rid] = subprocess.Popen([sys.executable, "-m", "toyrepair.orchestrate", "next", "--run", rid],
                                      cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                      env={**os.environ, "OMP_NUM_THREADS": "1"})
    for rid, p in procs.items():
        out, err = p.communicate()
        line = (out.strip().splitlines() or [f"ERROR {err.strip()[-300:]}"])[-1]
        parts = line.split()
        if parts and parts[0] in ("SESSION", "TRIAGE", "PENDING"):
            kind = "triage" if parts[0] == "TRIAGE" or parts[1] == "triage" else parts[1]
            tok = parts[2]
            st["inflight"][rid] = {"agent": kind, "token": tok}
            print(f"NEW {kind} {tok} {rid}" + (" (pending)" if parts[0] == "PENDING" else ""))
        elif parts and parts[0] == "DONE":
            st["done"].append(rid)
            print(f"DONE {rid} {' '.join(parts[1:])}")
            if rid not in st["eval_started"]:
                st["eval_started"].append(rid)
                log = DRV / "logs" / f"eval_{rid}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                subprocess.Popen(f"nohup {sys.executable} scripts/grid_share1.py eval --run {rid} > {log} 2>&1 &",
                                 shell=True, cwd=REPO, env={**os.environ, "OMP_NUM_THREADS": "1"})
        else:
            print(f"OTHER {rid} {line}")
    _save_state(st)
    print(f"inflight {len(st['inflight'])}: " + " ".join(f"{v['agent'][0]}:{k.replace('grid_cond_', '')}"
                                                         for k, v in st["inflight"].items()))


def record(token, summary):
    st = _state()
    rid = next((k for k, v in st["inflight"].items() if v["token"] == token), None)
    r = subprocess.run([sys.executable, "-m", "toyrepair.orchestrate", "record", "--token", token, "--summary",
                        summary], cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        print("RECORD FAILED", rid, r.stderr.strip()[-500:])
        return
    if rid:
        st["inflight"].pop(rid, None)
        _save_state(st)
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
        print("recorded", rid, {k: d.get(k) for k in ("accepted", "cost_usd", "audit_hits", "invalid")
                                if k in d} or r.stdout.strip()[-200:])
    except Exception:
        print("recorded", rid, r.stdout.strip()[-200:])


def eval_run(rid):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    import phase4
    run_dir = RUNS / "repair" / rid
    # reuse W2b's evaluation record, then place it in the grid layout
    phase4.OUT = DRV / "_p4tmp"
    phase4.eval_run(rid)
    run = json.loads((run_dir / "run.json").read_text())
    setting, seed, npm, arm = run["setting"], run["seed"], run["npm"], run["arm"]
    src = phase4.out_path(arm, setting, seed, npm)
    dst = res_dir(setting, seed, npm, arm + ("_rep1" if rid.endswith("_rep1") else ""))
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("ledger.jsonl", "versions.json", "run.json", "budget.json"):
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, dst / f)
    tmp = dst / "result.json.tmp"
    shutil.copy(src, tmp)
    os.replace(tmp, dst / "result.json")
    src.unlink()
    print("wrote", dst / "result.json")


def table():
    import numpy as np
    arms = ("B0", "B1a", "B1b") + CODE_ARMS + LLM_ARMS + ("A3_rep1",)
    rows = []
    for c in CELLS:
        row = {"cell": key_of(*c)}
        for a in arms:
            p = res_dir(*c, a) / "result.json"
            if a in ("B0", "B1a", "B1b"):
                row[a] = _baseline(a, *c)
            elif p.exists():
                d = json.loads(p.read_text())
                row[a] = {"D": d["final"]["D"], "removed": d["final"].get("removed"),
                          "collateral": d["final"].get("collateral"),
                          "cost": (d.get("output") or {}).get("cost_usd"),
                          "invalid": (d.get("output") or {}).get("invalid")}
        rows.append(row)
    L = ["# Share 1 (W3-1): D per cell and arm (seeds 5-9)", "",
         "| cell | " + " | ".join(arms) + " |", "|" + "---|" * (len(arms) + 1)]
    for r in rows:
        L.append(f"| {r['cell']} | " + " | ".join(f"{r[a]['D']:.3f}" if r.get(a) else "-" for a in arms) + " |")
    L += ["", "| arm | cells | mean D | mean Removed | mean Collateral |", "|---|---|---|---|---|"]
    for a in arms:
        v = [r[a] for r in rows if r.get(a)]
        if v:
            f = lambda k: np.nanmean([x[k] if x.get(k) is not None else np.nan for x in v])
            L.append(f"| {a} | {len(v)} | {f('D'):.3f} | {f('removed'):.3f} | {f('collateral'):.3f} |")
    (OUT / "share1_table.md").write_text("\n".join(L) + "\n")
    (OUT / "share1_table.json").write_text(json.dumps(rows, indent=1, default=float))
    print("\n".join(L))


B1B = {"k4": "R_hopB_bsup_mine_neg", "alt": "R_hop_bsup_neg"}
BASE_FILES = {"B0": lambda s: "B0_frozen", "B1a": lambda s: "R_hopB_bsup_neg_AMdist", "B1b": lambda s: B1B[s]}


def _baseline(arm, setting, seed, npm):
    """B0/B1a/B1b from W1-C's results/baselines (repo code, seeds 5-9). The complete set is on branch
    art/arms (commit 2d29e95); BASELINES points at a checkout of that directory."""
    base = Path(os.environ.get("BASELINES", REPO / "results" / "baselines"))
    p = base / setting / f"s{seed}_{'k4' if setting == 'alt' else setting}_n{npm}" / f"{BASE_FILES[arm](setting)}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    m = d.get("metrics", d)
    return {"D": m.get("tv_to_ideal"), "removed": 1 - m["neg_mass_ratio"] if "neg_mass_ratio" in m else None,
            "collateral": m.get("retain_tv")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("caches")
    sub.add_parser("manifest")
    for name in ("cache", "code"):
        p = sub.add_parser(name)
        if name == "code":
            p.add_argument("--arm", required=True)
        p.add_argument("--setting", required=True)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--npm", type=int, required=True)
    p = sub.add_parser("tick")
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--rep", action="store_true")
    p = sub.add_parser("record")
    p.add_argument("--token", required=True)
    p.add_argument("--summary", default="")
    sub.add_parser("eval").add_argument("--run", required=True)
    sub.add_parser("table")
    a = ap.parse_args(argv)
    if a.cmd == "caches":
        caches()
    elif a.cmd == "cache":
        cache(a.setting, a.seed, a.npm)
    elif a.cmd == "manifest":
        manifest()
    elif a.cmd == "code":
        code(a.arm, a.setting, a.seed, a.npm)
    elif a.cmd == "tick":
        tick(a.slots, a.rep)
    elif a.cmd == "record":
        record(a.token, a.summary)
    elif a.cmd == "eval":
        eval_run(a.run)
    elif a.cmd == "table":
        table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
