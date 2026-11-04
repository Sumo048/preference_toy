#!/usr/bin/env python3
"""A3LF driver: arm A3-LF (A3 with the label-free gate ``toyrepair.gate_lf``) on seeds 5-9.
Adapted from W3-1's ``scripts/grid_share1.py``; protocol in ``docs/A3LF_PREREG.md``.

    python scripts/a3lf.py caches [--extra]          # jobs manifest: cell + track caches per cell
    python scripts/a3lf.py cache --setting k4 --seed 5 --npm 4
    python scripts/a3lf.py tick [--slots 8] [--extra]  # advance the runs; prints NEW/DONE actions
    python scripts/a3lf.py record --token T --summary S
    python scripts/a3lf.py eval --run <run id>       # outside the loop: final eval + oracle shadow gate
    python scripts/a3lf.py table                     # results/a3lf/table.json

The loop process (``toyrepair.orchestrate`` / ``toyrepair-tool``) never imports the oracle for an A3-LF run.
``eval`` runs only after the run has stopped. It (1) evaluates the final version with ``eval_final``;
(2) re-runs every gate_lf check with the oracle gate ``toyrepair.gate.gate_check`` on the same batch (same
check index, same current/new versions) and appends one ``note`` record with event ``gate_shadow`` per check to the ledger;
(3) logs oracle purity of every atlas. None of this reaches a surgeon: the run is finished.

Results: results/a3lf/<setting>/s<seed>/n<npm>/ (result.json, ledger.jsonl, versions.json, run.json,
budget.json, purity.csv, patches/).
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
OUT = REPO / "results" / "a3lf"
DRV = OUT / "_driver"
ARM = "A3-LF"
# A3LF_PREREG Section 3: main cells in order (setting, seed 5..9, npm 4 then 16); extra cells at npm 4
CELLS = [(s, seed, n) for s in ("k4", "alt") for seed in range(5, 10) for n in (4, 16)]
EXTRA = [(s, seed, 4) for s in ("k1", "contig", "b3p1") for seed in range(5, 10)]


def key_of(setting, seed, npm):
    return f"cond_{setting}_s{seed}_n{npm}"


def run_id(setting, seed, npm):
    return f"a3lf_{key_of(setting, seed, npm)}"


def res_dir(setting, seed, npm):
    return OUT / setting / f"s{seed}" / f"n{npm}"


def _cells(extra):
    return CELLS + (EXTRA if extra else [])


def _write_manifest(name, lines):
    DRV.mkdir(parents=True, exist_ok=True)
    p = DRV / f"{name}.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    print(p, len(lines), "units")


def caches(extra=False):
    _write_manifest("caches", [
        {"id": f"cache_{key_of(*c)}", "cmd": ["python", "scripts/a3lf.py", "cache", "--setting", c[0],
                                             "--seed", str(c[1]), "--npm", str(c[2])]} for c in _cells(extra)])


def cache(setting, seed, npm):
    import torch
    torch.set_num_threads(1)
    from run_arm import cached_cell_for  # noqa
    from toyrepair import tracks
    cell = cached_cell_for("cond", setting, seed, npm)
    tracks.get(cell)
    print("cached", cell.key)


# ----------------------------------------------------------------------------- the loop

def _state():
    p = DRV / "llm_state.json"
    return json.loads(p.read_text()) if p.exists() else {"inflight": {}, "done": [], "eval_started": []}


def _save_state(st):
    DRV.mkdir(parents=True, exist_ok=True)
    p = DRV / "llm_state.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, p)


def _init(rid, setting, seed, npm):
    if (RUNS / "repair" / rid / "run.json").exists():
        return
    subprocess.run([sys.executable, "-m", "toyrepair.orchestrate", "init", "--run", rid, "--arm", ARM, "--grid",
                    "cond", "--setting", setting, "--seed", str(seed), "--npm", str(npm), "--phase4"],
                   check=True, cwd=REPO, stdout=subprocess.DEVNULL)


def tick(slots=8, extra=False):
    st = _state()
    active = []
    for c in _cells(extra):
        rid = run_id(*c)
        if rid in st["done"]:
            continue
        active.append((rid, c))
        if len(active) >= slots:
            break
    procs = {}
    for rid, c in active:
        if rid in st["inflight"]:
            continue
        _init(rid, *c)
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
                subprocess.Popen(f"nohup {sys.executable} scripts/a3lf.py eval --run {rid} > {log} 2>&1 &",
                                 shell=True, cwd=REPO, env={**os.environ, "OMP_NUM_THREADS": "1"})
        else:
            print(f"OTHER {rid} {line}")
    _save_state(st)
    print(f"inflight {len(st['inflight'])}: " + " ".join(f"{v['agent'][0]}:{k.replace('a3lf_cond_', '')}"
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
        print("recorded", rid, {k: d.get(k) for k in ("accepted", "cost_usd", "audit_hits", "cluster", "fallback")
                                if k in d})
    except Exception:
        print("recorded", rid, r.stdout.strip()[-200:])


# ----------------------------------------------------------------------------- outside the loop

def _shadow_records(run):
    """Ledger kinds are fixed (toyrepair.ledger.KINDS): shadow checks are ``note`` records with event gate_shadow."""
    from toyrepair import orchestrate as orc
    return [r for r in orc.ledger_records(run, "note") if (r.get("data") or {}).get("event") == "gate_shadow"]


def shadow_gates(run, be):
    """Oracle gate on the batch of every gate_lf check of a finished run -> ledger ``gate_shadow``."""
    from toyrepair import gate, orchestrate as orc
    done = {(r.get("data") or {}).get("check_index") for r in _shadow_records(run)}
    cell = be.cell(run)
    v0 = be.version(run, "v0")
    out = []
    for r in orc.ledger_records(run, "gate"):
        d = r.get("data") or {}
        res = d.get("result") or {}
        ci = res.get("check_index")
        if ci in done:
            continue
        vers = (res.get("numbers") or {}).get("versions") or {}
        cur, new = vers.get("current"), vers.get("new") or r.get("version")
        cl = be.cluster(run, d["cluster"], cur)
        og = gate.gate_check(cell, v0, be.version(run, cur), be.version(run, new), [int(u) for u in cl.uids], ci,
                             scope="cluster", cfg=gate.GateConfig(**run.gate_cfg))
        rec = {"check_index": ci, "session": r.get("session"), "version": new, "current": cur,
               "cluster": d["cluster"], "lf_accepted": bool(res.get("accepted")),
               "lf_passed": res.get("passed"), "oracle_accepted": bool(og.accepted),
               "oracle_passed": og.passed, "agree": bool(og.accepted) == bool(res.get("accepted")),
               "oracle": og.ledger_data()}
        orc.ledger_append(run, "note", {"event": "gate_shadow", **rec}, session=r.get("session"), version=new)
        out.append(rec)
    return out


def purity(run, be):
    from toyrepair.miner import log_purity
    cell = be.cell(run)
    p = Path(run.dir) / "purity.csv"
    if p.exists():
        p.unlink()
    for f in sorted((Path(run.dir) / "atlas").glob("v*.pkl")):
        from toyrepair.orchestrate import load_pickle
        try:
            log_purity(cell, load_pickle(str(f)), str(p))
        except Exception as e:  # noqa: BLE001
            print("purity failed", f, e)


def eval_run(rid):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    import phase4
    from toyrepair import orchestrate as orc
    from toyrepair.cli import RealBackend
    run = orc.load_run(rid)
    if not run.stop:
        raise SystemExit(f"{rid} has not stopped; eval runs outside the loop only")
    be = RealBackend()
    sh = shadow_gates(run, be)
    purity(run, be)
    phase4.OUT = DRV / "_p4tmp"
    phase4.eval_run(rid)
    src = phase4.out_path(run.arm, run.setting, run.seed, run.npm)
    rec = json.loads(src.read_text())
    allsh = [(r.get("data") or {}) for r in _shadow_records(run)]
    rec["shadow"] = {"n": len(allsh), "agree": sum(x["agree"] for x in allsh),
                     "lf_accept": sum(x["lf_accepted"] for x in allsh),
                     "oracle_accept": sum(x["oracle_accepted"] for x in allsh),
                     "both_accept": sum(x["lf_accepted"] and x["oracle_accepted"] for x in allsh),
                     "per_test_agree": {t: sum(bool(x["lf_passed"].get(t)) == bool(x["oracle_passed"].get(t))
                                               for x in allsh) for t in ("target", "progress", "drift",
                                                                         "other_forgotten", "between", "unit")}}
    dst = res_dir(run.setting, run.seed, run.npm)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("ledger.jsonl", "versions.json", "run.json", "budget.json", "purity.csv"):
        if (Path(run.dir) / f).exists():
            shutil.copy(Path(run.dir) / f, dst / f)
    tmp = dst / "result.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1, default=float))
    os.replace(tmp, dst / "result.json")
    from grid_share1 import copy_patch_stack
    copy_patch_stack(run.dir, run.current, dst)
    src.unlink()
    print("wrote", dst / "result.json", "D", rec["final"]["D"], "shadow", rec["shadow"], "new shadow", len(sh))


def table():
    rows = []
    for p in sorted(OUT.glob("*/s*/n*/result.json")):
        d = json.loads(p.read_text())
        o = d.get("output") or {}
        rows.append({"setting": d["setting"], "seed": d["seed"], "npm": d["npm"], "D": d["final"]["D"],
                     "removed": d["final"]["removed"], "collateral": d["final"]["collateral"],
                     "collateral_plan": d["final"].get("collateral_plan"), "mass_off": d["final"].get("mass_off"),
                     "cost_usd": o.get("cost_usd"), "sessions": o.get("sessions"), "gate_checks": o.get("gate_checks"),
                     "accepted": len(o.get("accepted_chain") or []), "stop": o.get("stop"), "invalid": o.get("invalid"),
                     "shadow": d.get("shadow")})
    (OUT / "table.json").write_text(json.dumps(rows, indent=1, default=float))
    for r in rows:
        print(r["setting"], r["seed"], r["npm"], round(r["D"], 4), r["stop"], r["shadow"] and
              f"{r['shadow']['agree']}/{r['shadow']['n']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("caches").add_argument("--extra", action="store_true")
    p = sub.add_parser("cache")
    p.add_argument("--setting", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--npm", type=int, required=True)
    p = sub.add_parser("tick")
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--extra", action="store_true")
    p = sub.add_parser("record")
    p.add_argument("--token", required=True)
    p.add_argument("--summary", default="")
    sub.add_parser("eval").add_argument("--run", required=True)
    sub.add_parser("table")
    a = ap.parse_args(argv)
    if a.cmd == "caches":
        caches(a.extra)
    elif a.cmd == "cache":
        cache(a.setting, a.seed, a.npm)
    elif a.cmd == "tick":
        tick(a.slots, a.extra)
    elif a.cmd == "record":
        record(a.token, a.summary)
    elif a.cmd == "eval":
        eval_run(a.run)
    elif a.cmd == "table":
        table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
