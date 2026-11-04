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


def step(token, summary, run=None):
    """record (if token) then next; on DONE start the final evaluation in the background."""
    from toyrepair import orchestrate as orc
    if not token and run:
        token = orc.load_run(run).pending or ""
    if token:
        r = orc.record(token, summary)
        run = r.get("run", run)
        brief = {k: r.get(k) for k in ("session", "role", "cluster", "fallback", "accepted", "calls",
                                        "submissions", "audit_hits", "error") if r.get(k) is not None}
        brief["cost"] = round(float(r.get("cost_usd") or 0), 3)
        print("REC", json.dumps(brief))
    with orc.run_lock(orc.run_dir(run)):
        act = orc.next_action(orc.load_run(run))
    print(run, act)
    if act.startswith("DONE"):
        log = open(OUT / "eval_logs.txt", "a")
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "eval", "--run", run],
                         stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(REPO),
                         env={**os.environ, "OMP_NUM_THREADS": "1"})


def _handback(path):
    """Last SubagentHandback message in a subagent transcript, or None."""
    last = None
    try:
        for line in open(path):
            if "SubagentHandback" not in line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = d.get("message") or {}
            for c in (m.get("content") if isinstance(m.get("content"), list) else []):
                if c.get("type") == "tool_use" and c.get("name") == "SubagentHandback":
                    last = (c.get("input") or {}).get("message")
    except OSError:
        return None
    return last


def auto(max_active=8, rep=False, start=True):
    """Advance every run whose pending subagent has handed back; start runs up to max_active.
    Prints LAUNCH lines (subagent to invoke), WAIT lines (pending, with idle minutes) and DONE lines."""
    import time
    from toyrepair import orchestrate as orc
    from toyrepair import transcripts as tr
    root = Path(os.environ["TOYPREF_RUNS"]) / "repair"
    order = llm_runs(False) + (llm_runs(True) if rep else [])
    active = 0
    fresh = []
    for rid, arm, armdir, s, sd, n in order:
        rp = root / rid / "run.json"
        if not rp.exists():
            continue
        r = orc.load_run(rid)
        if r.stop or r.invalid:
            if not res_path(s, sd, n, armdir).exists():
                print("NOEVAL", rid, r.stop or "invalid")
            continue
        b = orc.load_budget(r)
        if not r.pending and b.get("sessions", 0) == 0 and r.iteration == 0 and r.n_first_clusters is None:
            fresh.append(rid)
            continue
        active += 1
        act = None
        if r.pending:
            path = tr.find_subagent_for_token(r.pending)
            if path is None:
                with orc.run_lock(orc.run_dir(rid)):
                    act = orc.next_action(orc.load_run(rid))     # PENDING agent token
            else:
                hb = _handback(path)
                if hb is None:
                    idle = (time.time() - os.path.getmtime(path)) / 60
                    print(f"WAIT {rid} agent={Path(path).stem.replace('agent-', '')} idle={idle:.0f}m")
                    continue
                rec = orc.record(r.pending, hb)
                print("REC", rid, json.dumps({k: rec.get(k) for k in ("session", "role", "cluster", "accepted",
                                                                          "audit_hits") if rec.get(k) is not None}))
        if act is None:
            with orc.run_lock(orc.run_dir(rid)):
                act = orc.next_action(orc.load_run(rid))
        _emit(rid, act)
        if act.startswith("DONE"):
            active -= 1
    while start and active < max_active and fresh:
        rid = fresh.pop(0)
        with orc.run_lock(orc.run_dir(rid)):
            act = orc.next_action(orc.load_run(rid))
        _emit(rid, act)
        active += 1
    print("ACTIVE", active)


def _emit(rid, act):
    w = act.split()
    if w[0] in ("TRIAGE", "SESSION", "PENDING"):
        print("LAUNCH", w[1], rid, w[2])
    elif w[0] == "DONE":
        print("DONE", rid, act)
        log = open(OUT / "eval_logs.txt", "a")
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "eval", "--run", rid],
                         stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(REPO),
                         env={**os.environ, "OMP_NUM_THREADS": "1"})
    else:
        print("??", rid, act)


def eval_run(rid):
    import phase4
    from toyrepair import orchestrate as orc
    run = orc.load_run(rid)
    armdir = "A3_rep1" if rid.endswith("_A3_rep1") else run.arm
    phase4.out_path = lambda arm, s, sd, n: res_path(s, sd, n, armdir)
    phase4.eval_run(rid)
    export_patches(rid)


def export_patches(rid):
    """Copy the final version's patch stack (p*.json + p*.npz only; no caches) of a finished run to
    results/grid/<setting>/s<seed>/n<npm>/<arm>/patches/, with stack.json (version chain, patch order)."""
    import shutil
    from toyrepair import orchestrate as orc
    run = orc.load_run(rid)
    if not (run.stop or run.invalid):
        return None
    armdir = "A3_rep1" if rid.endswith("_A3_rep1") else run.arm
    dst = res_path(run.setting, run.seed, run.npm, armdir).parent / "patches"
    dst.mkdir(parents=True, exist_ok=True)
    vs = orc.versions(run)
    stack = list(vs[run.current]["patches"]) if run.current in vs else []
    src = Path(run.dir) / "patches"
    for pid in stack:
        for ext in (".json", ".npz"):
            f = src / f"{pid}{ext}"
            if f.exists():
                shutil.copy2(f, dst / f.name)
    chain, v = [], run.current
    while v and v != "v0":
        chain.append({"version": v, "parent": vs[v]["parent"], "call": vs[v].get("call"),
                      "session": vs[v].get("session"), "patches": vs[v]["patches"]})
        v = vs[v]["parent"]
    (dst / "stack.json").write_text(json.dumps({"run_id": rid, "version": run.current, "stop": run.stop,
                                                "invalid": run.invalid, "patches": stack,
                                                "chain": list(reversed(chain))}, indent=1))
    return dst, len(stack)


def export_all():
    root = Path(os.environ["TOYPREF_RUNS"]) / "repair"
    for rid, *_ in llm_runs(False) + llm_runs(True):
        if (root / rid / "run.json").exists():
            r = export_patches(rid)
            if r:
                print("PATCHES", rid, r[1], r[0].relative_to(REPO))


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
    """W1-C seed 5-9 baselines (repo code): results/baselines/<setting>/s<seed>_k<n_forget>_n<npm>/.
    B0 = B0_frozen.json, B1a = R_hopB_bsup_neg_AMdist.json, B1b = the other config file (= B1a when absent)."""
    kf = {"k1": 1, "k2": 2}.get(s, 4)
    d = REPO / "results" / "baselines" / s / f"s{sd}_k{kf}_n{n}"
    if not d.is_dir():
        return None
    if arm == "B0":
        p = d / "B0_frozen.json"
    elif arm == "B1a":
        p = d / "R_hopB_bsup_neg_AMdist.json"
    else:
        others = sorted(x for x in d.glob("*.json") if x.name not in ("B0_frozen.json", "R_hopB_bsup_neg_AMdist.json"))
        p = others[0] if others else d / "R_hopB_bsup_neg_AMdist.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    m = j.get("metrics", j)
    return {"D": m["tv_to_ideal"], "removed": 1 - m["neg_mass_ratio"], "collateral": m["retain_tv"]}


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
    sub.add_parser("patches")
    au = sub.add_parser("auto")
    au.add_argument("--max", type=int, default=8); au.add_argument("--rep", action="store_true")
    au.add_argument("--nostart", action="store_true")
    st = sub.add_parser("step")
    st.add_argument("--token", default=""); st.add_argument("--summary", default=""); st.add_argument("--run")
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
    elif a.cmd == "patches":
        export_all()
    elif a.cmd == "auto":
        auto(a.max, a.rep, not a.nostart)
    elif a.cmd == "step":
        step(a.token, a.summary, a.run)
    return 0


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(main())
