#!/usr/bin/env python3
"""Export the final patch stack (patch JSON + NPZ arrays, no caches) of every Phase 4 run, and check it rebuilds.

    python scripts/export_patches.py            # -> results/phase4/patches/<run_id>/{stack.json, patches/*}

Final stack: A3 runs from runs/repair/<run>/versions.json (current version's patch list); code arms (B4, D1, R0)
by walking each patch's parent_version back from the run's final version (results/phase4/.../<arm>_p4.json).
Each export is rebuilt with toyrepair.patches.rebuild (per-patch sha256 check) before it is kept.
"""
import json
import os
import shutil
import sys
import types
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
import toyrepair  # noqa: E402,F401

RUNS = Path(os.environ["TOYPREF_RUNS"]) / "repair"
OUT = R / "results" / "phase4" / "patches"


def code_stack(run_dir: Path, final: str):
    by_version = {}
    for f in (run_dir / "patches").glob("*.json"):
        p = json.loads(f.read_text())
        by_version[p["version"]] = p
    stack, v = [], final
    while v and v != "v0":
        p = by_version[v]
        stack.append(p["id"])
        v = p["parent_version"]
    return stack[::-1]


def runs():
    for res in sorted((R / "results" / "phase4").glob("k4/s*/n*/*_p4.json")):
        d = json.loads(res.read_text())
        run_id, arm = d["run_id"], d["arm"]
        run_dir = RUNS / run_id
        if arm.startswith("A"):
            r = json.loads((run_dir / "run.json").read_text())
            vs = json.loads((run_dir / "versions.json").read_text())
            final = r["current"]
            stack = list(vs[final]["patches"]) if final != "v0" else []
        else:
            final = d["output"]["version"]
            stack = code_stack(run_dir, final)
        yield run_id, arm, d, run_dir, final, stack


def main():
    from toyrepair.cli import cached_cell
    from toyrepair.patches import rebuild, state_sha256
    for run_id, arm, d, run_dir, final, stack in runs():
        dst = OUT / run_id
        if dst.exists():
            shutil.rmtree(dst)
        (dst / "patches").mkdir(parents=True)
        for pid in stack:
            for ext in (".json", ".npz"):
                shutil.copy2(run_dir / "patches" / f"{pid}{ext}", dst / "patches" / f"{pid}{ext}")
        cell = cached_cell(types.SimpleNamespace(cell_args=("cond", d["setting"], d["seed"], d["npm"])))
        v = rebuild(cell, str(dst), stack, final)
        meta = {"run_id": run_id, "arm": arm, "cell": ["cond", d["setting"], d["seed"], d["npm"]],
                "version": final, "patches": stack, "state_sha256": state_sha256(v),
                "final_D": d["final"]["D"]}
        if arm == "R0" and stack == [] and final != "v0":
            meta["note"] = "R0 versions are sampling-time controllers (runs/.../r0/), not weight patches"
        (dst / "stack.json").write_text(json.dumps(meta, indent=1))
        print(run_id, arm, final, len(stack), "patches, rebuild ok")


if __name__ == "__main__":
    main()
