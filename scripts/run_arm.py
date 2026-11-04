#!/usr/bin/env python3
"""Run one code arm on one cell, then its final evaluation (W1-C; docs/PLAN.md 9.2).

    python scripts/run_arm.py --arm B1a --setting k4 --seed 0 --npm 4 [--grid cond]
        [--max_rounds 1] [--measure_gate] [--out results/arms] [--force]

Arms: B0, B1a, B1b, B4, B5, R0, D1 (``toyrepair.arms``). The run directory is
``$TOYPREF_RUNS/repair/<run id>/`` (ledger, patches, scratch); the run id defaults to
``<arm>_<grid>_<setting>_s<seed>_n<npm>`` (+ ``--tag``). The result JSON (arm output + final
evaluation + timing) goes to ``<out>/<setting>/s<seed>/n<npm>/<arm>.json``; skip-if-exists.

Final evaluation: ``toyrepair.eval_final.evaluate`` (W1-A2). Until that module is merged the
script falls back to toypref's own ``Bank``/``compute`` (the method-map metrics, same fixed eval
bank), reached through ``cell.toypref_ctx()``.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import toyrepair  # noqa: E402,F401  (sets TOYPREF_RUNS before toypref is imported)
import torch  # noqa: E402

from toyrepair.arms import CODE_ARMS, get  # noqa: E402
from toyrepair.arms.base import _json_safe  # noqa: E402


class _Sampler:
    """CondFlow.sample over an arbitrary velocity net (same op order and batch split)."""

    def __init__(self, net):
        self.net = net

    @torch.no_grad()
    def sample(self, xi, c, batch=65536):
        out, dt = [], 1.0 / 20
        for i in range(0, xi.shape[0], batch):
            x = xi[i:i + batch].clone().float()
            cc = c[i:i + batch].float()
            for k in range(20):
                t = torch.full((x.shape[0],), 1.0 - k * dt)
                x = x - dt * self.net(x, t, cc)
            out.append(x)
        return torch.cat(out)


def final_eval(cell, out, arm, target_uids=None):
    try:
        from toyrepair import eval_final
        r = eval_final.evaluate(cell, out.final, arm, target_clusters=[target_uids] if target_uids else None)
        return {"source": "eval_final", "D": r.D, "removed": r.removed, "collateral": r.collateral,
                "mass_off": r.mass_off, "toypref": r.toypref, "target_removed": r.target_removed,
                "target_modes": r.target_modes, "collateral_plan": r.collateral_plan,
                "final_pool": r.final_pool}
    except NotImplementedError:
        from toypref.cond import experiment as E
        ed = cell.toypref_ctx(steps=0).ed
        bank = E.Bank(cell.ref, cell.seed, ed)
        met = E.compute(bank, _Sampler(out.final.net))
        return {"source": "toypref_compute_adapter", "D": met["tv_to_ideal"], "removed": 1.0 - met["neg_mass_ratio"],
                "collateral": met["retain_tv"], "mass_off": met["mass_off"], "toypref": met}


def cached_cell_for(grid, setting, seed, npm):
    """``load_cell`` through the shared cache ``runs/cells/<key>/cell_cache.pt`` (W2a)."""
    import types
    from toyrepair.cli import cached_cell
    return cached_cell(types.SimpleNamespace(cell_args=(grid, setting, seed, npm)))


def make_run(run_id: str, arm: str, grid: str, setting: str, seed: int, npm: int):
    from toyrepair.orchestrate import Run
    d = os.path.join(os.environ["TOYPREF_RUNS"], "repair", run_id)
    os.makedirs(d, exist_ok=True)
    return Run(id=run_id, dir=d, arm=arm, grid=grid, setting=setting, seed=seed, npm=npm)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(CODE_ARMS))
    ap.add_argument("--setting", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--npm", type=int, required=True)
    ap.add_argument("--grid", default="cond")
    ap.add_argument("--max_rounds", type=int, default=None, help="iterative arms: 1 = pilot (one repair)")
    ap.add_argument("--cluster", type=int, default=None, help="iterative arms: repair only this cluster (pilot target)")
    ap.add_argument("--targets", default=None, help="pilot target JSON (scripts/pilot.py targets): "
                    "repair its cluster and evaluate the removed share of its modes")
    ap.add_argument("--measure_gate", action="store_true", help="B1a/B1b: log one global gate check")
    ap.add_argument("--out", default="results/arms")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    out_path = REPO / a.out / a.setting / f"s{a.seed}" / f"n{a.npm}" / f"{a.arm}{a.tag}.json"
    if out_path.exists() and not a.force:
        print(f"exists: {out_path}")
        return 0
    t0, c0 = time.time(), time.process_time()
    tgt = json.load(open(a.targets)) if a.targets else None
    if tgt is not None and a.cluster is None:
        a.cluster = int(tgt["cluster"])
    cell = cached_cell_for(a.grid, a.setting, a.seed, a.npm)
    run_id = f"{a.arm}{a.tag}_{cell.key}"
    run = make_run(run_id, a.arm, a.grid, a.setting, a.seed, a.npm)
    kw = {}
    if a.max_rounds is not None and a.arm in ("B4", "R0", "D1"):
        kw["max_rounds"] = a.max_rounds
    if a.cluster is not None and a.arm in ("B4", "R0", "D1"):
        kw["target_cluster"] = a.cluster
    if a.measure_gate and a.arm in ("B1a", "B1b"):
        kw["measure_gate"] = True
    out = get(a.arm)(cell, run, **kw)
    t_arm = time.time() - t0
    ev = final_eval(cell, out, a.arm, target_uids=tgt["uids"] if tgt else None)
    rec = {"arm": a.arm, "cell": cell.key, "grid": a.grid, "setting": a.setting, "seed": a.seed, "npm": a.npm,
           "run_id": run_id, "run_dir": run.dir, "output": out.to_json(), "final": ev,
           "wall_s_arm": round(t_arm, 1), "wall_s_total": round(time.time() - t0, 1),
           "cpu_s": round(time.process_time() - c0, 1),
           "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
           "threads": torch.get_num_threads()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.tmp")
    tmp.write_text(json.dumps(_json_safe(rec), indent=1))
    os.replace(tmp, out_path)
    print(f"[{a.arm} {cell.key}] D {ev['D']:.4f} removed {ev['removed']:.3f} collateral {ev['collateral']:.3f} "
          f"stop {out.stop} gate_checks {out.gate_checks} ({rec['wall_s_total']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
