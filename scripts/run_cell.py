"""Run every arm (or a subset) on one sweep cell (seed, n_forget, n_neg).

Writes one JSON line per arm to ``$TOYPREF_RUNS/cells/<cell>/<arm>.json`` and
skips arms whose file exists, so a cell can be resumed or split across jobs.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from toypref import seeds
from toypref.arms import ARMS, ArmContext, run_arm
from toypref.config import DEFAULT, RUNS, base_path, cell_name
from toypref.editset import build_editset, make_pool
from toypref.flow import load_policy
from toypref.metrics import EvalBank, compute, make_probe
from toypref.mixture import sample_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_forget", type=int, required=True)
    ap.add_argument("--n_neg", type=int, required=True)
    ap.add_argument("--arms", default="all", help="comma list or 'all'")
    ap.add_argument("--steps", type=int, default=None, help="override fine-tune steps (smoke tests)")
    ap.add_argument("--lr", type=float, default=None, help="override fine-tune peak LR (pilot only)")
    ap.add_argument("--n_pos", type=int, default=None,
                    help="|T_p| (default config value 512); != 512 writes to $RUNS/cells_npos<N>")
    ap.add_argument("--out", default=None, help="cells root (default by n_pos, see --n_pos)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    n_pos = DEFAULT.n_pos if args.n_pos is None else args.n_pos
    if args.out is None:
        args.out = os.path.join(RUNS, "cells" if n_pos == DEFAULT.n_pos else f"cells_npos{n_pos}")

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    cfg = DEFAULT
    tcfg = cfg.edit if args.steps is None else replace(cfg.edit, steps=args.steps)
    if args.lr is not None:
        tcfg = replace(tcfg, lr=args.lr)
    mix = cfg.mixture
    s = args.seed
    ref = load_policy(base_path(s))

    pool = make_pool(ref, mix, s, cfg.pool_size)
    es = build_editset(pool, mix, s, args.n_forget, args.n_neg, n_pos)

    # oracle data: pi_ref's own samples in retained modes, its own stream
    opool = make_pool(ref, mix, s, cfg.n_oracle * 2, purpose=seeds.ORACLE_POOL)
    oracle_a = opool.a[np.isin(opool.mode, es.retain)][:cfg.n_oracle]

    Z = torch.randn(cfg.n_eval, mix.dim, generator=seeds.gen(s, seeds.EVAL_NOISE))
    rng_t = seeds.np_rng(s, seeds.TEST_SAMPLES)
    test_x, test_m = [], []
    for m in range(mix.n_modes):
        x, mm = sample_true(mix, cfg.n_test_per_mode, rng_t, modes=[m])
        test_x.append(x)
        test_m.append(mm)
    test_x, test_m = np.concatenate(test_x), np.concatenate(test_m)
    ideal_x, _ = sample_true(mix, cfg.n_ideal, seeds.np_rng(s, seeds.TEST_SAMPLES + 1), modes=es.retain)

    cell = cell_name(s, args.n_forget, args.n_neg)
    out_dir = os.path.join(args.out, cell)
    os.makedirs(out_dir, exist_ok=True)

    names = list(ARMS) if args.arms == "all" else args.arms.split(",")
    todo = [n for n in names if args.force or not os.path.exists(os.path.join(out_dir, f"{n}.json"))]
    print(f"[{cell}] forget={es.forget} |T_n|={len(es.neg_idx)} |T_p|={len(es.pos_idx)} "
          f"{len(todo)}/{len(names)} arms to run", flush=True)
    if not todo:
        return

    t0 = time.time()
    bank = EvalBank(ref, mix, Z, test_x, test_m, ideal_x, es.forget, es.a_neg, es.xi_neg, es.a_pos, s)
    print(f"[{cell}] eval bank {time.time() - t0:.1f}s", flush=True)

    probe = make_probe(bank)
    ctx = ArmContext(
        ref=ref, seed=s, xi_neg=es.xi_neg, a_neg=es.a_neg, neg_pool_idx=es.neg_idx,
        a_pos=es.a_pos, forget=es.forget, mix_cfg=mix, tcfg=tcfg, oracle_a=oracle_a,
        online_pool_fn=lambda pol: make_pool(pol, mix, s, cfg.online_pool, purpose=seeds.ONLINE_POOL),
        n_neg_per_mode=args.n_neg,
    )
    for name in todo:
        spec = ARMS[name]
        t1 = time.time()
        sampler, pol, info = run_arm(spec, ctx, probe=probe)
        t_train = time.time() - t1
        met, extra = compute(bank, sampler, pol)
        with torch.no_grad():
            np.save(os.path.join(out_dir, f"{name}_samples.npy"),
                    sampler.decode(Z[:2000]).numpy().astype(np.float16))
        rec = {
            "cell": cell, "seed": s, "n_forget": args.n_forget, "n_neg": args.n_neg,
            "n_neg_total": int(len(es.neg_idx)), "n_pos": int(len(es.pos_idx)),
            "forget": es.forget, "arm": name, "kind": spec.kind, "family": spec.family,
            "mix": spec.mix if spec.kind not in ("frozen", "bc", "rerank") else "",
            "negonly": spec.negonly, "negative_control": spec.negative_control,
            "extra_label_budget": spec.extra_label_budget, "privileged": spec.privileged,
            "train_s": round(t_train, 2), "eval_s": round(time.time() - t1 - t_train, 2),
            "steps": tcfg.steps, "config": cfg.to_dict() if name == todo[0] else None,
            **{k: v for k, v in info.items() if not k.startswith("train_history")},
            **met,
            "extra": extra,
            "train_history": info.get("train_history"),
            "train_history_r2": info.get("train_history_r2"),
        }
        tmp = os.path.join(out_dir, f".{name}.json.tmp")
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, os.path.join(out_dir, f"{name}.json"))
        print(f"[{cell}] {name:34s} neg_ratio {met['neg_mass_ratio']:.3f}  retain_tv {met['retain_tv']:.3f}  "
              f"off {met['mass_off']:.3f}  train {t_train:.1f}s eval {rec['eval_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
