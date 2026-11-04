"""Train the base (reference) flow policy for one seed and report its mode masses."""

import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from toypref import seeds
from toypref.config import DEFAULT, base_path
from toypref.mixture import assign_modes, mode_masses
from toypref.train import train_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--mode_std", type=float, default=None, help="mixture mode width (regime std=)")
    args = ap.parse_args()
    cfg = DEFAULT
    if args.mode_std is not None:
        from dataclasses import replace as _rep
        cfg = _rep(cfg, mixture=_rep(cfg.mixture, mode_std=args.mode_std, assign_radius=3 * args.mode_std))
    tcfg = cfg.base if args.steps is None else type(cfg.base)(**{**asdict(cfg.base), "steps": args.steps})
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    pol, hist = train_base(cfg.mixture, cfg.policy, tcfg, args.seed)
    path = base_path(args.seed, args.mode_std)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"ema": pol.state_dict(), "policy_cfg": asdict(cfg.policy), "train_cfg": asdict(tcfg),
                "history": hist}, path)
    pol.eval()
    Z = torch.randn(cfg.n_eval, 2, generator=seeds.gen(args.seed, seeds.EVAL_NOISE))
    m = mode_masses(cfg.mixture, assign_modes(cfg.mixture, pol.sample(Z)))
    rep = {"seed": args.seed, "mass": m.tolist(), "off": float(m[-1]), "final_loss": hist[-1]["loss"]}
    with open(os.path.join(os.path.dirname(path), "base_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
