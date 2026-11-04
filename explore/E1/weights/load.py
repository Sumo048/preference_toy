"""Load an E1 gated graft (method K10c) onto pi_ref and decode with it.

Self-contained: needs only torch, numpy and toypref's CondFlow (to load pi_ref). It does not import
explore/E1/graft.py or common.py.

Files per cell (k4_s0_n4, alt_s0_n4):
  graft_K10c_<cell>.pt    list of 2 dicts (one per recursive round), each with tensors
                            det_w (146, P), det_b (P,)  per-track linear detectors (P = 4 bad tracks)
                            mu_d, sd_d (146,)           detector input standardisation
                            w_in (128, 146), b_in (128,), w_out (2, 128)   the gated units
                            mu_f, sd_f (146,)           unit input standardisation
  graft_K10c_<cell>.json  config, cell, and the recorded final evaluation (D, Removed, Collateral, off)

The graft (both rounds summed) adds to the velocity of pi_ref's net, at every Euler step k:
  phi    = [SiLU(blocks[2].fc1(...)) (128), x_k (2), c_feat(c) (16)]          (146,)
  gate   = any_p((phi - mu_d) / sd_d @ det_w[:, p] + det_b[p] > 0)  AND  t_k > 1 - (k_gate - 0.5) / 20
  delta  = gate * w_out @ SiLU(w_in @ ((phi - mu_f) / sd_f) + b_in)
  v      = v_ref(x_k, t_k, c) + delta
The gate is hard: with it off the decode is bit-identical to pi_ref.

Usage:
  from load import load_graft_policy
  pol = load_graft_policy("k4_s0_n4")          # pol.sample(xi (N,2), c (N,)) -> actions (N,2)
  python explore/E1/weights/load.py            # checks both cells against their recorded D
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CELLS = ("k4_s0_n4", "alt_s0_n4")
K_STEPS = 20


def load_ref(seed: int):
    """pi_ref of a conditional seed, from ckpt/ (read only)."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from toypref.cond.task import CondFlow, frozen
    sd = torch.load(os.path.join(ROOT, "ckpt", "cond_base", f"s{seed}", "policy.pt"),
                    map_location="cpu", weights_only=False)
    ref = CondFlow(**sd["kw"])
    ref.load_state_dict(sd["state"])
    return frozen(ref)


def load_graft(cell: str):
    """(list of per-round tensor dicts, metadata dict) for one cell."""
    rounds = torch.load(os.path.join(HERE, f"graft_K10c_{cell}.pt"), map_location="cpu", weights_only=True)
    meta = json.load(open(os.path.join(HERE, f"graft_K10c_{cell}.json")))
    return rounds, meta


class GraftedPolicy:
    """pi_ref plus the gated paths beside its `out` layer. Exposes .sample(xi, c) like CondFlow."""

    def __init__(self, ref, rounds, k_gate: int):
        self.ref, self.net, self.rounds = ref, ref.net, rounds
        self.t_min = 1.0 - (k_gate - 0.5) / K_STEPS

    def velocity(self, x, t, c, return_gate=False):
        net = self.net
        cf = net.c_feat(c)
        cond = torch.cat([net.time_mlp(net.time_embed(t)), net.c_mlp(net.c_feat(c))], -1)
        h = net.inp(x)
        for b in net.blocks:
            z = b.act(b.fc1(torch.cat([b.norm(h), cond], -1)))
            h = h + b.fc2(z)
        v = net.out(F.silu(net.out_norm(h)))
        phi = torch.cat([z, x, cf], -1)                 # z = blocks[2] SiLU hidden (last block)
        fired = torch.zeros(x.shape[0], dtype=torch.bool)
        delta = torch.zeros_like(v)
        for r in self.rounds:
            on = (((phi - r["mu_d"]) / r["sd_d"]) @ r["det_w"] + r["det_b"] > 0).any(1) & (t > self.t_min)
            u = F.silu(((phi - r["mu_f"]) / r["sd_f"]) @ r["w_in"].T + r["b_in"]) @ r["w_out"].T
            delta = delta + u * on.float().unsqueeze(-1)
            fired |= on
        v = v + delta
        return (v, fired) if return_gate else v

    @torch.no_grad()
    def sample(self, xi, c, batch=32768):
        out = []
        for i in range(0, xi.shape[0], batch):
            x, cc = xi[i:i + batch].clone().float(), c[i:i + batch].float()
            for k in range(K_STEPS):
                t = torch.full((x.shape[0],), 1.0 - k / K_STEPS)
                x = x - (1.0 / K_STEPS) * self.velocity(x, t, cc)
            out.append(x)
        return torch.cat(out)


def load_graft_policy(cell: str) -> GraftedPolicy:
    rounds, meta = load_graft(cell)
    return GraftedPolicy(load_ref(meta["seed"]), rounds, meta["cfg"]["k_gate"])


if __name__ == "__main__":
    # Check: re-evaluate each graft with toypref's own metrics and compare with the recorded D.
    sys.path.insert(0, os.path.join(ROOT, "src"))
    os.environ.setdefault("TOYPREF_RUNS", os.path.join(ROOT, "runs"))
    from toypref.cond import experiment as E
    for cell in CELLS:
        pol = load_graft_policy(cell)
        meta = load_graft(cell)[1]
        s = meta["seed"]
        pool = E.make_pool(pol.ref, s, 200_000)
        ed = E.editset(pool, s, 4, meta["npm"], meta["layout"])
        m = E.compute(E.Bank(pol.ref, s, ed), pol)
        D, rec = m["tv_to_ideal"], meta["eval"]["D"]
        print(f"{cell}: D {D:.4f} (recorded {rec:.4f}) Removed {1 - m['neg_mass_ratio']:.3f} "
              f"Collateral {m['retain_tv']:.3f} {'OK' if abs(D - rec) < 1e-6 else 'MISMATCH'}")
