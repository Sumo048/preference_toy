"""E3 exploration: targeted training (parameter subsets, anchors, schedule) and the privileged floor.

One process = one cell. Builds the cell exactly as scripts/cond_cell.py does, then runs a list of
methods. Each method writes explore/E3/results/<setting>/s<seed>_n<npm>/<method>.json (skip if it exists).

Method spec (string, '-'-separated key=value; defaults reproduce B1a = R_hopB_bsup_neg_AMdist):
  p=<subset>    all | out | head (out_norm+out) | b0 | b1 | b2 | fc2 (all blocks' fc2) | cond (time/c MLPs)
                | lora<r> (rank-r adapters on every Linear in inp/blocks/out, base frozen)
                | lorab<r> (rank-r adapters on blocks' fc1/fc2 only)
  a=<anchor>    dist | map | local | lmap (local+map) | none | full (unmasked anchor pool, fresh noise)
  w=<float>     anchor weight; loss = (1-w)*method + w*anchor  (0.5 = B1a)
  st=<int>      steps (5000)
  lr=<float>    peak lr (1e-4)
  r=<rule>      reroute rule (hopB); m=1 adds mined pseudo-negatives (as R_hopB_bsup_mine_neg)
  o=1           ORACLE BC: flow-matching on 20,000 pi_ref samples whose oracle label is a retained
                mode (privileged; labels from CondMixture.assign). Ignores r/m; a/w still apply.
  on=map        oracle BC with the sample's own noise (default: fresh noise = density BC)

Edits read only (x, t, c) and activations; the oracle is used only by o=1 (the privileged floor)
and by the final compute().
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("TOYPREF_RUNS", os.path.join(ROOT, "runs"))

import numpy as np
import torch
import torch.nn as nn

from toypref.config import RUNS
from toypref.cond import experiment as E
from toypref.cond import targets as TG
from toypref.cond.task import CondFlow, frozen
from toypref.flow import FlowPolicy
from toypref.train import cosine

SETTINGS = {"k4": (4, "random"), "alt": (4, "alt")}
OUT = os.path.join(ROOT, "explore", "E3", "results")
P_ORACLE = 90_000   # fresh purpose id for the oracle-BC pool (not used by toypref)

DEFAULT = {"p": "all", "a": "dist", "w": "0.5", "st": "5000", "lr": "1e-4", "r": "hopB", "m": "0",
           "o": "0", "on": "dist"}


def parse(spec: str) -> dict:
    d = dict(DEFAULT)
    if spec not in ("B1a", "default"):
        for kv in spec.split("-"):
            k, v = kv.split("=")
            assert k in d, k
            d[k] = v
    return d


# ------------------------------------------------------------------ LoRA
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.t()) @ self.B.t()

    def merged(self) -> nn.Linear:
        lin = copy.deepcopy(self.base)
        with torch.no_grad():
            lin.weight += self.scale * self.B @ self.A
        return lin


def lora_wrap(pol: CondFlow, r: int, blocks_only: bool):
    net = pol.net
    names = []
    for i, b in enumerate(net.blocks):
        b.fc1 = LoRALinear(b.fc1, r)
        b.fc2 = LoRALinear(b.fc2, r)
        names += [f"blocks.{i}.fc1", f"blocks.{i}.fc2"]
    if not blocks_only:
        net.inp = LoRALinear(net.inp, r)
        net.out = LoRALinear(net.out, r)
        names += ["inp", "out"]
    return names


def lora_merge(pol: CondFlow):
    net = pol.net
    for b in net.blocks:
        if isinstance(b.fc1, LoRALinear):
            b.fc1 = b.fc1.merged()
            b.fc2 = b.fc2.merged()
    if isinstance(net.inp, LoRALinear):
        net.inp = net.inp.merged()
    if isinstance(net.out, LoRALinear):
        net.out = net.out.merged()


def select_params(pol: CondFlow, p: str):
    net = pol.net
    for q in pol.parameters():
        q.requires_grad_(False)
    if p.startswith("lora"):
        blocks_only = p.startswith("lorab")
        r = int(p[5:] if blocks_only else p[4:])
        lora_wrap(pol, r, blocks_only)
        params = [q for n, q in pol.named_parameters() if n.endswith(".A") or n.endswith(".B")]
    else:
        mods = {"all": [net], "out": [net.out], "head": [net.out_norm, net.out],
                "b0": [net.blocks[0]], "b1": [net.blocks[1]], "b2": [net.blocks[2]],
                "fc2": [b.fc2 for b in net.blocks], "cond": [net.time_mlp, net.c_mlp],
                "inp": [net.inp]}[p]
        params = [q for m in mods for q in m.parameters()]
    for q in params:
        q.requires_grad_(True)
    return params


# ------------------------------------------------------------------ cell
class Cell:
    def __init__(self, setting, seed, npm):
        k, layout = SETTINGS[setting]
        self.setting, self.seed, self.npm = setting, seed, npm
        assert seed <= 4, "E3 uses seeds 0-4 only"
        sd = torch.load(os.path.join(RUNS, "cond_base", f"s{seed}", "policy.pt"), map_location="cpu",
                        weights_only=False)
        ref = CondFlow(**sd["kw"]); ref.load_state_dict(sd["state"]); self.ref = frozen(ref)
        pool = E.make_pool(self.ref, seed, 200_000)
        self.ed = E.editset(pool, seed, k, npm, layout)
        anchor = E.make_pool(self.ref, seed, 20_000, purpose=E.P_ANCHOR)
        self.ctx = E.Ctx(ref=self.ref, seed=seed, ed=self.ed, pool=pool, anchor=anchor)
        self.bank = E.Bank(self.ref, seed, self.ed)
        self._rows = {}
        self._oracle = None

    def rows(self, rule, mine):
        key = (rule, mine)
        if key not in self._rows:
            ctx = self.ctx
            arm = E.Arm("x", "reroute", rule=rule, support="bsup", mine=bool(mine))
            rows = TG.build(ctx.ref, E.MIX, ctx.seed, ctx.ed.uid, ctx.ed.c, ctx.ed.xi, ctx.ed.a, arm.rule,
                            arm.support, transplant=False, tx_conditions=ctx.pool.c[:4096],
                            d0=4 * E.MIX.mode_std, pu=None, forget=ctx.ed.forget)
            if mine:
                bank, keep, _ = E.get_vote(ctx)
                P = ctx.anchor
                flagged = torch.nonzero(~keep).flatten()[:E.N_MINE]
                mrows = TG.build_mined(ctx.ref, E.MIX, ctx.seed, bank, P.c[flagged], P.xi[flagged], P.a[flagged],
                                       uid_offset=50_000_000, forget=ctx.ed.forget)
                if mrows.c.shape[0]:
                    rows = TG.Rows(c=torch.cat([rows.c, mrows.c]), noise=torch.cat([rows.noise, mrows.noise]),
                                   target=torch.cat([rows.target, mrows.target]), diag=rows.diag)
            self._rows[key] = rows
        return self._rows[key]

    def oracle_rows(self, n=20_000):
        """PRIVILEGED: pi_ref samples labelled by CondMixture.assign, retained modes only."""
        if self._oracle is None:
            P = E.make_pool(self.ref, self.seed, 3 * n, purpose=P_ORACLE)
            keep = np.flatnonzero(np.isin(P.mode, self.ed.retain))[:n]
            self._oracle = (P.c[keep], P.xi[keep], P.a[keep])
        return self._oracle


def anchor_fn(cell: Cell, kind: str):
    ctx = cell.ctx
    if kind == "none":
        return None
    if kind == "full":
        P, B = ctx.anchor, ctx.batch

        def loss(pol, g):
            j = E._pick(P.c.shape[0], B, g)
            t = FlowPolicy.sample_time(B, g)
            return pol.flow_loss(P.a[j], torch.randn(P.a[j].shape, generator=g), t, P.c[j]).mean()
        return loss
    return E.masked_anchor_loss(ctx, {"lmap": "local+map"}.get(kind, kind))


def run_method(cell: Cell, spec: str):
    d = parse(spec)
    ctx = cell.ctx
    B = ctx.batch
    t0 = time.time()
    if d["o"] == "1":
        oc, oxi, oa = cell.oracle_rows()

        def method(pol, g):
            j = E._pick(oc.shape[0], B, g)
            t = FlowPolicy.sample_time(B, g)
            noise = oxi[j] if d["on"] == "map" else torch.randn(oa[j].shape, generator=g)
            return pol.flow_loss(oa[j], noise, t, oc[j]).mean()
    else:
        rows = cell.rows(d["r"], int(d["m"]))
        method = E.step_fn(E.Arm("x", "reroute", rule=d["r"], support="bsup"), ctx, rows)
    al = anchor_fn(cell, d["a"])
    w = float(d["w"])
    if al is None:
        fn = method
    elif w == 0.5:
        def fn(pol, g):
            return 0.5 * (method(pol, g) + al(pol, g))
    else:
        def fn(pol, g):
            return (1 - w) * method(pol, g) + w * al(pol, g)

    steps, lr0 = int(d["st"]), float(d["lr"])
    torch.manual_seed(ctx.seed)
    pol = copy.deepcopy(ctx.ref)
    pol.train()
    params = select_params(pol, d["p"])
    n_train = sum(q.numel() for q in params)
    opt = torch.optim.AdamW(params, lr=lr0, betas=(0.9, 0.95), weight_decay=1e-10)
    g = E.gen(ctx.seed, E.P_TRAIN)
    probe = E.make_probe(cell.bank)
    hist = []
    for step in range(steps):
        lr = cosine(step, steps, min(500, steps // 10), lr0, 0.1 * lr0)
        for pg in opt.param_groups:
            pg["lr"] = lr
        loss = fn(pol, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 2500 == 0 and step + 1 < steps:
            pol.eval(); hist.append({"step": step + 1, **probe(pol)}); pol.train()
    lora_merge(pol)
    pol.eval()
    for q in pol.parameters():
        q.requires_grad_(False)
    # edit size
    ref_sd, new_sd = ctx.ref.state_dict(), pol.state_dict()
    dn = sum(float(((new_sd[k] - ref_sd[k]) ** 2).sum()) for k in ref_sd)
    nch = sum(int((new_sd[k] != ref_sd[k]).sum()) for k in ref_sd)
    ntot = sum(v.numel() for v in ref_sd.values())
    met = E.compute(cell.bank, pol)
    return {"method": spec, **d, "setting": cell.setting, "seed": cell.seed, "npm": cell.npm,
            "forget": cell.ed.forget, "D": met["tv_to_ideal"], "Removed": 1 - met["neg_mass_ratio"],
            "Collateral": met["retain_tv"], "mass_off": met["mass_off"], "n_trainable": n_train,
            "frac_changed": nch / ntot, "frob": math.sqrt(dn), "train_s": round(time.time() - t0, 1),
            "hist": hist, **{k: v for k, v in met.items() if k not in ("tv_to_ideal",)}}


def b0_record(cell: Cell):
    met = E.compute(cell.bank, cell.ref)
    return {"method": "B0", "setting": cell.setting, "seed": cell.seed, "npm": cell.npm, "forget": cell.ed.forget,
            "D": met["tv_to_ideal"], "Removed": 1 - met["neg_mass_ratio"], "Collateral": met["retain_tv"],
            "mass_off": met["mass_off"], "frac_changed": 0.0, "frob": 0.0, **met}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--npm", type=int, required=True)
    ap.add_argument("--methods", required=True, help="comma-separated specs; 'B0' and 'B1a' allowed")
    args = ap.parse_args()
    torch.set_num_threads(1)
    od = os.path.join(OUT, args.setting, f"s{args.seed}_n{args.npm}")
    os.makedirs(od, exist_ok=True)
    todo = [m for m in args.methods.split(",") if not os.path.exists(os.path.join(od, f"{m}.json"))]
    if not todo:
        return
    cell = Cell(args.setting, args.seed, args.npm)
    for m in todo:
        rec = b0_record(cell) if m == "B0" else run_method(cell, m)
        tmp = os.path.join(od, f".{m}.tmp")
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, os.path.join(od, f"{m}.json"))
        print(f"[{args.setting} s{args.seed} n{args.npm}] {m:40s} D {rec['D']:.3f} Rem {rec['Removed']:.3f} "
              f"Col {rec['Collateral']:.3f} off {rec['mass_off']:.3f} ({rec.get('train_s', 0)}s)", flush=True)


if __name__ == "__main__":
    main()
