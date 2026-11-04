"""TRKNG runner: build no-grid masks (builders.py) and feed each to the frozen B6 training step.

  python explore/TRKNG/trkng.py masks --setting k4 --seed 0 --npm 4 [--stage dev|confirm]
      Builds every builder config at every decode budget N on two independent fit draws, scores each
      against the oracle (evaluation only), computes the label-free checks, and writes
      results/<setting>/s<seed>_n<npm>/masks.json and masks.npz (pool flags/keep, draw A).
  python explore/TRKNG/trkng.py train --setting k4 --seed 0 --npm 4 --mask TA-N60000
      Trains B6's step (e3 spec a=track60k-w=1-st=20000: flow-matching BC on the kept samples of the
      60,000-sample pool, purpose 90,001, fresh noise, all weights, 20,000 steps, AdamW 1e-4 cosine)
      on the mask's kept set, then writes <mask>.json with D, Removed, Collateral.
  python explore/TRKNG/trkng.py train ... --mask B6      (B6 itself through this code path: reproduction check)

The cell (pi_ref, the 200k logged pool, the negatives, the eval bank) is built exactly as explore/E4/e4.py
builds it (e3.Cell without the seed assert, 7 settings).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "explore", "E3"))
sys.path.insert(0, HERE)
import e3  # noqa: E402  (puts src/ on sys.path)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from toypref.config import RUNS  # noqa: E402
from toypref.cond import experiment as E  # noqa: E402
from toypref.cond.task import CondFlow, frozen  # noqa: E402
from toypref.flow import FlowPolicy  # noqa: E402
from toypref.train import cosine  # noqa: E402

import builders as BL  # noqa: E402

SETTINGS = {"k1": (1, "random"), "k2": (2, "random"), "k4": (4, "random"),
            "contig": (4, "contig"), "alt": (4, "alt"), "pairs": (4, "pairs"), "b3p1": (4, "b3p1")}
e3.SETTINGS = SETTINGS
OUT = os.path.join(HERE, "results")
N_POOL, POOL_PURPOSE, DRAWB_PURPOSE = 60_000, 90_001, 90_002
STEPS, LR = 20_000, 1e-4
NS = (2_000, 8_000, 60_000)

# development configs (stage 1); the prereg freezes one per builder and one N
# f = activation steps used as features: the spec's (TA step 0; TB/TC steps 0,4,8) and step 19 (last
# Euler step). Development found the spec steps do not separate behaviours (noise-space preimages of
# different modes touch; 13% of kNN edges cross modes at step 0 vs 0.3% at step 19, k4 s0).
CONFIGS = {
    "TA": [{"f": "s0"}, {"f": "s19"}],
    "TB": [{"f": "s048", "mcs_frac": 0.01, "min_samples": 10},
           {"f": "s19", "mcs_frac": 0.01, "min_samples": 10},
           {"f": "s19", "mcs_frac": 0.005, "min_samples": 5}],
    "TC": [{"f": "s048", "knn": 10, "tau_mult": 1.5},
           {"f": "s19", "knn": 10, "tau_mult": 1.0}, {"f": "s19", "knn": 10, "tau_mult": 1.5},
           {"f": "s19", "knn": 10, "tau_mult": 2.0}],
}
FEATS = {"s0": (0,), "s048": (0, 4, 8), "s19": (19,)}
ALL_STEPS = (0, 4, 8, 19)
FROZEN_FILE = os.path.join(HERE, "frozen.json")


def cfg_name(b, cfg, N):
    s = "-".join(f"{k}={v}" for k, v in cfg.items())
    return f"{b}{'-' + s if s else ''}-N{N}"


class Cell(e3.Cell):
    def __init__(self, setting, seed, npm):  # = e4._init (e3.Cell.__init__ minus the seed assert)
        k, layout = SETTINGS[setting]
        self.setting, self.seed, self.npm = setting, seed, npm
        sd = torch.load(os.path.join(RUNS, "cond_base", f"s{seed}", "policy.pt"), map_location="cpu",
                        weights_only=False)
        ref = CondFlow(**sd["kw"]); ref.load_state_dict(sd["state"]); self.ref = frozen(ref)
        pool = E.make_pool(self.ref, seed, 200_000)
        self.ed = E.editset(pool, seed, k, npm, layout)
        anchor = E.make_pool(self.ref, seed, 20_000, purpose=E.P_ANCHOR)
        self.ctx = E.Ctx(ref=self.ref, seed=seed, ed=self.ed, pool=pool, anchor=anchor)
        self.bank = E.Bank(self.ref, seed, self.ed)
        self._rows, self._oracle, self._tracks, self.diag = {}, None, None, {}


def cell_dir(setting, seed, npm):
    return os.path.join(OUT, setting, f"s{seed}_n{npm}")


def draw(seed, purpose, n):
    """(c, xi) drawn as toypref's make_pool draws them; c is passed to pi_ref only."""
    g = E.gen(seed, purpose)
    c = torch.rand(n, generator=g)
    xi = torch.randn(n, 2, generator=g)
    return c, xi


# ------------------------------------------------------------------ masks
def mask_file(setting, seed, npm, b, N):
    return os.path.join(cell_dir(setting, seed, npm), f"masks_{b}_N{N}")


def run_masks(setting, seed, npm, configs, Ns):
    """One unit = one cell x the given builders x the given N (files masks_<b>_N<N>.json/.npz)."""
    od = cell_dir(setting, seed, npm)
    os.makedirs(od, exist_ok=True)
    todo = [(b, N) for b in configs for N in Ns if not os.path.exists(mask_file(setting, seed, npm, b, N) + ".json")]
    if not todo:
        return
    cell = Cell(setting, seed, npm)
    ref, ed = cell.ref, cell.ed
    P = E.make_pool(ref, seed, N_POOL, purpose=POOL_PURPOSE)       # B6's training pool
    acts_P, a_P = BL.activations(ref, P.xi, P.c, steps=ALL_STEPS)
    assert torch.equal(a_P, P.a), "hooked decode must reproduce pi_ref.sample"
    cB, xiB = draw(seed, DRAWB_PURPOSE, max(Ns))
    acts_B, _ = BL.activations(ref, xiB, cB, steps=ALL_STEPS)
    acts_N, _ = BL.activations(ref, ed.xi, ed.c, steps=ALL_STEPS)
    bad_or = np.isin(P.mode, ed.forget)                              # ORACLE: scoring only
    # reference fine partition for the balance check (builder-independent, label-free): k-means 64 on
    # PCA-32 of the pool's step-19 activations, fit on the first 20k pool samples
    from sklearn.cluster import KMeans
    F19 = BL.feat(acts_P, (19,))
    Z_ref = BL.pca_fit(F19[:20_000], seed).transform(F19)
    ref_cells = KMeans(n_clusters=64, n_init=1, random_state=seed).fit(Z_ref[:20_000]).predict(Z_ref)
    for b, N in todo:
        t0 = time.time()
        out, flags = {}, {}
        for dname, acts_fit in (("A", {s: v[:N] for s, v in acts_P.items()}),
                                ("B", {s: v[:N] for s, v in acts_B.items()})):
            pcas = {}
            for cfg in configs[b]:
                name = cfg_name(b, cfg, N)
                kw = {k: v for k, v in cfg.items() if k != "f"}
                st = FEATS[cfg["f"]]
                ts = time.time()
                Ff, Fp, Fn = (BL.feat(x, st) for x in (acts_fit, acts_P, acts_N))
                if b == "TA":
                    m = BL.build_TA(Ff, Fp, Fn, seed, **kw)
                else:
                    if cfg["f"] not in pcas:
                        pca = BL.pca_fit(Ff, seed)
                        pcas[cfg["f"]] = (pca.transform(Ff), pca.transform(Fp), pca.transform(Fn))
                    Zf, Zp, Zn = pcas[cfg["f"]]
                    m = (BL.build_TB if b == "TB" else BL.build_TC)(Zf, Zp, Zn, seed, **kw)
                sc = BL.score(m, bad_or)
                rec = out.setdefault(name, {"builder": b, "cfg": cfg, "N": N})
                if dname == "A":
                    rec.update(sc)
                    rec["info"] = {k: v for k, v in m["info"].items() if k != "sils"}
                    rec["sils"] = {str(k): v for k, v in m["info"].get("sils", {}).items()}
                    rec["balance"] = BL.balance(BL.keep_of(m), ref_cells, 64)
                    rec["build_s"] = round(time.time() - ts, 1)
                    flags[name] = m
                else:
                    mA = flags[name]
                    rec["stab_ari"] = BL.ari(mA["lab"], m["lab"])
                    rec["stab_flag_jacc"] = BL.jacc(mA["flag"], m["flag"])
                    rec["prec_B"], rec["rec_B"] = sc["prec"], sc["rec"]
                print(f"[{setting} s{seed} n{npm}] {name:44s} {dname} prec {sc['prec']:.3f} rec {sc['rec']:.3f} "
                      f"ncl {m['info']['n_clusters']} bad {m['info']['n_bad']} un {sc['unassigned']:.2f} "
                      f"({time.time() - ts:.0f}s)", flush=True)
        fb = mask_file(setting, seed, npm, b, N)
        np.savez_compressed(fb + ".npz", **{f"keep|{k}": np.packbits(BL.keep_of(m)) for k, m in flags.items()},
                            **{f"flag|{k}": np.packbits(m["flag"]) for k, m in flags.items()})
        meta = {"setting": setting, "seed": seed, "npm": npm, "forget": ed.forget, "n_neg": int(ed.c.shape[0]),
                "unit_s": round(time.time() - t0, 1), "masks": out}
        json.dump(meta, open(fb + ".tmp", "w"))
        os.replace(fb + ".tmp", fb + ".json")


# ------------------------------------------------------------------ training (frozen B6 step)
def b6_keep(cell, P):
    import tracks as TR
    tr = cell.tracks()
    bad, _ = TR.bad_tracks(tr, cell.ed.c, cell.ed.a)
    t = tr.assign(P.a, P.c)
    return (t >= 0) & ~np.isin(t, bad)


def run_train(setting, seed, npm, mask):
    od = cell_dir(setting, seed, npm)
    fn_out = os.path.join(od, f"{mask}.json")
    if os.path.exists(fn_out):
        return
    cell = Cell(setting, seed, npm)
    if mask == "B1a":  # e3's B1a (R_hopB_bsup_neg_AMdist), unchanged
        rec = e3.run_method(cell, "B1a")
        os.makedirs(od, exist_ok=True)
        json.dump(rec, open(os.path.join(od, ".B1a.tmp"), "w"))
        os.replace(os.path.join(od, ".B1a.tmp"), fn_out)
        print(f"[{setting} s{seed} n{npm}] B1a D {rec['D']:.3f} ({rec['train_s']}s)", flush=True)
        return
    ctx = cell.ctx
    P = E.make_pool(ctx.ref, ctx.seed, N_POOL, purpose=POOL_PURPOSE)
    if mask == "B6":
        keep_np = b6_keep(cell, P)
    else:
        b, N = mask.split("-")[0], mask.split("-N")[-1]
        z = np.load(mask_file(setting, seed, npm, b, N) + ".npz")
        keep_np = np.unpackbits(z[f"keep|{mask}"])[:N_POOL].astype(bool)
    keep = torch.from_numpy(keep_np)
    c, a = P.c[keep], P.a[keep]
    B = ctx.batch
    t0 = time.time()

    def fn(pol, g):  # e3.anchor_fn('track60k') loss, w=1
        j = E._pick(c.shape[0], B, g)
        t_ = FlowPolicy.sample_time(B, g)
        return pol.flow_loss(a[j], torch.randn(a[j].shape, generator=g), t_, c[j]).mean()

    # e3.run_method with p=all, st=20000, lr=1e-4 (B6 spec a=track60k-w=1-st=20000)
    torch.manual_seed(ctx.seed)
    pol = copy.deepcopy(ctx.ref)
    pol.train()
    params = e3.select_params(pol, "all")
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=1e-10)
    g = E.gen(ctx.seed, E.P_TRAIN)
    for step in range(STEPS):
        lr = cosine(step, STEPS, min(500, STEPS // 10), LR, 0.1 * LR)
        for pg in opt.param_groups:
            pg["lr"] = lr
        loss = fn(pol, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    pol.eval()
    for q in pol.parameters():
        q.requires_grad_(False)
    met = E.compute(cell.bank, pol)
    badm = np.isin(P.mode, cell.ed.forget)
    rec = {"mask": mask, "setting": setting, "seed": seed, "npm": npm, "forget": cell.ed.forget,
           "D": met["tv_to_ideal"], "Removed": 1 - met["neg_mass_ratio"], "Collateral": met["retain_tv"],
           "mass_off": met["mass_off"], "keep": int(keep_np.sum()),
           "keep_frac_bad": float(badm[keep_np].mean()), "train_s": round(time.time() - t0, 1),
           **{k: v for k, v in met.items() if k != "tv_to_ideal"}}
    os.makedirs(od, exist_ok=True)
    tmp = os.path.join(od, f".{mask}.tmp")
    json.dump(rec, open(tmp, "w"))
    os.replace(tmp, fn_out)
    print(f"[{setting} s{seed} n{npm}] {mask:40s} D {rec['D']:.3f} Rem {rec['Removed']:.3f} "
          f"Col {rec['Collateral']:.3f} keep {rec['keep']} ({rec['train_s']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["masks", "train"])
    ap.add_argument("--setting", required=True, choices=list(SETTINGS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--npm", type=int, required=True)
    ap.add_argument("--mask", default=None, help="comma-separated mask names (train)")
    ap.add_argument("--builders", default=None)
    ap.add_argument("--N", default=None)
    ap.add_argument("--stage", default="dev", choices=["dev", "confirm"])
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.stage == "confirm":
        assert os.path.exists(FROZEN_FILE), "confirm stage needs the frozen prereg configs"
    elif args.seed >= 5:
        raise SystemExit("seeds 5-9 are confirmation seeds: use --stage confirm after the prereg")
    if args.cmd == "masks":
        cfgs, Ns = CONFIGS, NS
        if args.stage == "confirm":
            fz = json.load(open(FROZEN_FILE))
            cfgs = {b: [fz["builders"][b]] for b in fz["builders"]}
            Ns = (int(fz["N"]),)
        if args.builders:
            cfgs = {b: cfgs[b] for b in args.builders.split(",")}
        if args.N:
            Ns = tuple(int(x) for x in args.N.split(","))
        run_masks(args.setting, args.seed, args.npm, cfgs, Ns)
    else:
        for m in args.mask.split(","):
            run_train(args.setting, args.seed, args.npm, m)


if __name__ == "__main__":
    main()
