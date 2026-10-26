"""Report figures, unconditional toy: train one arm on one cell and dump what the figures need.

Mirrors scripts/run_cell.py (same pool, edit set, anchors, ArmContext, finetune budget) but
skips the expensive log-density bank; the probe/final metrics use metrics.make_probe's formulas
(forgotten-mass ratio vs pi_ref, retain TV, off mass) via the repo's assign_modes/mode_masses.
usage: twod_unc_run.py --arm NAME [--seed 0 --n_forget 2 --n_neg 8]; --arm common writes the shared file.
TOYPREF_RUNS must point at the report_2d dir (base policy lives in $TOYPREF_RUNS/base/s0).
"""
import argparse, os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np, torch
from toypref import seeds
from toypref.arms import ARMS, ArmContext, run_arm
from toypref.config import DEFAULT, RUNS, base_path
from toypref.editset import build_editset, make_pool, apply_regime
from toypref.flow import load_policy
from toypref.mixture import assign_modes, mode_masses
from toypref.reroute import make_typicality

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_forget", type=int, default=2)
ap.add_argument("--n_neg", type=int, default=8)
ap.add_argument("--steps", type=int, default=None)
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
cfg, mix, s = DEFAULT, DEFAULT.mixture, args.seed
tcfg = cfg.edit
if args.steps:
    from dataclasses import replace
    tcfg = replace(tcfg, steps=args.steps)
ref = load_policy(base_path(s))
pool = make_pool(ref, mix, s, cfg.pool_size)
es = build_editset(pool, mix, s, args.n_forget, args.n_neg, cfg.n_pos)
es = apply_regime(es, pool, mix, s, cfg.n_pos)
d0 = cfg.d0_in_std * mix.mode_std
apool = make_pool(ref, mix, s, cfg.n_anchor, purpose=seeds.ANCHOR_POOL)
anchor_nf = torch.cdist(apool.a, es.a_neg).min(1).values > d0
opool = make_pool(ref, mix, s, cfg.n_oracle * 2, purpose=seeds.ORACLE_POOL)
oracle_a = opool.a[np.isin(opool.mode, es.retain)][:cfg.n_oracle]
Z = torch.randn(cfg.n_eval, mix.dim, generator=seeds.gen(s, seeds.EVAL_NOISE))
F, R, M = es.forget, es.retain, mix.n_modes
with torch.no_grad():
    lab_ref = assign_modes(mix, ref.sample(Z))
mref = mode_masses(mix, lab_ref)
ref_ret = np.isin(lab_ref, R)

def metrics(pol, n=None):
    ZZ = Z if n is None else Z[:n]
    with torch.no_grad():
        a = pol.sample(ZZ)
    a = torch.where(torch.isfinite(a), a, torch.full_like(a, 1e6))
    lab = assign_modes(mix, a); m = mode_masses(mix, lab)
    mr = mode_masses(mix, lab_ref[:len(ZZ)])
    pR = m[R] / max(m[R].sum(), 1e-12); qR = mr[R] / max(mr[R].sum(), 1e-12)
    return {"neg_mass_ratio": float(m[F].sum() / max(mr[F].sum(), 1e-12)),
            "retain_tv": 0.5 * float(np.abs(pR - qR).sum()), "mass_off": float(m[M]),
            "retain_noise_flip": float((lab != lab_ref[:len(ZZ)])[ref_ret[:len(ZZ)]].mean())}

out = os.path.join(RUNS, "report_unc"); os.makedirs(out, exist_ok=True)
NZ = 4000
if args.arm == "common":
    with torch.no_grad():
        ref_a = ref.sample(Z[:NZ]).numpy()
    np.savez(os.path.join(out, "common.npz"), Z=Z[:NZ].numpy(), ref_a=ref_a, xi_neg=es.xi_neg.numpy(),
             a_neg=es.a_neg.numpy(), forget=np.array(F), retain=np.array(R), centers=mix.centers(),
             lab_ref=lab_ref[:NZ])
    print("common written", F, es.xi_neg.shape); sys.exit(0)

probe = lambda pol: metrics(pol, 4000)
ctx = ArmContext(ref=ref, seed=s, xi_neg=es.xi_neg, a_neg=es.a_neg, neg_pool_idx=es.neg_idx, a_pos=es.a_pos,
                 forget=es.forget, mix_cfg=mix, tcfg=tcfg, oracle_a=oracle_a,
                 online_pool_fn=lambda pol: make_pool(pol, mix, s, cfg.online_pool, purpose=seeds.ONLINE_POOL),
                 n_neg_per_mode=args.n_neg, anchor_xi=apool.xi, anchor_a=apool.a, anchor_nf=anchor_nf, d0=d0,
                 typical=make_typicality(apool.a[:4000]))
t0 = time.time()
sampler, pol, info = run_arm(ARMS[args.arm], ctx, probe=probe)
fin = metrics(pol) if pol is not None else metrics(sampler)
g1 = np.linspace(-3, 3, 241, dtype=np.float32)
G = torch.tensor(np.stack(np.meshgrid(g1, g1), -1).reshape(-1, 2))
with torch.no_grad():
    a_fix = sampler.decode(Z[:NZ]).numpy()
    glab = assign_modes(mix, sampler.decode(G)).reshape(241, 241)
hist = [{k: h[k] for k in h if k == "step" or k.startswith("probe_")} for h in (info.get("train_history") or [])]
np.savez(os.path.join(out, f"{args.arm}.npz"), a_fix=a_fix, grid_lab=glab, final=json.dumps(fin),
         hist=json.dumps(hist), sdpo_pos_frac_in_forget=info.get("sdpo_pos_frac_in_forget", np.nan),
         train_s=time.time() - t0)
print(args.arm, fin, f"{time.time()-t0:.0f}s", flush=True)
