"""Every reported quantity, for one arm in one cell.

Two questions, many measurements of each:

DOWNWEIGHTING (did the forget modes lose mass?)
  mass on the forget modes vs pi_ref (absolute, ratio, per mode), the paired
  per-noise fate of noises pi_ref sent to a forget mode, the same for the LOGGED
  training negatives (memorisation) vs held-out ones (generalisation), and the
  change in log-density of held-out true forget-mode samples.

FORGETTING / COLLATERAL DAMAGE (did anything else move?)
  retained-mode masses vs pi_ref (TV, KL, min/max ratio, modes dropped, adjacent
  vs far from a forget mode), off-mode mass, paired per-noise flips of noises pi_ref
  sent to a retained mode and how far their actions moved, within-mode shape
  (centre error, spread), sample-level MMD / sliced-W2 against pi_ref's retained
  samples and against the ideal (true mixture renormalised over retained modes),
  held-out log-density on retained modes, velocity-field drift from pi_ref, and
  parameter drift.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .flow import FlowPolicy
from .mixture import OFF, MixtureConfig, assign_modes, mode_masses

MMD_BANDWIDTHS = (0.1, 0.3, 1.0, 3.0)
N_SUB = 3000
N_PROJ = 128


def mmd2(x: torch.Tensor, y: torch.Tensor) -> float:
    """Unbiased MMD^2 with a sum of RBF kernels."""
    def k(a, b):
        d2 = torch.cdist(a, b) ** 2
        return sum(torch.exp(-d2 / (2 * h * h)) for h in MMD_BANDWIDTHS)
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        return float("nan")
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    return float((kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
                 + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1)) - 2 * kxy.mean())


def sliced_w2(x: torch.Tensor, y: torch.Tensor, g: torch.Generator) -> float:
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    n = min(x.shape[0], y.shape[0])
    th = torch.randn(N_PROJ, x.shape[1], generator=g)
    th = th / th.norm(dim=1, keepdim=True)
    q = torch.linspace(0, 1, n)
    px = torch.quantile((x @ th.T), q, dim=0)
    py = torch.quantile((y @ th.T), q, dim=0)
    return float(((px - py) ** 2).mean().sqrt())


def _sub(x, n, g):
    if x.shape[0] <= n:
        return x
    return x[torch.randperm(x.shape[0], generator=g)[:n]]


class EvalBank:
    """Everything shared by all arms of one (seed, forget set): computed once."""

    def __init__(self, ref: FlowPolicy, mix: MixtureConfig, Z: torch.Tensor, test_x: np.ndarray,
                 test_m: np.ndarray, ideal_x: np.ndarray, forget: List[int], a_neg, xi_neg, a_pos,
                 seed: int, logprob_steps: int = 100):
        self.ref, self.mix, self.Z = ref, mix, Z
        self.forget = list(forget)
        self.retain = [m for m in range(mix.n_modes) if m not in self.forget]
        self.a_ref = ref.sample(Z)
        self.lab_ref = assign_modes(mix, self.a_ref)
        self.mass_ref = mode_masses(mix, self.lab_ref)
        self.test_x = torch.from_numpy(test_x)
        self.test_m = test_m
        self.ideal_x = torch.from_numpy(ideal_x)
        self.a_neg, self.xi_neg, self.a_pos = a_neg, xi_neg, a_pos
        self.seed = seed
        self.logprob_steps = logprob_steps
        self.ref_params = torch.cat([p.detach().flatten() for p in ref.parameters()])
        self.ref_lp_test = ref.log_prob(self.test_x, logprob_steps)
        self.ref_lp_neg = ref.log_prob(a_neg, logprob_steps)
        self.ref_lp_pos = ref.log_prob(a_pos, logprob_steps)
        # drift probe: pi_ref's own samples, fresh noise and time, fixed per seed
        g = torch.Generator().manual_seed(seed * 31 + 5)
        idx = torch.randperm(Z.shape[0], generator=g)[:4000]
        self.drift_a = self.a_ref[idx]
        self.drift_lab = self.lab_ref[idx.numpy()]
        self.drift_noise = torch.randn(self.drift_a.shape, generator=g)
        self.drift_t = FlowPolicy.sample_time(self.drift_a.shape[0], g)
        self.ref_retain_x = self.a_ref[torch.from_numpy(np.isin(self.lab_ref, self.retain))]


def make_probe(bank: "EvalBank", n: int = 4000):
    """Cheap per-checkpoint metrics on the first ``n`` common eval noises."""
    Z, lab_ref = bank.Z[:n], bank.lab_ref[:n]
    mref = mode_masses(bank.mix, lab_ref)
    F, R, M = bank.forget, bank.retain, bank.mix.n_modes
    ref_ret = np.isin(lab_ref, R)

    @torch.no_grad()
    def probe(policy):
        a = policy.sample(Z)
        a = torch.where(torch.isfinite(a), a, torch.full_like(a, 1e6))
        lab = assign_modes(bank.mix, a)
        m = mode_masses(bank.mix, lab)
        pR = m[R] / max(m[R].sum(), 1e-12)
        qR = mref[R] / max(mref[R].sum(), 1e-12)
        return {"neg_mass_ratio": float(m[F].sum() / max(mref[F].sum(), 1e-12)),
                "retain_tv": 0.5 * float(np.abs(pR - qR).sum()),
                "mass_off": float(m[M]),
                "retain_noise_flip": float((lab != lab_ref)[ref_ret].mean())}
    return probe


def compute(bank: EvalBank, sampler, policy: Optional[FlowPolicy]) -> (Dict, Dict):
    """-> (flat scalar metrics, nested extras such as the transition matrix)."""
    mix, F, R = bank.mix, bank.forget, bank.retain
    M = mix.n_modes
    out: Dict = {}
    extra: Dict = {}
    g = torch.Generator().manual_seed(bank.seed * 17 + 3)

    a = sampler.decode(bank.Z)
    finite = torch.isfinite(a).all(1)
    out["frac_nonfinite"] = float((~finite).float().mean())
    a = torch.where(finite[:, None], a, torch.full_like(a, 1e6))
    lab = assign_modes(mix, a)
    mass = mode_masses(mix, lab)
    mref = bank.mass_ref
    extra["mass"] = mass.tolist()
    extra["mass_ref"] = mref.tolist()
    for j in range(M):
        out[f"mass_m{j}"] = mass[j]
    out["mass_off"] = mass[M]

    # ---------------- downweighting --------------------------------------
    neg = mass[F].sum()
    neg_ref = mref[F].sum()
    out["neg_mass"] = neg
    out["neg_mass_ref"] = neg_ref
    out["neg_mass_ratio"] = neg / max(neg_ref, 1e-12)
    out["neg_mass_reduction"] = 1 - out["neg_mass_ratio"]
    ratios_F = mass[F] / np.maximum(mref[F], 1e-12)
    out["neg_mode_ratio_max"] = float(ratios_F.max())   # the least-forgotten mode
    out["neg_mode_ratio_min"] = float(ratios_F.min())

    ref_neg = np.isin(bank.lab_ref, F)
    ref_ret = np.isin(bank.lab_ref, R)
    same = lab == bank.lab_ref
    in_F, in_R, off = np.isin(lab, F), np.isin(lab, R), lab == OFF
    if ref_neg.any():
        out["neg_noise_stay_same"] = float(same[ref_neg].mean())
        out["neg_noise_to_other_neg"] = float((in_F & ~same)[ref_neg].mean())
        out["neg_noise_to_retain"] = float(in_R[ref_neg].mean())
        out["neg_noise_to_off"] = float(off[ref_neg].mean())
        out["disp_neg_mean"] = float(torch.linalg.norm(a - bank.a_ref, dim=1)[torch.from_numpy(ref_neg)].mean())

    # logged training negatives (memorisation) vs held-out eval noises above
    a_tn = sampler.decode(bank.xi_neg)
    lab_tn = assign_modes(mix, a_tn)
    out["train_neg_still_in_forget"] = float(np.isin(lab_tn, F).mean())
    out["train_neg_to_retain"] = float(np.isin(lab_tn, R).mean())
    out["train_neg_to_off"] = float((lab_tn == OFF).mean())
    if ref_neg.any():
        out["neg_generalisation_gap"] = (1 - out["train_neg_still_in_forget"]) - (
            1 - out["neg_noise_stay_same"] - out["neg_noise_to_other_neg"])

    # ---------------- forgetting / collateral -----------------------------
    out["retain_mass"] = mass[R].sum()
    out["retain_mass_ref"] = mref[R].sum()
    out["off_mass_delta"] = mass[M] - mref[M]
    pR = mass[R] / max(mass[R].sum(), 1e-12)
    qR = mref[R] / max(mref[R].sum(), 1e-12)
    out["retain_tv"] = 0.5 * float(np.abs(pR - qR).sum())
    out["retain_kl"] = float(np.sum(qR * np.log((qR + 1e-6) / (pR + 1e-6))))   # KL(ref || model)
    rr = mass[R] / np.maximum(mref[R], 1e-12)
    out["retain_ratio_min"] = float(rr.min())
    out["retain_ratio_max"] = float(rr.max())
    out["retain_ratio_mean"] = float(rr.mean())
    out["retain_n_below_0.5"] = int((rr < 0.5).sum())
    out["retain_n_below_0.1"] = int((rr < 0.1).sum())
    out["retain_mass_cv"] = float(mass[R].std() / max(mass[R].mean(), 1e-12))
    A = mix.adjacency()
    adj = [m for m in R if A[m, F].any()]
    far = [m for m in R if not A[m, F].any()]
    out["retain_adjacent_ratio_mean"] = float(np.mean([mass[m] / mref[m] for m in adj])) if adj else float("nan")
    out["retain_far_ratio_mean"] = float(np.mean([mass[m] / mref[m] for m in far])) if far else float("nan")
    ideal = np.zeros(M + 1)
    ideal[R] = 1.0 / len(R)
    out["tv_to_ideal"] = 0.5 * float(np.abs(mass - ideal).sum())
    out["tv_to_ideal_ref"] = 0.5 * float(np.abs(mref - ideal).sum())
    out["tv_vs_ref_full"] = 0.5 * float(np.abs(mass - mref).sum())

    if ref_ret.any():
        out["retain_noise_same"] = float(same[ref_ret].mean())
        out["retain_noise_flip"] = 1 - out["retain_noise_same"]
        out["retain_noise_to_other_retain"] = float((in_R & ~same)[ref_ret].mean())
        out["retain_noise_to_neg"] = float(in_F[ref_ret].mean())
        out["retain_noise_to_off"] = float(off[ref_ret].mean())
        d = torch.linalg.norm(a - bank.a_ref, dim=1).numpy()
        out["disp_retain_mean"] = float(d[ref_ret].mean())
        out["disp_retain_p90"] = float(np.quantile(d[ref_ret], 0.9))
        out["disp_retain_samemode_mean"] = float(d[ref_ret & same].mean()) if (ref_ret & same).any() else float("nan")

    T = np.zeros((M + 1, M + 1))
    lr_, lm_ = np.where(bank.lab_ref == OFF, M, bank.lab_ref), np.where(lab == OFF, M, lab)
    np.add.at(T, (lr_, lm_), 1)
    extra["transition_counts"] = T.astype(int).tolist()

    C = mix.centers()
    cerr, sratio = [], []
    for m in R:
        xm = a[torch.from_numpy(lab == m)]
        if xm.shape[0] >= 10:
            cerr.append(float(torch.linalg.norm(xm.mean(0) - torch.from_numpy(C[m]))))
            sratio.append(float(xm.std(0).mean() / mix.mode_std))
    out["retain_center_err_mean"] = float(np.mean(cerr)) if cerr else float("nan")
    out["retain_std_ratio_mean"] = float(np.mean(sratio)) if sratio else float("nan")
    out["retain_std_ratio_min"] = float(np.min(sratio)) if sratio else float("nan")
    out["retain_std_ratio_max"] = float(np.max(sratio)) if sratio else float("nan")

    a_fin = a[finite]
    a_ret = a[torch.from_numpy(in_R)]
    out["mmd2_to_ideal"] = mmd2(_sub(a_fin, N_SUB, g), _sub(bank.ideal_x, N_SUB, g))
    out["mmd2_retain_vs_ref_retain"] = mmd2(_sub(a_ret, N_SUB, g), _sub(bank.ref_retain_x, N_SUB, g))
    out["sw2_to_ideal"] = sliced_w2(_sub(a_fin, N_SUB, g), _sub(bank.ideal_x, N_SUB, g), g)
    out["sw2_retain_vs_ref_retain"] = sliced_w2(_sub(a_ret, N_SUB, g), _sub(bank.ref_retain_x, N_SUB, g), g)

    # ---------------- density-level (flow arms only) ----------------------
    if policy is not None:
        lp = policy.log_prob(bank.test_x, bank.logprob_steps)
        dl = (lp - bank.ref_lp_test).numpy()
        per = {int(m): float(dl[bank.test_m == m].mean()) for m in range(M)}
        extra["logp_delta_per_mode"] = per
        out["logp_delta_forget_mean"] = float(np.mean([per[m] for m in F]))
        out["logp_delta_retain_mean"] = float(np.mean([per[m] for m in R]))
        out["logp_delta_retain_min"] = float(np.min([per[m] for m in R]))
        out["logp_forget_mean"] = float(lp.numpy()[np.isin(bank.test_m, F)].mean())
        out["logp_retain_mean"] = float(lp.numpy()[np.isin(bank.test_m, R)].mean())
        out["logp_delta_train_neg"] = float((policy.log_prob(bank.a_neg, bank.logprob_steps) - bank.ref_lp_neg).mean())
        out["logp_delta_train_pos"] = float((policy.log_prob(bank.a_pos, bank.logprob_steps) - bank.ref_lp_pos).mean())

        with torch.no_grad():
            tt = bank.drift_t.reshape(-1, 1)
            x_t = tt * bank.drift_noise + (1 - tt) * bank.drift_a
            dv = ((policy.net(x_t, bank.drift_t) - bank.ref.net(x_t, bank.drift_t)) ** 2).sum(1).numpy()
        rmask, nmask = np.isin(bank.drift_lab, R), np.isin(bank.drift_lab, F)
        out["vdrift_retain"] = float(dv[rmask].mean()) if rmask.any() else float("nan")
        out["vdrift_forget"] = float(dv[nmask].mean()) if nmask.any() else float("nan")
        theta = torch.cat([p.detach().flatten() for p in policy.parameters()])
        out["param_drift_rel"] = float((theta - bank.ref_params).norm() / bank.ref_params.norm())

    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in out.items()}, extra
