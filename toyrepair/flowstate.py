"""Flow states, renoise, primitives and the branch tool (docs/PLAN.md 2, 5.4, 6.3).

Flow state: (x_k, t_k, c) at step k of one decode. A branch restarts the decode from a saved
flow state. Renoise at step k >= 1: x_hat = x_k - t_k * v(x_k, t_k, c);
x' = t_k * eps + (1 - t_k) * x_hat, eps ~ N(0, I) fresh. At step 0 x' is fresh noise.

Rescue: the branch ends outside B_i at a typical action (5th-NN distance to pi_ref samples at
c_i at or below their 90th percentile, as ``toypref.cond.targets``).
p(k): share of M renoise branches from step k that end in a rescue, averaged over members.
k_commit: earliest sampled step after which every later p(k) < 0.2; if none, the last step.
Class R: a primitive at or before k_commit rescues >= 50% of members; else U.

Owner: W1-A1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

DEFAULT_STEPS: Tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14, 16)
M_BRANCH = 32                 # renoise branches per member and step
M_EXTRA = 64                  # extra branches when the 95% Wilson interval contains P_COMMIT
P_COMMIT = 0.2                # k_commit threshold on p(k)
RESCUE_CLASS_R = 0.5          # class R if a primitive rescues >= 50% of members
PUSH_DIRS = 8
PUSH_ALPHAS = (0.5, 1.0, 2.0)  # x basin radius (90th pct distance of B_i samples to their mean)
PUSH_H = 3
TABLE_ROW_CAP = 40
TOP_PRIMITIVES = 5


@dataclass(frozen=True)
class FlowState:
    """x: (B, 2) float32 state x_k; k: step 0..20; c: (B,) float32. t_k = 1 - k/20."""
    x: torch.Tensor
    k: int
    c: torch.Tensor

    @property
    def t(self) -> float:
        return 1.0 - self.k / 20.0


@dataclass(frozen=True)
class Primitive:
    """A scripted change to a decode.

    kind "push": add total displacement alpha * u over steps k .. k+h-1 (alpha/h per step,
                 added to x after each Euler update); u (2,) unit vector; alpha in action units.
    kind "hop":  xi <- xi + tau * u at step 0 (k = 0, h = 0, alpha = tau, noise units).
    name: short id for tables, e.g. ``push(d3,1.0r,k6,h3)`` or ``hop(0.45,d7)``.
    """
    kind: str
    u: Tuple[float, float]
    alpha: float
    k: int
    h: int = PUSH_H
    name: str = ""


@dataclass(frozen=True)
class BranchRow:
    """One row of the branch table.

    step k, t_k, p (rescue share), ci (95% Wilson lo, hi), n (branches used per member),
    best: up to 5 (primitive name, rescue rate in [0, 1]) at this step (empty if none run)."""
    k: int
    t: float
    p: float
    ci: Tuple[float, float]
    n: int
    best: Tuple[Tuple[str, float], ...] = ()


@dataclass
class BranchTable:
    """cluster: cluster id; rows: <= 40 rows in step order; k_commit; failure_class "R" | "U";
    best_primitive: the best primitive overall (None if class U); n_decodes: decodes used."""
    cluster: int
    rows: List[BranchRow]
    k_commit: int
    failure_class: str
    best_primitive: Optional[Primitive] = None
    n_decodes: int = 0

    n_members: int = 0
    M: int = M_BRANCH
    meta: Dict = field(default_factory=dict)

    def render(self) -> str:
        """Plain-text table for the dossier (<= 40 rows)."""
        head = (f"branch table, cluster {self.cluster}: {self.n_members} members, M={self.M} renoise "
                f"branches/member/step (+{M_EXTRA} where the 95% CI holds {P_COMMIT}); "
                f"k_commit={self.k_commit}; class {self.failure_class}")
        if self.best_primitive is not None:
            head += f"; best primitive {self.best_primitive.name}"
        lines = [head, "step  t_k   p(k)   95% CI         n   best primitives (rescue rate)"]
        rows = self.rows[:TABLE_ROW_CAP]
        for r in rows:
            best = ", ".join(f"{nm} {rate:.2f}" for nm, rate in r.best[:TOP_PRIMITIVES]) or "-"
            lines.append(f"{r.k:>4}  {r.t:.2f}  {r.p:.3f}  [{r.ci[0]:.3f},{r.ci[1]:.3f}]  {r.n:>3}   {best}")
        if len(self.rows) > TABLE_ROW_CAP:
            lines.append(f"({len(self.rows) - TABLE_ROW_CAP} rows cut)")
        return "\n".join(lines)


# ---------------------------------------------------------------- basins, rescue
def _typical(cell, i: int, x: torch.Tensor) -> np.ndarray:
    """(P,) bool: 5th-NN distance of x to pi_ref's typical decodes at c_i <= their 90th pct
    (``toypref.cond.targets._typical`` on B_i's basins object)."""
    from toypref.cond import targets as TG
    return TG._typical(cell.basins[i], x).numpy()


def is_rescue(cell, i: int, x: torch.Tensor) -> np.ndarray:
    """(P,) bool: action x (P, 2) ends outside B_i at a typical action."""
    return (~cell.in_bad_basin(i, x)) & _typical(cell, i, x)


def basin_radius(cell, i: int) -> float:
    """90th-percentile distance of B_i's samples from their mean (action units)."""
    b = cell.basins[i]
    pts = b.pts[torch.from_numpy(b.lab == b.bad)]
    if pts.shape[0] == 0:
        return 0.0
    return float(torch.quantile(torch.linalg.norm(pts - pts.mean(0), dim=1), 0.9))


def basin_radii(cell) -> np.ndarray:
    """(N,) basin radius of every negative (cached on the cell)."""
    cache = cell.__dict__.setdefault("_w1a1_cache", {})
    if "radii" not in cache:
        cache["radii"] = np.array([basin_radius(cell, i) for i in range(len(cell.neg))])
    return cache["radii"]


# ---------------------------------------------------------------- flow states
def save_states(version, xi: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """(21, B, 2) states x_0..x_20 of the uninterrupted decode."""
    from . import policy
    return policy.decode(version, xi, c, save_states=True).states


def _primitive_hook(p: Primitive, n: int):
    """after_step hook that adds alpha/h * u after the Euler updates of steps k .. k+h-1."""
    du = torch.tensor(p.u, dtype=torch.float32) * (p.alpha / max(p.h, 1))

    def hook(k, x, rows):
        if p.k <= k < p.k + p.h:
            return x + du
        return x
    return hook


def restore_and_decode(version, state: FlowState, primitive: Optional[Primitive] = None) -> torch.Tensor:
    """Continue the decode from ``state`` to step 20, optionally applying ``primitive``.
    With no primitive, equals the uninterrupted decode bit for bit (5.4). Returns (B, 2).

    A hop applies only when ``state.k == 0`` (it moves the noise); a push applies at its steps
    that are >= state.k."""
    from . import policy
    x, hook = state.x, None
    if primitive is not None:
        if primitive.kind == "hop":
            if state.k != 0:
                raise ValueError("a hop acts at step 0; restore from k = 0")
            x = x + primitive.alpha * torch.tensor(primitive.u, dtype=torch.float32)
        elif primitive.kind == "push":
            hook = _primitive_hook(primitive, x.shape[0])
        else:
            raise ValueError(f"unknown primitive kind {primitive.kind!r}")
    return policy.decode(version, x, state.c, start_step=state.k, after_step=hook).action


@torch.no_grad()
def renoise(version, state: FlowState, g: torch.Generator) -> FlowState:
    """Renoise ``state`` (see module doc) with one fresh eps (B, 2) from ``g``; same k and c."""
    from . import policy
    eps = torch.randn(state.x.shape[0], 2, generator=g)
    if state.k == 0:
        return FlowState(x=eps, k=0, c=state.c)
    t = state.t
    v, _ = policy.velocity(version, state.x, torch.full((state.x.shape[0],), t), state.c)
    x_hat = state.x - t * v
    return FlowState(x=t * eps + (1.0 - t) * x_hat, k=state.k, c=state.c)


# ---------------------------------------------------------------- primitives
def _unit(u: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(u, dtype=np.float64).reshape(2)
    nrm = float(np.linalg.norm(a))
    if nrm == 0:
        raise ValueError("direction must be non-zero")
    return (float(a[0] / nrm), float(a[1] / nrm))


def push(u: Sequence[float], alpha: float, k: int, h: int = PUSH_H, name: str = "") -> Primitive:
    """Build a push primitive; u is normalised."""
    if not 0 <= k < 20 or h < 1:
        raise ValueError("push needs 0 <= k < 20 and h >= 1")
    uu = _unit(u)
    return Primitive("push", uu, float(alpha), int(k), int(min(h, 20 - k)),
                     name or f"push(({uu[0]:.2f},{uu[1]:.2f}),{alpha:.2f},k{k},h{h})")


def hop(tau: float, u: Sequence[float], name: str = "") -> Primitive:
    """Build a hop primitive (k = 0); u is normalised."""
    uu = _unit(u)
    return Primitive("hop", uu, float(tau), 0, 0, name or f"hop({tau:.2f},({uu[0]:.2f},{uu[1]:.2f}))")


def _dir8(j: int) -> Tuple[float, float]:
    a = 2 * np.pi * j / PUSH_DIRS
    return (float(np.cos(a)), float(np.sin(a)))


def hop_search(cell, version, rows: np.ndarray, g: Optional[torch.Generator] = None):
    """Toypref hop search (``toypref.cond.targets.target_one``, rule hop, no veto) under
    ``version`` for each negative row: march xi_i along 16 random unit directions in steps of
    0.05 up to 3.0; the smallest tau whose decode is in a non-bad basin, plus the 0.2 margin.
    Directions come from ``g`` (default: ``toypref.cond.targets.row_gen(seed, uid, 1)`` after
    the 128 basin noises, i.e. exactly toypref's directions). Returns a list of (tau, u) or None
    per row."""
    from toypref.cond import targets as TG
    from . import policy
    out = []
    taus = torch.arange(TG.HOP_STEP, TG.HOP_MAX + 1e-9, TG.HOP_STEP)
    for i in rows:
        i = int(i)
        if g is None:
            gi = TG.row_gen(cell.seed, int(cell.neg.uid[i]), 1)
            torch.randn(TG.M_DECODE, 2, generator=gi)          # basins_at's draw
        else:
            gi = g
        dirs = torch.randn(TG.HOP_DIRS, 2, generator=gi)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        Z = cell.neg.xi[i].reshape(1, 1, 2) + taus.reshape(1, -1, 1) * dirs.reshape(-1, 1, 2)
        n = Z.shape[0] * Z.shape[1]
        dec = policy.decode(version, Z.reshape(-1, 2), torch.full((n,), float(cell.neg.c[i]))).action
        b = cell.basins[i]
        cl = b.assign(dec).reshape(Z.shape[0], Z.shape[1])
        best = None
        for d in range(Z.shape[0]):
            hit = np.flatnonzero((cl[d] >= 0) & (cl[d] != b.bad))
            if len(hit):
                tau = float(taus[hit[0]]) + TG.HOP_MARGIN
                if best is None or tau < best[0]:
                    best = (tau, tuple(float(z) for z in dirs[d]))
        out.append(best)
    return out


def default_primitives(cell, cluster_members: np.ndarray, k: int, version=None) -> List[Primitive]:
    """8 directions x alpha in (0.5, 1, 2) x basin radius pushes at step k with h = 3, plus the
    toypref hop search exits (``toypref.cond.targets`` hop) for each member.

    The basin radius is the median over members of their B_i radius. Hops are step-0
    primitives: each member's exit (tau, u) becomes one shared primitive applied to every
    member (the per-member exits themselves always rescue their own member, by construction,
    so they do not count). Hops are returned only when ``k == 0`` or ``version`` is given
    (hop search needs a version; it runs under ``version``)."""
    rad = float(np.median(basin_radii(cell)[np.asarray(cluster_members)]))
    prims = []
    if k < 20:
        for j in range(PUSH_DIRS):
            for a in PUSH_ALPHAS:
                prims.append(push(_dir8(j), a * rad, k, PUSH_H, name=f"push(d{j},{a:g}r,k{k},h{PUSH_H})"))
    if version is not None:
        seen = set()
        for row, ex in zip(cluster_members, hop_search(cell, version, np.asarray(cluster_members))):
            if ex is None:
                continue
            key = (round(ex[0], 3), round(ex[1][0], 3), round(ex[1][1], 3))
            if key in seen:
                continue
            seen.add(key)
            prims.append(hop(ex[0], ex[1], name=f"hop({ex[0]:.2f},n{int(cell.neg.uid[int(row)])})"))
    return prims


# ---------------------------------------------------------------- statistics
def wilson(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial share."""
    if n <= 0:
        return (0.0, 1.0)
    den = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / den
    half = z * np.sqrt(max(p_hat * (1 - p_hat) / n + z * z / (4 * n * n), 0.0)) / den
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def k_commit(steps: Sequence[int], p: Sequence[float], thr: float = P_COMMIT) -> int:
    """Earliest sampled step after which every later p(k) < thr; else the last step.

    Reading used (W1-A1): the earliest sampled step k with p(k') < thr for every sampled
    k' >= k (from k on, renoising rescues less than thr). If no step qualifies (p of the last
    step >= thr), the last sampled step."""
    steps = list(steps)
    p = list(p)
    kc = steps[-1]
    for j in range(len(steps) - 1, -1, -1):
        if p[j] < thr:
            kc = steps[j]
        else:
            break
    return int(kc)


# ---------------------------------------------------------------- branch tool
def _rescue_matrix(cell, rows: np.ndarray, actions: torch.Tensor, per: int) -> np.ndarray:
    """(m, per) bool rescue of ``actions`` laid out member-major (member j owns rows j*per..)."""
    out = np.zeros((len(rows), per), dtype=bool)
    for j, i in enumerate(rows):
        out[j] = is_rescue(cell, int(i), actions[j * per:(j + 1) * per])
    return out


def _branch_p(cell, version, rows, states, k, n_per, g):
    """Rescue shares (m,) of n_per renoise branches per member from step k."""
    from . import policy
    m = len(rows)
    x = states[k].repeat_interleave(n_per, 0)
    c = cell.neg.c[torch.as_tensor(rows)].repeat_interleave(n_per, 0)
    st = renoise(version, FlowState(x=x, k=k, c=c), g)
    act = policy.decode(version, st.x, c, start_step=k).action
    return _rescue_matrix(cell, rows, act, n_per).reshape(m, n_per)


def primitive_rates(cell, version, rows: np.ndarray, states: torch.Tensor,
                    prims: Sequence[Primitive]) -> List[float]:
    """Rescue rate (share of members) of each primitive applied to every member's logged
    decode under ``version`` (states: (21, m, 2) logged states)."""
    from . import policy
    rows = np.asarray(rows)
    c = cell.neg.c[torch.as_tensor(rows)]
    rates = []
    for p in prims:
        if p.kind == "hop":
            act = restore_and_decode(version, FlowState(x=states[0], k=0, c=c), p)
        else:
            act = policy.decode(version, states[p.k], c, start_step=p.k, after_step=_primitive_hook(p, len(rows))).action
        rates.append(float(np.mean([is_rescue(cell, int(i), act[j:j + 1])[0] for j, i in enumerate(rows)])))
    return rates


def branch(cell, version, cluster, steps: Sequence[int] = DEFAULT_STEPS, M: int = M_BRANCH,
           primitives: Optional[List[Primitive]] = None, seed: int = 0) -> BranchTable:
    """The branch tool on all members of ``cluster`` (``toyrepair.miner.Cluster``).

    For each step: p(k) with M renoise branches per member (+64 where the Wilson interval
    contains 0.2). Then k_commit, then each primitive at k_commit and the two sampled steps
    before it; class R/U. Fresh noise comes from the mining pool stream. Counts decodes.

    ``primitives``: if given, these replace the default list (a push's own k is used as given;
    the default pushes are built at k_commit and the two sampled steps before it, the default
    hops at step 0). p(k)'s Wilson interval is over all member x branch outcomes pooled.
    Renoise eps come from ``pools.torch_generator(cell, MINING, stream=2, salt=seed)``."""
    from . import policy, pools
    t0 = policy.counter.total
    rows = np.asarray(cluster.members)
    steps = sorted(int(k) for k in steps)
    if not steps or steps[0] < 0 or steps[-1] > 19:
        raise ValueError("steps must lie in 0..19")
    m = len(rows)
    xi, c = cell.neg.xi[torch.as_tensor(rows)], cell.neg.c[torch.as_tensor(rows)]
    states = policy.decode(version, xi, c, save_states=True).states        # (21, m, 2)
    g = pools.torch_generator(cell, pools.MINING, pools.STREAM_BRANCH, salt=seed)
    ps, cis, ns, member_p = [], [], [], {}
    for k in steps:
        hits = _branch_p(cell, version, rows, states, k, M, g)
        n = M
        p_hat = float(hits.mean())
        lo, hi = wilson(p_hat, m * n)
        if lo <= P_COMMIT <= hi:
            extra = _branch_p(cell, version, rows, states, k, M_EXTRA, g)
            hits = np.concatenate([hits, extra], 1)
            n = M + M_EXTRA
            p_hat = float(hits.mean())
            lo, hi = wilson(p_hat, m * n)
        ps.append(p_hat)
        cis.append((lo, hi))
        ns.append(n)
        member_p[k] = hits.mean(1)
    kc = k_commit(steps, ps)
    j = steps.index(kc)
    prim_steps = steps[max(0, j - 2):j + 1]
    if primitives is None:
        prims = []
        for k in prim_steps:
            prims += default_primitives(cell, rows, k)
        prims += [p for p in default_primitives(cell, rows, 0, version=version) if p.kind == "hop"]
    else:
        prims = list(primitives)
    rates = primitive_rates(cell, version, rows, states, prims) if prims else []
    by_step: Dict[int, List[Tuple[str, float]]] = {}
    for p, r in zip(prims, rates):
        by_step.setdefault(p.k, []).append((p.name, r))
    table_rows = []
    for k, p_hat, ci, n in zip(steps, ps, cis, ns):
        best = sorted(by_step.get(k, []), key=lambda z: -z[1])[:TOP_PRIMITIVES]
        table_rows.append(BranchRow(k=k, t=1.0 - k / 20.0, p=p_hat, ci=ci, n=n, best=tuple(best)))
    for k in sorted(set(by_step) - set(steps)):          # primitives at unsampled steps
        best = sorted(by_step[k], key=lambda z: -z[1])[:TOP_PRIMITIVES]
        table_rows.append(BranchRow(k=k, t=1.0 - k / 20.0, p=float("nan"), ci=(float("nan"),) * 2, n=0,
                                    best=tuple(best)))
    table_rows.sort(key=lambda r: r.k)
    eligible = [(r, p) for p, r in zip(prims, rates) if p.k <= kc]
    best_r, best_p = max(eligible, key=lambda z: z[0]) if eligible else (0.0, None)
    fclass = "R" if best_r >= RESCUE_CLASS_R else "U"
    return BranchTable(cluster=int(cluster.id), rows=table_rows[:TABLE_ROW_CAP], k_commit=kc, failure_class=fclass,
                       best_primitive=best_p if fclass == "R" else None,
                       n_decodes=int(round(policy.counter.total - t0)), n_members=m, M=M,
                       meta={"best_rate": best_r, "prim_steps": prim_steps, "n_primitives": len(prims),
                             "member_p": {int(k): v.tolist() for k, v in member_p.items()},
                             "rows_cut": max(0, len(table_rows) - TABLE_ROW_CAP),
                             "best_name": best_p.name if best_p is not None else None})


# ---------------------------------------------------------------- inspect tool (6.4)
INSPECT_ROW_CAP = 21


@dataclass
class InspectTable:
    """One decode of one negative under a version, <= 21 rows.

    uid: negative uid; rows: dicts with step, t, x (2,), x_hat (2,), v_norm, in_bad (bool),
    and one entry per selected unit "<layer>[<idx>]" (activation at that step; NaN at step 20,
    where no network call happens)."""
    uid: int
    version_id: str
    rows: List[Dict]
    units: Tuple[Tuple[str, int], ...] = ()
    n_decodes: int = 1

    def render(self) -> str:
        unames = [f"{l}[{i}]" for l, i in self.units]
        head = f"inspect negative {self.uid} under {self.version_id}: x_hat = x_k - t_k v; in_B = x_hat in B_i"
        cols = "step  t_k   x_k              x_hat            |v|     in_B" + "".join(f"  {u:>14}" for u in unames)
        lines = [head, cols]
        for r in self.rows[:INSPECT_ROW_CAP]:
            s = (f"{r['step']:>4}  {r['t']:.2f}  ({r['x'][0]:+.3f},{r['x'][1]:+.3f})  "
                 f"({r['x_hat'][0]:+.3f},{r['x_hat'][1]:+.3f})  {r['v_norm']:6.3f}  {'yes' if r['in_bad'] else 'no ':>4}")
            s += "".join(f"  {r[u]:14.4f}" for u in unames)
            lines.append(s)
        return "\n".join(lines)


def row_of_uid(cell, uid: int) -> int:
    """Row position in ``cell.neg`` of negative ``uid`` (KeyError if absent)."""
    hit = np.flatnonzero(cell.neg.uid == int(uid))
    if len(hit) != 1:
        raise KeyError(f"no logged negative with uid {uid}")
    return int(hit[0])


def inspect(cell, version, uid: int, units: Sequence[Tuple[str, int]] = (), every: int = 1) -> InspectTable:
    """The inspect tool: one decode of negative ``uid`` from its logged noise under ``version``.

    units: (layer, index) pairs from ``policy.CAPTURE_LAYERS``; every: keep steps 0, every,
    2*every, ... and step 20 (<= 21 rows for every >= 1)."""
    from . import policy
    i = row_of_uid(cell, uid)
    units = tuple((str(l), int(j)) for l, j in units)
    cap = tuple(sorted({l for l, _ in units}, key=policy.CAPTURE_LAYERS.index))
    d = policy.decode(version, cell.neg.xi[i:i + 1], cell.neg.c[i:i + 1], save_states=True, capture=cap)
    keep = sorted(set(range(0, 21, max(1, int(every)))) | {20})
    xs = d.states[:, 0]
    rows = []
    for k in keep:
        if k < 20:
            v = d.v[k, 0]
            t = 1.0 - k / 20.0
            xh = xs[k] - t * v
            vn = float(v.norm())
        else:
            t, xh, vn = 0.0, xs[20], 0.0
        r = {"step": k, "t": t, "x": xs[k].tolist(), "x_hat": xh.tolist(), "v_norm": vn,
             "in_bad": bool(cell.in_bad_basin(i, xh.reshape(1, 2))[0])}
        for l, j in units:
            r[f"{l}[{j}]"] = float(d.acts[l][k, 0, j]) if k < 20 else float("nan")
        rows.append(r)
    return InspectTable(uid=int(uid), version_id=version.id, rows=rows[:INSPECT_ROW_CAP], units=units)
