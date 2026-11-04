"""targeted_train: gradient updates on a parameter subset with the locality penalty (docs/PLAN.md 2, 6.7). Owner: W1-B.

Objectives (toypref losses):
  "bc"   flow-matching BC to rescue targets (toypref rerouting rows; supports "log" | "nhood" | "bsup")
  "npo"  NPO on the selected negatives (toypref ``objectives.npo_term``), beta
  "kto"  KTO undesirable-only, beta
Anchors (locality penalty, toypref): "dist" (vote-masked density anchor, default), "map",
"local", "local+map", "none". Loss = 0.5 * (objective + anchor) as in toypref.
Optimiser and schedule as toypref ``train_arm``: AdamW(lr 1e-4, betas (0.9, 0.95)), cosine,
grad-clip 1.0, batch 256, minibatch stream gen(seed, P_TRAIN).

Reproduction: ``targeted_train`` with params ("all",), objective "bc", anchor "dist",
steps 5000 and data ``toypref_rows(cell, all uids, "hopB", "bsup")`` on parent v0 runs the same
float32 operations in the same order as toypref ``train_arm(ARMS["R_hopB_bsup_neg_AMdist"])``,
so the weights match bit for bit (tests/toyrepair/test_train.py).

Details (W1-B):
* NPO/KTO reference = the PARENT version (equal to pi_ref at v0, so toypref's arms reproduce).
* Anchors are built from pi_ref and ALL of the cell's logged negatives (the fixed reference
  set), whatever subset ``data`` trains on: anchor pool = toypref ``make_pool(ref, seed, 20000,
  P_ANCHOR)`` without mode labels; "dist"/"map" keep its first 4096 samples whose bad vote is
  below theta. No oracle is used anywhere in this module.
* Rules "own" (oracle geometry) and transplant supports are not offered.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

OBJECTIVES = ("bc", "npo", "kto")
ANCHORS = ("dist", "map", "local", "local+map", "none")
MAX_STEPS = 5000
PARAM_GROUPS = ("all", "inp", "blocks.0", "blocks.1", "blocks.2", "out", "out_norm", "grafts")
RULES = ("ours", "basin", "hop", "pu", "hopV", "hopB", "hopVB")
SUPPORTS = ("log", "nhood", "bsup")
N_VOTE_POOL = 4096
N_ANCHOR_POOL = 20_000


@dataclass
class TrainData:
    """Training rows for "bc": c (R,), noise (R, 2), target (R, 2) float32; or for "npo"/"kto":
    negatives by uid in ``neg_uids``. ``source`` documents the
    builder (e.g. "toypref_hopB_bsup", "rescue_targets", "sandbox:<file>").
    ``decodes``: pi_ref decodes the builder used (budget)."""
    c: Optional[torch.Tensor] = None
    noise: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    neg_uids: Optional[np.ndarray] = None
    source: str = ""
    decodes: float = 0.0


@dataclass(frozen=True)
class TrainSpec:
    """params: names from PARAM_GROUPS or explicit parameter names (e.g. "blocks.2.fc2.weight");
    objective in OBJECTIVES; steps <= MAX_STEPS; anchor in ANCHORS; beta (npo/kto); lr; batch."""
    params: tuple
    objective: str
    steps: int
    anchor: str = "dist"
    beta: float = 10.0
    lr: float = 1e-4
    batch: int = 256


# =============================================================== decode counting
class _Count:
    """Counts pi_ref decodes (rows through ``cell.ref.sample``) inside a ``with`` block."""

    def __init__(self, ref):
        self.ref, self.n = ref, 0

    def __enter__(self):
        orig = self.ref.sample
        self._had = "sample" in self.ref.__dict__      # nested counters: restore the outer one
        self._orig = orig

        def counted(xi, *a, **k):
            self.n += int(xi.shape[0])
            return orig(xi, *a, **k)
        self.ref.sample = counted
        return self

    def __exit__(self, *exc):
        if self._had:
            self.ref.sample = self._orig
        else:
            self.ref.__dict__.pop("sample", None)      # the class method shows through again
        return False


def _rows_of(cell, uids: np.ndarray) -> np.ndarray:
    """Row positions of ``uids`` in cell.neg, in toypref order."""
    uids = np.asarray(uids, dtype=np.int64).reshape(-1)
    known = set(cell.neg.uid.tolist())
    missing = [int(u) for u in uids if int(u) not in known]
    if missing:
        raise ValueError(f"unknown negative uids: {missing[:5]}")
    return np.flatnonzero(np.isin(cell.neg.uid, uids))


def _c_of(cell, rows):
    return cell.neg.c[torch.from_numpy(rows)]


# =============================================================== data builders
def rescue_rows(cell, uids: np.ndarray, targets: torch.Tensor, support: str = "bsup") -> TrainData:
    """BC rows from rescue targets: targets (m, 2), one per negative uid, with toypref supports
    ("log" | "nhood" | "bsup", streams ``row_gen(seed, uid, 2)`` as toypref)."""
    from toypref.cond import targets as TG
    if support not in SUPPORTS:
        raise ValueError(f"support must be one of {SUPPORTS}")
    uids = np.asarray(uids, dtype=np.int64).reshape(-1)
    targets = torch.as_tensor(np.asarray(targets, dtype=np.float32)).reshape(-1, 2)
    if targets.shape[0] != uids.shape[0]:
        raise ValueError("one target per uid")
    if not torch.isfinite(targets).all():
        raise ValueError("targets contain NaN/inf")
    pos = {int(u): i for i, u in enumerate(cell.neg.uid.tolist())}
    cs, ns, ts = [], [], []
    with _Count(cell.ref) as cnt:
        for u, tgt in zip(uids.tolist(), targets):
            if u not in pos:
                raise ValueError(f"unknown negative uid {u}")
            i = pos[u]
            ci, xi = float(cell.neg.c[i]), cell.neg.xi[i]
            gs = TG.row_gen(cell.seed, u, 2)
            if support == "log":
                sup = xi.reshape(1, 2)
            elif support == "nhood":
                sup = torch.cat([xi.reshape(1, 2), xi + TG.NHOOD_SIGMA * torch.randn(TG.NHOOD_K, 2, generator=gs)])
            else:
                b = cell.basins[i]
                jit = xi + TG.BSUP_SIGMA * torch.randn(TG.BSUP_K, 2, generator=gs)
                keep = b.assign(cell.ref.sample(jit, torch.full((TG.BSUP_K,), ci))) == b.bad
                sup = torch.cat([xi.reshape(1, 2), jit[torch.from_numpy(keep)][:TG.BSUP_KEEP]])
            cs += [ci] * len(sup)
            ns += list(sup)
            ts += [tgt] * len(sup)
    return TrainData(c=torch.tensor(cs, dtype=torch.float32), noise=torch.stack(ns), target=torch.stack(ts),
                     source=f"rescue_targets_{support}", decodes=float(cnt.n))


def toypref_rows(cell, uids: np.ndarray, rule: str = "hopB", support: str = "bsup", mine: bool = False) -> TrainData:
    """BC rows built by ``toypref.cond.targets.build`` for the negatives ``uids`` (rule in
    ours | basin | hop | pu | hopV | hopB | hopVB). With all uids, rule hopB and support
    bsup this reproduces the frozen recipe's rows bit for bit. ``mine``: add toypref's mined
    pseudo-negatives (vote-flagged anchor-pool samples, ``build_mined``) as in the ``_mine`` arms."""
    from toypref.cond import targets as TG
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES} ('own' uses oracle geometry and is not offered)")
    if support not in SUPPORTS:
        raise ValueError(f"support must be one of {SUPPORTS}")
    rows = _rows_of(cell, uids)
    if rows.size == 0:
        raise ValueError("no negatives selected")
    n = cell.neg
    r = torch.from_numpy(rows)
    with _Count(cell.ref) as cnt:
        pu = _pu_classifier(cell) if rule == "pu" else None
        R = TG.build(cell.ref, None, cell.seed, n.uid[rows], n.c[r], n.xi[r], n.a[r], rule, support,
                     transplant=False, tx_conditions=None, d0=None, pu=pu, forget=None)
        if mine:
            P = anchor_pool(cell)
            keep = vote_keep(cell)
            flagged = torch.nonzero(~keep).flatten()[:256]
            M = TG.build_mined(cell.ref, None, cell.seed, cell.vote, P["c"][flagged], P["xi"][flagged],
                               P["a"][flagged], uid_offset=50_000_000, forget=None)
            if M.c.shape[0]:
                R = TG.Rows(c=torch.cat([R.c, M.c]), noise=torch.cat([R.noise, M.noise]),
                            target=torch.cat([R.target, M.target]), diag=R.diag)
    return TrainData(c=R.c, noise=R.noise, target=R.target, neg_uids=n.uid[rows].copy(),
                     source=f"toypref_{rule}_{support}{'_mine' if mine else ''}", decodes=float(cnt.n))


def negatives(cell, uids: np.ndarray) -> TrainData:
    """NPO/KTO data: the logged negatives ``uids``."""
    rows = _rows_of(cell, uids)
    return TrainData(neg_uids=cell.neg.uid[rows].copy(), source="negatives")


def _pu_classifier(cell):
    from types import SimpleNamespace
    from toypref.cond import experiment as E
    cache = cell.__dict__.setdefault("_train_cache", {})
    if "pu" not in cache:
        upool = _make_pool(cell, 20_000, E.P_PU + 1)
        ed = SimpleNamespace(c=cell.neg.c, a=cell.neg.a)
        pool = SimpleNamespace(c=upool["c"], a=upool["a"])
        cache["pu"] = E.train_pu(ed, pool, cell.seed, prior=cell.n_forget / 8)
    return cache["pu"]


# =============================================================== anchors (no oracle)
def _make_pool(cell, n: int, purpose: int) -> Dict[str, torch.Tensor]:
    """toypref ``make_pool`` without mode labels (same streams, same decode batch)."""
    from toypref.cond import experiment as E
    g = E.gen(cell.seed, purpose)
    c = torch.rand(n, generator=g)
    xi = torch.randn(n, 2, generator=g)
    if cell.grid != "cond":
        c = torch.zeros(n)
    a = cell.ref.sample(xi, c)
    return {"c": c, "xi": xi, "a": a}


def anchor_pool(cell) -> Dict[str, torch.Tensor]:
    """The toypref anchor pool (20,000 pi_ref pairs, purpose P_ANCHOR); cached on the cell."""
    from toypref.cond import experiment as E
    cache = cell.__dict__.setdefault("_train_cache", {})
    if "anchor" not in cache:
        cache["anchor"] = _make_pool(cell, N_ANCHOR_POOL, E.P_ANCHOR)
    return cache["anchor"]


def vote_keep(cell) -> torch.Tensor:
    """(4096,) bool: anchor-pool samples NOT flagged by the bad vote (toypref ``get_vote``)."""
    cache = cell.__dict__.setdefault("_train_cache", {})
    if "keep" not in cache:
        P = anchor_pool(cell)
        bank = cell.vote
        v = bank.rate(P["xi"][:N_VOTE_POOL])
        cache["keep"] = torch.from_numpy(~(v >= bank.theta).numpy())
    return cache["keep"]


def _local(cell):
    from toypref.cond import targets as TG
    cache = cell.__dict__.setdefault("_train_cache", {})
    if "local" not in cache:
        n = cell.neg
        cache["local"] = TG.local_anchor_triplets(cell.ref, cell.seed, n.uid, n.c, n.a)
    return cache["local"]


def _flow_loss(net, a, noise, t, c):
    """toypref ``CondFlow.flow_loss`` on a bare net (same ops)."""
    tt = t.reshape(-1, 1)
    x_t = tt * noise + (1 - tt) * a
    return ((net(x_t, t, c) - (noise - a)) ** 2).mean(-1)


def _anchor_loss(cell, kind: str, B: int):
    from toypref.flow import FlowPolicy
    parts = []
    if kind in ("map", "dist", "local+map"):
        P, keep = anchor_pool(cell), vote_keep(cell)
        parts.append((P["c"][:N_VOTE_POOL][keep], P["xi"][:N_VOTE_POOL][keep], P["a"][:N_VOTE_POOL][keep]))
    if kind in ("local", "local+map"):
        parts.append(_local(cell))
    c = torch.cat([p[0] for p in parts])
    xi = torch.cat([p[1] for p in parts])
    a = torch.cat([p[2] for p in parts])
    mode = "dist" if kind == "dist" else "map"

    def loss(net, g):
        j = torch.randint(0, c.shape[0], (B,), generator=g)
        t = FlowPolicy.sample_time(B, g)
        noise = xi[j] if mode == "map" else torch.randn(a[j].shape, generator=g)
        return _flow_loss(net, a[j], noise, t, c[j]).mean()
    return loss


def _objective(cell, spec: TrainSpec, data: TrainData, ref_net):
    from toypref import objectives as O
    from toypref.flow import FlowPolicy
    B = spec.batch
    if spec.objective == "bc":
        if data.c is None or data.noise is None or data.target is None or data.c.shape[0] == 0:
            raise ValueError("objective bc needs rows (c, noise, target)")
        C, N, T = data.c.float(), data.noise.float(), data.target.float()
        if not (torch.isfinite(N).all() and torch.isfinite(T).all() and torch.isfinite(C).all()):
            raise ValueError("training rows contain NaN/inf")

        def bc(net, g):
            i = torch.randint(0, C.shape[0], (B,), generator=g)
            t = FlowPolicy.sample_time(B, g)
            return _flow_loss(net, T[i], N[i], t, C[i]).mean()
        return bc
    rows = _rows_of(cell, data.neg_uids if data.neg_uids is not None else cell.neg.uid)
    r = torch.from_numpy(rows)
    A, Cn = cell.neg.a[r], cell.neg.c[r]

    def lr(net, g):
        i = torch.randint(0, A.shape[0], (B,), generator=g)
        a, c = A[i], Cn[i]
        noise = torch.randn(a.shape, generator=g)
        t = FlowPolicy.sample_time(B, g)
        l_th = _flow_loss(net, a, noise, t, c)
        with torch.no_grad():
            l_ref = _flow_loss(ref_net, a, noise, t, c)
        rr = l_ref - l_th
        if spec.objective == "npo":
            return O.npo_term(rr, spec.beta).mean()
        k = min(8, B)
        with torch.no_grad():
            ar, nr = torch.roll(a, 1, 0)[:k], torch.roll(noise, 1, 0)[:k]
            z0 = torch.clamp((_flow_loss(ref_net, ar, nr, t[:k], c[:k]) - _flow_loss(net, ar, nr, t[:k], c[:k])).mean(), min=0.0)
        return (1.0 - torch.sigmoid(spec.beta * (z0 - rr))).mean()
    return lr


# =============================================================== parameter selection
def _core_name(name: str) -> str:
    for p in ("base.", "inner."):
        if name.startswith(p):
            name = name[len(p):]
    return name


def select_params(net, params: Sequence[str]) -> List[str]:
    """Full parameter names of ``net`` selected by group names / explicit names, in
    ``named_parameters`` order."""
    names = [n for n, _ in net.named_parameters()]
    out = set()
    for p in params:
        if p == "all":
            out |= set(names)
        elif p == "grafts":
            out |= {n for n in names if n.startswith("grafts.")}
        elif p in PARAM_GROUPS:
            out |= {n for n in names if not n.startswith("grafts.") and _core_name(n).startswith(p + ".")}
        else:
            hit = {n for n in names if n == p or (not n.startswith("grafts.") and _core_name(n) == p)}
            if not hit:
                raise ValueError(f"unknown parameter or group {p!r}; groups: {', '.join(PARAM_GROUPS)}")
            out |= hit
    if not out:
        raise ValueError("no parameters selected")
    return [n for n in names if n in out]


def validate_spec(spec: TrainSpec) -> None:
    if spec.objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if spec.anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}")
    if not 1 <= int(spec.steps) <= MAX_STEPS:
        raise ValueError(f"steps must be in 1..{MAX_STEPS}")
    if not (0 < spec.lr <= 1e-2) or not (1 <= spec.batch <= 4096) or not (0 < spec.beta <= 1000):
        raise ValueError("lr in (0, 1e-2], batch in 1..4096, beta in (0, 1000]")


# =============================================================== targeted_train
def targeted_train(cell, parent, new_id: str, spec: TrainSpec, data: TrainData, patch_id: Optional[str] = None,
                   probe=None):
    """Train a copy of ``parent`` (``policy.Version``) on ``data`` with ``spec``; the anchor is
    built from ``cell`` (pi_ref, vote). Deterministic given (cell.seed, spec, data).
    -> ``toyrepair.edits.EditResult`` (call "targeted_train"; the patch stores the full
    changed tensors). ``probe(net) -> dict`` (optional) is called every 1000 steps (history in
    ``record.args["history"]``)."""
    from toypref.cond import experiment as E
    from toypref.train import cosine
    from . import edits
    args = {"params": list(spec.params), "objective": spec.objective, "steps": int(spec.steps),
            "anchor": spec.anchor, "beta": spec.beta, "lr": spec.lr, "batch": spec.batch,
            "data": data.source, "n_rows": int(data.c.shape[0]) if data.c is not None else 0,
            "neg_uids": [int(u) for u in data.neg_uids] if data.neg_uids is not None else None}
    try:
        validate_spec(spec)
        with _Count(cell.ref) as cnt:
            ref_net = parent.net
            fn = _objective(cell, spec, data, ref_net)
            if spec.anchor != "none":
                al = _anchor_loss(cell, spec.anchor, spec.batch)

                def loss_fn(net, g):
                    return 0.5 * (fn(net, g) + al(net, g))
            else:
                loss_fn = fn
            torch.manual_seed(cell.seed)
            net = copy.deepcopy(parent.net)
            sel = select_params(net, spec.params)
    except (ValueError, edits.EditRejected) as e:
        return edits._rejected(parent, "targeted_train", args, str(e))
    net.train()
    named = dict(net.named_parameters())
    for n_, p in named.items():
        p.requires_grad_(n_ in sel)
    plist = [named[n_] for n_ in sel]
    opt = torch.optim.AdamW(plist, lr=spec.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    g = E.gen(cell.seed, E.P_TRAIN)
    hist = []
    for step in range(int(spec.steps)):
        lr = cosine(step, spec.steps, min(500, spec.steps // 10), spec.lr, 0.1 * spec.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        loss = loss_fn(net, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plist, 1.0)
        opt.step()
        if probe is not None and (step % 1000 == 0 or step == spec.steps - 1):
            net.eval()
            with torch.no_grad():
                hist.append({"step": step, "loss": float(loss.detach()), **probe(net)})
            net.train()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    pd = dict(parent.net.named_parameters())
    arrays = {n_: named[n_].detach().numpy().copy() for n_ in sel if not torch.equal(named[n_], pd[n_])}
    if not torch.isfinite(torch.cat([torch.as_tensor(a).reshape(-1) for a in arrays.values()] or [torch.zeros(1)])).all():
        return edits._rejected(parent, "targeted_train", args, "training diverged (NaN/inf weights)")
    if hist:
        args["history"] = hist
    res = edits._finish(parent, new_id, "targeted_train", args, arrays, patch_id,
                        decodes=float(cnt.n) + float(data.decodes))
    return res
