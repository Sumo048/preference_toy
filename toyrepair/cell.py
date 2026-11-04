"""One experiment cell: pi_ref, logged negatives, basins and the bad vote (docs/PLAN.md 5.1-5.3, 6.11).

``load_cell(grid, setting, seed, npm)`` rebuilds a cell exactly as toypref does:

* conditional (grid "cond"): ``scripts/cond_cell.py``: pi_ref from
  ``$TOYPREF_RUNS/cond_base/s<seed>/policy.pt``; ``make_pool(ref, seed, 200_000)``;
  ``editset(pool, seed, k, npm, layout)``.
* unconditional (grid "uncond"): ``scripts/run_cell.py``: pi_ref from
  ``$TOYPREF_RUNS/base/s<seed>/policy.pt``; ``make_pool(ref, mix, seed, 100_000)``;
  ``build_editset(pool, mix, seed, k, npm, n_pos=512)``. The unconditional policy is wrapped
  so it has the conditional interface ``net(x, t, c)`` / ``sample(xi, c)``; c is ignored and
  every logged negative has c = 0.

Basins (``toypref.cond.targets.basins_at`` with ``row_gen(seed, uid, 1)``) and the bad vote
(``toypref.cond.targets.NegBank``) are built lazily on first access and are bit-identical to
the ones toypref's frozen recipe uses.

Privacy: the forgotten/retained mode sets, the mode labels and the toypref ``Edit`` object live
in ``Cell._oracle`` (repr hidden). Only ``toyrepair.oracle`` reads it (``oracle.of(cell)``).
Nothing else in toyrepair may touch ``_oracle``.

Negative order: ``cell.neg`` keeps toypref's order (grouped by forgotten mode), which the
toypref arms need for bit-exact reproduction. Anything shown to an LLM role must use
``cell.public_order`` (sorted by uid) or uid-keyed ids, never the toypref row position.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

from . import RUNS  # noqa: F401  (sets TOYPREF_RUNS before toypref.config is imported)

GRIDS = ("cond", "uncond")
COND_SETTINGS = ("k1", "k2", "k4", "contig", "alt", "pairs", "b3p1")
UNCOND_SETTINGS = ("base_k1", "base_k2", "base_k4")
NPMS = (1, 4, 16, 64)
#: setting -> (n_forget, layout)
COND_SPEC = {"k1": (1, "random"), "k2": (2, "random"), "k4": (4, "random"),
             "contig": (4, "contig"), "alt": (4, "alt"), "pairs": (4, "pairs"), "b3p1": (4, "b3p1")}
UNCOND_SPEC = {"base_k1": 1, "base_k2": 2, "base_k4": 4}
COND_POOL_SIZE = 200_000
UNCOND_N_POS = 512


def cell_index(grid: str, setting: str, seed: int, npm: int) -> int:
    """1000*g + 100*s + 10*seed + n (PLAN.md 5.3). g: cond 0, uncond 1; s: position of the
    setting in COND_SETTINGS / UNCOND_SETTINGS; n: position of npm in (1, 4, 16, 64)."""
    if grid not in GRIDS:
        raise ValueError(f"grid must be one of {GRIDS}")
    settings = COND_SETTINGS if grid == "cond" else UNCOND_SETTINGS
    if setting not in settings:
        raise ValueError(f"unknown {grid} setting {setting!r}; expected one of {settings}")
    if npm not in NPMS:
        raise ValueError(f"npm must be one of {NPMS}")
    if not 0 <= seed <= 9:
        raise ValueError("seed must be in 0..9")
    return 1000 * GRIDS.index(grid) + 100 * settings.index(setting) + 10 * int(seed) + NPMS.index(npm)


def cell_key(grid: str, setting: str, seed: int, npm: int) -> str:
    """Human-readable id, e.g. ``cond_k4_s0_n4``."""
    return f"{grid}_{setting}_s{seed}_n{npm}"


@dataclass
class LoggedNegatives:
    """The cell's logged negatives (c_i, xi_i, a_i), in toypref order.

    uid: (N,) int64, row index in toypref's rollout pool (stable id of a negative)
    c:   (N,) float32, condition in [0, 1) (all 0 for the unconditional toy)
    xi:  (N, 2) float32, logged noise
    a:   (N, 2) float32, pi_ref action = decode(pi_ref, xi, c)
    """
    uid: np.ndarray
    c: torch.Tensor
    xi: torch.Tensor
    a: torch.Tensor

    def __len__(self) -> int:
        return int(self.uid.shape[0])


@dataclass
class _CellOracle:
    """Private. Read only through ``toyrepair.oracle``."""
    forget: List[int]
    retain: List[int]
    neg_modes: np.ndarray          # (N,) oracle mode of each logged negative
    toypref_edit: Any              # toypref Edit (cond) or EditSet (uncond)
    pool: Any                      # toypref Pool (cond) / RolloutPool (uncond) incl. mode labels


class _UncondNet(nn.Module):
    """Conditional interface over an unconditional VelocityNet: forward(x, t, c) ignores c."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.inner = net

    def forward(self, x, t, c=None):
        return self.inner(x, t)


class UncondAsCond(nn.Module):
    """Wraps toypref ``FlowPolicy`` so it has ``CondFlow``'s interface: ``.net(x, t, c)``,
    ``.sample(xi, c)``, ``.n_euler_steps`` (= 20). The condition is ignored.

    Layer names differ from the conditional net: no ``c_mlp``, ``cond`` is 64-dim (time only),
    and each block's ``fc1`` reads 128 + 64 = 192 inputs."""
    n_euler_steps = 20

    def __init__(self, pol):
        super().__init__()
        self.pol = pol
        self.net = _UncondNet(pol.net)
        self.kw = {"uncond": True}

    @torch.no_grad()
    def sample(self, xi, c=None, batch=65536):
        return self.pol.sample(xi, batch=batch)


@dataclass
class Cell:
    """One (grid, setting, seed, npm) cell.

    grid:     "cond" | "uncond"
    setting:  e.g. "k4" (cond) or "base_k2" (uncond)
    seed:     0..9 (0-4 development, 5-9 confirmation)
    npm:      negatives per forgotten mode, one of 1, 4, 16, 64
    index:    cell index of PLAN.md 5.3 (seeds the toyrepair pools, see ``toyrepair.pools``)
    n_forget: number of forgotten modes (public: it fixes the number of logged negatives)
    layout:   forget-set layout name ("random" for k1/k2/k4 and uncond)
    ref:      pi_ref, frozen, eval mode; ``ref.net(x, t, c) -> v`` (B, 2) and
              ``ref.sample(xi, c) -> a`` (B, 2), 20 Euler steps t: 1 -> 0
    neg:      logged negatives (toypref order)
    """
    grid: str
    setting: str
    seed: int
    npm: int
    index: int
    n_forget: int
    layout: str
    ref: nn.Module
    neg: LoggedNegatives
    _oracle: _CellOracle = field(repr=False)
    _bank: Optional[Any] = field(default=None, repr=False)

    @property
    def key(self) -> str:
        return cell_key(self.grid, self.setting, self.seed, self.npm)

    @property
    def public_order(self) -> np.ndarray:
        """(N,) row positions of ``neg`` sorted by uid; use this order for anything an LLM sees."""
        return np.argsort(self.neg.uid, kind="stable")

    @property
    def vote(self):
        """The bad vote: ``toypref.cond.targets.NegBank`` over all logged negatives (lazy).
        ``vote.rate(z (P,2)) -> (P,)`` share of negatives j with pi_ref(c_j, z) in B_j;
        ``vote.theta`` its threshold; ``vote.basins[i]`` the basin of negative i."""
        if self._bank is None:
            from toypref.cond import targets as TG
            n = self.neg
            self._bank = TG.NegBank(self.ref, self.seed, n.uid, n.c, n.xi, n.a)
        return self._bank

    @property
    def basins(self) -> list:
        """List of N ``toypref.cond.targets.Basins`` (lazy, shares the vote's construction).
        ``basins[i].assign(x (P,2)) -> (P,)`` basin cluster id (-1 = none); B_i is
        ``assign(x) == basins[i].bad``. Built from 128 pi_ref decodes at c_i."""
        return self.vote.basins

    def in_bad_basin(self, i: int, x: torch.Tensor) -> np.ndarray:
        """(P,) bool: is each action x (P, 2) in B_i, the bad basin of negative row i?"""
        b = self.basins[i]
        return b.assign(x) == b.bad

    def toypref_ctx(self, steps: int = 5000):
        """toypref ``Ctx`` for running toypref arms on this cell exactly as
        ``scripts/cond_cell.py`` does (cond only). For code arms (B1a, B1b, B4, B5); it carries
        the forget set, so never pass it to an LLM-facing path."""
        if self.grid != "cond":
            raise NotImplementedError("toypref_ctx is defined for the conditional toy only")
        from toypref.cond import experiment as E
        o = self._oracle
        anchor = E.make_pool(self.ref, self.seed, 20_000, purpose=E.P_ANCHOR)
        return E.Ctx(ref=self.ref, seed=self.seed, ed=o.toypref_edit, pool=o.pool, anchor=anchor, steps=steps)


def _load_cond(setting: str, seed: int, npm: int, runs: str) -> Cell:
    from toypref.cond import experiment as E
    from toypref.cond.task import CondFlow, frozen
    k, layout = COND_SPEC[setting]
    path = os.path.join(runs, "cond_base", f"s{seed}", "policy.pt")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    ref = CondFlow(**sd["kw"])
    ref.load_state_dict(sd["state"])
    ref = frozen(ref)
    pool = E.make_pool(ref, seed, COND_POOL_SIZE)
    ed = E.editset(pool, seed, k, npm, layout)
    assert len(set(ed.c.tolist())) == len(ed.c), "conditions must be unique"
    neg = LoggedNegatives(uid=np.asarray(ed.uid, dtype=np.int64), c=ed.c, xi=ed.xi, a=ed.a)
    orc = _CellOracle(forget=list(ed.forget), retain=list(ed.retain), neg_modes=pool.mode[ed.uid],
                      toypref_edit=ed, pool=pool)
    return Cell(grid="cond", setting=setting, seed=seed, npm=npm, index=cell_index("cond", setting, seed, npm),
                n_forget=k, layout=layout, ref=ref, neg=neg, _oracle=orc)


def _load_uncond(setting: str, seed: int, npm: int, runs: str) -> Cell:
    from toypref.config import DEFAULT
    from toypref.editset import build_editset, make_pool
    from toypref.flow import load_policy
    k = UNCOND_SPEC[setting]
    mix = DEFAULT.mixture
    pol = load_policy(os.path.join(runs, "base", f"s{seed}", "policy.pt"))
    pool = make_pool(pol, mix, seed, DEFAULT.pool_size)
    es = build_editset(pool, mix, seed, k, npm, UNCOND_N_POS)
    ref = UncondAsCond(pol).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    n = len(es.neg_idx)
    neg = LoggedNegatives(uid=np.asarray(es.neg_idx, dtype=np.int64), c=torch.zeros(n), xi=es.xi_neg, a=es.a_neg)
    orc = _CellOracle(forget=list(es.forget), retain=list(es.retain), neg_modes=np.asarray(es.mode_neg),
                      toypref_edit=es, pool=pool)
    return Cell(grid="uncond", setting=setting, seed=seed, npm=npm, index=cell_index("uncond", setting, seed, npm),
                n_forget=k, layout="random", ref=ref, neg=neg, _oracle=orc)


def load_cell(grid: str, setting: str, seed: int, npm: int, runs: Optional[str] = None) -> Cell:
    """Load one cell. ``runs`` defaults to ``$TOYPREF_RUNS`` (set by ``import toyrepair``).

    Cost: one pool decode (200k cond / 100k uncond, ~seconds on CPU). Basins and the vote are
    built lazily (``cell.vote``: 128 decodes per negative plus N*(N-1) for the threshold).
    """
    cell_index(grid, setting, seed, npm)  # validates
    runs = runs or os.environ["TOYPREF_RUNS"]
    if grid == "cond":
        return _load_cond(setting, seed, npm, runs)
    return _load_uncond(setting, seed, npm, runs)
