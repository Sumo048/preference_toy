"""Pinned experiment constants. Everything that changes a number lives here."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Tuple

from .flow import PolicyConfig
from .mixture import MixtureConfig
from .train import BaseTrainConfig, EditTrainConfig

ROOT = os.environ.get("TOYPREF_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#: /home is nearly full; run outputs live on the 1 TB rollout volume
RUNS = os.environ.get("TOYPREF_RUNS", "/grogu/datasets/rollout/soumojib/toypref/runs")


@dataclass(frozen=True)
class SweepConfig:
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    n_forget: Tuple[int, ...] = (1, 2, 3, 4)          # of 8 modes
    n_neg: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)     # logged negatives PER forgotten mode


@dataclass(frozen=True)
class ExperimentConfig:
    mixture: MixtureConfig = field(default_factory=MixtureConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    base: BaseTrainConfig = field(default_factory=BaseTrainConfig)
    edit: EditTrainConfig = field(default_factory=EditTrainConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    pool_size: int = 100_000          # pi_ref rollouts the edit sets are cut from
    n_pos: int = 512                  # |T_p|, fixed across the sweep
    n_eval: int = 20_000              # common eval noises
    n_test_per_mode: int = 128        # held-out TRUE samples per mode, for log-density
    n_ideal: int = 20_000             # true samples from the retained modes only
    n_oracle: int = 20_000            # pi_ref samples in retained modes (ORACLE arm)
    online_pool: int = 20_000         # pi_1 rollouts for the online loop's round 2
    n_anchor: int = 20_000            # unlabelled pi_ref (xi, a) pairs for the anchors
    d0_in_std: float = 4.0            # nf exclusion radius = d0_in_std * mode_std

    def to_dict(self):
        return asdict(self)


DEFAULT = ExperimentConfig()


def base_path(seed: int, mode_std: float = None) -> str:
    """Base policy for a seed; a non-default mode width gets its own base dir."""
    if mode_std is None or abs(mode_std - MixtureConfig().mode_std) < 1e-12:
        return os.path.join(RUNS, "base", f"s{seed}", "policy.pt")
    return os.path.join(RUNS, f"base_std{mode_std:g}", f"s{seed}", "policy.pt")


def parse_regime(regime: str) -> dict:
    """'std=0.45,cover=0.5,contam=0.2' -> dict; '' -> {} (the main sweep's setting)."""
    out = {}
    for part in filter(None, (regime or "").split(",")):
        k, v = part.split("=")
        if k not in ("std", "cover", "contam"):
            raise ValueError(f"unknown regime key {k!r}")
        out[k] = float(v)
    return out


def regime_tag(regime: str) -> str:
    d = parse_regime(regime)
    return "base" if not d else "_".join(f"{k}{v:g}" for k, v in sorted(d.items()))


def cell_name(seed: int, n_forget: int, n_neg: int) -> str:
    return f"s{seed}_k{n_forget}_n{n_neg}"
