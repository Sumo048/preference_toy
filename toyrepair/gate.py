"""The gate: accept or reject an edit with six fixed tests (docs/PLAN.md 10). Owner: W1-A2.

One of only two modules allowed to import ``toyrepair.oracle``.

Each gate check draws a fresh batch of N_G (c, xi) pairs from the GATE pool
(``toyrepair.pools.draw(cell, pools.GATE, N_G, index=check_index)``), decodes it with pi_ref,
the current and the new version through ``toyrepair.policy.decode``, and labels each action
with ``toyrepair.oracle.label``. Intervals: paired bootstrap over the batch, 2,000 resamples
(numpy Generator seeded from the gate pool stream, so a check is reproducible).

Target modes: ``oracle.target_modes(cell, uids)`` (cluster scope) or the union over all
first-atlas clusters (global scope). Masses are over the whole batch (all c).

Tests (all must pass):
  1 target:   mass on target modes falls by >= 50% of current, 95% lower bound of fall > 0
  2 progress: one-sided 95% upper bound of D(new) - D(current) < 0
  3 drift:    one-sided 95% upper bound of Collateral(new, pi_ref) <= 0.10
  4 other:    one-sided 95% UB of rise in mass on non-target forgotten modes <= 0.01
              (cluster scope only; passed and marked n/a in global scope)
  5 between:  one-sided 95% UB of rise in between-mode mass <= 0.01
  6 unit:     ``toyrepair.edits.unit_tests(new, current)`` (owner W1-B): shapes, no NaN,
              detectors read inputs/activations only, graft identity for each graft
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence, Tuple, Union

import numpy as np

N_G = 16_384
N_BOOT = 2_000
TARGET_FALL = 0.5
DRIFT_MAX = 0.10
OTHER_RISE_MAX = 0.01
BETWEEN_RISE_MAX = 0.01
ALPHA = 0.05
SCOPES = ("cluster", "global")
TEST_NAMES = ("target", "progress", "drift", "other_forgotten", "between", "unit")


@dataclass(frozen=True)
class GateConfig:
    """Thresholds (development values; Phase 3 may loosen tests 3-5 once to 0.15/0.02/0.02;
    Phase 5 freezes). ``accept_all`` is the A3-nogate arm: every test is still measured."""
    n_g: int = N_G
    n_boot: int = N_BOOT
    target_fall: float = TARGET_FALL
    drift_max: float = DRIFT_MAX
    other_rise_max: float = OTHER_RISE_MAX
    between_rise_max: float = BETWEEN_RISE_MAX
    accept_all: bool = False


@dataclass
class GateResult:
    """Result of one gate check.

    accepted:       all applicable tests passed (or cfg.accept_all)
    passed:         {test name in TEST_NAMES: bool}
    target_fall:    test 1 number: paired fall in target-mode mass, share of the batch
                    (mass_current - mass_new; positive = fewer target actions)
    target_fall_ci: 95% paired-bootstrap interval (lo, hi) of target_fall
    numbers:        every gate number (masses per mode for ref/current/new, D, Collateral,
                    bounds, rises) for the ledger; never shown to the surgeon
    check_index:    gate-pool check index used; n_g: batch size; scope: "cluster" | "global"
    cpu_s:          wall-clock seconds of the check
    """
    accepted: bool
    passed: Dict[str, bool]
    target_fall: float
    target_fall_ci: Tuple[float, float]
    numbers: Dict[str, float] = field(default_factory=dict)
    check_index: int = 0
    n_g: int = N_G
    scope: str = "cluster"
    cpu_s: float = 0.0

    def surgeon_view(self) -> Dict:
        """What the surgeon sees (docs/PLAN.md 7.2): pass/fail per test plus test 1's number and CI."""
        return {"accepted": self.accepted, "passed": dict(self.passed),
                "target_fall": self.target_fall, "target_fall_ci": list(self.target_fall_ci)}


def gate_check(cell, base, current, new, target_uids: Union[Sequence[int], Sequence[Sequence[int]]],
               check_index: int, scope: str = "cluster", cfg: GateConfig = GateConfig()) -> GateResult:
    """Run one gate check.

    cell: ``toyrepair.cell.Cell``; base / current / new: ``toyrepair.policy.Version`` (pi_ref,
    current version, candidate). target_uids: the target cluster's member negative uids
    (cluster scope) or one uid list per first-atlas cluster (global scope).
    check_index: the run's gate-check counter (0, 1, ...); it seeds the gate pool.
    The caller (orchestrate / cli) increments the budget and writes the ledger record."""
    raise NotImplementedError


def paired_bootstrap(stat_fn: Callable[[np.ndarray], float], n: int, n_boot: int,
                     rng: np.random.Generator) -> np.ndarray:
    """(n_boot,) float64 replicates of ``stat_fn(idx)``, idx (n,) int64 row indices drawn with
    replacement; the same idx serves every version in one replicate (paired)."""
    raise NotImplementedError
