"""The oracle: mode labels, purity, Removed, Collateral, D, toypref evaluation wrapper.

Owner: W1-A2.

ISOLATION (docs/PLAN.md 4.9, 6.6, 7.5): only ``toyrepair.gate`` and ``toyrepair.eval_final``
(and their tests) import this package, plus the miner's purity logging, which goes through
``purity`` below into a CSV/ledger only, never into a card. The CLI, dossier, sandbox and every
LLM-facing path must not import it. The repo directory is mode 700 so the ``sandbox`` user
cannot read it, and the sandbox audit hook blocks ``import toyrepair.oracle``.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .labels import OFF, N_MODES, counts, label, masses, negative_modes, target_modes  # noqa: F401


def of(cell):
    """The private oracle record of a cell (``toyrepair.cell._CellOracle``)."""
    return cell._oracle


def forget_modes(cell) -> List[int]:
    """Sorted forgotten mode indices of the cell."""
    return sorted(int(m) for m in cell._oracle.forget)


def retain_modes(cell) -> List[int]:
    """Sorted retained mode indices of the cell."""
    return sorted(int(m) for m in cell._oracle.retain)


def purity(cell, uids: Sequence[int]) -> float:
    """Cluster purity in [0, 1]: share of the members (by negative uid) whose oracle mode equals
    the cluster's majority mode. For the miner's CSV / ledger only (docs/PLAN.md 6.2).
    An empty cluster has purity nan."""
    m = negative_modes(cell, uids)
    if m.size == 0:
        return float("nan")
    _, cnt = np.unique(m, return_counts=True)
    return float(cnt.max() / m.size)
