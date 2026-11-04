"""targeted_train: gradient updates on a parameter subset with the locality penalty (docs/PLAN.md 2, 6.7). Owner: W1-B.

Objectives (toypref losses):
  "bc"   flow-matching BC to rescue targets (toypref rerouting rows; supports "log" | "nhood" | "bsup")
  "npo"  NPO on the selected negatives (toypref ``objectives.npo_term``), beta
  "kto"  KTO undesirable-only, beta
Anchors (locality penalty, toypref): "dist" (vote-masked density anchor, default), "map",
"local", "local+map", "none". Loss = 0.5 * (objective + anchor) as in toypref.
Optimiser and schedule as toypref ``train_arm``: AdamW(lr 1e-4, betas (0.9, 0.95)), cosine,
grad-clip 1.0, batch 256, minibatch stream gen(seed, P_TRAIN).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

OBJECTIVES = ("bc", "npo", "kto")
ANCHORS = ("dist", "map", "local", "local+map", "none")
MAX_STEPS = 5000
PARAM_GROUPS = ("all", "inp", "blocks.0", "blocks.1", "blocks.2", "out", "out_norm", "grafts")


@dataclass
class TrainData:
    """Training rows for "bc": c (R,), noise (R, 2), target (R, 2) float32; or for "npo"/"kto":
    negatives by uid in ``neg_uids``. ``source`` documents the
    builder (e.g. "toypref_hopB_bsup", "rescue_targets", "sandbox:<file>")."""
    c: Optional[torch.Tensor] = None
    noise: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    neg_uids: Optional[np.ndarray] = None
    source: str = ""


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


def rescue_rows(cell, uids: np.ndarray, targets: torch.Tensor, support: str = "bsup") -> TrainData:
    """BC rows from rescue targets: targets (m, 2), one per negative uid, with toypref supports
    ("log" | "nhood" | "bsup", streams ``row_gen(seed, uid, 2)`` as toypref)."""


def toypref_rows(cell, uids: np.ndarray, rule: str = "hopB", support: str = "bsup") -> TrainData:
    """BC rows built by ``toypref.cond.targets.build`` for the negatives ``uids`` (rule in
    ours | own | basin | hop | pu | hopV | hopB | hopVB). With all uids, rule hopB and support
    bsup this reproduces the frozen recipe's rows bit for bit."""
    raise NotImplementedError


def targeted_train(cell, parent, new_id: str, spec: TrainSpec, data: TrainData):
    """Train a copy of ``parent`` (``policy.Version``) on ``data`` with ``spec``; the anchor is
    built from ``cell`` (pi_ref, vote). Deterministic given (cell.seed, spec, data).
    -> ``toyrepair.edits.EditResult`` (call "targeted_train"; the patch stores the full
    changed tensors)."""
    raise NotImplementedError
