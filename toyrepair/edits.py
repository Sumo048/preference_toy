"""Edit API and the gate's unit-test hook (docs/PLAN.md 6.7, 10.2 test 6). Owner: W1-B.

Every edit call is a pure function of the parent ``toyrepair.policy.Version``: it returns an
``EditResult`` holding the new Version (id ``new_id``, chosen by the caller), the ``Patch``
to store (``toyrepair.patches.save_patch``) and the ``EditRecord`` for the ledger. The caller
(cli / orchestrate) assigns ids, saves the patch, counts the budget and writes the ledger.

Calls:
  graft_gated_path(layer, detector, n_units, w_in, w_out)
      n_units hidden units beside ``layer``: z = act(w_in @ src + b_in) * 1[detector fires];
      w_out @ z is ADDED to ``layer``'s output. Hard gate: exactly 0 when off, so with the
      detector off the velocity is bit-identical to the parent's.
  rank1(layer, u, v, s)                   W <- W + s * u v^T
  ablate_direction(layer, v, side)        v normalised. "pre": W <- (I - v v^T) W, b <- (I - v v^T) b
                                          "post": the next layer's input weights W_next <- W_next (I - v v^T)
  set_units(layer, idx, w_in, b, w_out)   write selected units' input rows/bias and outgoing columns
  targeted_train(...)                     ``toyrepair.train.targeted_train``

Layers: ``toyrepair.policy.EDITABLE_LAYERS`` (inp, blocks.{0,1,2}.fc1, blocks.{0,1,2}.fc2, out).
Detector sources: network inputs ("x", "t", "c", "inputs") and activations
(``toyrepair.policy.CAPTURE_LAYERS``, which includes "cond"); anything else is rejected.
Array arguments arrive as float32 numpy arrays (the CLI loads ``.npy`` files from scratch).

Per edit the record holds the fraction of weights changed, units added, the Frobenius norm of
the change, and decodes used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EDIT_CALLS = ("graft_gated_path", "rank1", "ablate_direction", "set_units", "targeted_train")
WEIGHT_CALLS = ("graft_gated_path", "rank1", "ablate_direction", "set_units")   # arm A2
TRAIN_CALLS = ("targeted_train",)                                              # arm A1
INPUT_SOURCES = ("x", "t", "c", "inputs")   # "inputs" = cat(x, t, c), 4 features
GRAFT_ACTS = ("silu", "relu", "linear")
N_IDENTITY_STATES = 10_000


@dataclass(frozen=True)
class Detector:
    """Linear detector: fires where w . features(source) + b > threshold (strict).

    source: one of INPUT_SOURCES or a name in policy.CAPTURE_LAYERS (activation at the same
    network call); w: (d,) float32 over that source's features; b: bias; threshold: scalar.
    k_range: optional (k_lo, k_hi): also require k_lo <= k < k_hi, evaluated from the network
    input t as k = round((1 - t) * 20) (t is a network input, so this is legal)."""
    source: str
    w: Tuple[float, ...]
    b: float = 0.0
    threshold: float = 0.0
    k_range: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class GatedPath:
    """A graft. layer: EDITABLE layer whose output receives the path; src: capture layer or
    input source read by w_in (default: the layer's own input); detector; n_units;
    w_in (n_units, d_src); b_in (n_units,); w_out (d_out(layer), n_units); act in GRAFT_ACTS."""
    layer: str
    src: str
    detector: Detector
    n_units: int
    w_in: np.ndarray
    b_in: np.ndarray
    w_out: np.ndarray
    act: str = "silu"


@dataclass
class EditRecord:
    """What one edit call did (ledger payload).

    call: one of EDIT_CALLS; args: JSON-safe arguments (arrays by scratch-relative path and
    sha256); parent / version: version ids; patch_id; frac_weights_changed in [0, 1] (share of
    pi_ref's parameter entries that differ from the parent); units_added; frob: Frobenius norm
    of the weight change over all changed tensors; decodes: decodes used; ok / error."""
    call: str
    args: Dict
    parent: str
    version: Optional[str]
    patch_id: Optional[str]
    frac_weights_changed: float = 0.0
    units_added: int = 0
    frob: float = 0.0
    decodes: float = 0.0
    ok: bool = True
    error: str = ""


@dataclass
class EditResult:
    """version: the new ``policy.Version`` (None if rejected); patch: ``patches.Patch`` (None if
    rejected); record: ``EditRecord``."""
    version: Optional[object]
    patch: Optional[object]
    record: EditRecord


@dataclass(frozen=True)
class UnitTestResult:
    """Gate test 6. passed: all checks pass; checks: {"shapes", "finite", "legal_detectors",
    "graft_identity": bool}; detail: short reason per failed check."""
    passed: bool
    checks: Dict[str, bool]
    detail: Dict[str, str] = field(default_factory=dict)


class EditRejected(ValueError):
    """Illegal edit: unknown layer, illegal detector source, wrong shapes, NaN/inf."""


def validate_detector(det: Detector, version) -> None:
    """Raise EditRejected unless the detector reads network inputs or activations only and its
    w has the source's feature size."""
    raise NotImplementedError


def graft_gated_path(parent, new_id: str, layer: str, detector: Detector, n_units: int,
                     w_in: np.ndarray, w_out: np.ndarray, b_in: Optional[np.ndarray] = None,
                     src: Optional[str] = None, act: str = "silu") -> EditResult:
    raise NotImplementedError


def rank1(parent, new_id: str, layer: str, u: np.ndarray, v: np.ndarray, s: float) -> EditResult:
    """u (d_out,), v (d_in,), s scalar: W <- W + s u v^T."""
    raise NotImplementedError


def ablate_direction(parent, new_id: str, layer: str, v: np.ndarray, side: str) -> EditResult:
    """side "pre" (v in the layer's output space) | "post" (v in the next layer's input space)."""
    raise NotImplementedError


def set_units(parent, new_id: str, layer: str, idx: Sequence[int], w_in: np.ndarray, b: np.ndarray,
              w_out: np.ndarray) -> EditResult:
    """idx (m,) unit indices of ``layer``; w_in (m, d_in); b (m,); w_out (d_next_out, m) written
    into the next linear layer's input columns (fc1 -> fc2; fc2/inp -> residual stream is not
    a unit layer: rejected)."""
    raise NotImplementedError


def graft_identity(base, new, n_states: int = N_IDENTITY_STATES, seed: int = 0) -> bool:
    """With every graft's detector forced off, velocities of ``new`` (minus its non-graft weight
    changes) and ``base`` agree bit for bit on n_states reference flow states."""
    raise NotImplementedError


def unit_tests(new, current) -> UnitTestResult:
    """Gate test 6 hook, called by ``toyrepair.gate.gate_check``: shapes, finite parameters
    and outputs, legal detectors, graft identity for each graft new relative to current."""
    raise NotImplementedError
