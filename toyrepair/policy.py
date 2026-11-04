"""Policy server: versions, decodes, activation capture (docs/PLAN.md 6.1).

A ``Version`` is pi_ref plus a stack of accepted patches (``toyrepair.patches``). Its ``net``
is a velocity network with the conditional interface ``net(x (B,2), t (B,), c (B,)) -> v (B,2)``
(possibly with gated paths grafted in). Decoding is the toypref sampler, bit for bit:
20 Euler steps, step k uses t_k = 1 - k/20 and does x_{k+1} = x_k - (1/20) * v(x_k, t_k, c).
Step 0 state is the noise xi, step 20 state is the action.

Layer names (conditional net; ``capture`` names)::

    cond                (B, 128)  cat(time_mlp(emb(t)) [64], c_mlp(fourier(c)) [64])
    inp                 (B, 128)  Linear(2 -> 128) on x_k
    blocks.{i}.fc1      (B, 128)  Linear(256 -> 128) on cat(LayerNorm(h), cond), pre-activation
    blocks.{i}.act      (B, 128)  SiLU(fc1)
    blocks.{i}.fc2      (B, 128)  Linear(128 -> 128) on act
    blocks.{i}          (B, 128)  block output h + fc2   (i = 0, 1, 2)
    out_norm            (B, 128)  LayerNorm(h), pre-activation
    out_act             (B, 128)  SiLU(out_norm)
    out                 (B, 2)    Linear(128 -> 2) = velocity v
Unconditional net: same names except ``cond`` is (B, 64) (time only) and fc1 reads 192 inputs.

Decode counting: every decode of one (xi, c) pair through all remaining steps counts 1 toward
the run's budget (``toyrepair.budget``); a single velocity call counts 1/20.

Owner: W1-A1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

N_STEPS = 20
DT = 1.0 / N_STEPS
CAPTURE_LAYERS: Tuple[str, ...] = (
    "cond", "inp",
    "blocks.0.fc1", "blocks.0.act", "blocks.0.fc2", "blocks.0",
    "blocks.1.fc1", "blocks.1.act", "blocks.1.fc2", "blocks.1",
    "blocks.2.fc1", "blocks.2.act", "blocks.2.fc2", "blocks.2",
    "out_norm", "out_act", "out",
)
#: layers an edit can target (weight-bearing Linear layers)
EDITABLE_LAYERS: Tuple[str, ...] = ("inp", "blocks.0.fc1", "blocks.0.fc2", "blocks.1.fc1", "blocks.1.fc2",
                                    "blocks.2.fc1", "blocks.2.fc2", "out")


def t_of_step(k: int) -> float:
    """t_k = 1 - k/20 for step k in 0..20."""
    return 1.0 - k * DT


@dataclass(frozen=True)
class LayerInfo:
    """name: capture name; kind: "linear" | "act" | "norm" | "block" | "cond";
    in_dim / out_dim: feature sizes (in_dim = 0 for non-linear kinds)."""
    name: str
    kind: str
    in_dim: int
    out_dim: int


@dataclass
class Version:
    """One policy state.

    id:       version id, e.g. ``v0`` (= pi_ref) or ``v3``; unique within a run
    parent:   parent version id (None for v0)
    patches:  patch ids applied on top of pi_ref, oldest first (``toyrepair.patches``)
    net:      velocity network, eval mode, requires_grad False; forward(x, t, c) -> v
    grid:     "cond" | "uncond"
    meta:     free-form (edit call, accepted flag, ...)
    """
    id: str
    parent: Optional[str]
    patches: Tuple[str, ...]
    net: nn.Module
    grid: str = "cond"
    meta: Dict = field(default_factory=dict)


@dataclass
class Decode:
    """Result of ``decode``.

    action:   (B, 2) float32, state at step 20
    states:   (21, B, 2) float32 x_k for k = 0..20 if ``save_states`` else None
    v:        (20, B, 2) float32 velocity at steps 0..19 if ``save_states`` else None
    acts:     {layer: (20, B, d) float32} activations at steps 0..19 for each captured layer
    n_decodes: number of decodes counted (= B)
    """
    action: torch.Tensor
    states: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    acts: Dict[str, torch.Tensor] = field(default_factory=dict)
    n_decodes: float = 0.0


def base_version(cell) -> Version:
    """v0: pi_ref of ``cell`` (``cell.ref.net``, deep-copied), no patches."""
    raise NotImplementedError


def velocity(version: Version, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor,
             capture: Sequence[str] = ()) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One network call. x (B, 2), t (B,), c (B,) float32 -> (v (B, 2), {layer: (B, d)}).
    Bit-identical to ``version.net(x, t, c)``. Counts 1/20 decode per row."""
    raise NotImplementedError


def decode(version: Version, xi: torch.Tensor, c: torch.Tensor, save_states: bool = False,
           capture: Sequence[str] = (), start_step: int = 0, batch: int = 65536) -> Decode:
    """Decode noises to actions with the toypref Euler sampler.

    xi: (B, 2) float32 state at ``start_step`` (the noise if 0); c: (B,) float32.
    Steps ``start_step .. 19`` run. With start_step = 0 and no capture the action equals
    ``CondFlow.sample(xi, c)`` bit for bit (same op order, same batch split).
    capture: layer names from CAPTURE_LAYERS. Counts B decodes.

    Batch caveat (measured in W0): CPU matmuls are not batch-size invariant. pi_ref decoding
    the same (xi, c) inside a batch of 200,000 and inside a batch of 4 differs by up to ~4e-7.
    Bit-identity tests (branch validity, graft identity) must compare decodes made with the
    same batch composition and size."""
    raise NotImplementedError


def layers(version: Version) -> List[LayerInfo]:
    """All capture layers of ``version`` with sizes, in forward order (grafted paths appear as
    ``<layer>.graft{j}`` with kind "graft")."""
    raise NotImplementedError


def load_version(run_dir: str, version_id: str, cell) -> Version:
    """Rebuild a version from pi_ref and its patch stack on disk (``toyrepair.patches.rebuild``)."""
    raise NotImplementedError
