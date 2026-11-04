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


class DecodeCounter:
    """Process-wide decode counter (6.1: the policy server counts decodes per run).

    ``total``: decodes since the last ``reset``; ``by_version``: {version id: decodes}. Callers
    that need per-run totals read ``total`` before and after a tool call (or ``reset`` first)."""

    def __init__(self):
        self.total = 0.0
        self.by_version: Dict[str, float] = {}

    def add(self, version_id: str, n: float) -> None:
        self.total += float(n)
        self.by_version[version_id] = self.by_version.get(version_id, 0.0) + float(n)

    def reset(self) -> float:
        """Zero the counter; returns the total before the reset."""
        t = self.total
        self.total = 0.0
        self.by_version = {}
        return t


counter = DecodeCounter()


# ---------------------------------------------------------------- network introspection
def _core(net: nn.Module) -> Tuple[nn.Module, str]:
    """(module that owns inp/blocks/out, its name prefix). The unconditional wrapper
    (``toyrepair.cell._UncondNet``) keeps the toypref VelocityNet in ``.inner``; a grafted
    ``toyrepair.edits.EditedNet`` keeps its weight-edited base net in ``.base``."""
    from .edits import EditedNet
    if isinstance(net, EditedNet):
        core, prefix = _core(net.base)
        return core, "base." + prefix
    if hasattr(net, "inp") and hasattr(net, "blocks"):
        return net, ""
    inner = getattr(net, "inner", None)
    if isinstance(inner, nn.Module) and hasattr(inner, "inp"):
        return inner, "inner."
    if isinstance(inner, nn.Module):
        # nested wrappers, e.g. R0's ControlledNet around an earlier ControlledNet
        # (W3-1 bug fix, PREREG Amendment 1: no method change)
        core, prefix = _core(inner)
        return core, "inner." + prefix
    raise ValueError("unknown velocity-network layout (no inp/blocks)")


def _is_cond(net: nn.Module) -> bool:
    return hasattr(_core(net)[0], "c_mlp")


class _Capture:
    """Forward hooks that record captured layers at every network call (bit-identical: hooks
    only copy outputs). ``out_act`` is SiLU(out_norm) recomputed with the same functional op as
    the forward; ``cond`` is cat(time_mlp, c_mlp) (time only for the unconditional net)."""

    def __init__(self, net: nn.Module, names: Sequence[str]):
        self.net = net
        self.names = tuple(names)
        bad = [n for n in self.names if n not in CAPTURE_LAYERS]
        if bad:
            raise ValueError(f"unknown capture layers {bad}; choose from {CAPTURE_LAYERS}")
        self.calls: Dict[str, List[torch.Tensor]] = {n: [] for n in self.names}
        self._handles = []
        self._tmp: Dict[str, torch.Tensor] = {}

    def __enter__(self):
        core, _ = _core(self.net)
        for n in self.names:
            if n == "cond":
                self._hook(core.time_mlp, "_t")
                if hasattr(core, "c_mlp"):
                    self._hook(core.c_mlp, "_c")
                    self._handles.append(core.c_mlp.register_forward_hook(self._cond_done_cond))
                else:
                    self._handles.append(core.time_mlp.register_forward_hook(self._cond_done_uncond))
            elif n == "out_act":
                self._handles.append(core.out_norm.register_forward_hook(
                    lambda m, i, o: self.calls["out_act"].append(torch.nn.functional.silu(o).detach().clone())))
            else:
                mod = core.get_submodule(n)
                self._handles.append(mod.register_forward_hook(
                    lambda m, i, o, n=n: self.calls[n].append(o.detach().clone())))
        return self

    def _hook(self, mod, key):
        self._handles.append(mod.register_forward_hook(lambda m, i, o, key=key: self._tmp.__setitem__(key, o.detach())))

    def _cond_done_cond(self, m, i, o):
        self.calls["cond"].append(torch.cat([self._tmp.pop("_t"), self._tmp.pop("_c")], -1).clone())

    def _cond_done_uncond(self, m, i, o):
        self.calls["cond"].append(self._tmp.pop("_t").clone())

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def _net_call(net: nn.Module, x, t, c, capture: Sequence[str]):
    """net(x, t, c) plus captured activations of this one call."""
    if not capture:
        return net(x, t, c), {}
    from .edits import EditedNet, run_forward
    if isinstance(net, EditedNet):          # grafted: hooks cannot see the gated paths
        return run_forward(net, x, t, c, capture=tuple(capture))
    fc = getattr(net, "forward_capture", None)
    if callable(fc):                       # optional protocol for nets that do their own capture
        return fc(x, t, c, tuple(capture))
    with _Capture(net, capture) as cap:
        v = net(x, t, c)
    return v, {n: cap.calls[n][-1] for n in capture}


# ---------------------------------------------------------------- public API
def base_version(cell) -> Version:
    """v0: pi_ref of ``cell`` (``cell.ref.net``, deep-copied), no patches."""
    import copy
    net = copy.deepcopy(cell.ref.net).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return Version(id="v0", parent=None, patches=(), net=net, grid=cell.grid)


@torch.no_grad()
def velocity(version: Version, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor,
             capture: Sequence[str] = ()) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One network call. x (B, 2), t (B,), c (B,) float32 -> (v (B, 2), {layer: (B, d)}).
    Bit-identical to ``version.net(x, t, c)``. Counts 1/20 decode per row."""
    v, acts = _net_call(version.net, x, t, c, capture)
    counter.add(version.id, x.shape[0] / N_STEPS)
    return v, acts


@torch.no_grad()
def decode(version: Version, xi: torch.Tensor, c: torch.Tensor, save_states: bool = False,
           capture: Sequence[str] = (), start_step: int = 0, batch: int = 65536,
           after_step=None, count: bool = True) -> Decode:
    """Decode noises to actions with the toypref Euler sampler.

    xi: (B, 2) float32 state at ``start_step`` (the noise if 0); c: (B,) float32.
    Steps ``start_step .. 19`` run. With start_step = 0 and no capture the action equals
    ``CondFlow.sample(xi, c)`` bit for bit (same op order, same batch split).
    capture: layer names from CAPTURE_LAYERS. Counts B decodes.

    Extra keywords (W1-A1): ``after_step(k, x, rows) -> x`` is called after the Euler update of
    step k (x is then x_{k+1}; rows is the slice of the batch) and may return a modified x
    (primitives use it); ``count=False`` skips the decode counter (tests only).
    With ``start_step > 0`` the entries of ``states`` / ``v`` / ``acts`` before start_step are NaN.

    Batch caveat (measured in W0): CPU matmuls are not batch-size invariant. pi_ref decoding
    the same (xi, c) inside a batch of 200,000 and inside a batch of 4 differs by up to ~4e-7.
    Bit-identity tests (branch validity, graft identity) must compare decodes made with the
    same batch composition and size."""
    if not 0 <= start_step <= N_STEPS:
        raise ValueError("start_step must be in 0..20")
    B = xi.shape[0]
    net = version.net
    capture = tuple(capture)
    outs, st_chunks, v_chunks = [], [], []
    act_chunks: Dict[str, List[torch.Tensor]] = {n: [] for n in capture}
    K, dt = N_STEPS, 1.0 / N_STEPS
    for i in range(0, B, batch):
        x = xi[i:i + batch].clone().float()
        cc = c[i:i + batch].float()
        n = x.shape[0]
        rows = slice(i, i + n)
        if save_states:
            st = torch.full((K + 1, n, 2), float("nan"))
            vv = torch.full((K, n, 2), float("nan"))
            st[start_step] = x
        acts_k: Dict[str, List[Optional[torch.Tensor]]] = {nm: [None] * K for nm in capture}
        for k in range(start_step, K):
            t = torch.full((n,), 1.0 - k * dt)
            v, a = _net_call(net, x, t, cc, capture)
            x = x - dt * v
            if after_step is not None:
                x = after_step(k, x, rows)
            if save_states:
                vv[k] = v
                st[k + 1] = x
            for nm in capture:
                acts_k[nm][k] = a[nm]
        outs.append(x)
        if save_states:
            st_chunks.append(st)
            v_chunks.append(vv)
        for nm in capture:
            d = next(z.shape[1] for z in acts_k[nm] if z is not None) if start_step < K else 0
            full = torch.full((K, n, d), float("nan"))
            for k in range(start_step, K):
                full[k] = acts_k[nm][k]
            act_chunks[nm].append(full)
    if B == 0:
        action = xi.clone().float()
    else:
        action = torch.cat(outs)
    if count:
        counter.add(version.id, B)
    res = Decode(action=action, n_decodes=float(B))
    if save_states and B:
        res.states = torch.cat(st_chunks, 1)
        res.v = torch.cat(v_chunks, 1)
    res.acts = {nm: torch.cat(act_chunks[nm], 1) for nm in capture} if B else {}
    return res


def layers(version: Version) -> List[LayerInfo]:
    """All capture layers of ``version`` with sizes, in forward order (grafted paths appear as
    ``<layer>.graft{j}`` with kind "graft"). A layer wrapped by a graft (a module exposing the
    original Linear as ``.base``) reports the original Linear's sizes."""
    core, prefix = _core(version.net)
    width = int(core.out_norm.normalized_shape[0])
    cond_dim = core.time_mlp[-1].out_features + (core.c_mlp[-1].out_features if hasattr(core, "c_mlp") else 0)
    out: List[LayerInfo] = []
    for n in CAPTURE_LAYERS:
        if n == "cond":
            out.append(LayerInfo(n, "cond", 0, cond_dim))
        elif n == "out_act" or n.endswith(".act"):
            out.append(LayerInfo(n, "act", 0, width))
        elif n == "out_norm":
            out.append(LayerInfo(n, "norm", 0, width))
        elif n.startswith("blocks.") and n.count(".") == 1:
            out.append(LayerInfo(n, "block", 0, width))
        else:
            m = core.get_submodule(n)
            lin = m if isinstance(m, nn.Linear) else getattr(m, "base", None)
            if not isinstance(lin, nn.Linear):
                raise ValueError(f"layer {n} is not a Linear (and has no .base Linear)")
            out.append(LayerInfo(n, "linear", lin.in_features, lin.out_features))
    from .edits import grafts_of            # W1-B graft modules (EditedNet.grafts)
    for g in grafts_of(version.net):
        out.append(LayerInfo(f"{g.layer}.graft_{g.gid}", "graft", 0, int(g.n_units)))
    return out


def load_version(run_dir: str, version_id: str, cell) -> Version:
    """Rebuild a version from pi_ref and its patch stack on disk (``toyrepair.patches.rebuild``).

    The patch stack is found by walking ``parent_version`` links of the patch JSON files in
    ``<run_dir>/patches/`` from ``version_id`` back to v0."""
    import json
    import os
    if version_id == "v0":
        return base_version(cell)
    pdir = os.path.join(run_dir, "patches")
    by_version = {}
    for f in sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []:
        if f.endswith(".json"):
            with open(os.path.join(pdir, f)) as fh:
                d = json.load(fh)
            by_version[d["version"]] = (d["id"], d["parent_version"])
    chain, v = [], version_id
    while v != "v0":
        if v not in by_version:
            raise FileNotFoundError(f"no patch makes version {v} in {pdir}")
        pid, v = by_version[v]
        chain.append(pid)
    from . import patches
    return patches.rebuild(cell, run_dir, list(reversed(chain)), version_id)
