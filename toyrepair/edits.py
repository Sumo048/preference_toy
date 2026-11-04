"""Edit API and the gate's unit-test hook (docs/PLAN.md 6.7, 10.2 test 6). Owner: W1-B.

Every edit call is a pure function of the parent ``toyrepair.policy.Version``: it returns an
``EditResult`` holding the new Version (id ``new_id``, chosen by the caller), the ``Patch``
to store (``toyrepair.patches.save_patch``) and the ``EditRecord`` for the ledger. The caller
(cli / orchestrate) assigns ids, saves the patch, counts the budget and writes the ledger.
The new version is built by ``patches.apply_patch(parent, patch)``, the same code that
``patches.rebuild`` replays, so a rebuilt version equals the live one bit for bit.

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
(``toyrepair.policy.CAPTURE_LAYERS``, which includes "cond", plus the layer-input names
``blocks.{i}.fc1.in``); anything else is rejected.
Array arguments arrive as float32 numpy arrays (the CLI loads ``.npy`` files from scratch).

Per edit the record holds the fraction of weights changed, units added, the Frobenius norm of
the change, and decodes used.

Implementation notes (W1-B):

* A version without grafts keeps toypref's own module (``CondVelocityNet`` or the uncond
  wrapper); weight edits change its tensors in place on a deep copy. A version with grafts
  has ``net = EditedNet(base, grafts)``. ``run_forward(net, x, t, c, capture)`` is the one
  forward used for grafted nets and for activation capture; on a plain net it is bit-identical
  to ``net(x, t, c)`` (same op order; tested).
* ``next layer`` for ablate "post": inp -> blocks.0.fc1 (its first 128 input columns, the
  normalised residual stream); blocks.i.fc1 -> blocks.i.fc2 (the SiLU output);
  blocks.i.fc2 -> blocks.{i+1}.fc1 first 128 columns, or ``out`` for i = 2; ``out`` has none.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EDIT_CALLS = ("graft_gated_path", "rank1", "ablate_direction", "set_units", "targeted_train")
WEIGHT_CALLS = ("graft_gated_path", "rank1", "ablate_direction", "set_units")   # arm A2
TRAIN_CALLS = ("targeted_train",)                                              # arm A1
INPUT_SOURCES = ("x", "t", "c", "inputs")   # "inputs" = cat(x, t, c), 4 features
GRAFT_ACTS = ("silu", "relu", "linear")
N_IDENTITY_STATES = 10_000
N_STEPS = 20
MAX_GRAFT_UNITS = 512
EDITABLE = ("inp", "blocks.0.fc1", "blocks.0.fc2", "blocks.1.fc1", "blocks.1.fc2",
            "blocks.2.fc1", "blocks.2.fc2", "out")
UNIT_LAYERS = ("blocks.0.fc1", "blocks.1.fc1", "blocks.2.fc1")


def _forward_order(n_blocks: int = 3) -> Tuple[str, ...]:
    o = ["x", "t", "c", "inputs", "cond", "inp"]
    for i in range(n_blocks):
        o += [f"blocks.{i}.fc1.in", f"blocks.{i}.fc1", f"blocks.{i}.act", f"blocks.{i}.fc2", f"blocks.{i}"]
    return tuple(o + ["out_norm", "out_act", "out"])


#: every legal source (inputs and activations) in the order one network call computes them
FORWARD_ORDER = _forward_order()
ACTIVATION_SOURCES = tuple(s for s in FORWARD_ORDER if s not in INPUT_SOURCES)
#: default graft source = the layer's own input
DEFAULT_SRC = {"inp": "x", "out": "out_act",
               **{f"blocks.{i}.fc1": f"blocks.{i}.fc1.in" for i in range(3)},
               **{f"blocks.{i}.fc2": f"blocks.{i}.act" for i in range(3)}}


# =============================================================== dataclasses (W0 contract)
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


# =============================================================== network structure
def core_of(net: nn.Module) -> Tuple[nn.Module, bool]:
    """(the toypref VelocityNet / CondVelocityNet inside ``net``, is_uncond)."""
    if isinstance(net, EditedNet):
        net = net.base
    if hasattr(net, "inner"):          # toyrepair.cell._UncondNet
        return net.inner, True
    return net, False


def base_of(net: nn.Module) -> nn.Module:
    """The plain (graft-free) network whose parameters are pi_ref's tensors."""
    return net.base if isinstance(net, EditedNet) else net


def grafts_of(net: nn.Module) -> List["Graft"]:
    return list(net.grafts) if isinstance(net, EditedNet) else []


def linear(net: nn.Module, layer: str) -> nn.Linear:
    """The nn.Linear of an EDITABLE layer name."""
    if layer not in EDITABLE:
        raise EditRejected(f"unknown layer {layer!r}; editable: {', '.join(EDITABLE)}")
    core, _ = core_of(net)
    m = core.get_submodule(layer)
    assert isinstance(m, nn.Linear)
    return m


def source_dims(net: nn.Module) -> Dict[str, int]:
    """Feature size of every legal source of ``net``."""
    core, unc = core_of(net)
    hid = core.inp.out_features
    cond = core.blocks[0].fc1.in_features - hid
    d = {"x": 2, "t": 1, "c": 1, "inputs": 4, "cond": cond, "inp": hid,
         "out_norm": hid, "out_act": hid, "out": core.out.out_features}
    for i in range(len(core.blocks)):
        d.update({f"blocks.{i}.fc1.in": hid + cond, f"blocks.{i}.fc1": hid, f"blocks.{i}.act": hid,
                  f"blocks.{i}.fc2": hid, f"blocks.{i}": hid})
    return d


# =============================================================== grafts
class Graft(nn.Module):
    """One gated path (see ``GatedPath``). Parameters w_in, b_in, w_out (trainable by
    ``targeted_train`` param group "grafts"); the detector is a fixed buffer."""

    def __init__(self, gid: str, layer: str, src: str, act: str, w_in, b_in, w_out,
                 det_source: str, det_w, det_b: float, det_thr: float, k_range=None):
        super().__init__()
        self.gid, self.layer, self.src, self.act = gid, layer, src, act
        self.det_source, self.det_b, self.det_thr = det_source, float(det_b), float(det_thr)
        self.k_range = None if k_range is None else (int(k_range[0]), int(k_range[1]))
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32)).clone()
        self.w_in = nn.Parameter(t(w_in), requires_grad=False)
        self.b_in = nn.Parameter(t(b_in), requires_grad=False)
        self.w_out = nn.Parameter(t(w_out), requires_grad=False)
        self.register_buffer("det_w", t(det_w))

    @property
    def n_units(self) -> int:
        return int(self.w_in.shape[0])

    def spec(self) -> Dict:
        return {"gid": self.gid, "layer": self.layer, "src": self.src, "act": self.act,
                "n_units": self.n_units, "det_source": self.det_source, "det_b": self.det_b,
                "det_thr": self.det_thr, "k_range": list(self.k_range) if self.k_range else None}

    def fires(self, feats: Dict[str, torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        f = feats[self.det_source]
        s = F.linear(f, self.det_w.reshape(1, -1)).reshape(-1) + self.det_b
        m = s > self.det_thr
        if self.k_range is not None:
            k = torch.round((1.0 - t) * N_STEPS)
            m = m & (k >= self.k_range[0]) & (k < self.k_range[1])
        return m

    def contribution(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = F.linear(feats[self.src], self.w_in, self.b_in)
        if self.act == "silu":
            z = F.silu(z)
        elif self.act == "relu":
            z = F.relu(z)
        return F.linear(z, self.w_out)


class EditedNet(nn.Module):
    """A velocity network with gated paths: ``base`` (toypref module, possibly weight-edited)
    plus ``grafts`` applied in order. ``forward(x, t, c)`` = ``run_forward(self, x, t, c)[0]``."""

    def __init__(self, base: nn.Module, grafts: Sequence[Graft] = ()):
        super().__init__()
        self.base = base
        self.grafts = nn.ModuleList(list(grafts))
        self.disabled: Tuple[str, ...] = ()       # graft ids forced off (identity tests only)

    def forward(self, x, t, c=None):
        return run_forward(self, x, t, c)[0]


def run_forward(net: nn.Module, x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor] = None,
                capture: Sequence[str] = (), disabled: Sequence[str] = (),
                fired: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One network call with graft insertion and activation capture.

    x (B, 2), t (B,), c (B,) or None (uncond) -> (v (B, 2), {name: (B, d)} for ``capture``).
    On a plain net this is bit-identical to ``net(x, t, c)``. ``disabled``: graft ids forced
    off. ``fired`` (optional dict) receives each graft's (B,) bool firing mask."""
    core, unc = core_of(net)
    grafts = grafts_of(net)
    off = set(disabled) | set(getattr(net, "disabled", ()))
    by_layer: Dict[str, List[Graft]] = {}
    for g in grafts:
        by_layer.setdefault(g.layer, []).append(g)
    B = x.shape[0]
    cc = torch.zeros(B, dtype=x.dtype) if c is None else c
    feats: Dict[str, torch.Tensor] = {"x": x, "t": t.reshape(-1, 1), "c": cc.reshape(-1, 1)}
    if grafts:
        feats["inputs"] = torch.cat([x, t.reshape(-1, 1), cc.reshape(-1, 1)], -1)

    def emit(name, y):
        for g in by_layer.get(name, ()):
            if g.gid in off:
                continue
            m = g.fires(feats, t)
            if fired is not None:
                fired[g.gid] = m
            if bool(m.any()):
                y = torch.where(m.reshape(-1, 1), y + g.contribution(feats), y)
        feats[name] = y
        return y

    if unc:
        cond = core.time_mlp(core.time_embed(t))
    else:
        cond = torch.cat([core.time_mlp(core.time_embed(t)), core.c_mlp(core.c_feat(cc))], -1)
    feats["cond"] = cond
    h = emit("inp", core.inp(x))
    for i, b in enumerate(core.blocks):
        fin = torch.cat([b.norm(h), cond], dim=-1)
        feats[f"blocks.{i}.fc1.in"] = fin
        y = emit(f"blocks.{i}.fc1", b.fc1(fin))
        z = b.act(y)
        feats[f"blocks.{i}.act"] = z
        w = emit(f"blocks.{i}.fc2", b.fc2(z))
        h = h + w
        feats[f"blocks.{i}"] = h
    on = core.out_norm(h)
    feats["out_norm"] = on
    oa = F.silu(on)
    feats["out_act"] = oa
    v = emit("out", core.out(oa))
    if "inputs" in capture and "inputs" not in feats:
        feats["inputs"] = torch.cat([x, t.reshape(-1, 1), cc.reshape(-1, 1)], -1)
    return v, {k: feats[k] for k in capture}


# =============================================================== versions and decodes (local adapter)
def _frozen(net: nn.Module) -> nn.Module:
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def base_version(cell):
    """v0 = pi_ref of ``cell``. Uses ``toyrepair.policy.base_version`` when W1-A1 has
    implemented it, else a local equivalent (deep copy of ``cell.ref.net``)."""
    from . import policy
    try:
        return policy.base_version(cell)
    except NotImplementedError:
        return policy.Version(id="v0", parent=None, patches=(), net=_frozen(copy.deepcopy(cell.ref.net)),
                              grid=cell.grid, meta={})


def make_version(parent, new_id: str, net: nn.Module, patch_id: str, meta: Optional[Dict] = None):
    from . import policy
    return policy.Version(id=new_id, parent=parent.id, patches=tuple(parent.patches) + (patch_id,),
                          net=_frozen(net), grid=getattr(parent, "grid", "cond"), meta=dict(meta or {}))


@torch.no_grad()
def decode_actions(net: nn.Module, xi: torch.Tensor, c: torch.Tensor, batch: int = 65536,
                   save_states: bool = False):
    """toypref Euler sampler on ``net`` (bit-identical to ``CondFlow.sample`` for a plain net).
    Returns actions (B, 2), or (actions, states (21, B, 2)) with ``save_states``."""
    K = N_STEPS
    dt = 1.0 / K
    outs, sts = [], []
    for i in range(0, xi.shape[0], batch):
        x = xi[i:i + batch].clone().float()
        cc = c[i:i + batch].float()
        st = [x.clone()] if save_states else None
        for k in range(K):
            t = torch.full((x.shape[0],), 1.0 - k * dt)
            x = x - dt * net(x, t, cc)
            if save_states:
                st.append(x.clone())
        outs.append(x)
        if save_states:
            sts.append(torch.stack(st))
    a = torch.cat(outs)
    return (a, torch.cat(sts, 1)) if save_states else a


def reference_states(net: nn.Module, n_states: int = N_IDENTITY_STATES, seed: int = 0, uncond: bool = False):
    """n_states reference flow states (x_k, t_k, c) from decodes of ``net``: ceil(n/20)
    noises x 20 steps (k = 0..19), noises and conditions from torch.Generator(seed)."""
    n_noise = -(-n_states // N_STEPS)
    g = torch.Generator().manual_seed(int(seed) + 7_777)
    c = torch.zeros(n_noise) if uncond else torch.rand(n_noise, generator=g)
    xi = torch.randn(n_noise, 2, generator=g)
    _, st = decode_actions(net, xi, c, save_states=True)
    x = st[:N_STEPS].reshape(-1, 2)
    t = torch.cat([torch.full((n_noise,), 1.0 - k / N_STEPS) for k in range(N_STEPS)])
    cc = c.repeat(N_STEPS)
    return x[:n_states], t[:n_states], cc[:n_states]


# =============================================================== validation
def _arr(a, name: str, shape=None) -> np.ndarray:
    try:
        x = np.asarray(a, dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        raise EditRejected(f"{name}: not a float array ({e})")
    if shape is not None and tuple(x.shape) != tuple(shape):
        raise EditRejected(f"{name}: shape {tuple(x.shape)}, expected {tuple(shape)}")
    if not np.all(np.isfinite(x)):
        raise EditRejected(f"{name}: contains NaN or inf")
    return x


def _check_source(name: str, net: nn.Module, before: Optional[str], what: str) -> int:
    dims = source_dims(net)
    if not isinstance(name, str) or name not in dims:
        raise EditRejected(f"{what} source {name!r} is not a network input or activation; "
                           f"legal: {', '.join(INPUT_SOURCES)} or {', '.join(ACTIVATION_SOURCES)}")
    if before is not None and FORWARD_ORDER.index(name) >= FORWARD_ORDER.index(before):
        raise EditRejected(f"{what} source {name!r} is computed after layer {before!r} (not causal)")
    return dims[name]


def validate_detector(det: Detector, version, layer: Optional[str] = None) -> None:
    """Raise EditRejected unless the detector reads network inputs or activations only and its
    w has the source's feature size. With ``layer``, the source must also be computed before
    that layer's output."""
    if type(det) is not Detector:
        raise EditRejected("detector must be a toyrepair.edits.Detector (linear, no code)")
    net = version.net if hasattr(version, "net") else version
    d = _check_source(det.source, net, layer, "detector")
    _arr(det.w, "detector.w", (d,))
    for nm in ("b", "threshold"):
        v = getattr(det, nm)
        if not isinstance(v, (int, float, np.floating, np.integer)) or not np.isfinite(float(v)):
            raise EditRejected(f"detector.{nm} must be a finite number")
    if det.k_range is not None:
        lo, hi = det.k_range
        if not (0 <= int(lo) < int(hi) <= N_STEPS):
            raise EditRejected("detector.k_range must satisfy 0 <= k_lo < k_hi <= 20")


# =============================================================== bookkeeping
def _param_dict(net: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in base_of(net).state_dict().items()}


def change_stats(parent_net: nn.Module, new_net: nn.Module) -> Tuple[float, int, float]:
    """(fraction of pi_ref parameter entries that differ, units added, Frobenius norm of the
    change incl. added graft arrays)."""
    a, b = _param_dict(parent_net), _param_dict(new_net)
    total = sum(v.numel() for v in a.values())
    diff, fro2 = 0, 0.0
    for k, v in a.items():
        d = (b[k] - v)
        diff += int((b[k] != v).sum())
        fro2 += float((d.double() ** 2).sum())
    old = {g.gid for g in grafts_of(parent_net)}
    units = 0
    for g in grafts_of(new_net):
        if g.gid in old:
            go = [h for h in grafts_of(parent_net) if h.gid == g.gid][0]
            for nm in ("w_in", "b_in", "w_out"):
                fro2 += float(((getattr(g, nm) - getattr(go, nm)).double() ** 2).sum())
            continue
        units += g.n_units
        for nm in ("w_in", "b_in", "w_out"):
            fro2 += float((getattr(g, nm).double() ** 2).sum())
    return diff / max(total, 1), units, fro2 ** 0.5


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _finish(parent, new_id: str, call: str, args: Dict, arrays: Dict[str, np.ndarray],
            patch_id: Optional[str], decodes: float = 0.0) -> EditResult:
    from . import patches as P
    pid = patch_id or f"p_{new_id}"
    patch = P.Patch(id=pid, call=call, args=args, arrays=arrays, parent_version=parent.id, version=new_id)
    new = P.apply_patch(parent, patch)
    patch.sha256 = P.state_sha256(new)
    frac, units, fro = change_stats(parent.net, new.net)
    rec_args = dict(args)
    rec_args["arrays"] = {k: {"shape": list(np.shape(v)), "sha256": _sha(np.asarray(v))} for k, v in arrays.items()}
    rec = EditRecord(call=call, args=rec_args, parent=parent.id, version=new_id, patch_id=pid,
                     frac_weights_changed=frac, units_added=units, frob=fro, decodes=decodes)
    return EditResult(version=new, patch=patch, record=rec)


def _rejected(parent, call: str, args: Dict, err: str) -> EditResult:
    return EditResult(version=None, patch=None,
                      record=EditRecord(call=call, args=_json_safe(args), parent=getattr(parent, "id", "?"),
                                        version=None, patch_id=None, ok=False, error=err))


def _json_safe(d):
    try:
        json.dumps(d)
        return d
    except TypeError:
        return {k: (v if isinstance(v, (int, float, str, bool, type(None), list, tuple)) else repr(v)[:80])
                for k, v in d.items()}


# =============================================================== edit calls
def graft_gated_path(parent, new_id: str, layer: str, detector: Detector, n_units: int,
                     w_in: np.ndarray, w_out: np.ndarray, b_in: Optional[np.ndarray] = None,
                     src: Optional[str] = None, act: str = "silu", patch_id: Optional[str] = None) -> EditResult:
    """Add ``n_units`` gated units beside ``layer``. w_in (n_units, d_src), b_in (n_units,)
    (default 0), w_out (d_out(layer), n_units); src defaults to the layer's own input."""
    args = {"layer": layer, "src": src, "act": act, "n_units": n_units}
    try:
        if layer not in EDITABLE:
            raise EditRejected(f"unknown layer {layer!r}; editable: {', '.join(EDITABLE)}")
        if act not in GRAFT_ACTS:
            raise EditRejected(f"act must be one of {GRAFT_ACTS}")
        n = int(n_units)
        if not 1 <= n <= MAX_GRAFT_UNITS:
            raise EditRejected(f"n_units must be in 1..{MAX_GRAFT_UNITS}")
        src = src or DEFAULT_SRC[layer]
        args["src"] = src
        validate_detector(detector, parent, layer)
        d_src = _check_source(src, parent.net, layer, "graft")
        d_out = linear(parent.net, layer).out_features
        wi = _arr(w_in, "w_in", (n, d_src))
        bi = _arr(np.zeros(n) if b_in is None else b_in, "b_in", (n,))
        wo = _arr(w_out, "w_out", (d_out, n))
        dw = _arr(detector.w, "detector.w")
    except EditRejected as e:
        return _rejected(parent, "graft_gated_path", args, str(e))
    args["detector"] = {"source": detector.source, "b": float(detector.b), "threshold": float(detector.threshold),
                        "k_range": list(detector.k_range) if detector.k_range is not None else None}
    return _finish(parent, new_id, "graft_gated_path", args,
                   {"w_in": wi, "b_in": bi, "w_out": wo, "det_w": dw}, patch_id)


def rank1(parent, new_id: str, layer: str, u: np.ndarray, v: np.ndarray, s: float,
          patch_id: Optional[str] = None) -> EditResult:
    """u (d_out,), v (d_in,), s scalar: W <- W + s u v^T."""
    args = {"layer": layer, "s": s}
    try:
        lin = linear(parent.net, layer)
        uu = _arr(u, "u", (lin.out_features,))
        vv = _arr(v, "v", (lin.in_features,))
        s = float(s)
        if not np.isfinite(s):
            raise EditRejected("s must be finite")
        args["s"] = s
    except (EditRejected, TypeError, ValueError) as e:
        return _rejected(parent, "rank1", args, str(e))
    return _finish(parent, new_id, "rank1", args, {"u": uu, "v": vv}, patch_id)


def next_reader(layer: str) -> Tuple[str, slice]:
    """(next Linear layer, its input columns) that reads ``layer``'s output (ablate "post")."""
    if layer == "inp":
        return "blocks.0.fc1", slice(0, 128)
    if layer.endswith(".fc1"):
        return layer[:-1] + "2", slice(None)
    if layer.endswith(".fc2"):
        i = int(layer.split(".")[1])
        return (f"blocks.{i + 1}.fc1", slice(0, 128)) if i < 2 else ("out", slice(None))
    raise EditRejected(f"layer {layer!r} has no next layer for side 'post'")


def ablate_direction(parent, new_id: str, layer: str, v: np.ndarray, side: str,
                     patch_id: Optional[str] = None) -> EditResult:
    """side "pre" (v in the layer's output space) | "post" (v in the next layer's input space)."""
    args = {"layer": layer, "side": side}
    try:
        lin = linear(parent.net, layer)
        if side not in ("pre", "post"):
            raise EditRejected("side must be 'pre' or 'post'")
        if side == "post":
            next_reader(layer)
        vv = _arr(v, "v", (lin.out_features,))
        if float(np.linalg.norm(vv)) < 1e-8:
            raise EditRejected("v must be non-zero")
    except EditRejected as e:
        return _rejected(parent, "ablate_direction", args, str(e))
    return _finish(parent, new_id, "ablate_direction", args, {"v": vv}, patch_id)


def set_units(parent, new_id: str, layer: str, idx: Sequence[int], w_in: Optional[np.ndarray],
              b: Optional[np.ndarray], w_out: Optional[np.ndarray], patch_id: Optional[str] = None) -> EditResult:
    """idx (m,) unit indices of ``layer``; w_in (m, d_in); b (m,); w_out (d_next_out, m) written
    into the next linear layer's input columns (fc1 -> fc2; fc2/inp -> residual stream is not
    a unit layer: rejected). Any of w_in, b, w_out may be None (left unchanged)."""
    args = {"layer": layer}
    try:
        if layer not in UNIT_LAYERS:
            raise EditRejected(f"set_units supports {', '.join(UNIT_LAYERS)} only (hidden units feeding fc2)")
        lin = linear(parent.net, layer)
        nxt = linear(parent.net, layer[:-1] + "2")
        ix = np.asarray(idx).reshape(-1)
        if ix.size == 0 or not np.issubdtype(ix.dtype, np.integer) and not np.all(ix == np.round(ix)):
            raise EditRejected("idx must be a non-empty list of integers")
        ix = ix.astype(np.int64)
        if ix.min() < 0 or ix.max() >= lin.out_features or len(set(ix.tolist())) != ix.size:
            raise EditRejected(f"idx must be distinct units in 0..{lin.out_features - 1}")
        m = ix.size
        arrays = {"idx": ix}
        if w_in is not None:
            arrays["w_in"] = _arr(w_in, "w_in", (m, lin.in_features))
        if b is not None:
            arrays["b"] = _arr(b, "b", (m,))
        if w_out is not None:
            arrays["w_out"] = _arr(w_out, "w_out", (nxt.out_features, m))
        if len(arrays) == 1:
            raise EditRejected("give at least one of w_in, b, w_out")
        args["idx"] = ix.tolist()
    except EditRejected as e:
        return _rejected(parent, "set_units", args, str(e))
    return _finish(parent, new_id, "set_units", args, arrays, patch_id)


# =============================================================== apply (used by patches.apply_patch)
@torch.no_grad()
def apply_call(net: nn.Module, call: str, args: Dict, arrays: Dict[str, np.ndarray], gid: str) -> nn.Module:
    """Apply one edit to ``net`` (already a deep copy) and return the resulting net."""
    T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))
    if call == "graft_gated_path":
        d = args["detector"]
        g = Graft(gid, args["layer"], args["src"], args["act"], arrays["w_in"], arrays["b_in"], arrays["w_out"],
                  d["source"], arrays["det_w"], d["b"], d["threshold"], d.get("k_range"))
        if isinstance(net, EditedNet):
            net.grafts.append(g)
            return net
        return EditedNet(net, [g])
    if call == "rank1":
        W = linear(net, args["layer"]).weight
        W.add_(float(args["s"]) * torch.outer(T(arrays["u"]), T(arrays["v"])))
        return net
    if call == "ablate_direction":
        v = T(arrays["v"])
        v = v / torch.linalg.norm(v)
        Pm = torch.eye(v.numel()) - torch.outer(v, v)
        if args["side"] == "pre":
            lin = linear(net, args["layer"])
            lin.weight.copy_(Pm @ lin.weight)
            if lin.bias is not None:
                lin.bias.copy_(Pm @ lin.bias)
        else:
            nm, cols = next_reader(args["layer"])
            W = linear(net, nm).weight
            W[:, cols] = W[:, cols] @ Pm
        return net
    if call == "set_units":
        lin = linear(net, args["layer"])
        ix = torch.as_tensor(np.asarray(arrays["idx"], dtype=np.int64))
        if "w_in" in arrays:
            lin.weight[ix] = T(arrays["w_in"])
        if "b" in arrays:
            lin.bias[ix] = T(arrays["b"])
        if "w_out" in arrays:
            linear(net, args["layer"][:-1] + "2").weight[:, ix] = T(arrays["w_out"])
        return net
    if call == "targeted_train":
        params = dict(net.named_parameters())
        for k, a in arrays.items():
            if k not in params:
                raise EditRejected(f"targeted_train patch names unknown parameter {k!r}")
            params[k].copy_(T(a))
        return net
    raise EditRejected(f"unknown edit call {call!r}")


# =============================================================== unit tests (gate test 6)
def _hybrid(base_net: nn.Module, new_net: nn.Module) -> nn.Module:
    """base's weights with all of new's grafts (= new minus its non-graft weight changes)."""
    gs = [copy.deepcopy(g) for g in grafts_of(new_net)]
    return EditedNet(copy.deepcopy(base_of(base_net)), gs) if gs else copy.deepcopy(base_of(base_net))


@torch.no_grad()
def graft_identity(base, new, n_states: int = N_IDENTITY_STATES, seed: int = 0) -> bool:
    """With every graft's detector forced off, velocities of ``new`` (minus its non-graft weight
    changes) and ``base`` agree bit for bit on n_states reference flow states. Also: on states
    where no new graft fires, the velocities agree bit for bit with the grafts on."""
    bnet = base.net if hasattr(base, "net") else base
    nnet = new.net if hasattr(new, "net") else new
    old = {g.gid for g in grafts_of(bnet)}
    new_ids = [g.gid for g in grafts_of(nnet) if g.gid not in old]
    if not new_ids:
        return True
    _, unc = core_of(bnet)
    x, t, c = reference_states(bnet, n_states, seed, uncond=unc)
    hyb = _hybrid(bnet, nnet)
    v_base, _ = run_forward(bnet, x, t, c)
    v_off, _ = run_forward(hyb, x, t, c, disabled=new_ids)
    if not torch.equal(v_base, v_off):
        return False
    fired: Dict[str, torch.Tensor] = {}
    v_on, _ = run_forward(hyb, x, t, c, fired=fired)
    quiet = torch.ones(x.shape[0], dtype=torch.bool)
    for gid in new_ids:
        if gid in fired:
            quiet &= ~fired[gid]
    return bool(torch.equal(v_on[quiet], v_base[quiet]))


def unit_tests(new, current) -> UnitTestResult:
    """Gate test 6 hook, called by ``toyrepair.gate.gate_check``: shapes, finite parameters
    and outputs, legal detectors, graft identity for each graft new relative to current."""
    checks, detail = {}, {}
    nnet, cnet = new.net, current.net
    # shapes: same parameter names and shapes as current; graft arrays consistent
    try:
        a, b = _param_dict(cnet), _param_dict(nnet)
        bad = [k for k in a if k not in b or tuple(a[k].shape) != tuple(b[k].shape)] + [k for k in b if k not in a]
        dims = source_dims(nnet)
        for g in grafts_of(nnet):
            n = g.n_units
            d_out = linear(nnet, g.layer).out_features
            if (tuple(g.w_in.shape) != (n, dims.get(g.src, -1)) or tuple(g.b_in.shape) != (n,)
                    or tuple(g.w_out.shape) != (d_out, n) or tuple(g.det_w.shape) != (dims.get(g.det_source, -1),)):
                bad.append(f"graft {g.gid}")
        checks["shapes"] = not bad
        if bad:
            detail["shapes"] = "mismatch: " + ", ".join(bad[:5])
    except Exception as e:  # noqa: BLE001
        checks["shapes"], detail["shapes"] = False, f"error: {e}"[:200]
    # finite parameters and outputs
    try:
        fin = all(bool(torch.isfinite(p).all()) for p in nnet.parameters())
        fin = fin and all(bool(torch.isfinite(bf).all()) for bf in nnet.buffers())
        if fin:
            _, unc = core_of(cnet)
            x, t, c = reference_states(cnet, 2000, seed=1, uncond=unc)
            with torch.no_grad():
                fin = bool(torch.isfinite(run_forward(nnet, x, t, c)[0]).all())
            if not fin:
                detail["finite"] = "non-finite velocity on reference states"
        else:
            detail["finite"] = "non-finite parameter"
        checks["finite"] = fin
    except Exception as e:  # noqa: BLE001
        checks["finite"], detail["finite"] = False, f"error: {e}"[:200]
    # legal detectors and causal sources
    try:
        if type(nnet) not in (EditedNet, type(base_of(cnet))):
            raise EditRejected(f"unexpected network type {type(nnet).__name__}")
        for g in grafts_of(nnet):
            if g.layer not in EDITABLE or g.act not in GRAFT_ACTS:
                raise EditRejected(f"graft {g.gid}: illegal layer/act")
            _check_source(g.src, nnet, g.layer, "graft")
            _check_source(g.det_source, nnet, g.layer, "detector")
        checks["legal_detectors"] = True
    except EditRejected as e:
        checks["legal_detectors"], detail["legal_detectors"] = False, str(e)[:200]
    # graft identity for each new graft
    try:
        ok = graft_identity(current, new)
        checks["graft_identity"] = ok
        if not ok:
            detail["graft_identity"] = "velocity differs with the new grafts' detectors off"
    except Exception as e:  # noqa: BLE001
        checks["graft_identity"], detail["graft_identity"] = False, f"error: {e}"[:200]
    return UnitTestResult(passed=all(checks.values()), checks=checks, detail=detail)
