"""Local adapters to other sessions' modules until the W2a merge (owner W1-A2).

* ``actions``: decode through ``toyrepair.policy.decode`` (W1-A1); while that is a stub, use a
  local copy of the toypref Euler sampler (``CondFlow.sample``: same op order, same batch split,
  so it is bit-identical to toypref).
* ``base_version``: ``toyrepair.policy.base_version`` or a local ``Version`` around a deep copy
  of ``cell.ref.net``.
* ``unit_tests``: ``toyrepair.edits.unit_tests`` (W1-B); while that is a stub, a local check of
  shapes and finiteness (see ``stub_unit_tests``).
"""

from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch

N_STEPS = 20
BATCH = 65536


@torch.no_grad()
def euler_decode(net, xi: torch.Tensor, c: torch.Tensor, batch: int = BATCH) -> torch.Tensor:
    """toypref ``CondFlow.sample`` with an explicit net: 20 Euler steps, t_k = 1 - k/20."""
    out = []
    dt = 1.0 / N_STEPS
    for i in range(0, xi.shape[0], batch):
        x = xi[i:i + batch].clone().float()
        cc = c[i:i + batch].float()
        for k in range(N_STEPS):
            t = torch.full((x.shape[0],), 1.0 - k * dt)
            x = x - dt * net(x, t, cc)
        out.append(x)
    if not out:
        return torch.zeros(0, 2)
    return torch.cat(out)


@torch.no_grad()
def actions(version, xi: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """(B, 2) float32 actions of ``version`` (a ``policy.Version``, or any module with
    ``.net(x, t, c)``) from noises xi (B, 2) at conditions c (B,)."""
    from .. import policy
    if isinstance(version, policy.Version):
        try:
            return policy.decode(version, xi, c).action
        except NotImplementedError:
            return euler_decode(version.net, xi, c)
    net = getattr(version, "net", version)
    return euler_decode(net, xi, c)


def base_version(cell):
    """v0 of the cell (``policy.base_version`` or the local adapter)."""
    from .. import policy
    try:
        return policy.base_version(cell)
    except NotImplementedError:
        net = copy.deepcopy(cell.ref.net).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return policy.Version(id="v0", parent=None, patches=(), net=net, grid=cell.grid)


def stub_unit_tests(new, current) -> Dict:
    """Local stand-in for ``edits.unit_tests`` (gate test 6): every parameter and buffer of
    ``new`` is finite, every tensor name shared with ``current`` has the same shape, and the
    net maps (4, 2) inputs to finite (4, 2) velocities. Graft identity and detector legality
    are W1-B's (reported as True with detail "stub")."""
    checks = {"shapes": True, "finite": True, "legal_detectors": True, "graft_identity": True}
    detail = {"legal_detectors": "stub", "graft_identity": "stub"}
    sd_n = new.net.state_dict()
    sd_c = current.net.state_dict()
    bad_shape = [k for k in sd_n if k in sd_c and tuple(sd_n[k].shape) != tuple(sd_c[k].shape)]
    if bad_shape:
        checks["shapes"] = False
        detail["shapes"] = f"shape differs: {bad_shape[:3]}"
    bad_fin = [k for k, v in sd_n.items() if torch.is_floating_point(v) and not torch.isfinite(v).all()]
    if bad_fin:
        checks["finite"] = False
        detail["finite"] = f"non-finite: {bad_fin[:3]}"
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4, 2, generator=g)
        t = torch.rand(4, generator=g)
        c = torch.rand(4, generator=g)
        try:
            v = new.net(x, t, c)
            if tuple(v.shape) != (4, 2):
                checks["shapes"] = False
                detail["shapes"] = f"velocity shape {tuple(v.shape)}"
            elif not torch.isfinite(v).all():
                checks["finite"] = False
                detail["finite"] = "non-finite velocity"
        except Exception as e:  # noqa: BLE001
            checks["shapes"] = False
            detail["shapes"] = f"forward failed: {type(e).__name__}: {e}"[:200]
    return {"passed": all(checks.values()), "checks": checks, "detail": detail, "stub": True}


def unit_tests(new, current) -> Dict:
    """Gate test 6 as a JSON-safe dict {passed, checks, detail, stub}."""
    from .. import edits
    try:
        r = edits.unit_tests(new, current)
    except NotImplementedError:
        return stub_unit_tests(new, current)
    return {"passed": bool(r.passed), "checks": {k: bool(v) for k, v in r.checks.items()},
            "detail": dict(r.detail), "stub": False}


def as_np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
