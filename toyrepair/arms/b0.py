"""B0: pi_ref, no change."""

from __future__ import annotations

from typing import Optional

from .base import ArmOutput, RunIO, Services, real_services


def run(cell, run, svc: Optional[Services] = None) -> ArmOutput:
    """Run the arm on ``cell`` inside run context ``run`` (toyrepair.orchestrate.Run): v0, no edit."""
    svc = svc or real_services()
    io = RunIO(run, svc)
    base = svc.base_version(cell)
    io.log("run", {"arm": "B0", "cell": cell.key})
    out = ArmOutput(arm="B0", version=base.id, final=base)
    io.log("note", {"stop": out.stop, **out.to_json()}, version=base.id)
    return out
