"""toyrepair: agentic policy repair on the toypref flow toy (docs/agentic_repair/docs/PLAN.md).

Import order matters: ``toypref.config`` reads ``TOYPREF_RUNS`` once, at import time, and its
default is a cluster path. Importing ``toyrepair`` (any submodule) first sets ``TOYPREF_RUNS``
to ``<repo>/runs`` when it is unset. Every toyrepair module that imports toypref imports
``toyrepair`` first (which Python does automatically for ``toyrepair.<sub>``).

Isolation: ``toyrepair.oracle`` is imported only by ``toyrepair.gate`` and
``toyrepair.eval_final`` (and their tests). ``toyrepair.cli`` imports neither toypref nor the
oracle at import time.
"""

from __future__ import annotations

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.environ.setdefault("TOYPREF_RUNS", os.path.join(REPO, "runs"))

__all__ = ["REPO", "RUNS"]

# One thread per process (W2a): the pilot runs up to 8 surgeon CLIs, the orchestrator and the
# jobs.py workers side by side on 4 cores, and default BLAS/torch threading oversubscribed them
# (load ~50). Set before numpy/torch initialise; an explicit environment value wins.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
