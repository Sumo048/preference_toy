#!/usr/bin/env bash
# Cloud-session setup for toyrepair (TOY_PLAN.md Phase 0; SESSION_PROMPTS.md rule 2).
# Idempotent. Run from anywhere: bash scripts/cloud_setup.sh
#
# 1. installs torch + deps and `pip install -e .` (toypref + toyrepair, console script toyrepair-tool)
# 2. unpacks ckpt/ into runs/ (runs/cond_base/s*/policy.pt, runs/base/s*/policy.pt, runs/cond/ from
#    ckpt/cond_cells.tgz if present); never writes into ckpt/
# 3. creates the unprivileged user `sandbox` and makes the repo mode 700, owned by the user that runs
#    Claude Code, so `sandbox` cannot read toypref or toyrepair/oracle (TOY_PLAN.md Section 6.6)
#
# Network note: the cloud proxy allows PyPI but not download.pytorch.org, so the CPU-only wheel index
# is tried first and the PyPI wheel (CUDA build, runs on CPU, pulls the nvidia-* libs) is the fallback.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
T0=$(date +%s)
log() { echo "[cloud_setup +$(( $(date +%s) - T0 ))s] $*"; }

# ---------------------------------------------------------------- 1. dependencies
TORCH_VER=2.14.0
if python -c "import torch, sys; sys.exit(0 if torch.__version__.split('+')[0] == '$TORCH_VER' else 1)" 2>/dev/null; then
  log "torch $TORCH_VER present"
else
  log "installing torch $TORCH_VER"
  pip install -q --timeout 20 --retries 0 --index-url https://download.pytorch.org/whl/cpu "torch==$TORCH_VER" 2>/dev/null \
    || pip install -q "torch==$TORCH_VER"
fi
python - <<'PY' || pip install -q -r requirements.txt matplotlib scikit-learn pytest
import importlib, sys
for m in ("numpy", "pandas", "matplotlib", "sklearn", "pytest"):
    importlib.import_module(m)
import numpy; sys.exit(0 if numpy.__version__ == "2.2.6" else 1)
PY
pip install -q --no-deps -e .
log "deps ok: $(python -c 'import torch, numpy; print("torch", torch.__version__, "numpy", numpy.__version__)')"

# ---------------------------------------------------------------- 2. checkpoints -> runs/
( cd ckpt && sha256sum --quiet -c SHA256SUMS ) || { echo "FAIL: ckpt checksums"; exit 1; }
log "ckpt checksums ok"
mkdir -p runs
for sub in cond_base base; do
  [ -d "ckpt/$sub" ] || continue
  for d in ckpt/$sub/s*; do
    dst="runs/$sub/$(basename "$d")"
    mkdir -p "$dst"
    for f in "$d"/*; do [ -e "$dst/$(basename "$f")" ] || cp "$f" "$dst/"; done
  done
done
if [ -f ckpt/cond_cells.tgz ]; then
  mkdir -p runs/cond && tar -xzf ckpt/cond_cells.tgz -C runs/cond --skip-old-files
  log "unpacked ckpt/cond_cells.tgz into runs/cond"
fi
log "runs/: $(ls runs/cond_base | wc -l) cond bases, $(ls runs/base 2>/dev/null | wc -l) uncond bases"

# ---------------------------------------------------------------- 3. sandbox user + repo isolation
OWNER="$(id -un)"
if ! id sandbox >/dev/null 2>&1; then
  if [ "$(id -u)" = 0 ]; then useradd --system --no-create-home --shell /usr/sbin/nologin sandbox
  else sudo useradd --system --no-create-home --shell /usr/sbin/nologin sandbox; fi
  log "created user sandbox"
fi
SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo
$SUDO chown "$OWNER" "$REPO"
$SUDO chmod 700 "$REPO"
# checks: Claude Code's user can read and write; sandbox cannot read
test -r "$REPO/src/toypref/cond/task.py" && touch "$REPO/.cloud_setup_rw" && rm "$REPO/.cloud_setup_rw" \
  || { echo "FAIL: repo not read/writable by $OWNER"; exit 1; }
if command -v runuser >/dev/null && [ "$(id -u)" = 0 ]; then
  if runuser -u sandbox -- cat "$REPO/src/toypref/cond/task.py" >/dev/null 2>&1; then
    echo "FAIL: sandbox user can read the repo"; exit 1
  fi
  log "isolation ok: sandbox cannot read $REPO (mode $(stat -c %a "$REPO"), owner $(stat -c %U "$REPO"))"
fi
export TOYPREF_RUNS="${TOYPREF_RUNS:-$REPO/runs}"
python -c "import toyrepair, toypref" && command -v toyrepair-tool >/dev/null && log "toyrepair importable, toyrepair-tool on PATH"
log "done in $(( $(date +%s) - T0 ))s"
