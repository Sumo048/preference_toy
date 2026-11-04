"""Sandbox: export, audit hook, runner (docs/PLAN.md 6.6).

The sandbox runs ``python -I`` in a subprocess as the unprivileged user ``sandbox``, cwd = the
run's scratch directory. It sees only the export:

  export/weights.pt     current version's state (+ graft arrays)
  export/model.py       self-contained velocity-network class (no toypref import), with
                        ``decode(xi, c)`` that counts decodes into decodes.json
  export/negatives.npz  logged negatives by uid (c, xi, a), public order
  export/basins.npz     each negative's basin points, labels, eps, bad id

The audit hook (``sys.addaudithook``) blocks imports of ``toypref`` and ``toyrepair.oracle``,
socket/network events and subprocess/os.system/exec events. Output is capped at 2000 tokens.
The sandbox has no write access to weights; it writes ``.npy`` arrays into scratch.

Owner: W1-B.

How the runner works (W1-B). The repo (which holds ``runs/``, so every scratch directory) is
mode 700 and owned by root, so the ``sandbox`` user cannot enter it. Each ``run`` therefore:

1. makes a fresh private work dir ``/var/tmp/tr_sbx/<random>/`` (parent root-owned, mode
   711, not listable; work dir owned by ``sandbox``, mode 700);
2. copies ``<scratch>/export/`` into ``<work>/export/`` (root-owned, read-only for sandbox)
   and the scratch's own files (regular files with an allowed extension, no symlinks) into
   ``<work>/``;
3. runs ``python -I _sandbox_main.py`` as uid/gid ``sandbox`` (no supplementary groups) there with a minimal
   environment, a timeout and one torch thread; ``_sandbox_main.py`` imports numpy and torch,
   installs the audit hook, then executes the code;
4. copies back new or changed regular files with an allowed extension (single link, size
   capped) from ``<work>/`` into the scratch directory, reads the decode counter, and deletes
   the work dir.
The export's ``model.py`` reproduces the harness forward (``toyrepair.edits.run_forward``) and
decode bit for bit (tested), grafts included.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Optional

SANDBOX_USER = "sandbox"
OUTPUT_TOKEN_CAP = 2000
CHARS_PER_TOKEN = 4.0          # W1-D measures the real ratio; the cap is in characters here
TIMEOUT_S = 300
BLOCKED_MODULES = ("toypref", "toyrepair.oracle", "toyrepair")
BLOCKED_EVENTS = ("socket.connect", "socket.bind", "subprocess.Popen", "os.system", "os.exec",
                  "os.posix_spawn", "os.fork")
SANDBOX_ROOT = "/var/tmp/tr_sbx"
COPY_EXT = (".npy", ".npz", ".json", ".txt", ".csv", ".py")
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
DECODE_FILE = "decodes.json"
MAIN = "_sandbox_main.py"
CODE = "_sandbox_code.py"


@dataclass
class SandboxResult:
    """stdout/stderr (truncated to the cap); returncode; decodes counted; timed_out; truncated."""
    stdout: str
    stderr: str
    returncode: int
    decodes: int = 0
    timed_out: bool = False
    truncated: bool = False


# =============================================================== the exported model (no toypref)
MODEL_SOURCE = r'''"""Self-contained copy of the policy network (export of the current version).

    import model
    net = model.load()                      # the current version (weights.pt next to this file)
    v, acts = net.velocity(x, t, c, capture=("blocks.1.act",))
    a = model.decode(net, xi, c)            # 20-step Euler decode: (B,2) actions; counts B decodes
    a, states, acts = model.decode(net, xi, c, save_states=True, capture=("blocks.2.act",))
        # states (21,B,2) = x_k for k=0..20; acts[name] (20,B,d) at steps 0..19
    neg = model.negatives()                 # dict uid, c, xi, a (public order, numpy)
    model.in_basin(i, x)                    # (P,) bool: is x (P,2) in negative i's bad basin
    model.basin(i)                          # dict pts, lab, eps, bad, thr (pi_ref decodes at c_i)

Shapes: x, xi (B,2) float32; t, c (B,) float32; step k uses t_k = 1 - k/20 and
x_{k+1} = x_k - v/20. Capture names: x t c inputs cond inp blocks.{i}.fc1.in blocks.{i}.fc1
blocks.{i}.act blocks.{i}.fc2 blocks.{i} out_norm out_act out.
"""
import json as _json
import math as _math
import os as _os

import numpy as _np
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F

HERE = _os.path.dirname(_os.path.abspath(__file__))
N_STEPS = 20
_COUNT = {"n": 0.0}


def _count(n):
    _COUNT["n"] += float(n)
    try:
        with open(_os.path.join(_os.getcwd(), "decodes.json"), "w") as f:
            _json.dump({"decodes": _COUNT["n"]}, f)
    except OSError:
        pass


class _TimeEmb(_nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = _torch.exp(-_math.log(10_000.0) * _torch.arange(half, dtype=t.dtype) / max(half - 1, 1))
        ang = t.reshape(-1, 1) * freqs.reshape(1, -1) * 1_000.0
        return _torch.cat([_torch.sin(ang), _torch.cos(ang)], dim=-1)


class _Block(_nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = _nn.LayerNorm(dim)
        self.fc1 = _nn.Linear(dim + cond_dim, dim)
        self.fc2 = _nn.Linear(dim, dim)
        self.act = _nn.SiLU()


class _Graft:
    def __init__(self, g):
        self.__dict__.update(g)

    def fires(self, feats, t):
        s = _F.linear(feats[self.det_source], self.det_w.reshape(1, -1)).reshape(-1) + self.det_b
        m = s > self.det_thr
        if self.k_range is not None:
            k = _torch.round((1.0 - t) * N_STEPS)
            m = m & (k >= self.k_range[0]) & (k < self.k_range[1])
        return m

    def contribution(self, feats):
        z = _F.linear(feats[self.src], self.w_in, self.b_in)
        if self.act == "silu":
            z = _F.silu(z)
        elif self.act == "relu":
            z = _F.relu(z)
        return _F.linear(z, self.w_out)


class Net(_nn.Module):
    """Velocity network v(x, t, c). ``conditional`` False: c is ignored (cond = time only)."""

    def __init__(self, conditional=True, hidden=128, n_layers=3, t_dim=64, c_freqs=8, dim=2):
        super().__init__()
        self.conditional = conditional
        self.time_embed = _TimeEmb(t_dim)
        self.time_mlp = _nn.Sequential(_nn.Linear(t_dim, t_dim), _nn.SiLU(), _nn.Linear(t_dim, t_dim))
        cond_dim = t_dim
        if conditional:
            self.c_freqs = c_freqs
            self.c_mlp = _nn.Sequential(_nn.Linear(2 * c_freqs, t_dim), _nn.SiLU(), _nn.Linear(t_dim, t_dim))
            cond_dim = 2 * t_dim
        self.inp = _nn.Linear(dim, hidden)
        self.blocks = _nn.ModuleList([_Block(hidden, cond_dim) for _ in range(n_layers)])
        self.out_norm = _nn.LayerNorm(hidden)
        self.out = _nn.Linear(hidden, dim)
        self.grafts = []

    def c_feat(self, c):
        k = _torch.arange(1, self.c_freqs + 1, dtype=c.dtype)
        a = 2 * _math.pi * c.reshape(-1, 1) * k.reshape(1, -1) / 2.0
        return _torch.cat([_torch.sin(a), _torch.cos(a)], -1)

    def forward(self, x, t, c=None):
        return self.velocity(x, t, c, count=False)[0]

    def velocity(self, x, t, c=None, capture=(), count=True):
        """One network call -> (v (B,2), {name: (B,d)}). Counts B/20 decodes."""
        if count:
            _count(x.shape[0] / N_STEPS)
        B = x.shape[0]
        cc = _torch.zeros(B, dtype=x.dtype) if c is None else c
        feats = {"x": x, "t": t.reshape(-1, 1), "c": cc.reshape(-1, 1)}
        if self.grafts:
            feats["inputs"] = _torch.cat([x, t.reshape(-1, 1), cc.reshape(-1, 1)], -1)
        by = {}
        for g in self.grafts:
            by.setdefault(g.layer, []).append(g)

        def emit(name, y):
            for g in by.get(name, ()):
                m = g.fires(feats, t)
                if bool(m.any()):
                    y = _torch.where(m.reshape(-1, 1), y + g.contribution(feats), y)
            feats[name] = y
            return y

        if self.conditional:
            cond = _torch.cat([self.time_mlp(self.time_embed(t)), self.c_mlp(self.c_feat(cc))], -1)
        else:
            cond = self.time_mlp(self.time_embed(t))
        feats["cond"] = cond
        h = emit("inp", self.inp(x))
        for i, b in enumerate(self.blocks):
            fin = _torch.cat([b.norm(h), cond], dim=-1)
            feats["blocks.%d.fc1.in" % i] = fin
            y = emit("blocks.%d.fc1" % i, b.fc1(fin))
            z = b.act(y)
            feats["blocks.%d.act" % i] = z
            w = emit("blocks.%d.fc2" % i, b.fc2(z))
            h = h + w
            feats["blocks.%d" % i] = h
        on = self.out_norm(h)
        feats["out_norm"] = on
        oa = _F.silu(on)
        feats["out_act"] = oa
        v = emit("out", self.out(oa))
        if "inputs" in capture and "inputs" not in feats:
            feats["inputs"] = _torch.cat([x, t.reshape(-1, 1), cc.reshape(-1, 1)], -1)
        return v, {k: feats[k] for k in capture}


def load(path=None):
    """The exported version as a Net (eval mode, no grad)."""
    sd = _torch.load(path or _os.path.join(HERE, "weights.pt"), map_location="cpu", weights_only=True)
    net = Net(conditional=sd["conditional"], **sd["kw"])
    net.load_state_dict(sd["state"])
    net.grafts = [_Graft(g) for g in sd["grafts"]]
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@_torch.no_grad()
def decode(net, xi, c, save_states=False, capture=(), start_step=0, batch=65536):
    """Euler decode from step ``start_step`` (xi is x_start). Counts B decodes."""
    xi = _torch.as_tensor(xi, dtype=_torch.float32)
    c = _torch.as_tensor(c, dtype=_torch.float32).reshape(-1)
    _count(xi.shape[0] * (N_STEPS - start_step) / N_STEPS)
    dt = 1.0 / N_STEPS
    outs, sts, acs = [], [], {k: [] for k in capture}
    for i in range(0, xi.shape[0], batch):
        x = xi[i:i + batch].clone().float()
        cc = c[i:i + batch].float()
        st, ac = [x.clone()], {k: [] for k in capture}
        for k in range(start_step, N_STEPS):
            t = _torch.full((x.shape[0],), 1.0 - k * dt)
            v, a = net.velocity(x, t, cc, capture, count=False)
            for nm in capture:
                ac[nm].append(a[nm])
            x = x - dt * v
            st.append(x.clone())
        outs.append(x)
        sts.append(_torch.stack(st))
        for nm in capture:
            acs[nm].append(_torch.stack(ac[nm]))
    a = _torch.cat(outs)
    if not save_states and not capture:
        return a
    return a, _torch.cat(sts, 1), {k: _torch.cat(v, 1) for k, v in acs.items()}


def negatives():
    z = _np.load(_os.path.join(HERE, "negatives.npz"))
    return {k: z[k] for k in z.files}


_B = {}


def basin(i):
    if not _B:
        z = _np.load(_os.path.join(HERE, "basins.npz"))
        _B.update({k: z[k] for k in z.files})
    o = _B["offset"]
    s = slice(int(o[i]), int(o[i + 1]))
    return {"pts": _B["pts"][s], "lab": _B["lab"][s], "eps": float(_B["eps"][i]),
            "bad": int(_B["bad"][i]), "thr": float(_B["thr"][i])}


def in_basin(i, x):
    b = basin(i)
    x = _torch.as_tensor(x, dtype=_torch.float32).reshape(-1, 2)
    d = _torch.cdist(x, _torch.as_tensor(b["pts"]))
    j = d.argmin(1).numpy()
    lab = b["lab"][j].copy()
    lab[(d.min(1).values > b["eps"]).numpy()] = -1
    return lab == b["bad"]
'''

README = """export/ (read-only). In Python: import model; net = model.load(); see help(model).
weights.pt: current version. negatives.npz: uid, c, xi, a (public order; index i = row i).
basins.npz: per-negative basin of pi_ref decodes at c_i (model.basin(i), model.in_basin(i, x)).
Write results as .npy files in the current directory; edit calls read them from there.
"""


def export(cell, version, scratch: str) -> str:
    """Write the export for ``version`` into ``<scratch>/export``. Returns that path."""
    import numpy as np
    import torch
    from . import edits
    out = os.path.join(scratch, "export")
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    net = version.net
    core, unc = edits.core_of(net)
    hid = core.inp.out_features
    kw = {"hidden": hid, "n_layers": len(core.blocks), "t_dim": core.time_mlp[0].in_features}
    if not unc:
        kw["c_freqs"] = core.c_freqs
    grafts = []
    for g in edits.grafts_of(net):
        s = g.spec()
        grafts.append({"layer": s["layer"], "src": s["src"], "act": s["act"], "det_source": s["det_source"],
                       "det_b": s["det_b"], "det_thr": s["det_thr"], "k_range": s["k_range"],
                       "w_in": g.w_in.detach().clone(), "b_in": g.b_in.detach().clone(),
                       "w_out": g.w_out.detach().clone(), "det_w": g.det_w.detach().clone()})
    state = {k: v.detach().clone() for k, v in core.state_dict().items()}
    torch.save({"conditional": not unc, "kw": kw, "state": state, "grafts": grafts, "version": version.id},
               os.path.join(tmp, "weights.pt"))
    with open(os.path.join(tmp, "model.py"), "w") as f:
        f.write(MODEL_SOURCE)
    with open(os.path.join(tmp, "README.txt"), "w") as f:
        f.write(README)
    order = cell.public_order
    n = cell.neg
    np.savez(os.path.join(tmp, "negatives.npz"), uid=n.uid[order], c=n.c.numpy()[order],
             xi=n.xi.numpy()[order], a=n.a.numpy()[order])
    bs = [cell.basins[int(i)] for i in order]
    off = np.cumsum([0] + [int(b.pts.shape[0]) for b in bs])
    np.savez(os.path.join(tmp, "basins.npz"), offset=off,
             pts=np.concatenate([b.pts.numpy() for b in bs]).astype(np.float32),
             lab=np.concatenate([b.lab for b in bs]).astype(np.int64),
             eps=np.array([b.eps for b in bs], dtype=np.float32), bad=np.array([b.bad for b in bs], dtype=np.int64),
             thr=np.array([b.thr for b in bs], dtype=np.float32))
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    return out


# =============================================================== audit hook
def audit_hook_source() -> str:
    """Python source of the audit hook prepended to every sandbox script."""
    return r'''
def _toyrepair_install_audit():
    import sys
    blocked_mods = ("toypref", "toyrepair")
    blocked_prefix = ("socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
                      "os.fork", "os.forkpty", "pty.", "ctypes.", "urllib.", "http.", "ftplib.", "smtplib.",
                      "telnetlib.", "imaplib.", "poplib.", "nntplib.", "webbrowser.", "sys.remote_exec",
                      "os.kill", "os.killpg", "signal.pthread_kill", "_posixsubprocess.", "mmap.")
    def hook(event, args):
        if event == "import":
            m = str(args[0] or "")
            if m.split(".")[0] in blocked_mods:
                raise ImportError("sandbox: import of %s is blocked" % m)
        elif event == "open":
            p = str(args[0]) if args else ""
            if "toypref" in p or "toyrepair" in p or "/oracle" in p:
                raise PermissionError("sandbox: opening %s is blocked" % p)
        elif event.startswith(blocked_prefix):
            raise PermissionError("sandbox: %s is blocked" % event)
    sys.addaudithook(hook)

    class _BlockFinder:
        """importlib.import_module raises no "import" audit event: block at the finder too."""
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name.split(".")[0] in blocked_mods:
                raise ImportError("sandbox: import of %s is blocked" % name)
            return None
    sys.meta_path[:] = [_BlockFinder] + [f for f in sys.meta_path if "__editable__" not in repr(f)]
    sys.path_hooks[:] = [h for h in sys.path_hooks if "__editable__" not in repr(h)]
    for m in [m for m in sys.modules if m.split(".")[0] in blocked_mods]:
        del sys.modules[m]
'''


def _main_source(work: str) -> str:
    return ("import sys, os\n"
            "import numpy, torch\n"
            "torch.set_num_threads(1)\n"
            f"sys.path.insert(0, {os.path.join(work, 'export')!r})\n"
            + audit_hook_source() +
            "_toyrepair_install_audit()\n"
            "del _toyrepair_install_audit\n"
            f"with open({CODE!r}) as _f:\n"
            "    _src = _f.read()\n"
            "_g = {'__name__': '__main__', '__builtins__': __builtins__}\n"
            f"exec(compile(_src, 'sandbox_code.py', 'exec'), _g)\n")


# =============================================================== runner
def _cap(text: str, limit: int):
    if len(text) <= limit:
        return text, False
    cut = len(text) - limit
    return text[:limit] + f"\n[... {cut} characters cut: output cap {OUTPUT_TOKEN_CAP} tokens]", True


def _snapshot(d: str) -> Dict[str, tuple]:
    out = {}
    for root, dirs, files in os.walk(d):
        rel = os.path.relpath(root, d)
        if rel.split(os.sep)[0] == "export":
            dirs[:] = []
            continue
        for f in files:
            p = os.path.join(root, f)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            out[os.path.normpath(os.path.join(rel, f))] = (st.st_size, st.st_mtime_ns)
    return out


def _copy_in(scratch: str, work: str, uid: Optional[int]) -> None:
    exp = os.path.join(scratch, "export")
    if os.path.isdir(exp):
        shutil.copytree(exp, os.path.join(work, "export"), symlinks=False)
        for root, dirs, files in os.walk(os.path.join(work, "export")):
            os.chmod(root, 0o755)
            for f in files:
                os.chmod(os.path.join(root, f), 0o644)
    for root, dirs, files in os.walk(scratch):
        rel = os.path.relpath(root, scratch)
        if rel.split(os.sep)[0] in ("export", "export.tmp", ".sbx_work"):
            dirs[:] = []
            continue
        for f in files:
            src = os.path.join(root, f)
            st = os.lstat(src)
            if not stat.S_ISREG(st.st_mode) or not f.endswith(COPY_EXT) or st.st_size > MAX_FILE_BYTES:
                continue
            dst = os.path.join(work, rel, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
    if uid is not None:
        for root, dirs, files in os.walk(work):
            if os.path.relpath(root, work).split(os.sep)[0] == "export":
                continue
            os.chown(root, uid, -1)
            for f in files:
                os.chown(os.path.join(root, f), uid, -1)


def _copy_out(work: str, scratch: str, before: Dict[str, tuple]) -> None:
    total = 0
    for rel, sig in _snapshot(work).items():
        if rel in (MAIN, CODE, DECODE_FILE) or before.get(rel) == sig or ".." in rel.split(os.sep):
            continue
        p = os.path.join(work, rel)
        st = os.lstat(p)
        if (not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or not rel.endswith(COPY_EXT)
                or st.st_size > MAX_FILE_BYTES or total + st.st_size > MAX_TOTAL_BYTES):
            continue
        total += st.st_size
        dst = os.path.join(scratch, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as fi, open(dst + ".tmp", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.replace(dst + ".tmp", dst)


def _sandbox_uid(user: Optional[str]) -> Optional[int]:
    if user is None:
        return None
    import pwd
    return pwd.getpwnam(user).pw_uid


def run(code: str, scratch: str, timeout_s: int = TIMEOUT_S, user: Optional[str] = SANDBOX_USER) -> SandboxResult:
    """Run ``code`` in the sandbox (``user=None`` only in tests where the user does not exist)."""
    os.makedirs(scratch, exist_ok=True)
    uid = _sandbox_uid(user)
    if uid is not None:
        os.makedirs(SANDBOX_ROOT, exist_ok=True)
        os.chown(SANDBOX_ROOT, 0, 0)
        os.chmod(SANDBOX_ROOT, 0o711)
        base = SANDBOX_ROOT
    else:
        base = os.path.join(scratch, ".sbx_work")
        os.makedirs(base, exist_ok=True)
    work = os.path.join(base, secrets.token_hex(12))
    os.makedirs(work, mode=0o700)
    try:
        _copy_in(scratch, work, uid)
        for name, src in ((MAIN, _main_source(work)), (CODE, code)):
            with open(os.path.join(work, name), "w") as f:
                f.write(src)
            os.chmod(os.path.join(work, name), 0o644)
        if os.path.exists(os.path.join(work, DECODE_FILE)):
            os.remove(os.path.join(work, DECODE_FILE))
        before = _snapshot(work)
        env = {"PATH": "/usr/bin:/bin", "HOME": work, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "LANG": "C.UTF-8", "TMPDIR": work}
        cmd = [sys.executable, "-I", MAIN]
        drop = {}
        if uid is not None:
            import pwd
            pw = pwd.getpwuid(uid)
            drop = {"user": uid, "group": pw.pw_gid, "extra_groups": []}
        timed_out = False
        p = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdin=subprocess.DEVNULL, start_new_session=True, **drop)
        try:
            out, err = p.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = p.communicate()
        decodes = 0
        try:
            with open(os.path.join(work, DECODE_FILE)) as f:
                decodes = int(round(float(json.load(f)["decodes"])))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        _copy_out(work, scratch, before)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    limit = int(OUTPUT_TOKEN_CAP * CHARS_PER_TOKEN)
    so = out.decode("utf-8", "replace")
    se = err.decode("utf-8", "replace")
    so, t1 = _cap(so, limit)
    se, t2 = _cap(se, max(limit - len(so), 400))
    if timed_out:
        se += f"\n[sandbox: killed after {timeout_s} s]"
    return SandboxResult(stdout=so, stderr=se, returncode=p.returncode if not timed_out else -9,
                         decodes=decodes, timed_out=timed_out, truncated=t1 or t2)
