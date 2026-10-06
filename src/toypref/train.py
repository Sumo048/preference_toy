"""Training loops: the base policy, and one shared fine-tuning loop for every arm.

Fine-tuning mirrors the preference repo's arm budget (5000 steps, AdamW
b=(0.9, 0.95), wd 1e-10, grad clip 1.0, cosine schedule with 500 warmup steps
decaying to 0.1x peak, no EMA, final weights evaluated). The toy is a full
fine-tune of a small MLP rather than LoRA on a 3B model, so the batch size and
peak LR are the toy's own (see ``EditTrainConfig``); they are the same for every
arm.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import torch

from . import seeds
from .flow import EMA, FlowPolicy, PolicyConfig
from .mixture import MixtureConfig, sample_true


@dataclass(frozen=True)
class BaseTrainConfig:
    steps: int = 30_000
    batch: int = 1024
    lr: float = 1e-3
    lr_min: float = 1e-5
    warmup: int = 1_000
    ema: float = 0.999
    grad_clip: float = 1.0


@dataclass(frozen=True)
class EditTrainConfig:
    steps: int = 5_000
    batch: int = 256
    lr: float = 1e-4
    lr_min_frac: float = 0.1
    warmup: int = 500
    weight_decay: float = 1e-10
    grad_clip: float = 1.0
    log_every: int = 500


def cosine(step, total, warmup, lr, lr_min):
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    prog = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * prog))


def train_base(mix: MixtureConfig, pcfg: PolicyConfig, tcfg: BaseTrainConfig, seed: int,
               log: Callable = print):
    torch.manual_seed(seed)
    pol = FlowPolicy(pcfg)
    ema = EMA(pol, tcfg.ema)
    opt = torch.optim.AdamW(pol.parameters(), lr=tcfg.lr, weight_decay=0.0)
    g = seeds.gen(seed, seeds.BASE_TRAIN)
    rng = seeds.np_rng(seed, seeds.BASE_TRAIN)
    hist = []
    for step in range(tcfg.steps):
        for pg in opt.param_groups:
            pg["lr"] = cosine(step, tcfg.steps, tcfg.warmup, tcfg.lr, tcfg.lr_min)
        x, _ = sample_true(mix, tcfg.batch, rng)
        a = torch.from_numpy(x)
        noise = torch.randn(a.shape, generator=g)
        t = FlowPolicy.sample_time(tcfg.batch, g)
        loss = pol.flow_loss(a, noise, t, reduce="mean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), tcfg.grad_clip)
        opt.step()
        ema.update(pol)
        if step % 2000 == 0 or step == tcfg.steps - 1:
            hist.append({"step": step, "loss": float(loss.detach())})
            log(f"[base s{seed}] step {step} loss {float(loss.detach()):.4f}")
    pol.load_state_dict(ema.shadow)
    return pol, hist


def finetune(init: FlowPolicy, step_loss: Callable, tcfg: EditTrainConfig, seed: int,
             log: Callable = None, probe: Callable = None) -> (FlowPolicy, List[Dict]):
    """Fine-tune a copy of ``init``. ``step_loss(policy, gen) -> (loss, stats)``.

    The minibatch generator is keyed by the experiment seed only, so every arm in a
    seed draws from the same stream.
    """
    import copy
    torch.manual_seed(seed)
    pol = copy.deepcopy(init)
    pol.train()
    for p in pol.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(pol.parameters(), lr=tcfg.lr, betas=(0.9, 0.95),
                            weight_decay=tcfg.weight_decay)
    g = seeds.gen(seed, seeds.TRAIN)
    hist = []
    t0 = time.time()
    for step in range(tcfg.steps):
        lr = cosine(step, tcfg.steps, tcfg.warmup, tcfg.lr, tcfg.lr * tcfg.lr_min_frac)
        for pg in opt.param_groups:
            pg["lr"] = lr
        loss, stats = step_loss(pol, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(pol.parameters(), tcfg.grad_clip)
        opt.step()
        if not torch.isfinite(loss.detach()):
            hist.append({"step": step, "loss": float("nan"), "diverged": True})
            break
        if step % tcfg.log_every == 0 or step == tcfg.steps - 1:
            rec = {"step": step, "loss": float(loss.detach()), "grad_norm": float(gn), "lr": lr,
                   "elapsed_s": round(time.time() - t0, 2)}
            rec.update({k: float(v) for k, v in stats.items()})
            if probe is not None:
                pol.eval()
                rec.update({f"probe_{k}": v for k, v in probe(pol).items()})
                pol.train()
            hist.append(rec)
            if log:
                log(rec)
    pol.eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    return pol, hist
