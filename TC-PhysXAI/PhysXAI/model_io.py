"""
TC-PhysXAI — shared TC-XFlow checkpoint loader
===============================================
One loader for every experiment, so that all of E1–E9 explain exactly the same
network that the training script evaluated:

  * state-dict key: 'model' | 'model_state' | 'model_state_dict' | 'state_dict'
  * architecture:   the checkpoint's own config ('model_cfg' | 'model_kwargs' ...),
                    passed through in FULL (filtered only by the constructor's real
                    parameters). Older XAI scripts kept just 7 keys and silently
                    reset the others (sigma_*, dim_ff, dropout, ...) to defaults.
  * EMA weights:    train_flowmatching.py always evaluates with the EMA shadow
                    (ck['ema']); the raw ck['model'] is NOT what was evaluated.
                    The shadow is applied here (same rule as evaluate_multi_model.py).
                    SWA checkpoints (is_swa=True) already store the average.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

_STATE_KEYS = ("model", "model_state", "model_state_dict", "state_dict")
_CFG_KEYS = ("model_cfg", "model_kwargs", "model_config", "cfg", "config")

_FALLBACK_CFG = {
    "obs_len": 8, "pred_len": 12, "d_cond": 256, "d_model": 256,
    "nhead": 8, "num_dec_layers": 4, "unet_in_ch": 13,
}


def load_tcxflow(ckpt_path: str, device, use_ema: bool = True, verbose: bool = True):
    """Return an eval-mode TCFlowMatching on `device`."""
    from Model.flow_matching_model import TCFlowMatching

    ck = torch.load(ckpt_path, map_location="cpu")
    name = Path(ckpt_path).name
    if verbose:
        print(f"  Loading checkpoint: {name}")

    # ── weights ──────────────────────────────────────────────────────────
    if all(torch.is_tensor(v) for v in ck.values()):
        state, cfg = ck, {}
        print("  [warn] bare state-dict checkpoint — using fallback architecture")
    else:
        state = next((ck[k] for k in _STATE_KEYS if k in ck and isinstance(ck[k], dict)), None)
        if state is None:
            raise KeyError(f"No model weights in checkpoint. Top-level keys: {list(ck.keys())}")
        cfg = next((dict(ck[k]) for k in _CFG_KEYS if isinstance(ck.get(k), dict) and ck[k]), {})
        if not cfg and "args" in ck:
            a = ck["args"]
            cfg = dict(vars(a)) if hasattr(a, "__dict__") else (dict(a) if isinstance(a, dict) else {})

    # ── architecture ─────────────────────────────────────────────────────
    accepted = set(inspect.signature(TCFlowMatching.__init__).parameters) - {"self"}
    if not any(k in cfg for k in _FALLBACK_CFG):
        print("  [warn] no architecture config in checkpoint — using fallback kwargs")
        cfg = {**_FALLBACK_CFG, **{k: v for k, v in cfg.items() if k in accepted}}
    kwargs = {k: v for k, v in cfg.items() if k in accepted}
    for k, v in _FALLBACK_CFG.items():
        kwargs.setdefault(k, v)
    model = TCFlowMatching(**kwargs)

    missing, unexpected = model.load_state_dict(state, strict=False)
    n_total = len(model.state_dict())
    if verbose:
        print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected "
              f"(of {n_total} tensors)")
    if len(missing) > 0.2 * n_total:
        raise RuntimeError(
            f"{len(missing)}/{n_total} weights missing — checkpoint does not match the "
            f"architecture built from its config. First missing: {missing[:5]}")

    # ── EMA shadow (what training-time evaluation actually used) ─────────
    is_swa = bool(ck.get("is_swa", False))
    ema = ck.get("ema")
    if use_ema and ema and not is_swa and isinstance(ema, dict):
        sd = model.state_dict()
        applied = 0
        for k, v in ema.items():
            if k in sd and torch.is_tensor(v) and sd[k].shape == v.shape:
                sd[k].copy_(v.to(sd[k].dtype))
                applied += 1
        if verbose:
            print(f"  ✓ EMA shadow applied ({applied} tensors) — matches training-time evaluation")
    elif verbose:
        why = "SWA average (already averaged)" if is_swa else (
            "use_ema=False" if not use_ema else "no 'ema' key — RAW weights")
        print(f"  ℹ EMA not applied: {why}")

    model.to(device).eval()
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
    return model
