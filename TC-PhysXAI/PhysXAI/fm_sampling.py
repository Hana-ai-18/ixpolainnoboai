"""
TC-PhysXAI — deterministic replica of TCFlowMatching.sample()'s generation step
===============================================================================
The XAI modules that re-integrate the flow (C4, C5, C6) must generate tracks the
same way `model.sample` does, otherwise they explain a different model:

  * initial noise  x0 ~ N(0, sigma_inference^2)       (sigma_inference = 0.04, NOT 1)
  * N = model.n_inference_steps Euler steps (default 10), each clamped to [-3, 3];
    N <= 1 -> single shot x0 + v(x0, 0, cond)
  * final clamp [-20, 20], then relative -> absolute with the last observed point

The candidate scoring / top-3 weighting / speed calibration of `sample` are NOT
applied: this returns raw candidates (what C5's V_noise is computed on).
"""

from __future__ import annotations

from typing import Optional

import torch


def sample_z0(model, batch: int, pred_len: int, device, generator: Optional[torch.Generator] = None):
    return torch.randn(batch, pred_len, 2, device=device, generator=generator) * model.sigma_inference


def fm_integrate(model, z0_rel: torch.Tensor, cond: torch.Tensor, last_obs: torch.Tensor,
                 ddim_steps: Optional[int] = None) -> torch.Tensor:
    """Returns ABSOLUTE normalised tracks [T_pred, B, 2] (differentiable w.r.t. cond/z0)."""
    N = ddim_steps if (ddim_steps is not None and ddim_steps > 1) else model.n_inference_steps
    dt = 1.0 / max(N, 1)
    B = z0_rel.shape[0]
    x = z0_rel
    if N <= 1:
        x = x + model.velocity(x, torch.zeros(B, device=x.device), cond)
    else:
        for step in range(N):
            t_b = torch.full((B,), step * dt, device=x.device)
            x = (x + dt * model.velocity(x, t_b, cond)).clamp(-3.0, 3.0)
    x = x.clamp(-20.0, 20.0)
    return model._from_relative(x, last_obs).permute(1, 0, 2)
