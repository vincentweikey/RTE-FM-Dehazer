from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Any, Callable
from functools import partial
from torchdiffeq import odeint


# ----------------------------------------- schedules -----------------------------------------
class LinearSchedule:
    """α(t)=t, σ(t)=1−t."""
    def alpha_t(self, t: Tensor) -> Tensor:
        return t

    def sigma_t(self, t: Tensor) -> Tensor:
        return 1 - t

    def alpha_dt_t(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def sigma_dt_t(self, t: Tensor) -> Tensor:
        return -torch.ones_like(t)


class GVPSchedule(LinearSchedule):
    """Generalized Variance Path: α=sin(πt/2), σ=cos(πt/2)."""
    def alpha_t(self, t: Tensor) -> Tensor:
        return torch.sin(math.pi / 2 * t)

    def sigma_t(self, t: Tensor) -> Tensor:
        return torch.cos(math.pi / 2 * t)

    def alpha_dt_t(self, t: Tensor) -> Tensor:
        return math.pi / 2 * torch.cos(math.pi / 2 * t)

    def sigma_dt_t(self, t: Tensor) -> Tensor:
        return -math.pi / 2 * torch.sin(math.pi / 2 * t)


# ------------- RTE Consistency -------------
class RTEConsistency4chFixed(nn.Module):
    """
        D   = 1 / (3μ_t + ε)   
        μ_a = κ * mean(|xt|) 
    """
    def __init__(self, kappa: float = 0.1, dx: float = 1.0):
        super().__init__()
        self.kappa = kappa   # scalar of abosrb 
        self.dx  = dx

    def forward(self, xt: Tensor, vt: Tensor) -> Tensor:
        B, C, H, W = xt.shape

        # 1. RTE parameters estimation
        mu_a = self.kappa * xt.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
        mu_s = 1.0 - mu_a                                       # fixed
        mu_t = mu_a + mu_s + 1e-6
        D = 1.0 / (3.0 * mu_t)                                  # D-index

        # 2. Laplacian
        lap_kernel = torch.tensor([[[[0, 1, 0],
                                    [1,-4, 1],
                                    [0, 1, 0]]]],
                                  dtype=xt.dtype, device=xt.device)
        lap = torch.nn.functional.conv2d(
            xt, lap_kernel.expand(C, 1, -1, -1),
            padding=1, groups=C) / (self.dx ** 2)   # [B,4,H,W]

        # 3. D + abosorb
        diff_term = D * lap
        absorb_term = mu_a * xt
        vt_rte = -diff_term - absorb_term

        # 4. RTE strict 
        return (vt - vt_rte).pow(2).mean()

# ----------------------------------------- core model -----------------------------------------
class FlowMatcher(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 schedule: str = "linear",
                 sigma_min: float = 0.0,
                 rte_weight: float = 0.5):
        
        super().__init__()
        self.net = backbone
        self.sigma_min = sigma_min
        self.sched = {"linear": LinearSchedule(), "gvp": GVPSchedule()}[schedule]
        
        self.rte_loss_fn = RTEConsistency4chFixed()
        self.rte_weight = rte_weight 

    # -------------------------------- training --------------------------------
    def compute_xt(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """xt = α(t)x1 + σ(t)x0 + σ_min·ε"""
        assert t.ndim == 1, f"t must be 1-D, got {t.shape}"
        t4d = t.view(-1, 1, 1, 1)         
        α = self.sched.alpha_t(t4d)
        σ = self.sched.sigma_t(t4d)
        xt = α * x1 + σ * x0
        if self.sigma_min > 0:
            xt += self.sigma_min * torch.randn_like(xt)
        return xt

    def compute_ut(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """target vector field: ut = α̇(t)x1 + σ̇(t)x0"""
        t4d = t.view(-1, 1, 1, 1)
        return self.sched.alpha_dt_t(t4d) * x1 + self.sched.sigma_dt_t(t4d) * x0

    def training_losses(self,
                        x1: Tensor,
                        x0: Optional[Tensor] = None,
                        **net_kwargs) -> Tensor:
        if x0 is None:
            x0 = torch.randn_like(x1)
        b = x1.size(0)
        t = torch.rand(b, device=x1.device, dtype=x1.dtype)   # 1-D
        xt = self.compute_xt(x0, x1, t)
        ut = self.compute_ut(x0, x1, t)
        vt = self.net(xt, t * 1000, **net_kwargs)             # keep 1-D
        
        fm_loss = (vt - ut).pow(2).mean()
        rte_loss = self.rte_loss_fn(xt, vt)
        
        return fm_loss + self.rte_weight * rte_loss

    # -------------------------------- sampling --------------------------------
    @torch.no_grad()
    def generate(self,
                 x0: Tensor,
                 num_steps: int = 50,
                 method: str = "euler",
                 atol: float = 1e-4,
                 rtol: float = 1e-2,
                 **net_kwargs) -> Tensor:
        """ODE sampling: x0 -> x1."""
        t = torch.linspace(0, 1, num_steps, device=x0.device)

        def ode_fn(_t: Tensor, _x: Tensor) -> Tensor:
            # _t is scalar, _x is (B,C,H,W)
            return self.net(_x, _t.expand(_x.size(0)) * 1000, **net_kwargs)

        traj = odeint(ode_fn, x0, t, method=method, atol=atol, rtol=rtol)
        return traj[-1]


# -------------------------------- quick test --------------------------------
if __name__ == "__main__":
    from models.unet import EfficientUNet
    net = EfficientUNet(3, 64, 3, num_res_blocks=1, attention_resolutions=[8])
    model = FlowMatcher(net)
    x1 = torch.randn(3, 3, 512, 512)
    x0 = torch.randn_like(x1)
    loss = model.training_losses(x1, x0)
    print("loss:", loss.item())
    with torch.no_grad():
        x1_hat = model.generate(x0, num_steps=25)
        print("sample shape:", x1_hat.shape)
