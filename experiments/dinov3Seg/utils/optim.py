"""
Optimization utilities: Muon optimizer and early stopping helper.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import Tensor
from torch import optim


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """
    Orthogonalize matrix updates using the Newton-Schulz iteration.

    >>> _ = torch.manual_seed(0)
    >>> G = torch.randn(4, 4)
    >>> out = zeropower_via_newtonschulz5(G, steps=2)
    >>> out.shape
    torch.Size([4, 4])
    """

    if G.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(optim.Optimizer):
    """
    Muon optimizer: orthogonalized momentum for >=2D weights plus AdamW for
    1D tensors.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_params: Optional[Iterable[Tensor]] = None,
        adamw_lr: float = 1e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_wd: float = 0.01,
    ) -> None:
        """
        Build the Muon optimizer.

        >>> _ = torch.manual_seed(0)
        >>> w = torch.nn.Parameter(torch.randn(2, 2))
        >>> opt = Muon([w], lr=0.01)
        >>> isinstance(opt, Muon)
        True
        """

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_params=list(adamw_params) if adamw_params is not None else None,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform one optimization step.

        >>> _ = torch.manual_seed(0)
        >>> w = torch.nn.Parameter(torch.randn(2, 2))
        >>> opt = Muon([w], lr=0.01)
        >>> loss = (w ** 2).sum()
        >>> loss.backward()
        >>> opt.step()
        """

        for group in self.param_groups:
            if group["adamw_params"] is not None:
                self._step_adamw(group)
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                state = self.state[p]
                buf = state.setdefault("momentum_buffer", torch.zeros_like(g))
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                if g.shape != p.shape:
                    g = g.view_as(p)
                p.data.add_(g, alpha=-lr)

    def _step_adamw(self, group) -> None:
        """
        Handle AdamW updates for the provided parameter subset.

        >>> base = torch.nn.Parameter(torch.randn(2, 2))
        >>> p = torch.nn.Parameter(torch.ones(2))
        >>> p.grad = torch.tensor([0.1, -0.2])
        >>> opt = Muon([base], adamw_params=[p])
        >>> opt._step_adamw(opt.param_groups[0])
        """

        lr = group["adamw_lr"]
        beta1, beta2 = group["adamw_betas"]
        wd = group["adamw_wd"]
        eps = 1e-8
        for p in group["adamw_params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
            denom = exp_avg_sq.sqrt().add_(eps)
            step_size = lr * torch.sqrt(torch.tensor(1 - beta2 ** state["step"])) / (
                1 - beta1 ** state["step"]
            )
            p.data.mul_(1 - lr * wd)
            p.data.addcdiv_(exp_avg, denom, value=-step_size.item())


class EarlyStopping:
    """
    Track validation metric improvements and stop once patience runs out.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        path: str = "checkpoint.pth",
        mode: str = "min",
    ) -> None:
        """
        Initialize the tracker.

        >>> es = EarlyStopping(patience=2, min_delta=0.1, path="tmp.pth")
        >>> es.best_score is None
        True
        """

        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def _is_improvement(self, value: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return value < self.best_score - self.min_delta
        return value > self.best_score + self.min_delta

    def __call__(self, metric: float, model: torch.nn.Module) -> None:
        """
        Update state with the latest validation metric.

        >>> es = EarlyStopping(patience=1, min_delta=0.0, path="tmp.pth", mode="max")
        >>> class Dummy(torch.nn.Module):
        ...     def state_dict(self):
        ...         return {}
        >>> model = Dummy()
        >>> es(0.5, model)
        >>> es(0.4, model)
        >>> es.early_stop
        True
        """

        if self._is_improvement(metric):
            self.best_score = metric
            self.save_checkpoint(metric, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, metric: float, model: torch.nn.Module) -> None:
        """
        Persist the model weights to disk.

        >>> es = EarlyStopping(path="tmp_checkpoint.pth")
        >>> class Dummy(torch.nn.Module):
        ...     def state_dict(self):
        ...         return {"w": torch.tensor([1.0])}
        >>> model = Dummy()
        >>> es.save_checkpoint(0.5, model)
        """

        torch.save(model.state_dict(), self.path)
