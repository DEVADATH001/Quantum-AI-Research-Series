"""Shared optimization utilities for quantum and classical learners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class OptimizerStepStats:
    """Diagnostics for a single optimizer step."""

    raw_grad_norm: float
    applied_grad_norm: float
    update_norm: float
    clipped: bool


class AdamOptimizer:
    """Lightweight Adam optimizer for NumPy parameter arrays with gradient clipping."""

    def __init__(
        self,
        lr: float = 0.03,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip: float | None = 1.0,
    ) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.grad_clip = grad_clip
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t = 0

    def set_learning_rate(self, lr: float) -> None:
        self.lr = float(lr)

    def step(self, params: np.ndarray, grads: np.ndarray) -> tuple[np.ndarray, OptimizerStepStats]:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        raw_grad_norm = float(np.linalg.norm(grads))
        applied_grads = np.asarray(grads, dtype=float).copy()
        clipped = False
        if self.grad_clip is not None and raw_grad_norm > self.grad_clip:
            applied_grads *= self.grad_clip / (raw_grad_norm + 1e-6)
            clipped = True

        applied_grad_norm = float(np.linalg.norm(applied_grads))
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * applied_grads
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (applied_grads * applied_grads)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        new_params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        update_norm = float(np.linalg.norm(new_params - params))
        return new_params, OptimizerStepStats(
            raw_grad_norm=raw_grad_norm,
            applied_grad_norm=applied_grad_norm,
            update_norm=update_norm,
            clipped=clipped,
        )
