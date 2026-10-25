"""Causal adaptation on frozen features. No ground-truth labels enter this module.

The default FP16 path retains the archived arithmetic, including mixed precision
in geometric warm-up. A zero/nonfinite candidate anchor can instead retain its
previous value. Set normalization='legacy' for the unguarded archived division.
"""
import numpy as np
import torch
from .config import FGGMConfig

BRANCHES = ("CLIP", "Baseline", "Geometry", "Ours")


class FGGM:
    """One independent instance per test stream/dataset.

    text: [C, d] normalized class vectors, FP16 on CUDA for paper arithmetic.
    step(feature, text_logits): feature [d], optional averaged view logits [C].
    Returns four score tensors in BRANCHES order. For multiview inputs, supply
    the mean of selected per-view logits, not recomputed aggregate logits.
    """

    def __init__(self, text: torch.Tensor, config: FGGMConfig | None = None):
        self.config = config or FGGMConfig()
        if text.ndim != 2 or min(text.shape) < 1:
            raise ValueError("text must have shape [classes, feature_dimension]")
        if text.dtype not in (torch.float16, torch.float32):
            raise TypeError("Use float16 for CUDA reproduction or float32 for CPU checks")
        if text.device.type == "cpu" and text.dtype == torch.float16:
            raise ValueError("CPU checks require float32; paper reproduction uses CUDA FP16")
        if not bool(torch.isfinite(text).all()) or bool((text.norm(dim=1) == 0).any()):
            raise ValueError("text vectors must be finite and nonzero")
        self.T = text.detach().clone()
        self.classes, self.dimension = text.shape
        self.reset()

    def reset(self):
        """Reset all adaptation state while retaining the class vectors."""
        self.V = torch.zeros_like(self.T)
        self.A = self.T.clone()
        self.n = self.age = 0
        d, device = self.dimension, self.T.device
        self.sum = torch.zeros(d, device=device, dtype=torch.float64)
        self.xx = torch.zeros((d, d), device=device, dtype=torch.float64)
        self.R = torch.eye(d, device=device, dtype=torch.float32)
        self.mu = torch.zeros(d, device=device, dtype=torch.float32)
        self.warm = torch.zeros(self.classes, device=device, dtype=torch.bool)
        self.prior = np.full(self.classes, 1 / self.classes)

    def _refresh(self):
        c, d = self.config, self.dimension
        if self.n < c.min_history or self.n % c.update_interval:
            return
        mu = self.sum / self.n
        covariance = self.xx / self.n - torch.outer(mu, mu)
        covariance = 0.5 * (covariance + covariance.T)
        scale = (covariance.trace() / d).clamp_min(c.covariance_floor)
        eta = d / (d + self.n)
        eye = torch.eye(d, device=self.T.device, dtype=torch.float64)
        inverse = (scale * torch.linalg.inv((1 - eta) * covariance + eta * scale * eye)).float()
        values, vectors = torch.linalg.eigh(0.5 * (inverse.double() + inverse.double().T))
        if not bool(torch.isfinite(values).all()) or not bool(values.min() > 0):
            raise FloatingPointError("The regularized transform is not positive definite")
        self.R = ((vectors * values.sqrt()[None]) @ vectors.T).float()
        self.mu = mu.float()
        self.age = self.n

    def _prototype_update(self, x, z):
        c = self.config
        z = z[None]
        q = z.softmax(-1).to(self.T.dtype)[0]
        admitted = q >= c.confidence_threshold
        beta = 1 - torch.exp(-q[admitted] / c.h)
        self.V[admitted] = ((1 - beta[:, None]) * self.V[admitted]
                            + beta[:, None] * x[None])
        previous = self.A
        candidate = c.alpha * previous + (1 - c.alpha) * self.V
        norm = candidate.norm(dim=-1, keepdim=True)
        if c.normalization == "guarded":
            valid = torch.isfinite(candidate).all(dim=-1, keepdim=True) & torch.isfinite(norm) & (norm > 0)
            safe_norm = torch.where(valid, norm, torch.ones_like(norm))
            self.A = torch.where(valid, candidate / safe_norm, previous)
        else:
            self.A = candidate / norm
        baseline = z.clone()
        baseline += c.logit_scale * x[None] @ self.A.T
        return z[0], baseline[0], admitted

    def _geometry(self, x, z, baseline):
        c = self.config
        xf, anchors = x.float(), self.A.float()
        original = c.logit_scale * (anchors @ xf)
        if not self.age:
            # Preserve historical parenthesization, rather than returning baseline.
            if c.geometry_weight == 0.5:
                return 0.5 * (baseline.float() + (z.float() + original))
            geometric = original
        else:
            query = self.R @ (xf - self.mu)
            centered = anchors - self.mu
            denominator = query.norm() * centered.norm(dim=-1)
            geometric = c.logit_scale * (query[None] @ centered.T)[0] / denominator.clamp_min(c.denominator_floor)
            usable = (denominator > c.denominator_floor) & torch.isfinite(denominator) & self.warm
            geometric = torch.where(usable, geometric, original)
        if c.geometry_weight == 0.5:
            return 0.5 * (baseline.float() + z.float() + geometric)
        if c.geometry_weight == 0:
            return baseline.float()
        return (1 - c.geometry_weight) * baseline.float() + c.geometry_weight * (z.float() + geometric)

    def _frequency(self, logits):
        c = self.config
        values = logits.cpu().numpy()
        # Both the correction and EMA retain their original NumPy operation order.
        if c.gamma == 0:
            output = values.copy()
        else:
            correction = c.gamma * np.log(self.classes * self.prior).astype("float32")
            output = values - correction
        tempered = values / np.float32(c.temperature)
        probability = np.exp(tempered - tempered.max())
        probability /= probability.sum()
        rate = c.frequency_rate if c.frequency_rate is not None else 1 / (10 * self.classes)
        self.prior = (1 - rate) * self.prior + rate * probability
        return torch.from_numpy(output).to(logits.device)

    @torch.no_grad()
    def step(self, feature: torch.Tensor, text_logits: torch.Tensor | None = None):
        x = feature.detach()
        if x.shape != (self.dimension,) or x.dtype != self.T.dtype or x.device != self.T.device:
            raise ValueError("feature must be [d] with the same dtype and device as text")
        if text_logits is None:
            text_logits = (self.config.logit_scale * x[None] @ self.T.T)[0]
        if text_logits.shape != (self.classes,) or text_logits.dtype != self.T.dtype or text_logits.device != self.T.device:
            raise ValueError("text_logits must be [C] with the same dtype and device as text")
        self._refresh()  # Historical features only, before seeing this observation.
        z, baseline, admitted = self._prototype_update(x, text_logits)
        self.warm |= admitted
        geometry = self._geometry(x, z, baseline)
        ours = self._frequency(geometry)  # Uses previous prior, then updates it.
        xd = x.double()
        self.xx += torch.outer(xd, xd)
        self.sum += xd
        self.n += 1
        return z, baseline, geometry, ours

    def step_with_logits(self, feature, text_logits):
        return self.step(feature, text_logits)
