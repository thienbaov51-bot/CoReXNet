"""Stable signed common-factor residualization for the frozen CI-ResMLP.

This is an independent innovation-1 candidate, not an extension of the frozen
history-centering v1. Only observed x is used. All factor estimates are detached;
one shared Baseline forecaster predicts individual residuals and one common
factor, whose signed channel loadings then reconstruct the forecast.
"""
import math

import torch
from torch import nn

from models import Baseline


class Model(Baseline):
    MODES = {"stable_factor", "mean_factor", "additive", "ungated", "identity"}

    def __init__(self, config, channels):
        super().__init__(config, channels)
        self.mode = config.get("i1_context_v2_mode", "stable_factor")
        if self.mode not in self.MODES:
            raise ValueError("unknown i1_context_v2_mode: " + self.mode)
        self.iterations = int(config.get("i1_context_v2_iterations", 6))
        if self.iterations < 1:
            raise ValueError("factor power iterations must be positive")
        strength = float(config.get("i1_context_v2_strength_init", 0.5))
        if not 0.0 < strength < 1.0:
            raise ValueError("strength initialization must lie in (0, 1)")
        # No RNG draw: every inherited weight exactly matches B0 at a fixed seed.
        self.factor_logit = nn.Parameter(torch.tensor(math.log(strength/(1-strength))))

    @staticmethod
    def _unit(x, dim):
        return x / x.square().sum(dim=dim, keepdim=True).sqrt().clamp_min(1e-8)

    def _principal_factor(self, x):
        """Return f [B,T,1], a [B,1,C], with mean(f**2) approximately 1.

        Three fixed cosine probes avoid a random seed dependence and reduce a
        single initialization's chance of missing the dominant eigendirection.
        X(X' q) needs O(B*T*C*K), rather than a dense C-by-C relation matrix.
        """
        batch, length, _ = x.shape
        pos = torch.arange(length, device=x.device, dtype=x.dtype) + 0.5
        frequencies = torch.arange(1, 4, device=x.device, dtype=x.dtype)
        probes = torch.cos(math.pi * pos[:, None] * frequencies[None, :] / length)
        probes = probes - probes.mean(dim=0, keepdim=True)
        q = self._unit(probes, 0).unsqueeze(0).expand(batch, -1, -1)
        for _ in range(self.iterations):
            q = self._unit(torch.bmm(x, torch.bmm(x.transpose(1, 2), q)), 1)
        projections = torch.bmm(x.transpose(1, 2), q)
        winner = projections.square().sum(dim=1).argmax(dim=-1)
        q = q.gather(2, winner[:, None, None].expand(-1, length, 1))
        factor = q * math.sqrt(length)
        loading = torch.bmm(q.transpose(1, 2), x) / math.sqrt(length)
        # PCA signs are arbitrary. Use the largest-magnitude loading to orient
        # each factor, consistently for its residualization and reconstruction.
        pivot = loading.abs().argmax(dim=-1, keepdim=True)
        sign = loading.gather(-1, pivot).sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return factor * sign, loading * sign

    @staticmethod
    def _mean_factor(x):
        factor = x.mean(dim=-1, keepdim=True)
        factor = factor / factor.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        loading = torch.bmm(factor.transpose(1, 2), x) / x.shape[1]
        return factor, loading

    @torch.no_grad()
    def _evidence(self, xn):
        # Power iteration/stability calculations run in float32 under AMP.
        with torch.autocast(device_type=xn.device.type, enabled=False):
            x = xn.detach().float()
            if self.mode == "mean_factor":
                factor_fn = self._mean_factor
            else:
                factor_fn = self._principal_factor
            factor, loading = factor_fn(x)
            if self.mode == "ungated":
                gate = torch.ones(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
            else:
                split = x.shape[1] // 2
                left, right = x[:, :split], x[:, split:]
                # Half-window level differences must not create a spurious
                # common loading agreement. Center each half before estimation.
                left = left - left.mean(dim=1, keepdim=True)
                right = right - right.mean(dim=1, keepdim=True)
                _, a_left = factor_fn(left)
                _, a_right = factor_fn(right)
                agreement = (self._unit(a_left, -1) * self._unit(a_right, -1)).sum(-1, keepdim=True)
                # Squared cosine is invariant to separate PCA sign conventions.
                gate = agreement.square().clamp(0.0, 1.0)
            return factor, loading, gate

    def _forecast(self, x):
        return self.head(self.encoder(self.embedding(x.transpose(1, 2)))).transpose(1, 2)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.length:
            raise ValueError("expected x of shape [batch, configured length, channels]")
        if self.mode == "identity":
            return super().forward(x)
        xn, mu, sigma = self.normalize(x)
        factor, loading, gate = self._evidence(xn)
        strength = self.factor_logit.sigmoid() * gate
        common = factor * loading
        individual = xn if self.mode == "additive" else xn - strength * common
        # All C residuals and the extra factor use exactly the inherited network;
        # no expert, channel embedding, temporal basis or extra output head.
        pred = self._forecast(torch.cat((individual, factor), dim=-1))
        normalized_forecast = pred[..., :-1] + strength * pred[..., -1:] * loading
        return normalized_forecast * sigma + mu


def _smoke():
    config = dict(length=96, horizon=96, dim=32, ffn_dim=64, layers=2, dropout=0.0)
    torch.manual_seed(2026)
    baseline = Baseline(config, 7)
    base_params = sum(p.numel() for p in baseline.parameters())
    x = torch.randn(3, 96, 7)
    results = {}
    for mode in sorted(Model.MODES):
        torch.manual_seed(2026)
        model = Model(dict(config, i1_context_v2_mode=mode), 7)
        assert all(torch.equal(value, model.state_dict()[name])
                   for name, value in baseline.state_dict().items())
        y = model(x)
        assert y.shape == (3, 96, 7) and torch.isfinite(y).all()
        assert sum(p.numel() for p in model.parameters()) == base_params + 1
        if mode == "identity":
            assert torch.equal(y, baseline(x))
        else:
            y.square().mean().backward()
            assert model.factor_logit.grad is not None
            assert torch.isfinite(model.factor_logit.grad).all()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            const = model(torch.ones_like(x))
            assert torch.isfinite(const).all()
            factor, loading, gate = model._evidence(model.normalize(x)[0])
            assert torch.isfinite(factor).all() and torch.isfinite(loading).all()
            assert ((gate >= 0) & (gate <= 1)).all()
            # Counterfactual shared linear forecaster has exact decomposition
            # equivalence in the absence of bias; this candidate gets its
            # additional inductive bias from separate nonlinear trajectories.
            if mode != "additive":
                w = torch.randn(96, 96)
                s = model.factor_logit.sigmoid() * gate
                residual = model.normalize(x)[0] - s * factor * loading
                combined = torch.einsum("bth,ts->bsh", residual, w) + s * torch.einsum("bth,ts->bsh", factor, w) * loading
                direct = torch.einsum("bth,ts->bsh", model.normalize(x)[0], w)
                assert torch.allclose(combined, direct, atol=2e-5, rtol=2e-5)
        results[mode] = dict(shape=list(y.shape), extra_params=1)
    # Sign-opposed channels retain a rank-one common driver which the mean
    # factor loses exactly. This is a mechanism check, not a forecast test.
    phase = torch.arange(96).float()
    driver = torch.sin(2*math.pi*phase/24)[None, :, None]
    opposed = driver * torch.tensor([1., -1., 1., -1.])[None, None, :]
    model = Model(config, 4)
    f, a = model._principal_factor(opposed)
    fm, am = model._mean_factor(opposed)
    assert torch.allclose(f*a, opposed, atol=1e-5, rtol=1e-5)
    assert torch.equal(fm*am, torch.zeros_like(opposed))
    print(results)


if __name__ == "__main__":
    _smoke()
