"""Origin-corrected common-factor contrasts: independent I1 audit candidate.

The frozen v2 factor estimator and strength are reused. Subtracting F(0) from
the common-factor forecast eliminates the artificial duplicated affine bias;
for every shared affine forecaster the complete split/reconstruct mapping then
equals the original B0 exactly (up to floating-point roundoff).
"""
import torch
from torch import nn

from models import Baseline
from i1_context_v2 import Model as FactorV2


class Model(FactorV2):
    MODES = {"stable_factor", "mean_factor", "additive", "ungated", "bias_only", "identity"}

    def __init__(self, config, channels):
        mode = config.get("i1_context_v3_mode", "stable_factor")
        if mode not in self.MODES:
            raise ValueError("unknown i1_context_v3_mode: " + mode)
        # V2 estimates observed factors but does not receive new learned modules.
        effective = dict(config)
        effective["i1_context_v2_mode"] = "stable_factor" if mode == "bias_only" else mode
        super().__init__(effective, channels)
        self.v3_mode = mode

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.length:
            raise ValueError("expected x of shape [batch, configured length, channels]")
        if self.v3_mode == "identity":
            return Baseline.forward(self, x)
        xn, mu, sigma = self.normalize(x)
        factor, loading, gate = self._evidence(xn)
        strength = self.factor_logit.sigmoid() * gate
        common = factor * loading
        if self.v3_mode in {"additive", "bias_only"}:
            individual = xn
        else:
            individual = xn - strength * common
        origin = torch.zeros_like(factor)
        # Origin has its own token but uses exactly the same inherited weights.
        # B0 LayerNorm acts on hidden features within each token, never channels.
        pred = self._forecast(torch.cat((individual, factor, origin), dim=-1))
        if self.v3_mode == "bias_only":
            normalized_forecast = pred[..., :-2] + strength * pred[..., -1:] * loading
        else:
            contrast = pred[..., -2:-1] - pred[..., -1:]
            normalized_forecast = pred[..., :-2] + strength * contrast * loading
        return normalized_forecast * sigma + mu


def _smoke():
    config = dict(length=96, horizon=96, dim=32, ffn_dim=64, layers=2, dropout=0.0)
    torch.manual_seed(2026)
    baseline = Baseline(config, 7)
    x = torch.randn(3, 96, 7)
    base_params = sum(p.numel() for p in baseline.parameters())
    modes = sorted(Model.MODES)
    for mode in modes:
        torch.manual_seed(2026)
        model = Model(dict(config, i1_context_v3_mode=mode), 7)
        assert all(torch.equal(value, model.state_dict()[name])
                   for name, value in baseline.state_dict().items())
        assert sum(p.numel() for p in model.parameters()) == base_params + 1
        y = model(x)
        assert y.shape == (3, 96, 7) and torch.isfinite(y).all()
        if mode == "identity":
            assert torch.equal(y, baseline(x))
        else:
            y.square().mean().backward()
            assert model.factor_logit.grad is not None and torch.isfinite(model.factor_logit.grad).all()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            assert torch.isfinite(model(torch.ones_like(x))).all()
        # Actual forward is exercised with a shared affine forecaster containing
        # a deliberately large, nonconstant horizon bias. No simplified hand
        # reconstruction or no-bias assumptions are used in this check.
        linear = nn.Linear(96, 96)
        with torch.no_grad():
            linear.bias.copy_(torch.linspace(-2, 3, 96))
        model._forecast = lambda z: linear(z.transpose(1, 2)).transpose(1, 2)
        xn, mu, sigma = model.normalize(x)
        direct = model._forecast(xn) * sigma + mu
        output = model(x)
        if mode in {"stable_factor", "mean_factor", "ungated"}:
            assert torch.allclose(output, direct, atol=3e-6, rtol=3e-6)
        elif mode == "bias_only":
            factor, loading, gate = model._evidence(xn)
            s = model.factor_logit.sigmoid() * gate
            expected_extra = s * linear.bias[None, :, None] * loading * sigma
            assert torch.allclose(output-direct, expected_extra, atol=3e-6, rtol=3e-6)
        elif mode == "additive":
            factor, loading, gate = model._evidence(xn)
            s = model.factor_logit.sigmoid() * gate
            no_bias_factor = torch.nn.functional.linear(factor.transpose(1, 2), linear.weight).transpose(1, 2)
            expected_extra = s * no_bias_factor * loading * sigma
            assert torch.allclose(output-direct, expected_extra, atol=3e-6, rtol=3e-6)
    print(dict(smoke="passed", modes=modes, extra_parameters=1,
               affine_with_bias_identity="stable_factor/mean_factor/ungated"))


if __name__ == "__main__":
    _smoke()
