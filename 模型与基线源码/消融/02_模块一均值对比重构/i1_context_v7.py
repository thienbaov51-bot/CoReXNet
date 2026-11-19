"""Gate-matched controls for the frozen v3 ungated factor predictor.

This is an audit version, not a new innovation. Every mode has gate=1, B0+one
strength, the same inherited initialization, and exactly C+2 network tokens.
The pca_zero mode accepts a v3 state_dict strictly and reproduces its ungated
forward. The mean and additive controls remove the old half-window gate
confound; pca_odd changes only the factor response's sign constraint.
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from i1_context_v2 import Model as FactorV2
from models import Baseline


class Model(FactorV2):
    V7_MODES = {"pca_zero", "mean_zero", "additive_zero", "pca_odd"}
    MODES = FactorV2.MODES

    def __init__(self, config, channels):
        mode = config.get("i1_context_v7_mode", "pca_zero")
        if mode not in self.V7_MODES:
            raise ValueError("unknown i1_context_v7_mode: " + mode)
        effective = dict(config)
        effective["i1_context_v2_mode"] = "mean_factor" if mode == "mean_zero" else "ungated"
        super().__init__(effective, channels)
        self.v7_mode = mode

    @torch.no_grad()
    def _evidence(self, xn):
        with torch.autocast(device_type=xn.device.type, enabled=False):
            observed = xn.detach().float()
            estimator = self._mean_factor if self.v7_mode == "mean_zero" else self._principal_factor
            factor, loading = estimator(observed)
            # Exactly one complete observed-window estimate. No half windows.
            gate = torch.ones(observed.shape[0], 1, 1, device=observed.device, dtype=observed.dtype)
            return factor, loading, gate

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.length:
            raise ValueError("expected x of shape [batch, configured length, channels]")
        xn, mu, sigma = self.normalize(x)
        factor, loading, gate = self._evidence(xn)
        strength = self.factor_logit.sigmoid()*gate
        # Preserve frozen v3's operation order for trained scalar values too.
        # (s*f)*a and s*(f*a) differ by float32 roundoff when s is not 0.5.
        common = factor*loading
        residual = xn if self.v7_mode == "additive_zero" else xn-strength*common
        alternate = -factor if self.v7_mode == "pca_odd" else torch.zeros_like(factor)
        pred = self._forecast(torch.cat((residual, factor, alternate), dim=-1))
        contrast = pred[..., -2:-1]-pred[..., -1:]
        if self.v7_mode == "pca_odd":
            contrast = 0.5*contrast
        normalized = pred[..., :-2]+strength*contrast*loading
        return normalized*sigma+mu


def _smoke(device):
    from i1_context_v3 import Model as OriginV3

    config = dict(length=96, horizon=96, dim=32, ffn_dim=64, layers=2, dropout=0.0)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2026)
    baseline = Baseline(config, 7).to(device)
    base_params = sum(p.numel() for p in baseline.parameters())
    x = torch.randn(4, 96, 7, device=device)
    signs = torch.tensor([1., -1., 1., -1.], device=device)[:, None, None]
    results = []
    try:
        for mode in sorted(Model.V7_MODES):
            torch.manual_seed(2026)
            model = Model(dict(config, i1_context_v7_mode=mode), 7).to(device)
            assert all(torch.equal(value, model.state_dict()[name])
                       for name, value in baseline.state_dict().items())
            assert sum(p.numel() for p in model.parameters()) == base_params+1
            original_forecast = model._forecast
            original_estimator = model._mean_factor if mode == "mean_zero" else model._principal_factor
            estimator_calls = []
            network_calls = []

            def record_estimator(z):
                estimator_calls.append(tuple(z.shape))
                return original_estimator(z)

            def record_forecast(z):
                network_calls.append(z.shape[-1])
                return original_forecast(z)

            estimator_name = "_mean_factor" if mode == "mean_zero" else "_principal_factor"
            setattr(model, estimator_name, record_estimator)
            model._forecast = record_forecast
            y = model(x)
            assert estimator_calls == [(4, 96, 7)]
            assert network_calls == [9]
            setattr(model, estimator_name, original_estimator)
            model._forecast = original_forecast
            assert y.shape == (4, 96, 7) and torch.isfinite(y).all()
            y.square().mean().backward()
            assert model.factor_logit.grad is not None and torch.isfinite(model.factor_logit.grad).all()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            assert torch.isfinite(model(torch.ones_like(x))).all()
            assert torch.isfinite(model(x[:, :, :1])).all()
            xn, mu, sigma = model.normalize(x)
            factor, loading, gate = model._evidence(xn)
            assert torch.equal(gate, torch.ones_like(gate))
            assert factor.dtype == loading.dtype == gate.dtype == torch.float32
            assert not factor.requires_grad and not loading.requires_grad
            torch.testing.assert_close(factor.square().mean(1), torch.ones_like(factor[:, 0]), atol=2e-6, rtol=2e-6)

            linear = nn.Linear(96, 96).to(device)
            with torch.no_grad():
                linear.bias.copy_(torch.linspace(-2, 3, 96, device=device))
            model._forecast = lambda z: linear(z.transpose(1, 2)).transpose(1, 2)
            direct = model._forecast(xn)*sigma+mu
            affine = model(x)
            if mode == "additive_zero":
                linear_factor = torch.nn.functional.linear(factor.transpose(1, 2), linear.weight).transpose(1, 2)
                expected = direct+model.factor_logit.sigmoid()*linear_factor*loading*sigma
            else:
                expected = direct
            affine_error = (affine-expected).abs().max().item()
            torch.testing.assert_close(affine, expected, atol=3e-6, rtol=3e-6)
            model._forecast = original_forecast

            original_evidence = model._evidence
            model._evidence = lambda _: (factor*signs, loading*signs, gate)
            sign_output = model(x)
            gauge_error = (sign_output-y.detach()).abs().max().item()
            if mode == "pca_odd":
                torch.testing.assert_close(sign_output, y.detach(), atol=2e-6, rtol=2e-6)
            model._evidence = original_evidence

            strict_load_pass = None
            parity_error = None
            loaded_parity_error = None
            if mode == "pca_zero":
                torch.manual_seed(2026)
                old = OriginV3(dict(config, i1_context_v3_mode="ungated"), 7).to(device)
                parity_error = (old(x)-model(x)).abs().max().item()
                assert parity_error == 0.0
                # A perturbed state establishes compatibility beyond matching
                # initial weights, without training or reading any checkpoint.
                with torch.no_grad():
                    old.factor_logit.copy_(torch.tensor(0.7, device=device))
                    old.embedding[0].bias.add_(0.13)
                    old.head.bias.add_(torch.linspace(-0.2, 0.3, 96, device=device))
                incompatible = model.load_state_dict(old.state_dict(), strict=True)
                assert not incompatible.missing_keys and not incompatible.unexpected_keys
                strict_load_pass = True
                loaded_parity_error = (old(x)-model(x)).abs().max().item()
                assert loaded_parity_error == 0.0

            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    amp = model(x)
                    loss = amp.float().square().mean()
                    f_amp, a_amp, g_amp = model._evidence(xn)
                loss.backward()
                assert torch.isfinite(amp).all()
                assert f_amp.dtype == a_amp.dtype == g_amp.dtype == torch.float32
                assert torch.equal(g_amp, torch.ones_like(g_amp))
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                if mode == "pca_zero":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        torch.testing.assert_close(old(x), model(x), atol=0, rtol=0)
                torch.backends.cuda.matmul.allow_tf32 = False
            results.append(dict(mode=mode, same_initialization=True, extra_parameters=1,
                                shared_trajectories=9, call_count_formula="C+2",
                                full_window_estimator_calls=1, half_window_estimator_calls=0,
                                gate_exactly_one=True, statistics_detached_fp32=True,
                                affine_max_abs_error=affine_error,
                                sign_gauge_max_abs_error=gauge_error,
                                frozen_v3_parity_max_abs_error=parity_error,
                                v3_strict_state_load=strict_load_pass,
                                v3_loaded_state_parity_max_abs_error=loaded_parity_error))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    result = dict(smoke="PASS", scope="Synthetic histories only; no metric/test/checkpoint reading",
                  device=str(device), torch=torch.__version__, results=results)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--qa-output")
    args = parser.parse_args()
    name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if name == "auto":
        name = "cpu"
    result = _smoke(torch.device(name))
    if args.qa_output:
        Path(args.qa_output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
