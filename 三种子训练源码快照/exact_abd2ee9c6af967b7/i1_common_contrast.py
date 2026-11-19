"""Shared common/individual forecast contrast with gate-matched controls.

MeanCore is an untrained observed-history frontend. The novelty under review is
the complete shared-F, origin-corrected forecasting operator, not mean pooling
or PCA. Existing v7 mean_zero and v3 ungated state_dicts load strictly into the
corresponding residual modes without changing the forward or retraining them.
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from i1_context_v2 import Model as FactorV2
from models import Baseline


class MeanCore(nn.Module):
    """Return a unit-RMS mean factor and signed observed OLS loadings.

    No parameter, persistent buffer, RNG draw, gate or half-window statistic.
    The frozen v2 arithmetic is reused for exact v7 mean_zero compatibility.
    """

    @torch.no_grad()
    def forward(self, normalized_history):
        with torch.autocast(device_type=normalized_history.device.type, enabled=False):
            observed = normalized_history.detach().float()
            return FactorV2._mean_factor(observed)


class Model(FactorV2):
    CONTRAST_MODES = {"mean_residual", "pca_residual", "mean_additive"}
    MODES = FactorV2.MODES

    def __init__(self, config, channels):
        mode = config.get("i1_common_contrast_mode", "mean_residual")
        if mode not in self.CONTRAST_MODES:
            raise ValueError("unknown i1_common_contrast_mode: " + mode)
        effective = dict(config)
        effective["i1_context_v2_mode"] = "ungated" if mode == "pca_residual" else "mean_factor"
        super().__init__(effective, channels)
        self.contrast_mode = mode
        self.mean_core = MeanCore()

    @torch.no_grad()
    def _evidence(self, xn):
        with torch.autocast(device_type=xn.device.type, enabled=False):
            observed = xn.detach().float()
            if self.contrast_mode == "pca_residual":
                factor, loading = self._principal_factor(observed)
            else:
                factor, loading = self.mean_core(observed)
            gate = torch.ones(observed.shape[0], 1, 1, device=observed.device, dtype=observed.dtype)
            return factor, loading, gate

    def _shared_contrast(self, normalized_history, factor, loading, gate):
        strength = self.factor_logit.sigmoid()*gate
        common = factor*loading
        residual = normalized_history if self.contrast_mode == "mean_additive" else normalized_history-strength*common
        origin = torch.zeros_like(factor)
        pred = self._forecast(torch.cat((residual, factor, origin), dim=-1))
        contrast = pred[..., -2:-1]-pred[..., -1:]
        return pred[..., :-2]+strength*contrast*loading

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.length:
            raise ValueError("expected x of shape [batch, configured length, channels]")
        normalized_history, mu, sigma = self.normalize(x)
        factor, loading, gate = self._evidence(normalized_history)
        normalized = self._shared_contrast(normalized_history, factor, loading, gate)
        return normalized*sigma+mu


def _smoke(device):
    from i1_context_v3 import Model as OriginV3
    from i1_context_v7 import Model as MatchedV7

    config = dict(length=96, horizon=96, dim=32, ffn_dim=64, layers=2, dropout=0.0)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2026)
    baseline = Baseline(config, 7).to(device)
    base_params = sum(p.numel() for p in baseline.parameters())
    x = torch.randn(4, 96, 7, device=device)
    results = []
    try:
        for mode in sorted(Model.CONTRAST_MODES):
            torch.manual_seed(2026)
            model = Model(dict(config, i1_common_contrast_mode=mode), 7).to(device)
            assert all(torch.equal(value, model.state_dict()[name])
                       for name, value in baseline.state_dict().items())
            assert sum(p.numel() for p in model.parameters()) == base_params+1
            assert list(model.mean_core.state_dict()) == []
            calls = []
            original_forecast = model._forecast

            def record_forecast(z):
                calls.append(z.shape[-1])
                return original_forecast(z)

            model._forecast = record_forecast
            y = model(x)
            assert calls == [9]
            model._forecast = original_forecast
            assert y.shape == (4, 96, 7) and torch.isfinite(y).all()
            y.square().mean().backward()
            assert model.factor_logit.grad is not None and torch.isfinite(model.factor_logit.grad).all()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            assert torch.isfinite(model(torch.ones_like(x))).all()
            assert torch.isfinite(model(x[:, :, :1])).all()
            xn, mu, sigma = model.normalize(x)
            factor, loading, gate = model._evidence(xn)
            assert factor.dtype == loading.dtype == torch.float32
            assert not factor.requires_grad and not loading.requires_grad
            assert torch.equal(gate, torch.ones_like(gate))

            linear = nn.Linear(96, 96).to(device)
            with torch.no_grad():
                linear.bias.copy_(torch.linspace(-2, 3, 96, device=device))
            model._forecast = lambda z: linear(z.transpose(1, 2)).transpose(1, 2)
            direct = model._forecast(xn)*sigma+mu
            affine = model(x)
            if mode == "mean_additive":
                linear_factor = torch.nn.functional.linear(factor.transpose(1, 2), linear.weight).transpose(1, 2)
                expected = direct+model.factor_logit.sigmoid()*linear_factor*loading*sigma
            else:
                expected = direct
            affine_error = (affine-expected).abs().max().item()
            torch.testing.assert_close(affine, expected, atol=3e-6, rtol=3e-6)
            model._forecast = original_forecast

            counterpart = None
            counterpart_name = None
            loaded_error = None
            parity_error = None
            if mode != "mean_additive":
                old_mode = "mean_zero" if mode == "mean_residual" else "pca_zero"
                torch.manual_seed(2026)
                counterpart = MatchedV7(dict(config, i1_context_v7_mode=old_mode), 7).to(device)
                counterpart_name = "v7 "+old_mode
                parity_error = (counterpart(x)-model(x)).abs().max().item()
                assert parity_error == 0.0
                with torch.no_grad():
                    counterpart.factor_logit.copy_(torch.tensor(0.7, device=device))
                    counterpart.embedding[0].bias.add_(0.13)
                    counterpart.head.bias.add_(torch.linspace(-0.2, 0.3, 96, device=device))
                loaded = model.load_state_dict(counterpart.state_dict(), strict=True)
                assert not loaded.missing_keys and not loaded.unexpected_keys
                loaded_error = (counterpart(x)-model(x)).abs().max().item()
                assert loaded_error == 0.0
                if mode == "pca_residual":
                    old_v3 = OriginV3(dict(config, i1_context_v3_mode="ungated"), 7).to(device)
                    old_v3.load_state_dict(model.state_dict(), strict=True)
                    assert (old_v3(x)-model(x)).abs().max().item() == 0.0

            if device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    amp = model(x)
                    loss = amp.float().square().mean()
                    f_amp, a_amp, gate_amp = model._evidence(xn)
                loss.backward()
                assert torch.isfinite(amp).all()
                assert f_amp.dtype == a_amp.dtype == torch.float32
                assert torch.equal(gate_amp, torch.ones_like(gate_amp))
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                if counterpart is not None:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        torch.testing.assert_close(counterpart(x), model(x), atol=0, rtol=0)
                        if mode == "pca_residual":
                            torch.testing.assert_close(old_v3(x), model(x), atol=0, rtol=0)
                torch.backends.cuda.matmul.allow_tf32 = False
            results.append(dict(mode=mode, same_initialization=True, extra_parameters=1,
                                shared_trajectories=9, call_count_formula="C+2",
                                mean_core_parameters=0, mean_core_state_keys=0,
                                full_window_estimator_calls=1, half_window_estimator_calls=0,
                                gate_exactly_one=True, statistics_detached_fp32=True,
                                affine_max_abs_error=affine_error,
                                counterpart=counterpart_name,
                                initialized_parity_max_abs_error=parity_error,
                                strict_state_loaded=counterpart is not None,
                                loaded_state_parity_max_abs_error=loaded_error))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    result = dict(smoke="PASS", scope="Synthetic histories only; no metrics/test/checkpoint reading",
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
