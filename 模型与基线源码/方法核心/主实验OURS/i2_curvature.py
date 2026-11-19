"""Shared prediction contrast for nonlinear common-driver amplitude response.

The frozen first mechanism is unchanged. Three observed nonlinear basis views
are centered and residualized against the original unit-RMS mean factor. They
are fit to X-fa, softly removed from the input, and forecast by the same F.
One origin response corrects both first-order and nonlinear reconstruction.

"Curvature" means a nonlinear remainder of amplitude response, not temporal
acceleration. Basis construction/orthogonalization are known primitives; the
complete shared-F subtraction/reconstruction operator is the candidate.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn

from i1_common_contrast import Model as FrozenCommonContrast
from models import Baseline


@contextmanager
def full_float32(device_type):
    """New statistics/reconstruction only: AMP and TF32 disabled, then restored."""
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@torch.no_grad()
def curvature_history(xn, factor, loading, mode, ridge):
    """Detached observed-only Q[B,L,3] and beta[B,3,C], in unit-RMS Q units.

    Each Q column is orthogonal to the constant and centered original f. The
    three columns need not be mutually orthogonal. The first factor/loadings
    are supplied unchanged by the frozen first mechanism.
    """
    with full_float32(xn.device.type):
        x, f, a = xn.detach().float(), factor.detach().float(), loading.detach().float()
        if mode == "adaptive_hinge":
            levels = torch.tensor([0.2, 0.5, 0.8], device=f.device, dtype=f.dtype)
            knots = torch.quantile(f.squeeze(-1), levels, dim=1).transpose(0, 1).unsqueeze(1)
            raw = torch.relu(f-knots)
        elif mode == "fixed_hinge":
            knots = torch.tensor([-0.75, 0.0, 0.75], device=f.device, dtype=f.dtype)[None, None, :]
            knots = knots.expand(f.shape[0], -1, -1)
            raw = torch.relu(f-knots)
        elif mode == "hermite":
            z = f.squeeze(-1)
            raw = torch.stack((z.square()-1, z.pow(3)-3*z, z.pow(4)-6*z.square()+3), dim=-1)
            knots = None
        else:
            raise ValueError("unknown nonlinear response basis: "+mode)
        factor_mean = f.mean(dim=1, keepdim=True)
        centered_factor = f-factor_mean
        basis_mean = raw.mean(dim=1, keepdim=True)
        centered = raw-basis_mean
        denominator = centered_factor.square().sum(dim=1, keepdim=True).clamp_min(1e-8)
        gamma = (centered_factor*centered).sum(dim=1, keepdim=True)/denominator
        orthogonal = centered-centered_factor*gamma
        rms = orthogonal.square().mean(dim=1, keepdim=True).sqrt()
        inactive = rms < 1e-4
        design = torch.where(inactive, torch.zeros_like(orthogonal), orthogonal/rms.clamp_min(1e-4))
        gram = torch.bmm(design.transpose(1, 2), design)
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        penalty = float(ridge)*trace/3+1e-5
        target = x-f*a
        cross = torch.bmm(design.transpose(1, 2), target)
        identity = torch.eye(3, device=x.device, dtype=x.dtype)[None]
        beta = torch.linalg.solve(gram+penalty[:, None, None]*identity, cross)
        component = torch.bmm(design, beta)
        return dict(design=design, beta=beta, component=component, raw_basis=raw,
                    basis_mean=basis_mean, linear_coefficient=gamma,
                    orthogonal_raw=orthogonal, rms=rms, inactive=inactive,
                    factor_mean=factor_mean, knots=knots, penalty=penalty)


class Model(FrozenCommonContrast):
    CURVATURE_MODES = {"adaptive_hinge", "fixed_hinge", "hermite"}

    def __init__(self, config, channels):
        mode = config.get("i2_curvature_mode", "adaptive_hinge")
        if mode not in self.CURVATURE_MODES:
            raise ValueError("unknown i2_curvature_mode: "+mode)
        ridge = float(config.get("i2_curvature_ridge", 0.03))
        if ridge <= 0:
            raise ValueError("i2_curvature_ridge must be positive")
        effective = dict(config, i1_common_contrast_mode="mean_residual")
        super().__init__(effective, channels)
        self.curvature_mode = mode
        self.curvature_ridge = ridge
        # No RNG draw. Exactly zero gives the first mechanism at initialization.
        self.curvature_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, return_components=False):
        if x.ndim != 3 or x.shape[1] != self.length:
            raise ValueError("expected x of shape [batch, configured length, channels]")
        xn, mu, sigma = self.normalize(x)
        # Do not wrap the first estimator in a new precision context: its
        # arithmetic/caller TF32 policy must remain that of frozen I1.
        factor, loading, gate = self._evidence(xn)
        terms = curvature_history(xn, factor, loading, self.curvature_mode, self.curvature_ridge)
        strength = self.factor_logit.sigmoid()*gate
        response_strength = self.curvature_logit.tanh()
        common = factor*loading
        first_residual = xn-strength*common
        residual = first_residual-response_strength*terms["component"]
        origin = torch.zeros_like(factor)
        # Keep frozen I1's C+2 GEMM layout. A single C+5 call changes BF16
        # reduction kernels and created ~.01 t=0 drift on synthetic CUDA.
        # Both passes use the same complete F/weights; there is one F(0).
        pred = self._forecast(torch.cat((residual, factor, origin), dim=-1))
        nonlinear_pred = self._forecast(terms["design"])
        zero = pred[..., -1:]
        first_contrast = pred[..., -2:-1]-zero
        with full_float32(x.device.type):
            # New contrast subtracts FP32 values before matrix reduction.
            # The first contrast above retains frozen I1's arithmetic.
            nonlinear_contrast = nonlinear_pred.float()-zero.float()
            nonlinear_future = torch.bmm(nonlinear_contrast, terms["beta"])
        normalized = pred[..., :-2]+strength*first_contrast*loading
        normalized = normalized+response_strength*nonlinear_future
        prediction = normalized*sigma+mu
        if not return_components:
            return prediction
        geometry = {name: value.detach() if torch.is_tensor(value) else value
                    for name, value in terms.items()}
        geometry.update(factor=factor.detach(), loading=loading.detach(), gate=gate.detach(),
                        first_common=common.detach(), first_residual=first_residual.detach(),
                        residual=residual.detach(), first_strength=strength.detach(),
                        response_strength=response_strength.detach(), mode=self.curvature_mode,
                        ridge=self.curvature_ridge, statistics_scope="observed history only")
        return prediction, geometry


def numpy_audit(xn, factor, loading, mode, ridge, actual):
    """Independent float64 NumPy basis/OLS equations on synthetic histories."""
    import numpy as np
    x = xn.detach().cpu().double().numpy()
    f = factor.detach().cpu().double().numpy()
    a = loading.detach().cpu().double().numpy()
    if mode == "adaptive_hinge":
        knots = np.quantile(f[..., 0], [0.2, 0.5, 0.8], axis=1).T[:, None, :]
        raw = np.maximum(f-knots, 0)
    elif mode == "fixed_hinge":
        knots = np.broadcast_to(np.array([-0.75, 0, 0.75])[None, None], (f.shape[0], 1, 3))
        raw = np.maximum(f-knots, 0)
    else:
        z = f[..., 0]
        raw = np.stack((z*z-1, z**3-3*z, z**4-6*z*z+3), axis=-1)
        knots = None
    fmean = f.mean(axis=1, keepdims=True)
    fc = f-fmean
    m = raw.mean(axis=1, keepdims=True)
    gamma = np.sum(fc*(raw-m), axis=1, keepdims=True)/np.maximum(np.sum(fc*fc, axis=1, keepdims=True), 1e-8)
    orth = raw-m-fc*gamma
    rms = np.sqrt(np.mean(orth*orth, axis=1, keepdims=True))
    Q = np.where(rms < 1e-4, 0, orth/np.maximum(rms, 1e-4))
    gram = Q.transpose(0, 2, 1)@Q
    penalty = ridge*np.trace(gram, axis1=-2, axis2=-1)/3+1e-5
    beta = np.linalg.solve(gram+penalty[:, None, None]*np.eye(3)[None], Q.transpose(0, 2, 1)@(x-f*a))
    reference = dict(design=Q, beta=beta, component=Q@beta, raw_basis=raw,
                     basis_mean=m, linear_coefficient=gamma, orthogonal_raw=orth,
                     rms=rms, factor_mean=fmean, penalty=penalty)
    if knots is not None:
        reference["knots"] = knots
    errors = {}
    for name, expected in reference.items():
        observed = actual[name].detach().cpu().double().numpy()
        errors[name] = float(np.max(np.abs(observed-expected)))
        np.testing.assert_allclose(observed, expected, atol=2e-5, rtol=4e-5)
    return errors


def smoke(device):
    config = dict(length=96, horizon=96, dim=256, ffn_dim=1024, layers=2, dropout=0.0,
                  i1_common_contrast_mode="mean_residual", i1_context_v2_strength_init=0.5)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2026)
    first = FrozenCommonContrast(config, 7).to(device)
    first_rng_after_init = torch.get_rng_state().clone()
    first_params = sum(p.numel() for p in first.parameters())
    x = torch.randn(3, 96, 7, device=device)
    rows = []
    try:
        for ridge in [0.03, 0.1]:
            for mode in ["adaptive_hinge", "fixed_hinge", "hermite"]:
                torch.manual_seed(2026)
                model = Model(dict(config, i2_curvature_mode=mode, i2_curvature_ridge=ridge), 7).to(device)
                assert torch.equal(torch.get_rng_state(), first_rng_after_init)
                assert all(torch.equal(value, model.state_dict()[name]) for name, value in first.state_dict().items())
                assert sum(p.numel() for p in model.parameters()) == first_params+1
                assert model.curvature_logit.item() == 0
                y, geometry = model(x, return_components=True)
                parity_initial = (y-first(x)).abs().max().item()
                torch.testing.assert_close(y, first(x), atol=3e-6, rtol=3e-6)
                xn, mu, sigma = model.normalize(x)
                f, a, gate = first._evidence(xn)
                assert torch.equal(f, geometry["factor"]) and torch.equal(a, geometry["loading"])
                assert torch.equal(gate, geometry["gate"])
                new_terms = curvature_history(xn, f, a, mode, ridge)
                assert all(not value.requires_grad for value in new_terms.values() if torch.is_tensor(value))
                assert all(value.dtype == torch.float32 for value in new_terms.values()
                           if torch.is_tensor(value) and value.dtype != torch.bool)
                numpy_errors = numpy_audit(xn, f, a, mode, ridge, new_terms)
                Q = geometry["design"]
                q_mean = Q.mean(dim=1).abs().max().item()
                linear_inner = ((f-f.mean(dim=1, keepdim=True))*Q).mean(dim=1).abs().max().item()
                active = ~geometry["inactive"].squeeze(1)
                rms_error = (Q.square().mean(dim=1)[active]-1).abs().max().item() if active.any() else 0
                assert q_mean < 2e-6 and linear_inner < 3e-6 and rms_error < 2e-6
                assert torch.equal(Q.masked_select(geometry["inactive"].expand_as(Q)),
                                   torch.zeros_like(Q.masked_select(geometry["inactive"].expand_as(Q))))
                with torch.no_grad():
                    changed = FrozenCommonContrast(config, 7).to(device)
                    changed.load_state_dict(first.state_dict(), strict=True)
                    changed.factor_logit.fill_(0.7)
                    changed.embedding[0].bias.add_(0.13)
                    changed.head.bias.add_(torch.linspace(-0.2, 0.3, 96, device=device))
                loaded = model.load_state_dict(changed.state_dict(), strict=False)
                assert loaded.missing_keys == ["curvature_logit"] and not loaded.unexpected_keys
                parity_loaded = (model(x)-changed(x)).abs().max().item()
                torch.testing.assert_close(model(x), changed(x), atol=3e-6, rtol=3e-6)
                calls = []
                original_forecast = model._forecast

                def record(z):
                    calls.append(z.shape[-1])
                    return original_forecast(z)

                model._forecast = record
                model.zero_grad(set_to_none=True)
                prediction = model(x)
                prediction.square().mean().backward()
                assert calls == [9, 3]
                assert model.curvature_logit.grad is not None and torch.isfinite(model.curvature_logit.grad)
                initial_response_gradient = model.curvature_logit.grad.item()
                assert abs(initial_response_gradient) > 1e-10
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                model._forecast = original_forecast
                with torch.no_grad():
                    model.curvature_logit.fill_(0.4)
                assert torch.isfinite(model(x)).all()
                constant, constant_geometry = model(torch.ones_like(x), return_components=True)
                assert torch.isfinite(constant).all()
                assert constant_geometry["inactive"].all()
                assert torch.equal(constant_geometry["design"], torch.zeros_like(constant_geometry["design"]))
                assert torch.isfinite(model(x[..., :1])).all()
                with torch.no_grad():
                    model.curvature_logit.fill_(-8)
                assert torch.isfinite(model(x)).all()

                linear = nn.Linear(96, 96).to(device)
                with torch.no_grad():
                    linear.bias.copy_(torch.linspace(-7, 11, 96, device=device))
                model._forecast = lambda z: linear(z.transpose(1, 2)).transpose(1, 2)
                direct = model._forecast(xn)*sigma+mu
                affine_max = 0.0
                for s_logit in [-1.1, 1.1]:
                    for t_logit in [-0.7, 0.0, 0.7]:
                        with torch.no_grad():
                            model.factor_logit.fill_(s_logit)
                            model.curvature_logit.fill_(t_logit)
                        affine = model(x)
                        affine_max = max(affine_max, (affine-direct).abs().max().item())
                        torch.testing.assert_close(affine, direct, atol=8e-6, rtol=5e-6)
                model._forecast = original_forecast
                amp_parity = None
                if device.type == "cuda":
                    torch.backends.cuda.matmul.allow_tf32 = True
                    with torch.no_grad():
                        model.curvature_logit.zero_()
                    loaded = model.load_state_dict(changed.state_dict(), strict=False)
                    assert loaded.missing_keys == ["curvature_logit"] and not loaded.unexpected_keys
                    model.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        amp, amp_geometry = model(x, return_components=True)
                        reference = changed(x)
                        amp_parity = (amp-reference).abs().max().item()
                        torch.testing.assert_close(amp, reference, atol=1e-5, rtol=1e-5)
                        loss = amp.float().square().mean()
                    loss.backward()
                    assert torch.isfinite(amp).all()
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    assert amp_geometry["design"].dtype == amp_geometry["beta"].dtype == torch.float32
                    # First evidence remains byte-identical under the caller's
                    # original TF32/AMP policy; new stats restore that policy.
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        amp_xn, _, _ = changed.normalize(x)
                        old_f, old_a, _ = changed._evidence(amp_xn)
                    assert torch.equal(old_f, amp_geometry["factor"])
                    assert torch.equal(old_a, amp_geometry["loading"])
                    assert torch.backends.cuda.matmul.allow_tf32 is True
                    with torch.no_grad():
                        model.curvature_logit.fill_(0.4)
                    model.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        nonzero_amp = model(x)
                        nonzero_loss = nonzero_amp.float().square().mean()
                    nonzero_loss.backward()
                    assert torch.isfinite(nonzero_amp).all()
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    torch.backends.cuda.matmul.allow_tf32 = False
                rows.append(dict(mode=mode, ridge=ridge, parameters=first_params+1,
                                 extra_parameters_vs_I1=1, extra_parameters_vs_B0=2,
                                 inherited_initial_weights_identical=True, inherited_evidence_identical=True,
                                 no_additional_initialization_RNG_draw=True,
                                 initial_t0_I1_parity_max_abs_error=parity_initial,
                                 noninitial_state_t0_I1_parity_max_abs_error=parity_loaded,
                                 bf16_t0_I1_parity_max_abs_error=amp_parity,
                                 initial_response_logit_gradient=initial_response_gradient,
                                 call_count=2, shared_tokens="C+5", actual_tokens_per_call=[9, 3],
                                 numpy_max_abs_errors=numpy_errors,
                                 design_mean_max_abs=q_mean, design_factor_inner_product_max_abs=linear_inner,
                                 active_design_RMS_squared_error=rms_error,
                                 affine_large_nonconstant_bias_max_abs_error=affine_max,
                                 constant_history_all_inactive=True, finite_gradients=True))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    result = dict(smoke="PASS", scope="Synthetic actual-width inputs only; no training/metrics/test/checkpoints",
                  device=str(device), torch=torch.__version__, config=config, results=rows,
                  caveats=["t0 is algebraically I1; layout roundoff is reported separately by backend",
                           "Q columns are orthogonal to 1/f, not mutually orthogonal",
                           "signed negative t does not guarantee history residual nonexpansion"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--qa-output")
    args = parser.parse_args()
    result = smoke(torch.device(args.device))
    if args.qa_output:
        Path(args.qa_output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
