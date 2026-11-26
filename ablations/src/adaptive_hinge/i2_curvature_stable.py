"""Independent FP64 history geometry for I2; frozen FP32 sources stay unchanged.

Only the new nonlinear basis statistics use FP64. Original normalization,
first-factor evidence, complete F weights/layout, and first contrast retain
the frozen I1/I2 arithmetic. Q/beta are cast once to FP32; input subtraction
and future reconstruction both use those same FP32 coordinates/loadings.
"""
import torch

from i2_curvature import Model as FrozenI2, full_float32


@torch.no_grad()
def stable_history(xn, factor, loading, mode, ridge, remove_linear=True):
    """Observed-only full-FP64 basis, projection, RMS and ridge geometry.

    Inactivity is decided from FP64 RMS using the unchanged absolute 1e-4
    threshold. Descriptors/Q/beta return as FP32; the actual component is
    recomputed by FP32 bmm with TF32 disabled, never from a FP64 product.
    """
    with full_float32(xn.device.type):
        # Inputs are the original first mechanism's FP32 values, then promoted.
        x = xn.detach().float().double()
        f = factor.detach().float().double()
        a = loading.detach().float().double()
        if mode == "adaptive_hinge":
            levels = torch.tensor([0.2, 0.5, 0.8], device=f.device, dtype=torch.float64)
            knots = torch.quantile(f.squeeze(-1), levels, dim=1).transpose(0, 1).unsqueeze(1)
            raw = torch.relu(f-knots)
        elif mode == "fixed_hinge":
            knots = torch.tensor([-0.75, 0, 0.75], device=f.device, dtype=torch.float64)[None, None]
            knots = knots.expand(f.shape[0], -1, -1)
            raw = torch.relu(f-knots)
        elif mode == "hermite":
            z = f.squeeze(-1)
            raw = torch.stack((z.square()-1, z.pow(3)-3*z, z.pow(4)-6*z.square()+3), dim=-1)
            knots = None
        else:
            raise ValueError("unknown nonlinear response basis: "+mode)
        factor_mean = f.mean(dim=1, keepdim=True)
        fc = f-factor_mean
        basis_mean = raw.mean(dim=1, keepdim=True)
        centered = raw-basis_mean
        denominator = fc.square().sum(dim=1, keepdim=True).clamp_min(1e-8)
        gamma = (fc*centered).sum(dim=1, keepdim=True)/denominator if remove_linear else torch.zeros_like(basis_mean)
        adjusted = centered-fc*gamma
        rms = adjusted.square().mean(dim=1, keepdim=True).sqrt()
        inactive = rms < 1e-4
        Q64 = torch.where(inactive, torch.zeros_like(adjusted), adjusted/rms.clamp_min(1e-4))
        gram = torch.bmm(Q64.transpose(1, 2), Q64)
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        penalty = float(ridge)*trace/3+1e-5
        cross = torch.bmm(Q64.transpose(1, 2), x-f*a)
        identity = torch.eye(3, device=x.device, dtype=torch.float64)[None]
        beta64 = torch.linalg.solve(gram+penalty[:, None, None]*identity, cross)
        design, beta = Q64.float(), beta64.float()
        component = torch.bmm(design, beta)
        return dict(design=design, beta=beta, component=component, raw_basis=raw.float(),
                    basis_mean=basis_mean.float(), linear_coefficient=gamma.float(),
                    orthogonal_raw=adjusted.float(), rms=rms.float(), inactive=inactive,
                    factor_mean=factor_mean.float(), knots=knots.float() if knots is not None else None,
                    penalty=penalty.float(), projection_applied=bool(remove_linear),
                    statistics_compute_dtype="float64_from_original_float32",
                    returned_float_dtype="float32", basis_knots_compute_dtype="float64",
                    inactive_threshold=1e-4, component_product_dtype="float32_no_TF32")


class Model(FrozenI2):
    STABLE_MODES = {"reference", "no_subtraction", "no_orthogonalization"}

    def __init__(self, config, channels):
        stable_mode = config.get("i2_stable_replacement_mode", "reference")
        if stable_mode not in self.STABLE_MODES:
            raise ValueError("unknown stable I2 mode: "+stable_mode)
        super().__init__(config, channels)
        self.stable_replacement_mode = stable_mode

    def forward(self, x, return_components=False):
        if x.ndim != 3 or x.shape[1] != self.length:
            raise ValueError("expected x of shape [batch, configured length, channels]")
        xn, mu, sigma = self.normalize(x)
        # The first estimator retains the parent's precision/caller TF32 policy.
        factor, loading, gate = self._evidence(xn)
        terms = stable_history(xn, factor, loading, self.curvature_mode, self.curvature_ridge,
                               self.stable_replacement_mode != "no_orthogonalization")
        strength = self.factor_logit.sigmoid()*gate
        response_strength = self.curvature_logit.tanh()
        common = factor*loading
        first_residual = xn-strength*common
        residual = first_residual if self.stable_replacement_mode == "no_subtraction" else first_residual-response_strength*terms["component"]
        origin = torch.zeros_like(factor)
        pred = self._forecast(torch.cat((residual, factor, origin), dim=-1))
        nonlinear_pred = self._forecast(terms["design"])
        zero = pred[..., -1:]
        first_contrast = pred[..., -2:-1]-zero
        with full_float32(x.device.type):
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
                        stable_replacement_mode=self.stable_replacement_mode,
                        ridge=self.curvature_ridge, statistics_scope="observed history only")
        return prediction, geometry


if __name__ == "__main__":
    from i2_curvature_stable_qa import main
    main()
