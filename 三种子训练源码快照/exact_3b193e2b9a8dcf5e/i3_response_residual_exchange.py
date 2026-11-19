"""History-response coordinates route residual codes through a small decoder.

Exploratory extension after the user rejected the former small I3 gain.
The accepted complete additive I2 remains an exact zero-initialized parent.
Four observed loading coordinates describe donor/receiver relations; this
is low-rank statistical exchange, not a causal or physical graph. Known
factor mixing and bottleneck decoders are primitives, not claimed as new.
"""
import torch
from torch import nn
from i2_curvature_stable import Model as Parent, stable_history
from i2_curvature import full_float32

MODES = ('signed_response', 'absolute_response', 'shuffled_response')


@torch.no_grad()
def routing_coordinates(loading, beta):
    """All modes use the same four active, bounded history coordinates."""
    with full_float32(beta.device.type):
        raw = torch.cat((loading, beta), dim=1).detach().double()
        rms = raw.square().mean(dim=-1, keepdim=True).sqrt()
        active = rms >= 1e-4
        signed = torch.where(active, torch.tanh(raw/rms.clamp_min(1e-4)), torch.zeros_like(raw)).float()
        absolute = signed.abs()
        shifted = torch.roll(signed, shifts=1, dims=-1)
        return dict(signed_response=signed, absolute_response=absolute,
                    shuffled_response=shifted), active.float()


class Model(Parent):
    def __init__(self, config, channels):
        mode = config.get('i3_response_exchange_mode', MODES[0])
        if mode not in MODES:
            raise ValueError('unknown response exchange mode: '+str(mode))
        if config.get('i2_curvature_mode', 'fixed_hinge') != 'fixed_hinge':
            raise ValueError('requires fixed-hinge I2')
        if config.get('i2_stable_replacement_mode', 'no_subtraction') != 'no_subtraction':
            raise ValueError('requires unchanged additive I2')
        super().__init__(dict(config, i2_curvature_mode='fixed_hinge', i2_curvature_ridge=float(config.get('i2_curvature_ridge', .03)),
                              i2_stable_replacement_mode='no_subtraction'), channels)
        self.response_exchange_mode = mode
        width = int(config.get('i3_response_exchange_width', 64))
        if width < 1:
            raise ValueError('positive bottleneck width required')
        dim = self.head.in_features
        # These CPU-created modules consume only their own deterministic RNG.
        # Preserve the parent's global initialization/training RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026+406)
            self.exchange_down = nn.Linear(dim, width)
            self.exchange_up = nn.Linear(width, self.horizon)
            nn.init.zeros_(self.exchange_up.weight)
            nn.init.zeros_(self.exchange_up.bias)
        self.exchange_activation = nn.GELU()

    def forward(self, x, return_components=False):
        if x.ndim != 3 or x.shape[1] != self.length:
            raise ValueError('expected configured X[B,L,C]')
        xn, mu, sigma = self.normalize(x)
        factor, loading, gate = self._evidence(xn)
        terms = stable_history(xn, factor, loading, self.curvature_mode, self.curvature_ridge, True)
        strength = self.factor_logit.sigmoid()*gate
        residual = xn-strength*(factor*loading)
        views = torch.cat((residual, factor, torch.zeros_like(factor)), dim=-1)
        # Exact original C+2 complete-F layout. Retain already computed codes.
        codes = self.encoder(self.embedding(views.transpose(1, 2)))
        pred = self.head(codes).transpose(1, 2)
        nonlinear_pred = self._forecast(terms['design'])
        zero = pred[..., -1:]
        first_contrast = pred[..., -2:-1]-zero
        with full_float32(x.device.type):
            nonlinear_contrast = nonlinear_pred.float()-zero.float()
            nonlinear_future = torch.bmm(nonlinear_contrast, terms['beta'])
        normalized = pred[..., :-2]+strength*first_contrast*loading
        normalized = normalized+self.curvature_logit.tanh()*nonlinear_future
        parent_prediction = normalized*sigma+mu
        coordinates, active = routing_coordinates(loading, terms['beta'])
        with full_float32(x.device.type):
            contrast_codes = codes[:, :-2].float()-codes[:, -1:].float()
            # Compute all three same-shaped donor routes in every mode.
            pooled = {name: torch.bmm(h, contrast_codes)/h.abs().sum(dim=-1, keepdim=True).clamp_min(1e-5)
                      for name, h in coordinates.items()}
        # Every learned weight is shared across all four exchanged tokens.
        decoded = self.exchange_up(self.exchange_activation(self.exchange_down(pooled[self.response_exchange_mode])))
        with full_float32(x.device.type):
            scatter = coordinates['absolute_response'] if self.response_exchange_mode == 'absolute_response' else coordinates['signed_response']
            denominator = active.sum(dim=1, keepdim=True).clamp_min(1)
            correction = torch.bmm(decoded.float().transpose(1, 2), scatter)/denominator
        prediction = parent_prediction+sigma*correction
        if not return_components:
            return prediction
        return prediction, dict(parent_prediction=parent_prediction.detach(), correction=correction.detach(),
          coordinates=coordinates, active=active, history_only_detached_coordinates=True,
          F_tokens=[x.shape[-1]+2, 3], decoder_tokens=4, mode=self.response_exchange_mode,
          donor_shuffle='fixed cyclic shift by one channel; receiver unchanged',
          benchmark_test_already_exposed=True, untouched_confirmation=False)
