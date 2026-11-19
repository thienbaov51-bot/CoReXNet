"""Shared CI temporal residual-MLP skeleton, without SOFTS STAR mechanisms."""
import torch
from torch import nn

class ResidualFFN(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.ffn=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(hidden,dim),nn.Dropout(dropout))
        self.norm=nn.LayerNorm(dim)
    def forward(self,z):
        return self.norm(z+self.ffn(z))

class Baseline(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        self.length=config.get('length',96); self.horizon=config.get('horizon',96)
        dim=config.get('dim',128); hidden=config.get('ffn_dim',dim*4)
        self.embedding=nn.Sequential(nn.Linear(self.length,dim),nn.Dropout(config.get('dropout',0.0)))
        self.encoder=nn.Sequential(*[ResidualFFN(dim,hidden,config.get('dropout',0.0))
                                    for _ in range(config.get('layers',2))])
        self.head=nn.Linear(dim,self.horizon)

    def normalize(self,x):
        mu=x.mean(1,keepdim=True).detach()
        sigma=(x.var(1,keepdim=True,unbiased=False)+1e-5).sqrt().detach()
        return (x-mu)/sigma,mu,sigma

    def forward(self,x):
        xn,mu,sigma=self.normalize(x)
        z=self.encoder(self.embedding(xn.transpose(1,2)))
        return self.head(z).transpose(1,2)*sigma+mu

def create_model(config, channels):
    if config['model']=='I3_response_deflated_power_factor':
        from i3_response_deflated_power_factor import Model
        return Model(config,channels)
    if config['model']=='I3_output_metric_response_supervision_low_weight':
        from i3_output_metric_response_supervision_low_weight import Model
        return Model(config,channels)
    if config['model']=='I3_response_coordinate_whitening':
        from i3_response_coordinate_whitening import Model
        return Model(config,channels)
    if config['model']=='I3_output_metric_response_supervision':
        from i3_output_metric_response_supervision import Model
        return Model(config,channels)
    if config['model']=='I3_crossfit_response_calibration':
        from i3_crossfit_response_calibration import Model
        return Model(config,channels)
    if config['model']=='I3_block_influence_shrinkage':
        from i3_block_influence_shrinkage import Model
        return Model(config,channels)
    if config['model']=='I3_response_view_supervision':
        from i3_response_view_supervision import Model
        return Model(config,channels)
    if config['model']=='I3_training_response_coherence':
        from i3_training_response_coherence_v2 import Model
        return Model(config,channels)
    if config['model']=='I3_causal_lag_response':
        from i3_causal_lag_response import Model
        return Model(config,channels)
    if config['model']=='I3_driver_moment_calibration':
        from i3_driver_moment_calibration import Model
        return Model(config,channels)
    if config['model']=='I3_signed_directional_support':
        from i3_signed_directional_support import Model
        return Model(config,channels)
    if config['model']=='B0': return Baseline(config,channels)
    if config['model']=='I3_directional_support':
        from i3_directional_support import Model
        return Model(config,channels)
    if config['model']=='I3_state_transport':
        from i3_state_transport import Model
        return Model(config,channels)
    if config['model']=='I3_response_consistency_v2':
        from i3_response_consistency_v2 import Model
        return Model(config,channels)
    if config['model']=='I3_response_consistency':
        from i3_response_consistency import Model
        return Model(config,channels)
    if config['model']=='I2_exposure_commutator':
        from i2_exposure_commutator import Model
        return Model(config,channels)
    if config['model']=='I2_curvature':
        from i2_curvature import Model
        return Model(config,channels)
    if config['model']=='I2_curvature_stable':
        from i2_curvature_stable import Model
        return Model(config,channels)
    if config['model']=='I2_curvature_replacement':
        from i2_curvature_replacement import Model
        return Model(config,channels)
    if config['model']=='I1_dynamics':
        from i1_dynamics import Model
        return Model(config,channels)
    if config['model']=='I1_common_contrast':
        from i1_common_contrast import Model
        return Model(config,channels)
    if config['model']=='I1_origin_replacement':
        from i1_origin_replacement import Model
        return Model(config,channels)
    if config['model']=='I1_lagged_common':
        from i1_lagged_common import Model
        return Model(config,channels)
    if config['model']=='I1_lagged_independent':
        from i1_lagged_independent import Model
        return Model(config,channels)
    if config['model']=='I1_context':
        from i1_context import Model
        return Model(config,channels)
    if config['model']=='I1_spectral':
        from i1_spectral import Model
        return Model(config,channels)
    if config['model']=='I1_temporal':
        from i1_temporal import Model
        return Model(config,channels)
    if config['model']=='I1_context_v2':
        from i1_context_v2 import Model
        return Model(config,channels)
    if config['model']=='I1_context_v3':
        from i1_context_v3 import Model
        return Model(config,channels)
    if config['model']=='I1_context_v4':
        from i1_context_v4 import Model
        return Model(config,channels)
    if config['model']=='I1_context_v5':
        from i1_context_v5 import Model
        return Model(config,channels)
    if config['model']=='I1_context_v6':
        from i1_context_v6 import Model
        return Model(config,channels)
    if config['model']=='I1_context_v7':
        from i1_context_v7 import Model
        return Model(config,channels)
    if config['model']=='I1_temporal_v2':
        from i1_temporal_v2 import Model
        return Model(config,channels)
    if config['model']=='I1_spatial':
        from i1_spatial import Model
        return Model(config,channels)
    if config['model']=='I1_stationarity':
        from i1_stationarity import Model
        return Model(config,channels)
    if config['model']=='I3_anchored_mixed_response':
        from i3_anchored_mixed_response import Model
        return Model(config,channels)
    if config['model']=='I3_heterogeneous_response_budget':
        from i3_heterogeneous_response_budget import Model
        return Model(config,channels)
    if config['model']=='I3_driver_centered_response_transport':
        from i3_driver_centered_response_transport import Model
        return Model(config,channels)
    if config['model']=='I3_slope_only_response_modulation':
        from i3_slope_only_response_modulation import Model
        return Model(config,channels)
    if config['model']=='I3_orientation_invariant_loading_gate':
        from i3_orientation_invariant_loading_gate import Model
        return Model(config,channels)
    if config['model']=='I3_normalized_response_shape_calibration':
        from i3_normalized_response_shape_calibration import Model
        return Model(config,channels)
    if config['model']=='I3_response_temporal_calibration':
        from i3_response_temporal_coherence import Model
        return Model(config,channels)
    if config['model']=='I3_channel_response_temporal_calibration':
        from i3_channel_response_temporal_calibration import Model
        return Model(config,channels)
    if config['model']=='I3_response_amplitude_shape_calibration':
        from i3_response_amplitude_shape_calibration import Model
        return Model(config,channels)
    if config['model']=='I3_response_residual_exchange':
        from i3_response_residual_exchange import Model
        return Model(config,channels)
    if config['model']=='I3_local_response_transport':
        from i3_local_response_transport import Model
        return Model(config,channels)
    if config['model']=='I3_local_response_transport_supported':
        from i3_local_response_transport_supported import Model
        return Model(config,channels)
    if config['model']=='I3_joint_response_trajectory':
        from i3_joint_response_trajectory import Model
        return Model(config,channels)
    raise ValueError('Unknown model '+config['model'])
