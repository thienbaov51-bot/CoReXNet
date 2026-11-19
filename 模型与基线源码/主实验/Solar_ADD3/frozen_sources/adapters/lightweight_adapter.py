"""Thin adapters for byte-preserved official DLinear/SparseTSF/CycleNet.

Origins are absolute indices of the first forecast row (input x_end), exactly
the official CycleNet data loader's s_end. No future observation is inspected.
Vendor imports are isolated; no model/data-provider package from another repo
is left in sys.modules. This module performs no training or data loading.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

GIT_PINS = {
    'DLinear': ('cure-lab/LTSF-Linear', '0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6'),
    'SparseTSF': ('lss-1138/SparseTSF', 'b8c2740eecc84d8095ffce49ba5acafe68e53bb8'),
    'CycleNet': ('ACAT-SCUT/CycleNet', 'd807e51fc2dcd143885ee639d97965a7ab0926f4'),
}
CORE_HASHES = {
    'DLinear/models/DLinear.py': '0893b53cb6473d6bdca7aeca514cb3ee12efa6df227c29c4469571c9711451cc',
    'SparseTSF/models/SparseTSF.py': '7f4afe758bd7ad71b939b88e633acd271f275f4e85dd73b724233c7b1a041948',
    'SparseTSF/layers/Embed.py': '3743a14dd505f9caafb512727b10bec2d4a84d84cacc6f8936f33a5071e2a0a2',
    'CycleNet/models/CycleNet.py': '28e868d7d4211082e490ed9f19e323568e5fcd0a3beb29af78ce791bcaa31c05',
}
_IMPORT_LOCK = threading.RLock()
_MISSING = object()


def _mapping(config):
    return dict(config) if isinstance(config, dict) else vars(config).copy()


def _dataset_key(dataset):
    name = str(dataset or '').lower()
    return {'ecl': 'electricity', 'electricity': 'electricity'}.get(name, name)


def official_training_preset(model_name, dataset, horizon=96, channels=None):
    """Author script settings; PEMS is explicitly a benchmark extension.

    L=96/seed=2026/fixed chronological splits and batch caps are campaign
    controls. Scripts used L336 for DLinear and L720 for SparseTSF, so this is
    controlled retraining of their architecture, not paper-table reproduction.
    lr_schedule type3 means official decay .8, after_epoch below is authoritative.
    """
    if model_name not in GIT_PINS:
        raise ValueError('unsupported lightweight model: ' + str(model_name))
    key = _dataset_key(dataset)
    is_pems = key.startswith('pems')
    known = {'etth1', 'etth2', 'ettm1', 'ettm2', 'electricity', 'weather', 'traffic'}
    if key not in known and not is_pems:
        raise ValueError('dataset requires an explicit author/extension preset')
    h = int(horizon)
    if h not in (96, 192, 336, 720):
        raise ValueError('this campaign locks four prediction horizons')
    common = {'seed': 2026, 'length': 96, 'horizon': h, 'loss': 'MSE',
              'optimizer': 'Adam', 'weight_decay': 0.0, 'grad_clip': None,
              'evaluate_test': False, 'amp': True,
              'recipe_scope': 'benchmark_extension_no_author_PEMS_script' if is_pems
                              else 'author_script_hyperparameters_controlled_L96_retraining'}
    if model_name == 'DLinear':
        lr = {'etth1': .005, 'etth2': .05, 'ettm1': .0001,
              'electricity': .001, 'weather': .0001, 'traffic': .05}.get(key, .001)
        if key == 'ettm2':
            lr = .001 if h in (96, 192) else .01
        batch = 8 if key == 'ettm1' else 16 if key in ('electricity', 'weather', 'traffic') else 32
        common.update(lr=lr, epochs=10, patience=3, batch_size=batch,
                      lr_schedule='type1', lr_decay_factor=.5,
                      after_epoch_lr='base_lr * 0.5 ** (completed_epoch_1based - 1)',
                      official_input_length=336, official_config={'individual': False})
    elif model_name == 'SparseTSF':
        period = 4 if key in ('ettm1', 'ettm2', 'weather') else 24
        lr = .03 if key in ('etth2', 'traffic') else .02
        batch = 128 if key in ('electricity', 'traffic') else 256
        common.update(lr=lr, epochs=30, patience=5, batch_size=batch,
                      lr_schedule='type3', lr_decay_factor=.8,
                      after_epoch_lr='base_lr if completed_epoch_1based < 3 else base_lr * 0.8 ** (completed_epoch_1based - 3)',
                      official_input_length=720,
                      official_config={'model_type': 'linear', 'period_len': period, 'd_model': 128})
    else:
        cycle = 288 if is_pems else 96 if key in ('ettm1', 'ettm2') else 144 if key == 'weather' else 168 if key in ('electricity', 'traffic') else 24
        lr = .002 if key == 'traffic' else .005
        batch = 64 if key in ('electricity', 'traffic') else 256
        common.update(lr=lr, epochs=30, patience=5, batch_size=batch,
                      lr_schedule='type3', lr_decay_factor=.8,
                      after_epoch_lr='base_lr if completed_epoch_1based < 3 else base_lr * 0.8 ** (completed_epoch_1based - 3)',
                      official_input_length=96,
                      official_config={'model_type': 'mlp', 'd_model': 512, 'cycle': cycle, 'use_revin': True})
    if channels is not None:
        common['official_batch_size'] = common['batch_size']
        common['batch_size'] = min(common['batch_size'], 64 if int(channels) > 64 else 256)
    common['model'] = model_name
    common['dataset'] = dataset
    return common


def _vendor_root(config):
    if config.get('lightweight_vendor_root'):
        return Path(config['lightweight_vendor_root']).resolve()
    if config.get('official_vendor_root'):
        return Path(config['official_vendor_root']).resolve()
    return Path(__file__).resolve().parents[2] / 'main_experiments' / 'MAIN_L96_SINGLE_SEED_20261009' / 'vendor' / 'lightweight'


def _checked_file(root, relative):
    path = root / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != CORE_HASHES[relative]:
        raise ValueError('official source byte identity mismatch: ' + relative)
    return path


def _import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _sparse_embed_alias(root):
    """Satisfy the author's unused absolute import, then restore all aliases."""
    previous = {key: sys.modules.get(key, _MISSING) for key in ('layers', 'layers.Embed')}
    embed = _import_file(_checked_file(root, 'SparseTSF/layers/Embed.py'), '_paper5_sparse_official_embed')
    package = ModuleType('layers')
    package.__path__ = [str(root / 'SparseTSF' / 'layers')]
    package.Embed = embed
    sys.modules['layers'] = package
    sys.modules['layers.Embed'] = embed
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is _MISSING:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old


def _official_class(root, model_name):
    relative = model_name + '/models/' + model_name + '.py'
    path = _checked_file(root, relative)
    with _IMPORT_LOCK:
        if model_name == 'SparseTSF':
            with _sparse_embed_alias(root):
                return _import_file(path, '_paper5_official_sparse_core').Model
        return _import_file(path, '_paper5_official_' + model_name).Model


class OfficialLightweightAdapter(nn.Module):
    def __init__(self, model_name, config, channels, dataset=None):
        super().__init__()
        if model_name not in GIT_PINS:
            raise ValueError('unknown official lightweight model')
        cfg = _mapping(config)
        dataset = dataset or cfg.get('dataset')
        defaults = official_training_preset(model_name, dataset, cfg.get('horizon', cfg.get('pred_len', 96))) if dataset else {'official_config': {}}
        effective = {**defaults['official_config'], **cfg, **cfg.get('official_config', {})}
        self.model_name = model_name
        self.seq_len = int(effective.get('seq_len', effective.get('length', 96)))
        self.pred_len = int(effective.get('pred_len', effective.get('horizon', 96)))
        self.channels = int(channels)
        if self.seq_len < 2 or self.pred_len < 1 or self.channels < 1:
            raise ValueError('invalid dimensions')
        args = {'seq_len': self.seq_len, 'pred_len': self.pred_len, 'enc_in': self.channels}
        if model_name == 'DLinear':
            args['individual'] = bool(effective.get('individual', False))
            self.normalization_description = 'external train-only StandardScaler; official moving-average decomposition (kernel25); no extra instance normalization'
        elif model_name == 'SparseTSF':
            args.update(period_len=int(effective.get('period_len', 24)),
                        d_model=int(effective.get('d_model', 128)), model_type=effective.get('model_type', 'linear'))
            if args['model_type'] != 'linear':
                raise ValueError('SparseTSF comparison is the original linear branch, not journal MLP')
            if args['period_len'] < 1 or self.seq_len % args['period_len'] or self.pred_len % args['period_len']:
                raise ValueError('SparseTSF period must divide both input and forecast lengths')
            self.normalization_description = 'external train-only StandardScaler; official history mean subtraction/add-back, no extra RevIN'
        else:
            args.update(cycle=int(effective.get('cycle', 24)), d_model=int(effective.get('d_model', 512)),
                        model_type=effective.get('model_type', 'mlp'), use_revin=bool(effective.get('use_revin', True)))
            if args['cycle'] < 1:
                raise ValueError('invalid CycleNet cycle')
            self.normalization_description = 'external train-only StandardScaler; official use_revin=True by default, mean/sample variance+1e-5 normalization and inverse; author phase retained'
        self.official_config = args.copy()
        self.requires_origin_indices = model_name == 'CycleNet'
        self.official_git_sha = GIT_PINS[model_name][1]
        self.official_source_hashes = {key: value for key, value in CORE_HASHES.items() if key.startswith(model_name + '/')}
        self.core = _official_class(_vendor_root(cfg), model_name)(SimpleNamespace(**args))

    def predict_windows(self, x, origin_indices=None):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.channels):
            raise ValueError('expected history [B,seq_len,channels]')
        if self.requires_origin_indices:
            if origin_indices is None:
                raise ValueError('CycleNet needs absolute forecast-origin indices; zeros are not a valid fallback')
            if not torch.is_tensor(origin_indices):
                origin_indices = torch.as_tensor(origin_indices, device=x.device)
            if origin_indices.ndim != 1 or origin_indices.numel() != x.shape[0]:
                raise ValueError('origins must be one absolute forecast-start index per batch row')
            if origin_indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError('origin indices must have integer dtype')
            cycle_index = origin_indices.to(device=x.device, dtype=torch.long) % self.official_config['cycle']
            # Official loader already returns s_end; official core adds L internally.
            result = self.core(x, cycle_index)
        else:
            result = self.core(x)
        if result.shape != (x.shape[0], self.pred_len, self.channels):
            raise ValueError('official model returned an invalid B,H,C shape')
        return result

    def forward(self, x, origin_indices=None):
        return self.predict_windows(x, origin_indices)


def create_lightweight(model_name, config, channels, dataset=None):
    return OfficialLightweightAdapter(model_name, config, channels, dataset)
