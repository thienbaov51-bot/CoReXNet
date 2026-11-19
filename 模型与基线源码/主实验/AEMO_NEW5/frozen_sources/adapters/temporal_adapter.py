"""Thin wrappers over pinned, byte-preserved official temporal model cores.

The comparison supplies standardized history [B,L,C], without timestamps.
PatchTST/ModernTCN run their entire official forward; TimeMixer uses its
official mark=None forecast path with future temporal features disabled.
No official normalization/decomposition/mixer/head is replaced here.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from threading import RLock
from types import SimpleNamespace

import torch
from torch import nn

_IMPORT_LOCK = RLock()
_PREFIXES = ('layers', 'models', 'utils', 'exp', 'data_provider')
_MODEL_CACHE = {}


def _vendor(config=None):
    config = config or {}
    default = Path(__file__).resolve().parents[2]/'main_experiments/MAIN_L96_SINGLE_SEED_20261009/vendor/temporal'
    return Path(config.get('official_vendor_root', config.get('temporal_vendor_root',
        os.environ.get('PAPER5_TEMPORAL_VENDOR_ROOT', default)))).resolve()


def _manifest(config=None):
    config = config or {}
    explicit = config.get('official_temporal_manifest_path')
    if explicit is not None:
        path = Path(explicit).resolve()
    else:
        vendor = _vendor(config)
        candidates = [vendor/'OFFICIAL_TEMPORAL_MANIFEST.json',
                      vendor.parent/'OFFICIAL_TEMPORAL_MANIFEST.json',
                      vendor.parent.parent/'OFFICIAL_TEMPORAL_MANIFEST.json']
        path = next((p for p in candidates if p.is_file()), candidates[-1])
    return json.loads(path.read_text(encoding='utf-8-sig'))


def official_temporal_preset(model_name, dataset, horizon=96, config=None):
    """Return recorded constructor args and author optimization defaults."""
    manifest = _manifest(config)
    return manifest['repositories'][model_name]['presets'][dataset][str(horizon)]


@contextmanager
def _isolated_official_import(root):
    # Entire load+constructor runs under a lock. Old global modules are restored;
    # imported class/function globals keep their actual own-repository objects.
    with _IMPORT_LOCK:
        def local(name):
            return any(name == p or name.startswith(p+'.') for p in _PREFIXES)
        prior_modules = {n:m for n,m in sys.modules.items() if local(n)}
        prior_path = sys.path[:]
        for name in list(sys.modules):
            if local(name):
                del sys.modules[name]
        sys.path.insert(0, str(root))
        try:
            yield
        finally:
            for name in list(sys.modules):
                if local(name):
                    del sys.modules[name]
            sys.modules.update(prior_modules)
            sys.path[:] = prior_path


def _official_model(model_name, args, config):
    vendor = _vendor(config)
    data = _manifest(config)['repositories'][model_name]
    repo = vendor/model_name
    root = repo/data['official_entry_subdirectory']
    entry = repo/data['model_entry']
    # Verify all retained official layers/models, not only the top-level entry.
    for rel in data['full_layers_and_models_original_byte_closure']:
        assert hashlib.sha256((repo/rel).read_bytes()).hexdigest() == data['inventory'][rel]['sha256'], rel
    with _isolated_official_import(root):
        spec = importlib.util.spec_from_file_location('_paper5_official_temporal_'+model_name, entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Model(SimpleNamespace(**args))


class OfficialTemporalAdapter(nn.Module):
    def __init__(self, model_name, config, channels, dataset=None):
        super().__init__()
        self.model_name = model_name
        self.length = int(config.get('length',96))
        self.horizon = int(config.get('horizon',96))
        self.channels = int(channels)
        dataset = dataset or config.get('dataset')
        if dataset is None:
            raise ValueError('Official dataset preset must be explicit')
        preset = official_temporal_preset(model_name, dataset, self.horizon, config)
        args = dict(preset['official_args'])
        args.update(config.get('official_args',{}))
        args.update(config.get('official_config',{}))
        args.update(seq_len=self.length, pred_len=self.horizon,
                    enc_in=self.channels, dec_in=self.channels, c_out=self.channels)
        if model_name == 'TimeMixer':
            args.update(task_name='long_term_forecast', use_future_temporal_feature=False)
        self.official_args = args
        self.training_defaults = dict(preset['training'])
        self.adaptation_scope = preset['transfer_scope']
        self.core = _official_model(model_name, args, config)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.length,self.channels):
            raise ValueError('Expected configured [B,L,C]')
        if self.model_name == 'TimeMixer':
            y = self.core(x, None, None, None)
        else:
            y = self.core(x)
        if tuple(y.shape) != (x.shape[0],self.horizon,self.channels):
            raise RuntimeError('Official output shape mismatch: '+str(tuple(y.shape)))
        return y


def create_temporal(model_name, config, channels, dataset=None):
    if model_name not in ('PatchTST','TimeMixer','ModernTCN'):
        raise ValueError('Unknown official temporal model: '+str(model_name))
    return OfficialTemporalAdapter(model_name, config, channels, dataset)
