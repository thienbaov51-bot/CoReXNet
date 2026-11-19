"""Fixed classical comparators for the added single-seed, history-only campaign.

Transformer executes the byte-preserved TSL encoder/decoder, including both
decoder attentions and official embeddings/LayerNorms. Its decoder receives
the last label_len observed values followed by zeros; timestamps are None.
LSTM is a standard two-layer PyTorch LSTM with a direct joint H*C head, not
an implementation claimed to originate from a particular forecasting paper.
Training recipes and upstream byte identities are in CLASSICAL_RECIPES.json.
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


def _recipes_path(config=None):
    config = config or {}
    explicit = config.get('classical_recipes_path')
    return Path(explicit).resolve() if explicit else Path(__file__).with_name('CLASSICAL_RECIPES.json')


def classical_preset(model_name, dataset, horizon=96, config=None):
    """Return a fresh JSON-compatible, predeclared recipe for the actual task."""
    if model_name == 'VanillaTransformer':
        model_name = 'Transformer'
    data = json.loads(_recipes_path(config).read_text(encoding='utf-8-sig'))
    return data['presets'][model_name][dataset][str(int(horizon))]


def _vendor(config):
    default = Path(__file__).with_name('vendor_classical') / 'Time-Series-Library'
    # This override points to the repository root, not the parent vendor folder.
    return Path(config.get('official_classical_vendor_root',
        os.environ.get('PAPER5_CLASSICAL_VENDOR_ROOT', default))).resolve()


@contextmanager
def _isolated_import(root):
    # Constructors are inside the same context. Previously loaded official
    # packages are restored even on error; class globals retain their own refs.
    with _IMPORT_LOCK:
        def local(name):
            return any(name == p or name.startswith(p + '.') for p in _PREFIXES)
        prior = {n: m for n, m in sys.modules.items() if local(n)}
        old_path = sys.path[:]
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
            sys.modules.update(prior)
            sys.path[:] = old_path


def _official_transformer(args, config):
    recipes = json.loads(_recipes_path(config).read_text(encoding='utf-8-sig'))
    data = recipes['official_transformer_source']
    root = _vendor(config)
    for rel, expected in data['runtime_source_sha256'].items():
        actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Official Transformer source byte mismatch: {rel}')
    with _isolated_import(root):
        entry = root / data['entry']
        spec = importlib.util.spec_from_file_location('_main20261010_official_transformer', entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Model(SimpleNamespace(**args))


class TransformerAdapter(nn.Module):
    def __init__(self, config, channels, dataset=None):
        super().__init__()
        self.length = int(config.get('length', 96))
        self.horizon = int(config.get('horizon', 96))
        self.channels = int(channels)
        dataset = dataset or config.get('dataset')
        if dataset is None:
            raise ValueError('dataset is required for a fixed classical recipe')
        preset = classical_preset('Transformer', dataset, self.horizon, config)
        args = dict(preset['official_config'])
        args.update(config.get('official_config', {}))
        args.update(task_name='long_term_forecast', seq_len=self.length,
                    pred_len=self.horizon, enc_in=self.channels,
                    dec_in=self.channels, c_out=self.channels)
        self.label_len = int(args['label_len'])
        if not 1 <= self.label_len <= self.length:
            raise ValueError('label_len must fit entirely within observed history')
        if self.length + self.horizon + self.label_len >= 5000:
            raise ValueError('Requested task exceeds official positional encoding support')
        self.official_config = dict(args)
        self.core = _official_transformer(args, config)

    def decoder_input(self, x):
        return torch.cat((x[:, -self.label_len:, :],
                          x.new_zeros(x.shape[0], self.horizon, self.channels)), dim=1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.length, self.channels):
            raise ValueError(f'Expected [B,{self.length},{self.channels}], received {tuple(x.shape)}')
        y = self.core(x, None, self.decoder_input(x), None)
        if y.shape != (x.shape[0], self.horizon, self.channels):
            raise RuntimeError(f'Official Transformer output shape mismatch: {tuple(y.shape)}')
        return y


class LSTMAdapter(nn.Module):
    """Two-layer unidirectional joint-channel LSTM, direct full-horizon head."""
    def __init__(self, config, channels, dataset=None):
        super().__init__()
        self.length = int(config.get('length', 96))
        self.horizon = int(config.get('horizon', 96))
        self.channels = int(channels)
        dataset = dataset or config.get('dataset')
        if dataset is None:
            raise ValueError('dataset is required for a fixed classical recipe')
        preset = classical_preset('LSTM', dataset, self.horizon, config)
        args = dict(preset['official_config'])
        args.update(config.get('official_config', {}))
        hidden = int(args['hidden_size'])
        layers = int(args['num_layers'])
        if layers != 2 or hidden < 1:
            raise ValueError('This standard comparator requires two LSTM layers and a positive width')
        self.official_config = dict(args, input_size=self.channels,
                                    output_size=self.horizon * self.channels)
        self.core = nn.LSTM(self.channels, hidden, layers, batch_first=True,
                            dropout=float(args['dropout']), bidirectional=False)
        self.head = nn.Linear(hidden, self.horizon * self.channels, bias=True)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.length, self.channels):
            raise ValueError(f'Expected [B,{self.length},{self.channels}], received {tuple(x.shape)}')
        # A new zero recurrent state is used for each observed window. No future
        # targets, autoregressive teacher forcing, auxiliary head, or RevIN.
        _, (hidden, _) = self.core(x)
        return self.head(hidden[-1]).reshape(x.shape[0], self.horizon, self.channels)


def create_classical(model_name, config, channels, dataset=None):
    if model_name in ('Transformer', 'VanillaTransformer'):
        return TransformerAdapter(config, channels, dataset)
    if model_name == 'LSTM':
        return LSTMAdapter(config, channels, dataset)
    raise ValueError(f'Unknown classical model: {model_name}')
