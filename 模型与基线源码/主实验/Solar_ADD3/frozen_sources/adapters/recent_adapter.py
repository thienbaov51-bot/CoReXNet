"""Faithful official MixLinear/GTR/PhaseFormer adapters, with isolated imports.

Original vendor modules execute unchanged. GTR receives only known global
forecast-origin indices; neither targets nor held-out statistics are inputs.
The complex FFT MixLinear core is always FP32, as in author default training.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

_VENDOR = Path(__file__).resolve().parent / 'vendor_recent'
_FOLDERS = {'MixLinear':'MixLinear', 'GTR':'GTR', 'PhaseFormer':'PhaseFormer_TSL'}
_DEFAULTS = {
    'MixLinear': dict(seq_len=96,pred_len=96,enc_in=7,period_len=24,lpf=4,alpha=.5),
    'GTR': dict(seq_len=96,pred_len=96,enc_in=7,cycle=24,d_model=512,
                dropout=0.,use_revin=1,individual=0),
    'PhaseFormer': dict(task_name='long_term_forecast',seq_len=96,pred_len=96,enc_in=7,
        period_len=24,latent_dim=32,phase_encoder_hidden=64,predictor_hidden=128,
        phase_layers=2,phase_attn_heads=8,phase_attn_dropout=.1,phase_attn_use_relpos=1,
        phase_attn_window=None,phase_attention_dim=None,phase_num_routers=1,
        phase_use_pos_embed=1,phase_pos_dropout=0.,use_revin=1,revin_affine=0,revin_eps=1e-5,
        phase_encoder_use_mlp=False,phase_encoder_dropout=0.,predictor_use_mlp=False,predictor_dropout=0.),
}


class _PrivateLoader(importlib.abc.Loader):
    def __init__(self, path, finder, package=False):
        self.path, self.finder, self.package = path, finder, package

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self.package:
            # Namespace shells avoid importing author experiment dispatchers.
            module.__path__ = [str(self.path)]
            return
        module.__file__ = str(self.path)
        bp = dict(vars(builtins))
        original_import = builtins.__import__
        prefix = self.finder.prefix

        def private_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name.split('.')[0] in {'models','layers','utils'}:
                mapped = prefix + '.' + name
                result = original_import(mapped, globals, locals, fromlist, 0)
                if not fromlist:
                    return sys.modules[prefix + '.' + name.split('.')[0]]
                return result
            return original_import(name, globals, locals, fromlist, level)

        bp['__import__'] = private_import
        module.__dict__['__builtins__'] = bp
        exec(compile(self.path.read_bytes(), str(self.path), 'exec'), module.__dict__)


class _PrivateFinder(importlib.abc.MetaPathFinder):
    def __init__(self, prefix, path):
        self.prefix, self.path = prefix, path

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.prefix:
            p = self.path
        elif fullname.startswith(self.prefix + '.'):
            p = self.path.joinpath(*fullname[len(self.prefix)+1:].split('.'))
        else:
            return None
        if p.is_dir():
            return importlib.util.spec_from_loader(fullname, _PrivateLoader(p,self,True),is_package=True)
        py = p.with_suffix('.py')
        if py.is_file():
            return importlib.util.spec_from_loader(fullname,_PrivateLoader(py,self))
        return None


def vendor_root(config):
    return Path(config.get('official_vendor_root',config.get('recent_vendor_root',_VENDOR))).resolve()


def verify_vendor(name, root=None, manifest_path=None):
    root = Path(root).resolve() if root is not None else _VENDOR.resolve()
    manifest_path = Path(manifest_path) if manifest_path is not None else root/'OFFICIAL_RECENT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    proof = manifest['models'][name]
    if proof['folder'] != _FOLDERS[name]:
        raise RuntimeError('Official model folder identity mismatch')
    for row in proof['files']:
        relative = Path(row['relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Vendor inventory path escapes its official folder')
        path = root/proof['folder']/relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('Official vendor bytes changed: '+str(path))
    return proof


def _official_class(name, root=None, manifest_path=None):
    root = Path(root).resolve() if root is not None else _VENDOR.resolve()
    verify_vendor(name,root,manifest_path)
    root_id = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    prefix = '_paper5_add5_official_'+name.lower()+'_'+root_id
    if not any(isinstance(f,_PrivateFinder) and f.prefix == prefix for f in sys.meta_path):
        sys.meta_path.insert(0,_PrivateFinder(prefix,root/_FOLDERS[name]))
    return importlib.import_module(prefix+'.models.'+name).Model


def resolved_model_config(name, config, channels):
    if name not in _DEFAULTS:
        raise ValueError('Unsupported recent model: '+str(name))
    cfg = dict(_DEFAULTS[name])
    cfg.update({k:config[k] for k in cfg if k in config})
    cfg.update(config.get('official_config',config.get('official_model_config',{})))
    cfg.update(seq_len=int(config.get('seq_len',config.get('lookback',96))),
               pred_len=int(config.get('pred_len',config.get('horizon',96))),enc_in=int(channels))
    if cfg['seq_len'] < 2 or cfg['pred_len'] < 1 or cfg['enc_in'] < 1:
        raise ValueError('Invalid forecast dimensions')
    if name == 'MixLinear':
        period = int(cfg['period_len'])
        if period < 2 or period % 2 or cfg['seq_len'] % period:
            raise ValueError('Official MixLinear reshape/convolution requires an even period dividing L')
        if not 1 <= int(cfg['lpf']) <= cfg['seq_len']//period:
            raise ValueError('lpf must be fixed before training within the available FFT bins')
    if name == 'GTR' and int(cfg['cycle']) < 1:
        raise ValueError('GTR requires a positive known-position cycle')
    return cfg


class RecentModel(nn.Module):
    def __init__(self,name,config,channels):
        super().__init__()
        self.model_name = name
        self.official_config = resolved_model_config(name,config,channels)
        self.vendor_root = vendor_root(config)
        self.manifest_path = Path(config.get('official_recent_manifest_path',
                                            self.vendor_root/'OFFICIAL_RECENT_MANIFEST.json')).resolve()
        proof = verify_vendor(name,self.vendor_root,self.manifest_path)
        self.official_source_hashes = {row['relative_path']:row['sha256'] for row in proof['files']}
        self.core = _official_class(name,self.vendor_root,self.manifest_path)(SimpleNamespace(**self.official_config))

    def forward(self,x,origin_indices=None):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.official_config['seq_len'],self.official_config['enc_in']):
            raise ValueError('Expected observed history [B,L,C] with frozen dimensions')
        if self.model_name == 'GTR':
            if origin_indices is None:
                raise ValueError('GTR requires known global forecast-origin indices; use predict_windows')
            origins = torch.as_tensor(origin_indices,device=x.device)
            if origins.ndim != 1 or origins.shape[0] != x.shape[0] or origins.dtype not in (torch.int32,torch.int64):
                raise ValueError('Expected one integral global forecast-origin index per observed window')
            # The author dataset returns cycle_index[s_end], not cycle_index[s_begin].
            cycle = torch.remainder(origins.long(),int(self.official_config['cycle']))
            result = self.core(x,cycle)
        elif self.model_name == 'MixLinear':
            # No FFT length/power-of-two or complex dtype alteration is made.
            with torch.autocast(device_type=x.device.type,enabled=False):
                result = self.core(x.float())
        else:
            result = self.core(x,None,None,None)
        if isinstance(result,tuple):
            result = result[0]
        if result.shape != (x.shape[0],self.official_config['pred_len'],x.shape[2]):
            raise RuntimeError('Official prediction shape does not match all-channel full-horizon task')
        return result

    def predict_windows(self,x,origin_indices=None):
        return self.forward(x,origin_indices)


def create_recent(model_name,config,channels,dataset=None):
    return RecentModel(model_name,config,channels)


def make_model(config):
    name = config.get('model',config.get('model_name'))
    channels = config.get('channels',config.get('enc_in'))
    if channels is None:
        raise ValueError('make_model requires the frozen channel count')
    return create_recent(name,config,int(channels),config.get('dataset'))
