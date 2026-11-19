"""Seven unchanged, previously executed model cores for fresh dataset runs.

This module performs no data loading, training, scoring or dataset selection.
An explicit preset-dataset alias only selects the byte-preserved wrapper's
constructor defaults; the complete frozen official_config supplies the new
dataset's predeclared dimensions and hyperparameters.
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

MODELS = ('OURS', 'DLinear', 'SparseTSF', 'PatchTST', 'ModernTCN', 'MixLinear', 'PhaseFormer')
_HERE = Path(__file__).resolve().parent
_OURS_NAMES = {'models', 'i3_response_residual_exchange', 'i2_curvature_stable',
               'i2_curvature', 'i1_common_contrast', 'i1_context_v2'}


def package_root(config):
    return Path(config.get('frozen_model_sources_root', _HERE)).resolve()


def verify_sources(config):
    root = package_root(config)
    proof_path = Path(config.get('model_reuse_manifest_path', root/'MODEL_REUSE_MANIFEST.json')).resolve()
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    if proof['status'] != 'PREVIOUSLY_EXECUTED_SEVEN_CORES_BYTE_PRESERVED':
        raise ValueError('Unrecognized model source manifest')
    if tuple(proof['models']) != MODELS:
        raise ValueError('Model panel changed')
    for relative, record in proof['source_inventory'].items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Source inventory path escapes package')
        p = root/rel
        if p.stat().st_size != record['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Frozen model source bytes changed: '+relative)
    return root, proof


class _OwnLoader(importlib.abc.Loader):
    def __init__(self, path, finder, namespace=False):
        self.path, self.finder, self.namespace = path, finder, namespace

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self.namespace:
            module.__path__ = [str(self.path)]
            return
        module.__file__ = str(self.path)
        custom = dict(vars(builtins))
        original_import = builtins.__import__
        prefix = self.finder.prefix
        def own_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name.split('.')[0] in _OURS_NAMES:
                result = original_import(prefix+'.'+name, globals, locals, fromlist, 0)
                return result if fromlist else sys.modules[prefix+'.'+name.split('.')[0]]
            return original_import(name, globals, locals, fromlist, level)
        custom['__import__'] = own_import
        module.__dict__['__builtins__'] = custom
        exec(compile(self.path.read_bytes(), str(self.path), 'exec'), module.__dict__)


class _OwnFinder(importlib.abc.MetaPathFinder):
    def __init__(self, prefix, root):
        self.prefix, self.root = prefix, root
    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.prefix:
            return importlib.util.spec_from_loader(fullname, _OwnLoader(self.root,self,True),is_package=True)
        if fullname.startswith(self.prefix+'.'):
            p = self.root.joinpath(*fullname[len(self.prefix)+1:].split('.')).with_suffix('.py')
            if p.is_file():
                return importlib.util.spec_from_loader(fullname,_OwnLoader(p,self))
        return None


def _ours(config, channels, root):
    prefix = '_add3_original_ours_'+hashlib.sha256(str(root).encode()).hexdigest()[:12]
    if not any(isinstance(f,_OwnFinder) and f.prefix==prefix for f in sys.meta_path):
        sys.meta_path.insert(0,_OwnFinder(prefix,root/'frozen_sources/ours'))
    original = importlib.import_module(prefix+'.models')
    return original.create_model(dict(config,model='I3_response_residual_exchange'),channels)


def _wrapper(name, root):
    module_name = '_add3_original_'+name+'_'+hashlib.sha256(str(root).encode()).hexdigest()[:12]
    if module_name not in sys.modules:
        p = root/'frozen_sources/adapters'/(name+'.py')
        spec = importlib.util.spec_from_file_location(module_name,p)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name,None)
            raise
    return sys.modules[module_name]


def make_model(config):
    cfg = dict(config)
    name = cfg.get('display_model',cfg.get('model'))
    if name not in MODELS:
        raise ValueError('Only the user-confirmed seven models are supported')
    channels = int(cfg.get('channels',cfg.get('enc_in',0)))
    if channels < 1 or int(cfg.get('length',96)) != 96 or int(cfg.get('horizon',96)) not in (96,192,336,720):
        raise ValueError('Expected frozen positive C, L96 and one of four horizons')
    root, _ = verify_sources(cfg)
    if name == 'OURS':
        return _ours(cfg,channels,root)
    if not cfg.get('official_config'):
        raise ValueError('Complete official constructor configuration must be predeclared')
    cfg.update(seq_len=96,pred_len=int(cfg['horizon']),enc_in=channels)
    preset_dataset = cfg.get('adapter_preset_dataset')
    if not preset_dataset:
        raise ValueError('Explicit prior constructor-preset alias required')
    if name in ('DLinear','SparseTSF'):
        cfg['official_vendor_root'] = str(root/'vendor/lightweight')
        return _wrapper('lightweight_adapter',root).create_lightweight(name,cfg,channels,preset_dataset)
    if name in ('PatchTST','ModernTCN'):
        cfg['official_vendor_root'] = str(root/'vendor/temporal')
        cfg['official_temporal_manifest_path'] = str(root/'vendor/temporal/OFFICIAL_TEMPORAL_MANIFEST.json')
        return _wrapper('temporal_adapter',root).create_temporal(name,cfg,channels,preset_dataset)
    cfg['official_vendor_root'] = str(root/'vendor/recent')
    cfg['official_recent_manifest_path'] = str(root/'vendor/recent/OFFICIAL_RECENT_MANIFEST.json')
    return _wrapper('recent_adapter',root).create_recent(name,cfg,channels,preset_dataset)
