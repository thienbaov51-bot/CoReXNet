"""New ADD5 campaign utilities; original window/scaler/seed/optimizer arithmetic.

The four copied definitions preserve the previous comparator runner's source;
new wrappers add cohort/source guards and device-aware evaluation, never recipes.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time


def initialize_tensor_runtime(dependency_root=None):
    global np,torch
    folder = Path(dependency_root) if dependency_root is not None else Path(__file__).parent / 'dependencies'
    if folder.is_dir():
        sys.path.insert(0, str(folder.resolve()))
    import numpy as np
    import torch
    return torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    if exclusive:
        with path.open('xb') as stream:
            stream.write(payload)
        return
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('xb') as stream:
        stream.write(payload)
    os.replace(temporary, path)


def inside(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('workspace-relative path required: ' + str(relative))
    result = (root / relative).resolve()
    if result != root and root not in result.parents:
        raise ValueError('path escapes this cohort')
    return result


def expected_hash(meta):
    return meta if isinstance(meta, str) else meta['sha256']


def verify_inventory(root, inventory):
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('nonempty source inventory required')
    for name, meta in inventory.items():
        path = inside(root, name)
        if not path.is_file() or sha(path) != expected_hash(meta):
            raise ValueError('source identity mismatch: ' + name)
        if isinstance(meta, dict) and 'bytes' in meta and path.stat().st_size != meta['bytes']:
            raise ValueError('source byte count mismatch: ' + name)


def effective_config(config):
    result = dict(config)
    # Flat fields are authoritative; an optional recipe.training must agree.
    recipe = config.get('recipe', {})
    training = recipe.get('training', {}) if isinstance(recipe, dict) else {}
    for key, value in training.items():
        if key in result and result[key] != value:
            raise ValueError('conflicting recipe and flat field: ' + key)
        result[key] = value
    return result


def validate_config(config, run_id, plan):
    c = effective_config(config)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', run_id) or c.get('run_id') != run_id:
        raise ValueError('unsafe/mismatched run identity')
    if c['model'] not in plan['models'] or c['dataset'] not in plan['datasets']:
        raise ValueError('undeclared model/dataset')
    if c['seed'] != 2026 or c['length'] != 96 or c['horizon'] not in plan['horizons']:
        raise ValueError('fixed seed/length/horizon mismatch')
    if c.get('evaluate_test') is not False or c.get('initialization') != 'from_scratch_no_old_checkpoint':
        raise ValueError('training must be fresh and test-free')
    if c.get('amp', False) is not False or c.get('adapter') not in ('classical', 'recent'):
        raise ValueError('ADD5 requires an explicit adapter and FP32 recipe')
    if c.get('loss', 'MSE') not in ('MSE', 'MAE') or c.get('optimizer', 'Adam') not in ('Adam', 'AdamW'):
        raise ValueError('unsupported locked objective/optimizer')
    for key in ['batch_size', 'epochs', 'patience', 'reserve_gpu_mib', 'test_batch_size', 'test_reserve_gpu_mib']:
        if not isinstance(c.get(key), int) or c[key] < 1:
            raise ValueError('positive locked integer required: ' + key)
    if max(c['reserve_gpu_mib'], c['test_reserve_gpu_mib']) > plan.get('memory_budget_mib', 22500):
        raise ValueError('individual reservation exceeds declared GPU budget')
    if c.get('resource_qa_pass') is not True:
        raise ValueError('recipe-specific resource/shape QA admission is not approved')
    d = plan['data'][c['dataset']]
    if c['expected_data_sha256'] != d['sha256'] or c['channels'] != d['channels']:
        raise ValueError('dataset/channel identity mismatch')
    for key, val in [('seq_len', 96), ('pred_len', c['horizon']), ('enc_in', c['channels'])]:
        if key in c and c[key] != val:
            raise ValueError('top-level shape alias mismatch: ' + key)
        if key in c.get('official_config', {}) and c['official_config'][key] != val:
            raise ValueError('official shape alias mismatch: ' + key)
    return c


def load_plan(root, expected_sha, verify_sources=True):
    root = Path(root).resolve()
    path = root / 'COHORT_PRETRAIN_PLAN.json'
    if not expected_sha or sha(path) != expected_sha:
        raise ValueError('explicit immutable cohort plan SHA required')
    plan = read_json(path)
    count = plan.get('expected_count', plan.get('run_count'))
    if not isinstance(count, int) or count < 1 or len(plan['runs']) != count:
        raise ValueError('dynamic cohort count mismatch')
    if plan['seed'] != 2026 or plan['length'] != 96:
        raise ValueError('cohort seed/length mismatch')
    if str(root) != plan['cohort_root']:
        raise ValueError('this plan belongs to a different run directory')
    declared = {(m, d, h) for m in plan['models'] for d in plan['datasets'] for h in plan['horizons']}
    actual = set()
    for run_id, entry in plan['runs'].items():
        c = validate_config(entry['config'], run_id, plan)
        actual.add((c['model'], c['dataset'], c['horizon']))
        if not entry.get('source_sha256') or not entry.get('config_sha256'):
            raise ValueError('run config/source fingerprints missing')
        for name in entry['source_sha256']:
            if Path(name).name != name or not name.endswith('.py'):
                raise ValueError('run snapshot must consist of explicit basename Python sources')
    if len(actual) != count or actual != declared:
        raise ValueError('not the complete predeclared grid')
    if not plan.get('tool_source_sha256'):
        raise ValueError('entrypoint source identities required')
    if verify_sources:
        verify_inventory(root, plan['source_inventory'])
    return plan


def verify_tool(plan, path):
    if sha(path) != plan['tool_source_sha256'][Path(path).name]:
        raise ValueError('pipeline/controller/evaluator source is not authorized')


def authorize(root, plan_sha, authorization_sha):
    path = Path(root) / 'RUN_AUTHORIZATION.json'
    if not authorization_sha or sha(path) != authorization_sha:
        raise ValueError('explicit ROOT authorization SHA required')
    value = read_json(path)
    if value['status'] != 'ROOT_AUTHORIZED' or value['pretrain_plan_sha256'] != plan_sha:
        raise ValueError('authorization does not bind this pretrain plan')
    if value['cohort_root'] != str(Path(root).resolve()):
        raise ValueError('authorization belongs to another cohort')
    return value


def registry_ids(root, name):
    folder = Path(root) / name
    return {p.stem for p in folder.glob('*.json')} if folder.exists() else set()


def train_complete(root, run_id, entry):
    output = Path(root) / 'results' / run_id
    required = ['metrics.json', 'config.json', 'history.json', 'best.pt']
    if not all((output / name).is_file() for name in required):
        return False
    config = read_json(output / 'config.json')
    metrics = read_json(output / 'metrics.json')
    if config != entry['config'] or metrics['config'] != config or 'test' in metrics:
        raise ValueError('completed run config/early-test collision: ' + run_id)
    if metrics['code_sha256'] != entry['source_sha256'] or metrics['data_sha256'] != config['expected_data_sha256']:
        raise ValueError('completed training provenance mismatch: ' + run_id)
    for name, h in entry['source_sha256'].items():
        if sha(output / 'code_snapshot' / name) != h:
            raise ValueError('snapshot identity changed: ' + run_id)
    history = read_json(output / 'history.json')
    if len(history) != metrics['epochs_run'] or not history or history[0]['epoch'] != 1:
        raise ValueError('invalid complete epoch history')
    if not 1 <= metrics['best_epoch'] <= len(history) <= effective_config(config)['epochs']:
        raise ValueError('invalid best validation epoch')
    return True


def make_model(config):
    c = dict(config)
    c.setdefault('seq_len', c['length'])
    c.setdefault('pred_len', c['horizon'])
    c.setdefault('enc_in', c['channels'])
    c.setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    dependency_root = c.get('dependency_root')
    if dependency_root and Path(dependency_root).is_dir():
        sys.path.insert(0, str(Path(dependency_root).resolve()))
    module_name = c['adapter'] + '_adapter'
    import importlib
    adapter = importlib.import_module(module_name)
    if hasattr(adapter, 'make_model'):
        return adapter.make_model(c)
    if c['adapter'] == 'classical' and hasattr(adapter, 'create_classical'):
        return adapter.create_classical(c['model'], c, c['channels'], c['dataset'])
    if c['adapter'] == 'recent' and hasattr(adapter, 'create_recent'):
        return adapter.create_recent(c['model'], c, c['channels'], c['dataset'])
    raise ValueError('adapter needs make_model(config) or its explicit create_* API')


def predict(model, x, origins):
    pred = model.predict_windows(x, origins) if hasattr(model, 'predict_windows') else model(x)
    if not isinstance(pred, torch.Tensor) or pred.ndim != 3 or pred.shape[0] != x.shape[0] or pred.shape[2] != x.shape[2]:
        raise ValueError('adapter must return a B,H,C tensor')
    return pred


def evaluate(model, windows, split, batch_size, amp=False):
    # CUDA arithmetic/order is the original comparator evaluator. The device
    # variable permits the same implementation in CPU-only synthetic fixtures.
    model.eval()
    device = windows.values.device
    with torch.inference_mode():
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        channel_se = torch.zeros(windows.channels, device=device, dtype=torch.float64)
        indices = windows.indices[split]
        for chunk in indices.split(batch_size):
            x, y = windows.batch(chunk)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                pred = predict(model, x, chunk)
            if pred.shape != y.shape:
                raise ValueError('forecast shape differs from the frozen horizon')
            difference = pred.float() - y
            totals[0] += difference.square().sum(dtype=torch.float64)
            totals[1] += difference.abs().sum(dtype=torch.float64)
            totals[2] += difference.numel()
            channel_se += difference.square().sum((0, 1), dtype=torch.float64)
        totals = totals.cpu().numpy()
        channel_mse = (channel_se / (len(indices) * windows.horizon)).cpu().tolist()
        return {'mse': float(totals[0] / totals[2]), 'mae': float(totals[1] / totals[2]),
                'elements': int(totals[2]), 'windows': len(indices), 'channel_mse': channel_mse}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False


class Windows:
    def __init__(self, path, length, horizon, device, pilot=False):
        z = np.load(path, allow_pickle=False)
        a = np.asarray(z['data'], dtype=np.float32)
        bounds = list(map(int, z['split_counts']))
        if len(bounds) != 3 or bounds[-1] != len(a):
            raise ValueError('split_counts must contain absolute train/val/test end offsets')
        if not np.isfinite(a).all():
            raise ValueError('non-finite data: resolve preprocessing before any training')
        mean = a[:bounds[0]].mean(axis=0, dtype=np.float64).astype(np.float32)
        std = a[:bounds[0]].std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        self.values = torch.as_tensor((a - mean) / std, device=device)
        self.mean, self.std = mean, std
        self.channels = a.shape[1]
        self.length, self.horizon, self.bounds = length, horizon, bounds
        self.xoff = torch.arange(-length, 0, device=device)
        self.yoff = torch.arange(horizon, device=device)
        self.indices = {}
        for name, start, end in [('train',length,bounds[0]),('val',bounds[0],bounds[1]),('test',bounds[1],bounds[2])]:
            ix = torch.arange(start, end-horizon+1, device=device)
            if len(ix) == 0: raise ValueError(f'empty {name} split')
            # Pilot uses deterministic evenly-spaced windows, never test windows.
            cap = 12000 if name == 'train' else 4096
            if pilot and name != 'test' and len(ix) > cap:
                ix = ix[torch.linspace(0,len(ix)-1,cap,device=device).long()]
            self.indices[name] = ix

    def batch(self, indices):
        return self.values[indices[:,None]+self.xoff], self.values[indices[:,None]+self.yoff]


def make_optimizer(model, config):
    kind = config.get('optimizer', 'Adam')
    cls = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}[kind]
    return cls(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.0))


def make_scheduler(optimizer, config, steps):
    kind = config.get('lr_schedule', 'type1')
    lr = config['lr']
    if kind == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'],
                 eta_min=lr * config.get('eta_min_ratio', 0.0)), False
    if kind == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=steps,
                 epochs=config['epochs'], pct_start=config.get('pct_start', 0.3), max_lr=lr), True
    if kind == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.0), False
    if kind == 'type1':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: config.get('first_epoch_lr_ratio', 1.0) if e == 0 else 0.5 ** max(0, e-1)), False
    if kind == 'type3':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: config.get('first_epoch_lr_ratio', 1.0) if e == 0 else config.get('lr_decay_factor', 0.9) ** max(0, e-3)), False
    if kind == 'type2':
        mapping = {2:5e-5, 4:1e-5, 6:5e-6, 8:1e-6, 10:5e-7, 15:1e-7, 20:5e-8}
        def ratio(e):
            keys = [k for k in mapping if k<=e]
            return mapping[max(keys)]/lr if keys else 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, ratio), False
    raise ValueError('unsupported predeclared learning-rate schedule: '+kind)
