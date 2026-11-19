"""Lazy official-model dispatch. Each training job owns a separate process."""
import sys
from pathlib import Path

GROUPS = {
    'DLinear': 'lightweight', 'SparseTSF': 'lightweight', 'CycleNet': 'lightweight',
    'PatchTST': 'temporal', 'TimeMixer': 'temporal', 'ModernTCN': 'temporal',
    'iTransformer': 'crosschannel', 'SOFTS': 'crosschannel', 'DUET': 'crosschannel',
}


def create_model(config, channels):
    # Runtime dependencies are isolated from the pre-existing Python installation.
    deps = Path(__file__).resolve().parent.parent / 'dependencies'
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    name = config['model']
    if GROUPS[name] == 'lightweight':
        from lightweight_adapter import create_lightweight
        return create_lightweight(name, config, channels, config['dataset'])
    if GROUPS[name] == 'temporal':
        from temporal_adapter import create_temporal
        return create_temporal(name, config, channels, config['dataset'])
    from crosschannel_adapter import create_crosschannel
    return create_crosschannel(name, config, channels, config['dataset'])
