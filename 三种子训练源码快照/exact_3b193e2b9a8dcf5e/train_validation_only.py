"""Train and validate with unchanged main kernel on a train/validation prefix.

This entry point never constructs test indices. The input NPZ must contain
only data[:original_validation_end] and its two absolute split end offsets.
Prefix/scaler identity must be verified against the original full NPZ before
using this entry point; the frozen original train.py remains byte unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import train


class PrefixWindows:
    def __init__(self, path, length, horizon, device, pilot=False):
        if pilot:
            raise ValueError('sensitivity requires full train/validation budget')
        z = np.load(path, allow_pickle=False)
        a = np.asarray(z['data'], dtype=np.float32)
        bounds = list(map(int, z['split_counts']))
        if len(bounds) != 2 or bounds[-1] != len(a):
            raise ValueError('prefix split_counts must contain absolute train/validation end offsets')
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
        for name, start, end in [('train', length, bounds[0]), ('val', bounds[0], bounds[1])]:
            ix = torch.arange(start, end-horizon+1, device=device)
            if len(ix) == 0:
                raise ValueError(f'empty {name} split')
            self.indices[name] = ix

    def batch(self, indices):
        return self.values[indices[:, None]+self.xoff], self.values[indices[:, None]+self.yoff]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    args, _ = parser.parse_known_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if config.get('evaluate_test') is not False:
        raise ValueError('train/validation-only wrapper requires explicit evaluate_test=false')
    if config.get('phase') == 'pilot':
        raise ValueError('sensitivity requires full train/validation budget')
    train.Windows = PrefixWindows
    train.main()


if __name__ == '__main__':
    main()
