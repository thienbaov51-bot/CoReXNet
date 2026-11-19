"""Explicit future one-attempt original-batch CUDA QA; default PENDING."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import random
import shutil
import time
import traceback
from types import SimpleNamespace

from root18_production_common import (local_root, verify_manifest, gate_old_cohorts,
    gpu_idle_except, exclusive, pending, config_sha, require, process_identity, sha)


def probe(pin):
    root, manifest, recipe, _ = verify_manifest(pin)
    admission = gate_old_cohorts(); admission.update(gpu_idle_except())
    require(shutil.disk_usage(root).free >= 1536 << 20, 'PENDING: actual disk floor')
    exclusive(root / 'ROOT18_PROBE_CLAIM.json', {'status': 'ACTUAL_ONE_ATTEMPT_ROOT18_PROBE_CLAIM',
        'manifest_sha256': pin, 'utc_epoch': time.time(), 'pid_identity': process_identity(__import__('os').getpid()),
        'old_cohort_admission': admission, 'automatic_retry': False, 'training_started': False})
    # Scientific libraries and source constructors are reached only after actual admission.
    import numpy as np
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one actual CUDA device required')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = False
    modules = {}
    for name, filename in (('RootPurgeLinear', 'RootPurge_classes_exact.py'),):
        spec = importlib.util.spec_from_file_location('root18_probe_' + name, root / filename)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); modules[name] = module
    rows = []
    for row_recipe in sorted(recipe['runs'], key=lambda row: row['run_id']):
        config = dict(row_recipe['config_draft']); rid = row_recipe['run_id']
        row = {'run_id': rid, 'synthetic_only': True, 'measurement_array_read': False}
        try:
            gpu_idle_except(__import__('os').getpid())
            require(shutil.disk_usage(root).free >= 1536 << 20, 'actual disk floor drained during one probe')
            random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            resident = torch.zeros((config['prefix_rows'], config['channels']), device='cuda')
            calendar_channels = manifest['data_metadata'][config['dataset']]['calendar_channels']
            marks_resident = torch.zeros((config['prefix_rows'], calendar_channels), device='cuda')
            indices_resident = torch.arange(config['prefix_rows'], device='cuda')
            permutation_resident = torch.arange(config['prefix_rows'], device='cuda')
            xoffsets = torch.arange(-config['seq_len'], 0, device='cuda')
            yoffsets = torch.arange(config['pred_len'], device='cuda')
            module = modules[config['model']]; args = SimpleNamespace(**config)
            require(config['model'] == 'RootPurgeLinear', 'Root-only original class required')
            model = module.RootPurgeModel(args, module.BaseLinearModel).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            x = torch.randn(config['batch_size'], config['seq_len'], config['channels'], device='cuda')
            y = torch.randn(config['batch_size'], config['pred_len'], config['channels'], device='cuda')
            marks = torch.zeros(config['batch_size'], config['seq_len'], 4, device='cuda') if config.get('freq') == 'h' else None
            model.train()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = model(x, y)['loss']
                require(bool(torch.isfinite(loss)), 'actual synthetic loss nonfinite')
                loss.backward(); optimizer.step()
            model.eval()
            with torch.inference_mode():
                prediction = model(x)
                require(prediction.shape == y.shape and bool(torch.isfinite(prediction).all()), 'actual forecast shape/nonfinite')
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(); reserved = torch.cuda.max_memory_reserved()
            parameters = sum(parameter.numel() for parameter in model.parameters())
            gpu_reserve = max(1200, (max(peak, reserved) + (1 << 20) - 1) // (1 << 20) + 512)
            disk_reserve = max(64 << 20, parameters * 12 + (16 << 20))
            require(gpu_reserve <= 22500 and disk_reserve <= 4 << 30, 'actual original-batch recipe exceeds production cap')
            config.update(reserve_gpu_mib=gpu_reserve, reserve_disk_bytes=disk_reserve)
            row.update(status='PASS', optimizer_steps=2, precision='FP32_TF32', config_sha256=config_sha(config),
                fullprefix_shape=[config['prefix_rows'], config['channels']], peak_gpu_bytes=peak,
                calendar_shape=[config['prefix_rows'], calendar_channels],
                peak_gpu_reserved_bytes=reserved, parameter_count=parameters, reserve_gpu_mib=gpu_reserve,
                reserve_disk_bytes=disk_reserve, config=config, device_name=torch.cuda.get_device_name(),
                torch_version=torch.__version__, loss_policy='EXACT_ROOTPURGE_REGULARIZED_LOSS')
            del resident, marks_resident, indices_resident, permutation_resident, xoffsets, yoffsets, model, optimizer, x, y, marks, prediction, loss
            gc.collect(); torch.cuda.empty_cache()
        except Exception as error:
            row.update(status='FAIL', error=str(error), traceback=traceback.format_exc()); rows.append(row); break
        rows.append(row)
    receipt = {'status': 'PASS_ALL18_ACTUAL_ORIGINAL_BATCH_SYNTHETIC_CUDA_ONLY' if len(rows) == 18 and all(row['status'] == 'PASS' for row in rows) else 'FAILED_ONE_ATTEMPT_ROOT18_CUDA_QA_REQUIRES_ROOT_REVIEW',
        'utc_epoch': time.time(), 'rows': rows, 'manifest_sha256': pin, 'recipe_map_sha256': manifest['recipe_map_sha256'],
        'probe_claim_sha256': sha(root / 'ROOT18_PROBE_CLAIM.json'),
        'source_inventory': manifest['files'], 'old_cohort_admission': admission,
        'measurement_array_read': False, 'research_scores_read': False, 'training_started': False,
        'automatic_retry': False}
    exclusive(root / 'ACTUAL_ROOT18_SYNTHETIC_CUDA_RECEIPT.json', receipt)
    print(json.dumps({'status': receipt['status'], 'row_count': len(rows)}))
    require(receipt['status'] == 'PASS_ALL18_ACTUAL_ORIGINAL_BATCH_SYNTHETIC_CUDA_ONLY', 'one-attempt QA failed; no retry')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--probe-once', action='store_true'); parser.add_argument('--manifest-sha256')
    args = parser.parse_args()
    if not args.probe_once or not args.manifest_sha256:
        print(json.dumps(pending())); return
    probe(args.manifest_sha256)


if __name__ == '__main__':
    main()
