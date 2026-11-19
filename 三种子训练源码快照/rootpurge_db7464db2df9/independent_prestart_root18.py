"""Reviewed future score-free prestart auditor; default PENDING, no array import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from root18_production_common import (load_frozen, gate_old_cohorts, verify_controller_identity,
    exclusive, receipt_name, pending, require, config_sha, validate_evidence, DATASETS)


def review(pin, authpin, run_id=None, owned=None, active_configs=None):
    root, plan, recipe, documents, freeze_pin = load_frozen(pin, authpin)
    old_admission = gate_old_cohorts()
    from root18_resources import collect_telemetry, validate_resource_admission
    owned = {} if owned is None else owned; active_configs = {} if active_configs is None else active_configs
    candidate = plan['runs'][run_id] if run_id is not None else None
    resource = collect_telemetry(root, owned); validate_resource_admission(resource, active_configs, candidate)
    controller = None
    if run_id is not None:
        from root18_admission_guard import metadata, safe_path, sha
        start_path = safe_path(root, 'ROOT18_ACTUAL_START.json'); start = metadata(start_path)
        controller = verify_controller_identity(start, pin, authpin)
        require(controller['pid'] == os.getpid(), 'only explicitly authorized owned controller may audit queued runs')
        controller['actual_start_sha256'] = sha(start_path)
    # Every physical/source/canonical/resource gate precedes observed prefix decoding.
    import numpy as np
    datasets = {}; total = 0
    for dataset in sorted(DATASETS):
        meta = plan['data_metadata'][dataset]
        new_path = root / 'data' / (dataset + '_trainval.npz')
        old_path = Path(plan['original_root']) / 'data' / (dataset + '_trainval.npz')
        with np.load(old_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
            require(set(old.files) == set(new.files) == {'data', 'split_counts'}, 'prefix fields differ')
            original = old['data']; copied = new['data']; bounds = new['split_counts']
            require(original.dtype == copied.dtype == np.dtype('float32')
                    and copied.shape == (meta['prefix_rows'], meta['channels'])
                    and bounds.tolist() == meta['bounds'][:2]
                    and np.array_equal(old['split_counts'], bounds)
                    and np.array_equal(original.view(np.uint32), copied.view(np.uint32))
                    and np.isfinite(copied).all(), 'actual full prefix bitwise identity differs')
            mean = copied[:bounds[0]].mean(0, dtype=np.float64).astype(np.float32)
            std = copied[:bounds[0]].std(0, dtype=np.float64).astype(np.float32); std[std < 1e-6] = 1
            require(hashlib.sha256(mean.tobytes()).hexdigest() == meta['scaler_mean_sha256']
                    and hashlib.sha256(std.tobytes()).hexdigest() == meta['scaler_std_sha256'], 'TRAIN-only scaler SHA differs')
            total += copied.size
        calendar = np.load(root / 'data' / (dataset + '_known_calendar.npy'), allow_pickle=False)
        require(calendar.dtype == np.dtype('float32') and calendar.shape == (meta['prefix_rows'], meta['calendar_channels'])
                and np.isfinite(calendar).all(), 'actual known calendar shape differs')
        datasets[dataset] = {'prefix_sha256': meta['prefix_sha256'], 'calendar_sha256': meta['calendar_sha256'],
            'scaler_mean_sha256': meta['scaler_mean_sha256'], 'scaler_std_sha256': meta['scaler_std_sha256'],
            'prefix_rows': meta['prefix_rows'], 'channels': meta['channels'], 'prefix_bitwise_elements': meta['prefix_rows'] * meta['channels']}
        del original, copied, bounds, mean, std, calendar
    # The CPU comparison can take time. Refresh resource ownership after all six datasets.
    resource = collect_telemetry(root, owned); validate_resource_admission(resource, active_configs, candidate)
    name = receipt_name('ACTUAL_INDEPENDENT_ROOT18_PRESTART_REVIEW_'); now = time.time()
    pre = {'status': 'PASS_ACTUAL_INDEPENDENT_ROOT18_SOURCE_PREFIX_RESOURCE_PRESTART',
        'receipt_name': name, 'utc_epoch': now, 'expires_after_seconds': 120,
        'plan_sha256': pin, 'authorization_sha256': authpin, 'freeze_sha256': freeze_pin,
        'files': plan['files'], 'qa_sha256': plan['qa_sha256'], 'selection_sha256': plan['selection_sha256'],
        'auditor_source_sha256': plan['files']['independent_prestart_root18.py'],
        'datasets': datasets, 'prefix_bitwise_elements_checked': total,
        'test_scores_accessed': False, 'new_validation_scores_accessed': False,
        'old_cohort_admission': old_admission, 'resource_telemetry': resource,
        'active_configs': active_configs, 'owned': owned, 'candidate_run_id': run_id,
        'controller_identity': controller, 'automatic_retry': False, 'training_started_by_auditor': False}
    rid = run_id or sorted(plan['runs'])[0]
    virtual = {'status': 'ROOT_EXPLICIT_ONE_RUN_ROOT18_CAPABILITY_AFTER_FRESH_INDEPENDENT_AUDIT',
        'run_id': rid, 'plan_sha256': pin, 'run_config_sha256': config_sha(plan['runs'][rid]),
        'automatic_retry': False, 'test_allowed': False, 'utc_epoch': now, 'expires_after_seconds': 120}
    validate_evidence(plan, recipe, rid, pin, **documents, pre=pre, capability=virtual, now=now)
    exclusive(root / name, pre)
    return name, pre


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--review-start-once', action='store_true')
    parser.add_argument('--plan-sha256'); parser.add_argument('--authorization-sha256'); args = parser.parse_args()
    if not args.review_start_once or not args.plan_sha256 or not args.authorization_sha256:
        print(json.dumps(pending())); return
    name, pre = review(args.plan_sha256, args.authorization_sha256)
    print(json.dumps({'status': pre['status'], 'receipt_name': name, 'plan_sha256': args.plan_sha256}))


if __name__ == '__main__':
    main()
