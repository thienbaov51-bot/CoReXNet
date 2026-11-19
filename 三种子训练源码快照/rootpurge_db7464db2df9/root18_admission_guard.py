"""Prospective root18 admission contract; stdlib only, no scores or arrays.

Production helpers are local prepared source. Actual independent selection,
CUDA measurements, source/resource reviews and ROOT actions remain required.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time

ORIGINAL = '/root/autodl-tmp/paper5/models2026_pilot_20261011'
CORRECTED = ORIGINAL + '/fact_freq_corrected4'
CHILD = ORIGINAL + '/rootpurge_retain18'
ORIGINAL_PLAN_SHA = '16a9dacee118f30b966489e0e17ad90274cfd56828ceaffb0c612b6fca9144ed'
RECIPE_SHA = 'bd429b65e92507502e1d95233fe7db5d26bd31909182e2b325c2b9286867ba37'
DATASETS = frozenset(('Electricity', 'PEMS03', 'PEMS04', 'PEMS08', 'Solar', 'AEMO5'))
MODELS = ('RootPurgeLinear',)
from root18_retained_contract import (RETENTION_SHA, REFERENCES_SHA, metadata_bytes, validate_retained_metadata)
HORIZONS = (192, 336, 720)
RUN_IDS = frozenset(f'{model}_{ds}_L96_H{h}_seed2026' for model in MODELS for ds in DATASETS for h in HORIZONS)


def require(value, message):
    if not value:
        raise ValueError(message)


def safe_path(root, name):
    relative = PurePosixPath(name)
    require(bool(relative.parts) and relative.as_posix() == name and not relative.is_absolute()
            and '..' not in relative.parts and '\\' not in name and ':' not in name, 'unsafe root18 path')
    path = Path(root) / name
    require(all(not entry.is_symlink() for entry in (path, *path.parents)), 'root18 leaf/ancestor symlink forbidden')
    return path


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def metadata(path):
    require(path.name not in ('VALIDATION_RESULT.json', 'history.json', 'TEST_RESULT.json') and path.suffix == '.json', 'root18 score decode forbidden')
    return metadata_bytes(path.read_bytes())

def config_sha(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_plan(plan, recipe):
    require(plan['status'] == 'ROOT_FROZEN_ROOT18_VALIDATION_ONLY' and plan['child_root'] == CHILD
            and plan['original_root'] == ORIGINAL and plan['original_plan_sha256'] == ORIGINAL_PLAN_SHA
            and plan['recipe_map_sha256'] == RECIPE_SHA and plan['length'] == 96
            and plan['horizons'] == list(HORIZONS) and plan['seed'] == 2026
            and plan['test_allowed'] is False and plan['all_windows'] is True
            and plan['no_test_scores_for_selection'] is True and set(plan['runs']) == RUN_IDS, 'root18 plan/protocol not frozen')
    require(plan['resource_policy'] == {'max_workers': 8, 'gpu_budget_mib': 22500,
            'active_disk_cap_bytes': 4 << 30, 'actual_disk_floor_bytes': 1536 << 20,
            'unknown_gpu_process_blocks_admission': True}
            and plan['automatic_retry'] is False and plan['h96_retraining_allowed'] is False
            and plan['weights_copied'] is False, 'root18 resource/one-attempt/H96 policy differs')
    require(plan['root_retained_model_only'] is True and plan['two_model_pair_selected'] is False
            and plan['retention_record_sha256'] == RETENTION_SHA and plan['h96_references_sha256'] == REFERENCES_SHA, 'Root18 conditional retained-model-only authority differs')
    require(recipe['run_count'] == 18 and recipe['source_parent_plan_sha256'] == ORIGINAL_PLAN_SHA,
            'root18 exact prospective author recipe identity differs')
    recipes = {row['run_id']: row for row in recipe['runs']}
    require(set(recipes) == RUN_IDS and set(plan['data_metadata']) == DATASETS, 'root18 exact membership differs')
    require({'root18_retained_contract.py', 'RootPurge_classes_exact.py', 'root18_trainval_runtime_pending.py',
             'root18_admission_guard.py', 'root18_production_common.py', 'root18_resources.py',
             'root18_controller.py', 'synthetic_cuda18.py', 'independent_prestart_root18.py',
             'start_root18.py', 'FUTURE_ROOT18_SOURCE_RECIPE_MAP_PENDING.json'} <= set(plan['files'])
            and plan['files']['FUTURE_ROOT18_SOURCE_RECIPE_MAP_PENDING.json'] == RECIPE_SHA
            and plan['files']['RootPurge_classes_exact.py'] == recipe['source_pin_inventory']['source_qa_package/RootPurge_classes_exact.py'],
            'root18 mandatory exact author/runtime source inventory missing')
    for name in plan['files']:
        path = PurePosixPath(name)
        require(path.name not in ('VALIDATION_RESULT.json', 'history.json') and path.suffix != '.pt', 'root18 source inventory may not contain score/weights')
        if path.suffix in ('.npz', '.npy'):
            require(name in {'data/' + ds + suffix for ds in DATASETS for suffix in ('_trainval.npz', '_known_calendar.npy')},
                    'root18 only TRAIN+VAL prefix/known calendar arrays permitted')
    for rid, config in plan['runs'].items():
        scientific = {key: value for key, value in config.items() if key not in ('reserve_gpu_mib', 'reserve_disk_bytes')}
        require(scientific == recipes[rid]['config_draft'], 'root18 unapproved source recipe alteration')
        require(type(config['reserve_gpu_mib']) is int and 0 < config['reserve_gpu_mib'] <= 22500
                and type(config['reserve_disk_bytes']) is int and config['reserve_disk_bytes'] >= 64 << 20, 'root18 actual resource reserves missing')
        require(config['model'] == 'RootPurgeLinear' and config['seq_len'] == 96 and config['pred_len'] in HORIZONS
                and config['seed'] == 2026 and config['precision'] == 'FP32_TF32' and config['batch_size'] == 64
                and config['lr'] == .0005 and config['train_steps'] == 5000 and config['regu_coef'] == .125
                and config['purge_order'] == 1 and config['lr_schedule'] == 'type3', 'Root18 exact author model/loss/5000step/B64/type3 budget')
        meta = plan['data_metadata'][config['dataset']]
        ds = config['dataset']
        require(plan['files']['data/' + ds + '_trainval.npz'] == meta['prefix_sha256']
                and plan['files']['data/' + ds + '_known_calendar.npy'] == meta['calendar_sha256']
                and meta['prefix_rows'] == config['prefix_rows'] and meta['channels'] == config['channels'],
                'root18 mandatory physical prefix/calendar inventory differs')
        require(meta['prefix_sha256'] == config['expected_prefix_sha256'] == recipes[rid]['existing_trainval_prefix_sha256']
                and meta['calendar_sha256'] == config['expected_calendar_sha256'] == recipes[rid]['existing_calendar_sha256']
                and meta['scaler_mean_sha256'] == recipes[rid]['existing_train_scaler_mean_sha256']
                and meta['scaler_std_sha256'] == recipes[rid]['existing_train_scaler_std_sha256'], 'root18 prefix/calendar/scaler identity differs')
        train_end, prefix_end = meta['bounds'][:2]
        require(recipes[rid]['expected_train_windows'] == train_end - config['seq_len'] - config['pred_len'] + 1
                and recipes[rid]['expected_validation_windows'] == prefix_end - train_end - config['pred_len'] + 1,
                'root18 all-window source recipe count differs')


def validate_evidence(plan, recipe, rid, pin, auth, selection, qa, pre, capability, references, closure, root_review, independent_review, freeze, now, probe_claim):
    """Pure metadata validator. It does not read physical artifacts or metrics."""
    validate_plan(plan, recipe)
    require(rid in RUN_IDS and (capability is None or capability['run_id'] == rid), 'H96 training and unknown run forbidden')
    validate_retained_metadata(selection, closure, references)
    require(plan['selection_sha256'] == plan['retention_record_sha256'] == RETENTION_SHA
            and plan['h96_references_sha256'] == REFERENCES_SHA, 'specific retained Root18 scientific source identities')
    require(auth['status'] == 'ROOT_AUTHORIZED_ROOT18_AFTER_SOURCE_RESOURCE_REVIEW_SUBJECT_TO_FRESH_INDEPENDENT_PRESTART'
            and auth['plan_sha256'] == pin and auth['files'] == plan['files'] and auth['run_ids'] == sorted(RUN_IDS)
            and auth['automatic_retry'] is False and auth['new_test_allowed'] is False
            and auth['controller_one_attempt_allowed'] is True,
            'root18 actual ROOT authorization not closed')
    require(qa['status'] == 'PASS_ALL18_ACTUAL_ORIGINAL_BATCH_SYNTHETIC_CUDA_ONLY'
            and qa['recipe_map_sha256'] == RECIPE_SHA and qa['source_inventory'] == plan['files']
            and qa['manifest_sha256'] == plan['source_manifest_sha256']
            and qa['measurement_array_read'] is False and qa['research_scores_read'] is False
            and len(qa['rows']) == 18 and {row['run_id'] for row in qa['rows']} == RUN_IDS, 'actual root18 CUDA resource QA not closed')
    require(qa['probe_claim_sha256'] == plan['probe_claim_sha256']
            and probe_claim['status'] == 'ACTUAL_ONE_ATTEMPT_ROOT18_PROBE_CLAIM'
            and probe_claim['manifest_sha256'] == plan['source_manifest_sha256']
            and probe_claim['automatic_retry'] is False and probe_claim['training_started'] is False
            and probe_claim['utc_epoch'] <= qa['utc_epoch']
            and probe_claim['old_cohort_admission'] == qa['old_cohort_admission'], 'actual one-attempt Root18 CUDA claim/source/chronology identity')
    admission = qa['old_cohort_admission']
    require(admission['status'] == 'ACTUAL_OLD16_COMPLETE_CONTROLLERS_EXITED_SOURCE_SHA_CLOSED'
            and admission['original_plan_sha256'] == ORIGINAL_PLAN_SHA
            and admission['corrected_plan_sha256'] == closure['corrected4_plan_sha256']
            and not admission['old_worker_or_controller_pids'] and not admission['gpu_compute_pids']
            and admission['new_scores_accessed'] is False, 'root18 QA old16/GPU-idle admission differs')
    for row in qa['rows']:
        config = plan['runs'][row['run_id']]
        require(row['status'] == 'PASS' and row['optimizer_steps'] == 2 and row['precision'] == 'FP32_TF32'
                and row['config_sha256'] == config_sha(config) and row['fullprefix_shape'] == [config['prefix_rows'], config['channels']]
                and row['calendar_shape'] == [config['prefix_rows'], plan['data_metadata'][config['dataset']]['calendar_channels']]
                and row['config'] == config and row['synthetic_only'] is True and row['measurement_array_read'] is False
                and row['loss_policy'] == 'EXACT_ROOTPURGE_REGULARIZED_LOSS',
                'actual root18 original recipe/loss/fullprefix QA differs')
        require(all(type(row[key]) is int and row[key] > 0 for key in ('peak_gpu_bytes', 'peak_gpu_reserved_bytes', 'parameter_count'))
                and bool(row['device_name']) and bool(row['torch_version']), 'root18 actual CUDA resource/device measurement missing')
        reserve = max(1200, (max(row['peak_gpu_bytes'], row['peak_gpu_reserved_bytes']) + (1 << 20) - 1) // (1 << 20) + 512)
        require(config['reserve_gpu_mib'] == row['reserve_gpu_mib'] == reserve
                and config['reserve_disk_bytes'] == row['reserve_disk_bytes'] == max(64 << 20, row['parameter_count'] * 12 + (16 << 20)),
                'root18 actual resource reserve formula differs')
    snapshot = {'recipe_map_sha256': RECIPE_SHA, 'files': plan['files'], 'qa_sha256': plan['qa_sha256'], 'probe_claim_sha256': plan['probe_claim_sha256'],
                'selection_sha256': plan['selection_sha256'], 'child_root': CHILD}
    for review, status in ((root_review, 'PASS_ROOT_ROOT18_SOURCE_RESOURCE_REVIEW'),
                           (independent_review, 'PASS_INDEPENDENT_ROOT18_SOURCE_RESOURCE_REVIEW')):
        require(review['status'] == status and review['snapshot'] == snapshot
                and review['new_scores_accessed'] is False and review['test_accessed'] is False,
                'root18 actual source/resource independent review not closed')
    for key in ('qa_sha256', 'probe_claim_sha256', 'selection_sha256', 'root_source_resource_review_sha256', 'independent_source_resource_review_sha256'):
        require(auth[key] == plan[key], 'root18 ROOT authorization scientific evidence SHA differs')
    require(freeze['status'] == 'ACTUAL_ROOT18_PLAN_SOURCE_RESOURCE_FROZEN_NO_START'
            and freeze['plan_sha256'] == pin and freeze['qa_sha256'] == plan['qa_sha256']
            and qa['utc_epoch'] <= freeze['utc_epoch']
            and freeze['training_started'] is False and freeze['scores_accessed'] is False,
            'actual root18 freeze not closed')
    # Static evidence is fully checked before an explicit prestart auditor loads
    # any prefix. This branch neither writes a prestart nor admits a worker.
    if pre is None and capability is None:
        return
    require(pre['status'] == 'PASS_ACTUAL_INDEPENDENT_ROOT18_SOURCE_PREFIX_RESOURCE_PRESTART'
            and pre['plan_sha256'] == pin and pre['files'] == plan['files']
            and pre['qa_sha256'] == plan['qa_sha256'] and pre['selection_sha256'] == plan['selection_sha256']
            and pre['auditor_source_sha256'] == plan['files']['independent_prestart_root18.py']
            and pre['test_scores_accessed'] is False and pre['new_validation_scores_accessed'] is False,
            'root18 independent actual prestart not closed')
    require(type(pre['datasets']) is dict and len(pre['datasets']) == 6 and set(pre['datasets']) == DATASETS
            and pre['prefix_bitwise_elements_checked'] == sum(m['prefix_rows'] * m['channels'] for m in plan['data_metadata'].values())
            and qa['utc_epoch'] <= freeze['utc_epoch'] <= pre['utc_epoch']
            and 0 <= now - pre['utc_epoch'] <= pre['expires_after_seconds'] == 120,
            'root18 actual fresh bitwise prestart/freeze chronology differs')
    for ds, row in pre['datasets'].items():
        meta = plan['data_metadata'][ds]
        require(row == {key: meta[key] for key in ('prefix_sha256', 'calendar_sha256', 'scaler_mean_sha256',
                'scaler_std_sha256', 'prefix_rows', 'channels')} | {'prefix_bitwise_elements': meta['prefix_rows'] * meta['channels']},
                'root18 actual prestart per-dataset prefix/calendar/scaler proof differs')
    require(capability['status'] == 'ROOT_EXPLICIT_ONE_RUN_ROOT18_CAPABILITY_AFTER_FRESH_INDEPENDENT_AUDIT'
            and capability['plan_sha256'] == pin and capability['run_config_sha256'] == config_sha(plan['runs'][rid])
            and capability['automatic_retry'] is False and capability['test_allowed'] is False
            and 0 <= now - capability['utc_epoch'] <= capability['expires_after_seconds'] == 120
            and pre['utc_epoch'] <= capability['utc_epoch'] <= pre['utc_epoch'] + pre['expires_after_seconds'] == pre['utc_epoch'] + 120,
            'root18 fresh one-run capability not closed')


def verify_files(root, plan):
    require(str(root) == CHILD and root.resolve() == root, 'fixed ROOT-confirmed new root18 child required')
    require(Path(__file__).absolute().parent == root, 'active root18 guard must be the frozen child source')
    recipe_path = safe_path(root, 'FUTURE_ROOT18_SOURCE_RECIPE_MAP_PENDING.json')
    require(sha(recipe_path) == RECIPE_SHA, 'root18 exact recipe SHA required')
    validate_plan(plan, metadata(recipe_path))
    require(sha(safe_path(root, 'root18_admission_guard.py')) == plan['files']['root18_admission_guard.py'], 'root18 active guard source SHA differs')
    for name, expected in plan['files'].items():
        require(sha(safe_path(root, name)) == expected, 'root18 frozen source/prefix/calendar SHA differs')


def validate_running_controller_state(state):
    require(state['status'] == 'RUNNING' and type(state['failed']) is dict and not state['failed']
            and state['admission_halt'] is None, 'root18 halted/failed/nonrunning controller blocks kernel admission')


def admit_run(root, rid, pin, authpin, cappin, capname):
    require(str(root) == CHILD and os.name == 'posix' and os.geteuid() == 0, 'future ROOT Linux process required')
    require(re.fullmatch(r'ROOT_ROOT18_RUN_CAPABILITY_[0-9a-f]{32}\.json', capname) is not None, 'unsafe root18 capability name')
    plan_path = safe_path(root, 'ROOT18_PLAN.json'); require(sha(plan_path) == pin, 'exact root18 plan SHA required')
    plan = metadata(plan_path); verify_files(root, plan)
    cap_path = safe_path(root, capname); require(sha(cap_path) == cappin, 'root18 exact one-run capability SHA differs')
    capability = metadata(cap_path)
    pre_name = capability['prestart_name']
    require(re.fullmatch(r'ACTUAL_INDEPENDENT_ROOT18_PRESTART_REVIEW_[0-9a-f]{32}\.json', pre_name) is not None, 'unsafe root18 prestart name')
    paths = {'auth': ('ROOT_ROOT18_START_AUTHORIZATION.json', authpin),
             'selection': ('ROOT18_RETENTION_RECORD.json', plan['selection_sha256']),
             'qa': ('ACTUAL_ROOT18_SYNTHETIC_CUDA_RECEIPT.json', plan['qa_sha256']),
             'probe_claim': ('ROOT18_PROBE_CLAIM.json', plan['probe_claim_sha256']),
             'pre': (pre_name, capability['prestart_sha256']),
             'capability': (capname, cappin),
             'references': ('ROOTPURGE_H96_REFERENCES.json', plan['h96_references_sha256']),
             'closure': ('CANONICAL12_WHOLE_PHYSICAL_CLOSURE.json', plan['canonical_closure_sha256']),
             'root_review': ('ROOT_ROOT18_SOURCE_RESOURCE_REVIEW.json', plan['root_source_resource_review_sha256']),
             'independent_review': ('INDEPENDENT_ROOT18_SOURCE_RESOURCE_REVIEW.json', plan['independent_source_resource_review_sha256']),
             'freeze': ('ACTUAL_ROOT18_FREEZE_RECEIPT.json', capability['freeze_sha256'])}
    documents = {}
    for key, (name, expected) in paths.items():
        path = safe_path(root, name); require(sha(path) == expected, 'root18 actual admission receipt SHA differs'); documents[key] = metadata(path)
    require(documents['pre']['receipt_name'] == pre_name
            and documents['pre']['authorization_sha256'] == documents['capability']['authorization_sha256'] == authpin
            and documents['pre']['freeze_sha256'] == documents['capability']['freeze_sha256']
            and documents['freeze']['authorization_sha256'] == authpin,
            'root18 capability actual admission SHA binding differs')
    validate_evidence(plan, metadata(safe_path(root, 'FUTURE_ROOT18_SOURCE_RECIPE_MAP_PENDING.json')), rid, pin,
                      **documents, now=time.time())
    from root18_production_common import verify_controller_identity, process_identity, validate_canonical
    validate_canonical(documents['selection'], documents['closure'], documents['references'])
    start_path = safe_path(root, 'ROOT18_ACTUAL_START.json')
    require(sha(start_path) == capability['controller_actual_start_sha256'], 'root18 capability controller start SHA differs')
    physical = verify_controller_identity(metadata(start_path), pin, authpin)
    expected = capability['controller_identity']
    require(physical['pid'] == os.getppid() == expected['pid']
            and physical['start_ticks'] == expected['start_ticks'] and physical['argv'] == expected['argv']
            and documents['pre']['controller_identity']['actual_start_sha256'] == capability['controller_actual_start_sha256']
            and documents['pre']['owned'] == capability['owned']
            and documents['pre']['active_configs'] == capability['active_configs']
            and documents['pre']['candidate_run_id'] == rid,
            'root18 worker must be owned by the explicit single controller')
    from root18_resources import collect_telemetry, validate_resource_admission
    # A later queued worker can start between this capability and our guard.
    # Re-read the single controller's current state and include every declared
    # owned worker; the candidate itself has no CUDA context before this guard.
    deadline = time.time() + 5
    while True:
        state = metadata(safe_path(root, 'ROOT18_STATE.json'))
        if rid in state['active'] or time.time() >= deadline:
            break
        time.sleep(0.05)
    require(state['plan_sha256'] == pin and state['authorization_sha256'] == authpin
            and state['actual_start_sha256'] == capability['controller_actual_start_sha256']
            and state['test_accessed'] is False and state['scores_released'] is False,
            'root18 current controller state identity differs')
    validate_running_controller_state(state)
    owned_now = state['active']; self_identity = process_identity(os.getpid())
    own = owned_now[rid]
    require(own['pid'] == os.getpid() and own['start_ticks'] == self_identity['start_ticks']
            and own['argv'] == self_identity['argv'] and own['controller_pid'] == os.getppid(),
            'root18 current owned worker identity differs')
    others = {r: v for r, v in owned_now.items() if r != rid}
    for worker in others.values():
        argv = worker['argv']
        require(worker['controller_pid'] == os.getppid()
                and argv[argv.index('--plan-sha256') + 1] == pin
                and argv[argv.index('--authorization-sha256') + 1] == authpin,
                'root18 other worker belongs to another controller/plan')
    telemetry = collect_telemetry(root, others)
    validate_resource_admission(telemetry, {r: plan['runs'][r] for r in others}, plan['runs'][rid])
    from root18_production_common import verify_references
    verify_references(documents['references'])
    require(not safe_path(root, 'results/' + rid).exists(), 'never automatically retrain root18 run')
    claim = safe_path(root, 'ROOT18_RUN_CLAIM_' + rid + '.json')
    with claim.open('x', encoding='utf-8') as stream:
        json.dump({'run_id': rid, 'plan_sha256': pin, 'authorization_sha256': authpin,
                   'capability_name': capname, 'capability_sha256': cappin, 'utc_epoch': time.time(),
                   'worker_identity': process_identity(os.getpid()),
                   'controller_actual_start_sha256': capability['controller_actual_start_sha256'],
                   'automatic_retry': False, 'test_allowed': False}, stream, indent=2)
    return plan
