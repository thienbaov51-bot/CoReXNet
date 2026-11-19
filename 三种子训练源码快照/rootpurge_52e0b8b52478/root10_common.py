"""Distinct environment-only recovery10 helpers. Import/default perform no I/O."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
import uuid

from collect_root18_recovery10 import (ORIGINAL18, PLAN_SHA as ORIGINAL18_PLAN_SHA,
    MODEL_SHA, KERNEL_SHA, RECIPE_SHA, RUN_IDS, CORE, metadata_bytes, safe_path,
    identity, valid_sha, require, scan_global, load_reviewed_static_guard)
from root18_retained_contract import reference_index

CHILD = '/root/autodl-tmp/paper5/models2026_pilot_20261011/rootpurge_recovery10'
DATASETS = frozenset(('Electricity', 'PEMS03', 'PEMS04', 'PEMS08', 'Solar', 'AEMO5'))
SOURCE_FILES = ('collect_root18_recovery10.py', 'root18_retained_contract.py',
    'RootPurge_classes_exact.py', 'root18_trainval_runtime_pending.py',
    'FUTURE_ROOT18_SOURCE_RECIPE_MAP_PENDING.json', 'root18_admission_guard.py',
    'root18_resources.py', 'root10_common.py', 'root10_controller.py',
    'project_original_cuda18.py', 'independent_prestart_root10.py', 'start_root10.py', 'operate_root10_pending.py')


def sha(path):
    return identity(Path(path))['sha256']


def metadata(path):
    require(Path(path).name not in ('VALIDATION_RESULT.json', 'history.json', 'TEST_RESULT.json')
            and Path(path).suffix == '.json', 'recovery numerical decoding forbidden')
    return metadata_bytes(Path(path).read_bytes())


def config_sha(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def atomic(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def receipt_name(prefix):
    return prefix + uuid.uuid4().hex + '.json'


def local_root():
    root = Path(CHILD)
    require(os.name == 'posix' and os.geteuid() == 0 and Path(__file__).absolute().parent == root
            and root.resolve() == root, 'PENDING: exact separate ROOT recovery10 child required')
    return root


def pinned(root, name, pin):
    require(valid_sha(pin), 'exact recovery metadata SHA required')
    path = safe_path(root, name); require(sha(path) == pin, 'recovery metadata SHA differs: ' + name)
    return metadata(path)


def validate_recovery_evidence(evidence, original_plan):
    require(evidence['status'] == 'ACTUAL_ROOT18_HALTED_EIGHT_COMPLETE_TEN_UNATTEMPTED_OPAQUE_RECOVERY_EVIDENCE'
            and evidence['original_root'] == ORIGINAL18 and evidence['original_plan_sha256'] == ORIGINAL18_PLAN_SHA
            and evidence['original_source_files'] == original_plan['files']
            and evidence['all18_original_recipe_configs'] == original_plan['runs']
            and evidence['source_data_calendar_scaler_identity_metadata'] == original_plan['data_metadata']
            and evidence['numerical_artifacts_hashed_not_decoded'] is True
            and evidence['new_scores_accessed'] is False and evidence['arrays_opened'] is False
            and evidence['framework_imported'] is False and evidence['training_started'] is False
            and evidence['test_accessed'] is False and evidence['automatic_retry'] is False
            and evidence['recovery_training_authorized'] is False and evidence['joint_gpu_authorization_granted'] is False,
            'original recovery evidence identity differs')
    done = evidence['original_complete8']; queue = evidence['untouched10_run_ids']
    require(type(done) is dict and len(done) == 8 and type(queue) is list and len(queue) == len(set(queue)) == 10
            and set(done).isdisjoint(queue) and set(done) | set(queue) == RUN_IDS, 'original eight/ten recovery partition differs')
    for rid, row in done.items():
        require(row['source_root'] == ORIGINAL18 and row['plan_sha256'] == ORIGINAL18_PLAN_SHA
                and row['config'] == original_plan['runs'][rid] and row['numerical_values_decoded'] is False
                and set(row['files']) == {*CORE, 'COMPLETE_RECEIPT.json'}, 'original completed8 physical record differs')
    require(evidence['gpu_count'] == 1 and type(evidence['gpu_count']) is int
            and evidence['gpu_compute_pids'] == [] and evidence['global_old_runtime_controller_pids'] == [],
            'original evidence idle GPU/global census differs')
    references = reference_index(evidence['retained_root6_references'])
    require(len(references) == 6, 'native Root6 retained references differ')
    return set(queue), done, references


def verify_original(evidence):
    """Preserve parent failure/source/8 complete files; no old numerical value reads."""
    root = Path(ORIGINAL18)
    plan = pinned(root, 'ROOT18_PLAN.json', ORIGINAL18_PLAN_SHA)
    queue, done, references = validate_recovery_evidence(evidence, plan)
    for name, expected in evidence['original_failure_evidence_inventory'].items():
        require(identity(safe_path(root, name)) == expected, 'original failed cohort bytes changed: ' + name)
    for name, field in (('ROOT18_ACTUAL_START.json', 'original_actual_start_sha256'),
                       ('ROOT18_STATE.json', 'original_state_sha256'),
                       ('ROOT18_TRAINING_BOUNDARY.json', 'original_boundary_sha256')):
        require(sha(safe_path(root, name)) == evidence[field], 'original failure metadata changed')
    for rid in queue:
        require(not safe_path(root, 'results/' + rid).exists() and not safe_path(root, 'logs/' + rid + '.log').exists()
                and not safe_path(root, 'ROOT18_RUN_CLAIM_' + rid + '.json').exists(), 'original pending10 acquired artifacts')
    for path in root.glob('ROOT_ROOT18_RUN_CAPABILITY_*.json'):
        require(metadata(path)['run_id'] in done, 'original pending10 acquired capability')
    for reference in references.values():
        folder = Path(reference['source_root']) / 'results' / reference['run_id']
        for name, expected in reference['physical']['files'].items():
            require(identity(safe_path(folder, name)) == expected, 'retained Root6 original physical identity differs')
    return plan, queue, done, references


def validate_plan(plan, manifest, evidence, original_plan):
    queue, done, _ = validate_recovery_evidence(evidence, original_plan)
    require(plan['status'] == 'ROOT_FROZEN_ROOT10_ENVIRONMENT_ONLY_RECOVERY_VALIDATION_ONLY'
            and plan['child_root'] == CHILD and plan['original18_root'] == ORIGINAL18
            and plan['original18_plan_sha256'] == ORIGINAL18_PLAN_SHA and plan['runs'] == original_plan['runs']
            and plan['authorized_run_ids'] == sorted(queue) and plan['preserved_complete8'] == done
            and plan['data_metadata'] == original_plan['data_metadata'] and plan['files'] == manifest['files']
            and plan['test_allowed'] is False and plan['automatic_retry'] is False
            and plan['weights_copied'] is False and plan['original_completed8_retraining_allowed'] is False
            and plan['h96_retraining_allowed'] is False and plan['scientific_runtime_sha256'] == KERNEL_SHA
            and plan['scientific_model_sha256'] == MODEL_SHA and plan['all18_recipe_sha256'] == RECIPE_SHA,
            'recovery plan must retain original full18 recipes and authorize only untouched10')
    require(plan['scientific_math_changed'] is False and plan['two_model_pair_selected'] is False
            and plan['resource_policy'] == {'max_workers':8,'gpu_budget_mib':22500,'active_disk_cap_bytes':4<<30,
                'actual_disk_floor_bytes':1536<<20,'gpu_count':1,'unknown_gpu_process_blocks_admission':True},
            'new recovery resource policy/scientific scope differs')
    require(plan['files']['RootPurge_classes_exact.py'] == MODEL_SHA
            and plan['files']['root18_trainval_runtime_pending.py'] == KERNEL_SHA
            and plan['files']['FUTURE_ROOT18_SOURCE_RECIPE_MAP_PENDING.json'] == RECIPE_SHA,
            'immutable original science bytes differ')
    require(plan['files']['collect_root18_recovery10.py'] == evidence['collector_source_sha256'],
            'reviewed original evidence collector source pin differs')
    for name, pin in original_plan['files'].items():
        if name.startswith('data/'):
            require(plan['files'][name] == pin, 'original prefix/calendar byte identity differs')


def verify_files(root, plan):
    require(str(root) == CHILD and root.resolve() == root, 'exact distinct recovery child required')
    for name, pin in plan['files'].items():
        require(sha(safe_path(root, name)) == pin, 'recovery frozen source/prefix SHA differs: ' + name)


def verify_manifest(pin):
    root = local_root(); manifest = pinned(root, 'ROOT10_SOURCE_MANIFEST.json', pin)
    require(manifest['status'] == 'REVIEWED_DISTINCT_ROOT10_ENVIRONMENT_RECOVERY_SOURCE_DEPLOYMENT_ONLY'
            and manifest['child_root'] == CHILD and manifest['original18_root'] == ORIGINAL18
            and manifest['original18_plan_sha256'] == ORIGINAL18_PLAN_SHA
            and set(SOURCE_FILES) <= set(manifest['files']) and manifest['training_started'] is False
            and manifest['test_allowed'] is False, 'reviewed recovery source manifest required')
    for name, sourcepin in manifest['files'].items():
        require(sha(safe_path(root, name)) == sourcepin, 'new recovery source/prefix identity differs')
    require(manifest['files']['RootPurge_classes_exact.py'] == MODEL_SHA
            and manifest['files']['root18_trainval_runtime_pending.py'] == KERNEL_SHA
            and manifest['files']['FUTURE_ROOT18_SOURCE_RECIPE_MAP_PENDING.json'] == RECIPE_SHA,
            'new child scientific bytes must be exact original')
    evidence = pinned(root, 'ROOT18_RECOVERY10_EVIDENCE.json', manifest['recovery_evidence_sha256'])
    original_plan, queue, done, references = verify_original(evidence)
    require(set(manifest['files']) == set(SOURCE_FILES) | {name for name in original_plan['files'] if name.startswith('data/')},
            'exact new runtime source and original prefix/calendar member set required')
    require(manifest['authorized_run_ids'] == sorted(queue) and manifest['data_metadata'] == original_plan['data_metadata'], 'deployment recovery partition differs')
    snapshot = {'files': manifest['files'], 'recovery_evidence_sha256': manifest['recovery_evidence_sha256'],
        'original18_plan_sha256': ORIGINAL18_PLAN_SHA, 'child_root': CHILD, 'authorized_run_ids': sorted(queue)}
    for name, key, status in (('ROOT_ROOT10_LOCAL_SOURCE_REVIEW.json', 'root_source_review_sha256', 'PASS_ROOT_ROOT10_LOCAL_SOURCE_REVIEW'),
                           ('INDEPENDENT_ROOT10_LOCAL_SOURCE_REVIEW.json', 'independent_source_review_sha256', 'PASS_INDEPENDENT_ROOT10_LOCAL_SOURCE_REVIEW')):
        proof = pinned(root, name, manifest[key])
        require(proof['status'] == status and proof['snapshot'] == snapshot
                and proof['actual_scores_read'] is False and proof['actual_arrays_read'] is False
                and proof['scientific_math_changed'] is False, 'independent new wrapper source review required')
    return root, manifest, evidence, original_plan


def validate_static(plan, pin, auth, qa, claim, root_review, independent_review, freeze, manifest, evidence, original_plan, original_qa):
    validate_plan(plan, manifest, evidence, original_plan)
    ids = plan['authorized_run_ids']
    require(auth['status'] == 'ROOT_AUTHORIZED_DISTINCT_UNATTEMPTED_ROOT10_AFTER_NEW_SOURCE_RESOURCE_REVIEWS'
            and auth['plan_sha256'] == pin and auth['files'] == plan['files'] and auth['run_ids'] == ids
            and auth['recovery_evidence_sha256'] == plan['recovery_evidence_sha256']
            and auth['controller_one_attempt_allowed'] is True and auth['automatic_retry'] is False
            and auth['new_test_allowed'] is False and auth['original_completed8_retraining_allowed'] is False
            and auth['joint_gpu_authorization_allowed'] is False, 'new ROOT recovery10 authorization required')
    require(qa['status'] == 'ACTUAL_CUDA18_MEASURED_RESOURCE_PROJECTION_TO_UNTOUCHED10_NO_NEW_KERNEL'
            and qa['manifest_sha256'] == plan['source_manifest_sha256'] and qa['source_inventory'] == plan['files']
            and qa['recovery_evidence_sha256'] == plan['recovery_evidence_sha256']
            and {row['run_id'] for row in qa['rows']} == set(ids) and len(qa['rows']) == 10
            and qa['measurement_array_read'] is False and qa['research_scores_read'] is False
            and qa['automatic_retry'] is False and qa['training_started'] is False, 'fresh recovery actual synthetic10 QA required')
    require(qa['original_cuda18_sha256'] == original_plan['qa_sha256']
            and qa['original_scientific_runtime_sha256'] == KERNEL_SHA and qa['original_scientific_model_sha256'] == MODEL_SHA
            and qa['new_kernel_probe_performed'] is False and qa['scientific_math_changed'] is False
            and qa['original_measured_rows_reused'] is True, 'original actual CUDA18 measurement projection required')
    measured_rows={row['run_id']:row for row in original_qa['rows']}
    require(original_qa['status'] == 'PASS_ALL18_ACTUAL_ORIGINAL_BATCH_SYNTHETIC_CUDA_ONLY'
            and original_qa['source_inventory'] == original_plan['files']
            and len(measured_rows) == len(original_qa['rows']) == 18
            and qa['rows'] == [measured_rows[rid] for rid in ids],
            'projection must reuse original actual measured CUDA18 rows exactly')
    require(claim['status'] == 'ROOT_EXPLICIT_METADATA_ONLY_ORIGINAL_CUDA18_PROJECTION_CLAIM'
            and claim['manifest_sha256'] == plan['source_manifest_sha256'] and claim['automatic_retry'] is False
            and claim['training_started'] is False and claim['new_kernel_probe_performed'] is False
            and claim['utc_epoch'] <= qa['utc_epoch']
            and qa['probe_claim_sha256'] == plan['probe_claim_sha256'], 'actual exclusive recovery probe claim differs')
    from root18_resources import validate_resource_admission
    validate_resource_admission(qa['resource_telemetry'],{},None)
    validate_resource_admission(claim['resource_telemetry'],{},None)
    require(qa['resource_telemetry']['gpu_compute_pids'] == claim['resource_telemetry']['gpu_compute_pids'] == [],
            'actual projection must prove current idle GPU; no joint admission')
    for row in qa['rows']:
        cfg = plan['runs'][row['run_id']]
        measured_gpu = max(1200, (max(row['peak_gpu_bytes'], row['peak_gpu_reserved_bytes']) + (1 << 20) - 1) // (1 << 20) + 512)
        measured_disk = max(64 << 20, row['parameter_count'] * 12 + (16 << 20))
        require(row['status'] == 'PASS' and row['optimizer_steps'] == 2 and row['config'] == cfg
                and row['config_sha256'] == config_sha(cfg) and row['fullprefix_shape'] == [cfg['prefix_rows'], cfg['channels']]
                and row['calendar_shape'] == [cfg['prefix_rows'], plan['data_metadata'][cfg['dataset']]['calendar_channels']]
                and row['precision'] == 'FP32_TF32' and row['loss_policy'] == 'EXACT_ROOTPURGE_REGULARIZED_LOSS'
                and row['synthetic_only'] is True and row['measurement_array_read'] is False
                and all(type(row[key]) is int and row[key] > 0 for key in ('peak_gpu_bytes','peak_gpu_reserved_bytes','parameter_count'))
                and cfg['reserve_gpu_mib'] >= measured_gpu and cfg['reserve_disk_bytes'] >= measured_disk
                and row['reserve_gpu_mib'] == cfg['reserve_gpu_mib'] and row['reserve_disk_bytes'] == cfg['reserve_disk_bytes'],
                'fresh actual fullresident/batch64/reserve evidence differs')
    snapshot = {'files': plan['files'], 'source_manifest_sha256': plan['source_manifest_sha256'],
        'qa_sha256': plan['qa_sha256'], 'probe_claim_sha256': plan['probe_claim_sha256'],
        'recovery_evidence_sha256': plan['recovery_evidence_sha256'], 'authorized_run_ids': ids, 'child_root': CHILD}
    for review, status in ((root_review, 'PASS_ROOT_ROOT10_SOURCE_RESOURCE_REVIEW'),
                           (independent_review, 'PASS_INDEPENDENT_ROOT10_SOURCE_RESOURCE_REVIEW')):
        require(review['status'] == status and review['snapshot'] == snapshot and review['new_scores_accessed'] is False
                and review['test_accessed'] is False and review['scientific_math_changed'] is False, 'new independent recovery source/resource review required')
    for key in ('source_manifest_sha256','recovery_evidence_sha256','qa_sha256','probe_claim_sha256',
                'root_source_resource_review_sha256','independent_source_resource_review_sha256'):
        require(auth[key] == plan[key], 'ROOT recovery authorization proof pins differ')
    require(freeze['status'] == 'ACTUAL_ROOT10_SOURCE_RESOURCE_PLAN_FROZEN_NO_START'
            and freeze['plan_sha256'] == pin
            and freeze['qa_sha256'] == plan['qa_sha256'] and qa['utc_epoch'] <= freeze['utc_epoch']
            and freeze['training_started'] is False and freeze['scores_accessed'] is False, 'new recovery actual freeze required')


def load_frozen(pin, authpin):
    root = local_root(); plan = pinned(root, 'ROOT10_PLAN.json', pin); verify_files(root, plan)
    _, manifest, evidence, original_plan = verify_manifest(plan['source_manifest_sha256'])
    docs = {'auth': pinned(root, 'ROOT_ROOT10_START_AUTHORIZATION.json', authpin),
        'qa': pinned(root, 'ACTUAL_ROOT10_ORIGINAL_CUDA18_RESOURCE_PROJECTION.json', plan['qa_sha256']),
        'claim': pinned(root, 'ROOT10_PROBE_CLAIM.json', plan['probe_claim_sha256']),
        'root_review': pinned(root, 'ROOT_ROOT10_SOURCE_RESOURCE_REVIEW.json', plan['root_source_resource_review_sha256']),
        'independent_review': pinned(root, 'INDEPENDENT_ROOT10_SOURCE_RESOURCE_REVIEW.json', plan['independent_source_resource_review_sha256']),
        'freeze': metadata(safe_path(root, 'ACTUAL_ROOT10_FREEZE_RECEIPT.json'))}
    require(plan['recovery_evidence_sha256'] == manifest['recovery_evidence_sha256'], 'recovery evidence plan/source binding differs')
    require(docs['freeze']['authorization_sha256'] == authpin, 'recovery actual freeze authorization SHA differs')
    original_qa = pinned(Path(ORIGINAL18), 'ACTUAL_ROOT18_SYNTHETIC_CUDA_RECEIPT.json', original_plan['qa_sha256'])
    original_rows = {row['run_id']:row for row in original_qa['rows']}
    require(docs['qa']['rows'] == [original_rows[rid] for rid in plan['authorized_run_ids']],
            'projection must retain original actual CUDA18 rows exactly')
    validate_static(plan, pin, manifest=manifest, evidence=evidence, original_plan=original_plan, original_qa=original_qa, **docs)
    return root, plan, manifest, evidence, original_plan, docs, sha(root / 'ACTUAL_ROOT10_FREEZE_RECEIPT.json')


def process_identity(pid):
    from root18_resources import _process_identity, _nvidia_query
    gpu = sorted(set(_nvidia_query('pid', 'compute-apps')))
    physical = _process_identity(pid, gpu)
    require(sorted(set(_nvidia_query('pid', 'compute-apps'))) == gpu, 'GPU membership changed during strict process sampling')
    if not physical['alive']:
        raise ProcessLookupError('owned recovery process exited or is a non-GPU zombie')
    return {'pid': pid, **{key: physical[key] for key in ('uid','start_ticks','parent_pid','argv')}}


def no_other_producers():
    forbidden = {ORIGINAL18 + '/' + name for name in ('root18_controller.py','root18_trainval_runtime_pending.py')}
    parent = Path(ORIGINAL18).parent
    for child, names in (('',('minimum12_controller.py','trainval_runtime.py')),
            ('fact_freq_corrected4',('corrected4_controller.py','trainval_runtime.py')),
            ('vpnet_minimum6',('vpnet6_controller.py','vpnet6_runtime.py')),
            ('vpnet18_after_selection',('vpnet18_controller.py','vpnet18_runtime.py'))):
        forbidden |= {str(parent / child / name) for name in names}
    scan_global(Path('/proc'), forbidden)


def verify_controller_identity(start, pin, authpin):
    require(start['status'] == 'ACTUAL_ROOT10_ONE_CONTROLLER_STARTED_NO_TEST'
            and start['plan_sha256'] == pin and start['authorization_sha256'] == authpin
            and start['automatic_retry'] is False and start['test_allowed'] is False,
            'recovery actual controller attempt identity differs')
    physical = process_identity(start['pid'])
    require(physical['uid'] == 0 and physical['start_ticks'] == start['start_ticks']
            and physical['argv'] == start['argv'] and CHILD + '/root10_controller.py' in physical['argv'],
            'owned recovery controller PID/argv/starttick differs')
    return physical


def validate_prestart(pre, plan, pin, authpin, freezepin, now, run_id=None, documents=None):
    require(type(documents) is dict and pre['freeze_utc_epoch'] == documents['freeze']['utc_epoch']
            and pre['qa_projection_utc_epoch'] == documents['qa']['utc_epoch']
            and documents['qa']['utc_epoch'] <= documents['freeze']['utc_epoch'] <= pre['utc_epoch'],
            'actual projection/freeze/fresh prestart chronology differs')
    require(pre['status'] == 'PASS_ACTUAL_INDEPENDENT_ROOT10_SOURCE_PREFIX_RESOURCE_PRESTART'
            and pre['plan_sha256'] == pin and pre['authorization_sha256'] == authpin
            and pre['freeze_sha256'] == freezepin and pre['files'] == plan['files']
            and pre['recovery_evidence_sha256'] == plan['recovery_evidence_sha256']
            and pre['qa_sha256'] == plan['qa_sha256'] and pre['candidate_run_id'] == run_id
            and pre['expires_after_seconds'] == 120 and 0 <= now - pre['utc_epoch'] <= 120
            and pre['test_scores_accessed'] is False and pre['new_validation_scores_accessed'] is False
            and pre['automatic_retry'] is False and pre['training_started_by_auditor'] is False
            and pre['auditor_source_sha256'] == plan['files']['independent_prestart_root10.py'], 'fresh independent recovery prestart differs')
    require(set(pre['datasets']) == DATASETS and pre['prefix_bitwise_elements_checked']
            == sum(meta['prefix_rows'] * meta['channels'] for meta in plan['data_metadata'].values()), 'full prefix prestart coverage differs')
    for dataset, row in pre['datasets'].items():
        meta = plan['data_metadata'][dataset]
        require(row == {key: meta[key] for key in ('prefix_sha256','calendar_sha256','scaler_mean_sha256',
            'scaler_std_sha256','prefix_rows','channels')} | {'prefix_bitwise_elements':meta['prefix_rows'] * meta['channels']}, 'prestart scaler/data/calendar identity differs')
    from root18_resources import validate_resource_admission
    validate_resource_admission(pre['resource_telemetry'], pre['active_configs'],
        None if run_id is None else plan['runs'][run_id])
    if run_id is None:
        require(not pre['owned'] and not pre['active_configs'] and pre['controller_identity'] is None, 'initial recovery start must be GPU idle')


def pending():
    return {'status':'PENDING_DISTINCT_ROOT10_EVIDENCE_SOURCE_RESOURCE_ROOT_AUTH_FREEZE_FRESH_PRESTART_REQUIRED',
        'training_started':False,'test_accessed':False,'new_scores_accessed':False,'remote_action_performed':False}
