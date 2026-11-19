"""Score-free future root18 helpers. Importing this module performs no I/O."""
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from root18_admission_guard import (CHILD, ORIGINAL, CORRECTED, ORIGINAL_PLAN_SHA,
    RECIPE_SHA, RUN_IDS, MODELS, DATASETS, require, safe_path, sha, metadata,
    config_sha, validate_plan, validate_evidence, verify_files)

SOURCE_FILES = ('root18_retained_contract.py', 'RootPurge_classes_exact.py',
    'root18_trainval_runtime_pending.py', 'root18_admission_guard.py',
    'root18_production_common.py', 'root18_resources.py', 'root18_controller.py',
    'synthetic_cuda18.py', 'independent_prestart_root18.py', 'start_root18.py',
    'FUTURE_ROOT18_SOURCE_RECIPE_MAP_PENDING.json')
from root18_retained_contract import RETENTION_SHA, REFERENCES_SHA, validate_retained_metadata, reference_index

DOCUMENTS = {'selection': ('ROOT18_RETENTION_RECORD.json', 'selection_sha256'),
    'qa': ('ACTUAL_ROOT18_SYNTHETIC_CUDA_RECEIPT.json', 'qa_sha256'),
    'probe_claim': ('ROOT18_PROBE_CLAIM.json', 'probe_claim_sha256'),
    'references': ('ROOTPURGE_H96_REFERENCES.json', 'h96_references_sha256'),
    'closure': ('CANONICAL12_WHOLE_PHYSICAL_CLOSURE.json', 'canonical_closure_sha256'),
    'root_review': ('ROOT_ROOT18_SOURCE_RESOURCE_REVIEW.json', 'root_source_resource_review_sha256'),
    'independent_review': ('INDEPENDENT_ROOT18_SOURCE_RESOURCE_REVIEW.json', 'independent_source_resource_review_sha256')}


def exclusive(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def local_root():
    root = Path(CHILD)
    require(os.name == 'posix' and os.geteuid() == 0 and Path(__file__).absolute().parent == root,
            'PENDING: explicit future ROOT Linux child required')
    require(root.resolve() == root, 'future child symlink forbidden')
    return root


def pinned(root, name, expected):
    path = safe_path(root, name)
    require(sha(path) == expected, 'root18 physical metadata SHA differs: ' + name)
    return metadata(path)


def validate_canonical(selection, closure, references):
    validate_retained_metadata(selection, closure, references)

def verify_references(references):
    for rid, reference in reference_index(references).items():
        folder = Path(reference['source_root']) / 'results' / rid
        for filename, identity in reference['physical']['files'].items():
            path = folder / filename
            require(all(not p.is_symlink() for p in (path, *path.parents)) and sha(path) == identity['sha256']
                    and path.stat().st_size == identity['bytes'], 'retained Root6 original all5 physical byte SHA differs')
        receipt = metadata(folder / 'COMPLETE_RECEIPT.json')
        require(receipt['status'] == 'COMPLETE_VALIDATION_ONLY' and receipt['run_id'] == rid
                and receipt['plan_sha256'] == ORIGINAL_PLAN_SHA and receipt['test_accessed'] is False,
                'retained Root H96 exact completion source identity')
        for key, filename in (('checkpoint_sha256', 'best.pt'), ('config_sha256', 'config.json'),
                              ('history_sha256', 'history.json'), ('validation_result_sha256', 'VALIDATION_RESULT.json')):
            require(receipt[key] == reference['physical']['files'][filename]['sha256'], 'retained all5 completion links differ')


def verify_manifest(pin):
    root = local_root(); manifest = pinned(root, 'ROOT18_SOURCE_MANIFEST.json', pin)
    require(manifest['status'] == 'REVIEWED_FUTURE_ROOT18_SOURCE_DEPLOYMENT_ONLY_NO_SCIENTIFIC_FREEZE'
            and manifest['child_root'] == CHILD and manifest['original_plan_sha256'] == ORIGINAL_PLAN_SHA
            and manifest['recipe_map_sha256'] == RECIPE_SHA and manifest['training_started'] is False
            and manifest['test_allowed'] is False and set(SOURCE_FILES) <= set(manifest['files']), 'future source manifest differs')
    for name, expected in manifest['files'].items():
        require(sha(safe_path(root, name)) == expected, 'root18 deployed source/prefix SHA differs: ' + name)
    recipe = metadata(safe_path(root, 'FUTURE_ROOT18_SOURCE_RECIPE_MAP_PENDING.json'))
    draft = {row['run_id']: {**row['config_draft'], 'reserve_gpu_mib': 1200, 'reserve_disk_bytes': 64 << 20} for row in recipe['runs']}
    plan = {'status': 'ROOT_FROZEN_ROOT18_VALIDATION_ONLY', 'child_root': CHILD, 'original_root': ORIGINAL,
        'original_plan_sha256': ORIGINAL_PLAN_SHA, 'recipe_map_sha256': RECIPE_SHA, 'length': 96,
        'horizons': [192, 336, 720], 'seed': 2026, 'test_allowed': False, 'all_windows': True,
        'no_test_scores_for_selection': True, 'runs': draft, 'data_metadata': manifest['data_metadata'], 'files': manifest['files'],
        'resource_policy': {'max_workers': 8, 'gpu_budget_mib': 22500, 'active_disk_cap_bytes': 4 << 30,
            'actual_disk_floor_bytes': 1536 << 20, 'unknown_gpu_process_blocks_admission': True},
        'automatic_retry': False, 'h96_retraining_allowed': False, 'weights_copied': False,
        'root_retained_model_only': True, 'two_model_pair_selected': False,
        'retention_record_sha256': RETENTION_SHA, 'h96_references_sha256': REFERENCES_SHA}
    validate_plan(plan, recipe)
    require(manifest['selection_sha256'] == manifest['retention_record_sha256'] == RETENTION_SHA
            and manifest['h96_references_sha256'] == REFERENCES_SHA, 'exact ROOT conditional retention source pins')
    documents = {key: pinned(root, name, manifest[field]) for key, name, field in (
        ('selection', 'ROOT18_RETENTION_RECORD.json', 'selection_sha256'),
        ('closure', 'CANONICAL12_WHOLE_PHYSICAL_CLOSURE.json', 'canonical_closure_sha256'),
        ('references', 'ROOTPURGE_H96_REFERENCES.json', 'h96_references_sha256'))}
    require(manifest['canonical_closure_sha256'] == documents['selection']['physical_closure_sha256']
            == documents['references']['physical_closure_sha256'], 'exact ROOT audited raw physical closure SHA binding')
    validate_canonical(**documents); verify_references(documents['references'])
    return root, manifest, recipe, documents


def process_identity(pid):
    # Reuse the reviewed metadata sampler: double stat read rejects PID reuse
    # during sampling and all four UID fields must remain ROOT.
    from root18_resources import _process_identity
    physical = _process_identity(pid)
    if not physical['alive']:
        raise ProcessLookupError('owned root18 process has exited')
    return {'pid': pid, **{key: physical[key] for key in ('start_ticks', 'parent_pid', 'uid', 'argv')}}


def scan_old_processes(proc_root, forbidden):
    """Global argv census; empty kernel/zombie argv is legitimate here only."""
    for proc in proc_root.iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            raw = (proc / 'cmdline').read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        # This census detects old executable paths, not ownership. Owned/GPU
        # processes still require strict UID, live argv and double-stat identity.
        argv = {part for part in raw.split(b'\0') if part}
        require(not {name.encode('utf-8') for name in forbidden}.intersection(argv), 'old controller/worker still exists')


def gate_old_cohorts():
    """All old train controllers/workers must have exited; score bytes remain opaque."""
    parent = Path(ORIGINAL); child = Path(CORRECTED)
    original = pinned(parent, 'MINIMUM12_PLAN.json', ORIGINAL_PLAN_SHA)
    boundary = metadata(safe_path(parent, 'MINIMUM12_TRAINING_BOUNDARY.json'))
    require(boundary['status'] == 'COMPLETE_VALIDATION_TRAINING'
            and boundary['plan_sha256'] == ORIGINAL_PLAN_SHA and set(boundary['done']) == set(original['runs'])
            and not boundary['failed'] and boundary['test_accessed'] is False, 'original12 not complete')
    corrected = metadata(safe_path(child, 'CORRECTED4_PLAN.json'))
    corrected_sha = sha(safe_path(child, 'CORRECTED4_PLAN.json'))
    boundary4 = metadata(safe_path(child, 'CORRECTED4_TRAINING_BOUNDARY.json'))
    require(boundary4['status'] == 'COMPLETE_CORRECTED4_VALIDATION_TRAINING'
            and boundary4['plan_sha256'] == corrected_sha and set(boundary4['done']) == set(corrected['runs'])
            and len(boundary4['done']) == 4 and not boundary4['failed'] and boundary4['test_accessed'] is False,
            'corrected4 not complete')
    for cohort, plan, state_name, start_name in ((parent, original, 'MINIMUM12_STATE.json', 'MINIMUM12_ACTUAL_START.json'),
            (child, corrected, 'CORRECTED4_STATE.json', 'CORRECTED4_ACTUAL_START.json')):
        state = metadata(safe_path(cohort, state_name)); start = metadata(safe_path(cohort, start_name))
        require(not state['active'] and not state['queue'] and not state['failed'] and set(state['done']) == set(plan['runs']), 'old cohort not physically idle')
        require(not (Path('/proc') / str(start['pid'])).exists(), 'old controller PID not exited')
        for name, expected in plan['files'].items():
            require(sha(safe_path(cohort, name)) == expected, 'old frozen source/data SHA differs')
    forbidden = {ORIGINAL + '/minimum12_controller.py', ORIGINAL + '/trainval_runtime.py',
                 CORRECTED + '/corrected4_controller.py', CORRECTED + '/trainval_runtime.py'}
    scan_old_processes(Path('/proc'), forbidden)
    return {'status': 'ACTUAL_OLD16_COMPLETE_CONTROLLERS_EXITED_SOURCE_SHA_CLOSED',
        'original_plan_sha256': ORIGINAL_PLAN_SHA, 'corrected_plan_sha256': corrected_sha,
        'original_boundary_sha256': sha(safe_path(parent, 'MINIMUM12_TRAINING_BOUNDARY.json')),
        'corrected_boundary_sha256': sha(safe_path(child, 'CORRECTED4_TRAINING_BOUNDARY.json')),
        'old_worker_or_controller_pids': [], 'new_scores_accessed': False}


def gpu_idle_except(allowed_pid=None):
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    pids = sorted(set(int(v.strip()) for v in pids.splitlines() if v.strip()))
    require(not pids or pids == [allowed_pid], 'PENDING: unowned GPU compute process exists')
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    require(len(used) == 1 and float(used[0]) <= 22500, 'one GPU and 22500MiB actual cap required')
    return {'gpu_compute_pids': pids, 'gpu_used_mib': float(used[0])}


def load_frozen(pin, authpin):
    root = local_root(); plan = pinned(root, 'ROOT18_PLAN.json', pin); verify_files(root, plan)
    manifest_path = safe_path(root, 'ROOT18_SOURCE_MANIFEST.json')
    _, manifest, recipe, _ = verify_manifest(sha(manifest_path))
    require(plan['source_manifest_sha256'] == sha(manifest_path) and plan['files'] == manifest['files'], 'frozen source manifest differs')
    auth = pinned(root, 'ROOT_ROOT18_START_AUTHORIZATION.json', authpin)
    require(auth['controller_one_attempt_allowed'] is True and auth['automatic_retry'] is False
            and auth['plan_sha256'] == pin and auth['run_ids'] == sorted(RUN_IDS)
            and auth['new_test_allowed'] is False, 'explicit ROOT one-controller authorization required')
    documents = {key: pinned(root, name, plan[field]) for key, (name, field) in DOCUMENTS.items()}
    documents['auth'] = auth
    freeze_path = safe_path(root, 'ACTUAL_ROOT18_FREEZE_RECEIPT.json')
    documents['freeze'] = metadata(freeze_path)
    require(documents['freeze']['authorization_sha256'] == authpin, 'actual freeze authorization differs')
    validate_canonical(documents['selection'], documents['closure'], documents['references'])
    verify_references(documents['references'])
    validate_evidence(plan, recipe, sorted(RUN_IDS)[0], pin, **documents, pre=None, capability=None, now=time.time())
    return root, plan, recipe, documents, sha(freeze_path)


def receipt_name(prefix):
    return prefix + uuid.uuid4().hex + '.json'


def validate_prestart_identity(pre, plan, pin, authpin, freeze_pin, now):
    require(pre['status'] == 'PASS_ACTUAL_INDEPENDENT_ROOT18_SOURCE_PREFIX_RESOURCE_PRESTART'
            and pre['plan_sha256'] == pin and pre['authorization_sha256'] == authpin
            and pre['freeze_sha256'] == freeze_pin and pre['files'] == plan['files']
            and pre['expires_after_seconds'] == 120 and 0 <= now - pre['utc_epoch'] <= 120,
            'root18 prestart identity/expiry differs')


def validate_initial_prestart(pre, plan, recipe, documents, pin, authpin, freeze_pin, now):
    validate_prestart_identity(pre, plan, pin, authpin, freeze_pin, now)
    require(pre['candidate_run_id'] is None and pre['controller_identity'] is None
            and not pre['owned'] and not pre['active_configs'], 'initial start requires GPU-idle independent audit')
    from root18_resources import validate_resource_admission
    validate_resource_admission(pre['resource_telemetry'], {}, None)
    rid = sorted(RUN_IDS)[0]
    virtual = {'status': 'ROOT_EXPLICIT_ONE_RUN_ROOT18_CAPABILITY_AFTER_FRESH_INDEPENDENT_AUDIT',
        'run_id': rid, 'plan_sha256': pin, 'run_config_sha256': config_sha(plan['runs'][rid]),
        'automatic_retry': False, 'test_allowed': False, 'utc_epoch': pre['utc_epoch'], 'expires_after_seconds': 120}
    validate_evidence(plan, recipe, rid, pin, **documents, pre=pre, capability=virtual, now=now)


def verify_controller_identity(start, pin, authpin):
    require(start['status'] == 'ACTUAL_ROOT18_ONE_CONTROLLER_STARTED_NO_TEST'
            and start['plan_sha256'] == pin and start['authorization_sha256'] == authpin
            and start['automatic_retry'] is False and start['test_allowed'] is False,
            'root18 controller attempt identity differs')
    physical = process_identity(start['pid'])
    require(physical['uid'] == 0 and physical['start_ticks'] == start['start_ticks']
            and physical['argv'] == start['argv'] and CHILD + '/root18_controller.py' in physical['argv'],
            'root18 controller owned PID/argv/starttick differs')
    return physical


def pending():
    return {'status': 'PENDING_FUTURE_ROOT18_CANONICAL_SELECTION_ACTUAL_QA_REVIEWS_FREEZE_FRESH_PRESTART_REQUIRED',
            'training_started': False, 'test_accessed': False, 'remote_action_performed': False}
