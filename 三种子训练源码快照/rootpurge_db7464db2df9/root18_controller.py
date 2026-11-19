"""Future owned root18 one-attempt TRAIN+VAL controller; default PENDING."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from root18_production_common import (load_frozen, pinned, validate_initial_prestart, exclusive,
    atomic, pending, require, sha, metadata, process_identity, receipt_name, config_sha)


def complete(root, rid, plan, pin):
    """Hash opaque numeric artifacts; decode only the score-free completion receipt."""
    from root18_admission_guard import safe_path
    folder = root / 'results' / rid; receipt = metadata(safe_path(root, 'results/' + rid + '/COMPLETE_RECEIPT.json'))
    require(receipt['status'] == 'COMPLETE_VALIDATION_ONLY' and receipt['run_id'] == rid
            and receipt['plan_sha256'] == pin and receipt['test_accessed'] is False, 'completion identity differs')
    expected_config = __import__('hashlib').sha256(json.dumps(plan['runs'][rid], indent=2, allow_nan=False).encode()).hexdigest()
    require(receipt['config_sha256'] == expected_config, 'completion original configuration differs')
    for name, field in (('best.pt', 'checkpoint_sha256'), ('config.json', 'config_sha256'),
            ('history.json', 'history_sha256'), ('VALIDATION_RESULT.json', 'validation_result_sha256')):
        path = folder / name
        require(all(not item.is_symlink() for item in (path, *path.parents)) and sha(path) == receipt[field], 'opaque completion file SHA differs')
    return {**receipt, 'complete_receipt_sha256': sha(folder / 'COMPLETE_RECEIPT.json')}


def controller(pin, authpin, prepin, prename):
    root, plan, recipe, documents, freeze_pin = load_frozen(pin, authpin)
    pre = pinned(root, prename, prepin); validate_initial_prestart(pre, plan, recipe, documents, pin, authpin, freeze_pin, time.time())
    claim = metadata(root / 'ROOT18_START_CLAIM.json')
    require(claim['plan_sha256'] == pin and claim['authorization_sha256'] == authpin
            and claim['prestart_sha256'] == prepin and claim['prestart_name'] == prename
            and claim['automatic_retry'] is False, 'ROOT start claim differs')
    # The launcher writes PID/starttick evidence immediately after Popen. No scientific import occurs here.
    deadline = time.time() + 5
    while not (root / 'ROOT18_LAUNCH_RECEIPT.json').exists() and time.time() < deadline:
        time.sleep(0.05)
    launch = metadata(root / 'ROOT18_LAUNCH_RECEIPT.json'); identity = process_identity(os.getpid())
    require(launch['pid'] == os.getpid() and launch['start_ticks'] == identity['start_ticks']
            and launch['argv'] == identity['argv'] and launch['claim_sha256'] == sha(root / 'ROOT18_START_CLAIM.json')
            and launch['plan_sha256'] == pin and launch['authorization_sha256'] == authpin
            and launch['prestart_sha256'] == prepin and launch['prestart_name'] == prename,
            'actual controller owned launch identity differs')
    validate_initial_prestart(pre, plan, recipe, documents, pin, authpin, freeze_pin, time.time())
    start = {'status': 'ACTUAL_ROOT18_ONE_CONTROLLER_STARTED_NO_TEST', 'utc_epoch': time.time(),
        'pid': os.getpid(), 'start_ticks': identity['start_ticks'], 'argv': identity['argv'],
        'plan_sha256': pin, 'authorization_sha256': authpin, 'freeze_sha256': freeze_pin,
        'prestart_sha256': prepin, 'prestart_name': prename, 'claim_sha256': sha(root / 'ROOT18_START_CLAIM.json'),
        'launch_sha256': sha(root / 'ROOT18_LAUNCH_RECEIPT.json'), 'source_inventory': plan['files'],
        'automatic_retry': False, 'test_allowed': False}
    exclusive(root / 'ROOT18_ACTUAL_START.json', start)
    (root / 'logs').mkdir(); (root / 'results').mkdir()
    from root18_resources import collect_telemetry, validate_resource_admission, PendingAdmission
    from independent_prestart_root18 import review
    queue = sorted(plan['runs']); active = {}; done = {}; failed = {}; halt = None
    def snapshot(status='RUNNING'):
        atomic(root / 'ROOT18_STATE.json', {'status': status, 'utc_epoch': time.time(), 'plan_sha256': pin,
            'authorization_sha256': authpin, 'actual_start_sha256': sha(root / 'ROOT18_ACTUAL_START.json'),
            'done': done, 'active': {r: v['owned'] for r, v in active.items()}, 'queue': queue, 'failed': failed,
            'admission_halt': halt, 'scores_released': False, 'test_accessed': False, 'automatic_retry': False})
    snapshot()
    while queue or active:
        for rid, item in list(active.items()):
            code = item['process'].poll()
            if code is None:
                continue
            item['log'].close(); active.pop(rid)
            try:
                require(code == 0, 'worker nonzero exit: ' + str(code)); done[rid] = complete(root, rid, plan, pin)
            except Exception as error:
                failed[rid] = {'returncode': code, 'error': str(error), 'automatic_retry': False}
                halt = 'FAILED_WORKER_REQUIRES_ROOT_REVIEW'
        if failed:
            halt = 'FAILED_WORKER_REQUIRES_ROOT_REVIEW'
        for rid in list(queue) if halt is None else []:
            owned = {r: v['owned'] for r, v in active.items()}; configs = {r: plan['runs'][r] for r in active}
            try:
                telemetry = collect_telemetry(root, owned)
                validate_resource_admission(telemetry, configs, plan['runs'][rid])
            except PendingAdmission as error:
                if not active:
                    halt = 'PENDING_RESOURCE_REQUIRES_ROOT_REVIEW: ' + str(error)
                    break
                continue
            except Exception as error:
                halt = 'UNOWNED_OR_INVALID_RESOURCE_REQUIRES_ROOT_REVIEW: ' + str(error); break
            try:
                name, fresh = review(pin, authpin, rid, owned, configs)
                capability_name = receipt_name('ROOT_ROOT18_RUN_CAPABILITY_')
                capability = {'status': 'ROOT_EXPLICIT_ONE_RUN_ROOT18_CAPABILITY_AFTER_FRESH_INDEPENDENT_AUDIT',
                    'utc_epoch': time.time(), 'expires_after_seconds': 120, 'run_id': rid, 'plan_sha256': pin,
                    'run_config_sha256': config_sha(plan['runs'][rid]), 'authorization_sha256': authpin,
                    'freeze_sha256': freeze_pin, 'prestart_name': name, 'prestart_sha256': sha(root / name),
                    'controller_actual_start_sha256': sha(root / 'ROOT18_ACTUAL_START.json'),
                    'controller_identity': {key: start[key] for key in ('pid', 'start_ticks', 'argv')},
                    'owned': owned, 'active_configs': configs, 'automatic_retry': False, 'test_allowed': False}
                exclusive(root / capability_name, capability)
                argv = ['/root/miniconda3/bin/python', '-u', str(root / 'root18_trainval_runtime_pending.py'), '--execute-reviewed-run',
                    '--run-id', rid, '--plan-sha256', pin, '--authorization-sha256', authpin,
                    '--capability-sha256', sha(root / capability_name), '--capability-name', capability_name]
                log = (root / 'logs' / (rid + '.log')).open('x')
                process = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                worker = process_identity(process.pid)
                require(worker['argv'] == argv and worker['parent_pid'] == os.getpid() and worker['uid'] == 0, 'spawned worker ownership differs')
                active[rid] = {'process': process, 'log': log, 'owned': {'pid': process.pid,
                    'start_ticks': worker['start_ticks'], 'argv': argv, 'controller_pid': os.getpid()}}
                queue.remove(rid); snapshot()
            except Exception as error:
                halt = 'FRESH_ADMISSION_OR_START_REQUIRES_ROOT_REVIEW: ' + str(error); break
        snapshot('STOPPED_ADMISSIONS_WAITING_OWNED_WORKERS' if halt else 'RUNNING')
        if halt and not active:
            break
        time.sleep(3)
    if len(done) == 18 and not queue and not active and not failed and halt is None:
        try:
            for rid, previous in done.items():
                require(complete(root, rid, plan, pin) == previous, 'completed opaque run identity changed before whole18 closure')
            from root18_production_common import verify_references
            from root18_admission_guard import verify_files
            verify_references(documents['references']); verify_files(root, plan)
            validate_resource_admission(collect_telemetry(root, {}), {}, None)
        except Exception as error:
            halt = 'WHOLE24_OPAQUE_PHYSICAL_CLOSURE_REQUIRES_ROOT_REVIEW: ' + str(error)
    from root18_retained_contract import reference_index
    references = reference_index(documents['references'])
    success = len(done) == 18 and not queue and not active and not failed and halt is None
    boundary = {'status': 'COMPLETE_ROOT18_VALIDATION_TRAINING' if success else 'INCOMPLETE_REQUIRES_ROOT_REVIEW_NO_RETRY',
        'utc_epoch': time.time(), 'plan_sha256': pin, 'authorization_sha256': authpin,
        'actual_start_sha256': sha(root / 'ROOT18_ACTUAL_START.json'), 'done': done, 'queue': queue,
        'active': {}, 'failed': failed, 'admission_halt': halt, 'canonical_h96_references': references,
        'new_training_count': 18, 'h96_reference_count': 6, 'weights_copied': False,
        'all24_best_sha256': {**{rid: row['checkpoint_sha256'] for rid, row in done.items()},
                            **{rid + '_L96_H96_seed2026': row['physical']['files']['best.pt']['sha256'] for rid, row in references.items()}},
        'scores_released': False, 'test_accessed': False, 'automatic_retry': False,
        'full24_best_freeze_and_separate_puretest_review_still_required': True}
    exclusive(root / 'ROOT18_TRAINING_BOUNDARY.json', boundary); snapshot(boundary['status'])
    require(success, 'root18 incomplete; no automatic retry or resume')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--execute-reviewed-controller', action='store_true')
    for name in ('plan-sha256', 'authorization-sha256', 'prestart-sha256', 'prestart-name'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if not args.execute_reviewed_controller or not all((args.plan_sha256, args.authorization_sha256, args.prestart_sha256, args.prestart_name)):
        print(json.dumps(pending())); return
    controller(args.plan_sha256, args.authorization_sha256, args.prestart_sha256, args.prestart_name)


if __name__ == '__main__':
    main()
