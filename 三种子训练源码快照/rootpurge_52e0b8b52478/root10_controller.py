"""Future owned root10 one-attempt TRAIN+VAL controller; default PENDING."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from root10_common import (load_frozen, pinned, validate_prestart, exclusive,
    atomic, pending, require, sha, metadata, process_identity, receipt_name, config_sha)


def complete(root, rid, plan, pin, authpin):
    """Hash every native numerical artifact; decode only metadata completion links."""
    from root10_common import CORE, safe_path
    prefix = 'results/' + rid + '/'
    receipt = metadata(safe_path(root, prefix + 'COMPLETE_RECEIPT.json'))
    require(rid in plan['authorized_run_ids'] and receipt['status'] == 'COMPLETE_VALIDATION_ONLY'
            and receipt['run_id'] == rid and receipt['plan_sha256'] == pin and receipt['test_accessed'] is False,
            'native completed recovery identity differs')
    expected = __import__('hashlib').sha256(json.dumps(plan['runs'][rid],indent=2,allow_nan=False).encode()).hexdigest()
    require(receipt['config_sha256'] == expected, 'original unchanged native config serialization differs')
    claim_path = safe_path(root, 'ROOT10_RUN_CLAIM_' + rid + '.json'); claim = metadata(claim_path)
    require(claim['status'] == 'ACTUAL_ONE_UNATTEMPTED_ROOT10_OWNED_RUN_CLAIM' and claim['run_id'] == rid
            and claim['plan_sha256'] == pin and claim['authorization_sha256'] == authpin
            and claim['original18_plan_sha256'] == plan['original18_plan_sha256']
            and claim['recovery_evidence_sha256'] == plan['recovery_evidence_sha256']
            and claim['automatic_retry'] is False and claim['test_allowed'] is False,
            'native exclusive recovery attempt differs')
    import re
    require(re.fullmatch(r'ROOT_ROOT10_RUN_CAPABILITY_[0-9a-f]{32}\.json',claim['capability_name']), 'unsafe native recovery capability')
    cap_path = safe_path(root,claim['capability_name']); cap = metadata(cap_path)
    require(sha(cap_path) == claim['capability_sha256'] and cap['run_id'] == rid and cap['plan_sha256'] == pin
            and cap['authorization_sha256'] == authpin and cap['run_config_sha256'] == config_sha(plan['runs'][rid])
            and cap['controller_actual_start_sha256'] == claim['controller_actual_start_sha256']
            and cap['automatic_retry'] is False and cap['test_allowed'] is False,
            'native recovery complete capability links differ')
    for name, field in CORE.items():
        require(sha(safe_path(root,prefix+name)) == receipt[field], 'native recovery all5 physical completion links differ')
    return {**receipt,'complete_receipt_sha256':sha(safe_path(root,prefix+'COMPLETE_RECEIPT.json'))}


def capture_started_worker(unverified, rid, process, log, evidence, sampler, controller_pid):
    """Preserve the successful Popen handle before any fallible ownership read."""
    require(type(process.pid) is int and process.pid > 0, 'successful Popen PID differs')
    unverified[rid]={'process':process,'log':log,'evidence':evidence}
    worker=sampler(process.pid)
    require(worker['argv']==evidence['expected_argv'] and worker['parent_pid']==controller_pid
            and worker['uid']==0, 'spawned worker ownership differs')
    return {'pid':process.pid,'start_ticks':worker['start_ticks'],
        'argv':evidence['expected_argv'],'controller_pid':controller_pid}


def controller(pin, authpin, prepin, prename):
    root, plan, manifest, evidence, original_plan, documents, freeze_pin = load_frozen(pin, authpin)
    pre = pinned(root, prename, prepin); validate_prestart(pre, plan, pin, authpin, freeze_pin, time.time(),documents=documents)
    claim = metadata(root / 'ROOT10_START_CLAIM.json')
    require(claim['plan_sha256'] == pin and claim['authorization_sha256'] == authpin
            and claim['prestart_sha256'] == prepin and claim['prestart_name'] == prename
            and claim['automatic_retry'] is False, 'ROOT start claim differs')
    # The launcher writes PID/starttick evidence immediately after Popen. No scientific import occurs here.
    deadline = time.time() + 5
    while not (root / 'ROOT10_LAUNCH_RECEIPT.json').exists() and time.time() < deadline:
        time.sleep(0.05)
    launch = metadata(root / 'ROOT10_LAUNCH_RECEIPT.json'); identity = process_identity(os.getpid())
    require(launch['pid'] == os.getpid() and launch['start_ticks'] == identity['start_ticks']
            and launch['argv'] == identity['argv'] and launch['claim_sha256'] == sha(root / 'ROOT10_START_CLAIM.json')
            and launch['plan_sha256'] == pin and launch['authorization_sha256'] == authpin
            and launch['prestart_sha256'] == prepin and launch['prestart_name'] == prename,
            'actual controller owned launch identity differs')
    validate_prestart(pre, plan, pin, authpin, freeze_pin, time.time(),documents=documents)
    start = {'status': 'ACTUAL_ROOT10_ONE_CONTROLLER_STARTED_NO_TEST', 'utc_epoch': time.time(),
        'pid': os.getpid(), 'start_ticks': identity['start_ticks'], 'argv': identity['argv'],
        'plan_sha256': pin, 'authorization_sha256': authpin, 'freeze_sha256': freeze_pin,
        'prestart_sha256': prepin, 'prestart_name': prename, 'claim_sha256': sha(root / 'ROOT10_START_CLAIM.json'),
        'launch_sha256': sha(root / 'ROOT10_LAUNCH_RECEIPT.json'), 'source_inventory': plan['files'],
        'automatic_retry': False, 'test_allowed': False}
    exclusive(root / 'ROOT10_ACTUAL_START.json', start)
    (root / 'logs').mkdir(); (root / 'results').mkdir()
    from root18_resources import collect_telemetry, validate_resource_admission, PendingAdmission
    from independent_prestart_root10 import review
    queue = list(plan['authorized_run_ids']); active = {}; unverified = {}; unverified_exits = {}; done = {}; failed = {}; halt = None
    def snapshot(status='RUNNING'):
        atomic(root / 'ROOT10_STATE.json', {'status': status, 'utc_epoch': time.time(), 'plan_sha256': pin,
            'authorization_sha256': authpin, 'actual_start_sha256': sha(root / 'ROOT10_ACTUAL_START.json'),
            'done': done, 'active': {r: v['owned'] for r, v in active.items()}, 'queue': queue, 'failed': failed,
            'unverified_started_workers': {r:v['evidence'] for r,v in unverified.items()},
            'unverified_worker_exits': unverified_exits,
            'admission_halt': halt, 'scores_released': False, 'test_accessed': False, 'automatic_retry': False})
    snapshot()
    while queue or active or unverified:
        for rid,item in list(unverified.items()):
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();unverified.pop(rid)
            unverified_exits[rid]={**item['evidence'],'returncode':code,'exit_observed_by_poll':True,
                'ownership_verified':False,'eligible_for_complete_receipt':False}
        for rid, item in list(active.items()):
            code = item['process'].poll()
            if code is None:
                continue
            item['log'].close(); active.pop(rid)
            try:
                require(code == 0, 'worker nonzero exit: ' + str(code)); done[rid] = complete(root, rid, plan, pin, authpin)
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
            log = None
            try:
                name, fresh = review(pin, authpin, rid, owned, configs)
                capability_name = receipt_name('ROOT_ROOT10_RUN_CAPABILITY_')
                capability = {'status': 'ROOT_EXPLICIT_ONE_RUN_ROOT10_CAPABILITY_AFTER_FRESH_INDEPENDENT_AUDIT',
                    'utc_epoch': time.time(), 'expires_after_seconds': 120, 'run_id': rid, 'plan_sha256': pin,
                    'run_config_sha256': config_sha(plan['runs'][rid]), 'authorization_sha256': authpin,
                    'freeze_sha256': freeze_pin, 'prestart_name': name, 'prestart_sha256': sha(root / name),
                    'controller_actual_start_sha256': sha(root / 'ROOT10_ACTUAL_START.json'),
                    'controller_identity': {key: start[key] for key in ('pid', 'start_ticks', 'argv')},
                    'owned': owned, 'active_configs': configs, 'automatic_retry': False, 'test_allowed': False}
                exclusive(root / capability_name, capability)
                capability_pin = sha(root / capability_name)
                argv = ['/root/miniconda3/bin/python', '-u', str(root / 'root18_trainval_runtime_pending.py'), '--execute-reviewed-run',
                    '--run-id', rid, '--plan-sha256', pin, '--authorization-sha256', authpin,
                    '--capability-sha256', capability_pin, '--capability-name', capability_name]
                log = (root / 'logs' / (rid + '.log')).open('x')
                process = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                # Keep every successful Popen handle before attempting owned
                # identity sampling. An unverified PID is never declared owned.
                birth_evidence={
                    'status':'UNVERIFIED_STARTED_ROOT10_WORKER_REQUIRES_ROOT_REVIEW','pid':process.pid,
                    'run_id':rid,'expected_argv':argv,'plan_sha256':pin,'authorization_sha256':authpin,
                    'capability_name':capability_name,'capability_sha256':capability_pin,
                    'ownership_verified':False,'automatic_retry':False}
                owned_worker=capture_started_worker(unverified,rid,process,log,birth_evidence,process_identity,os.getpid())
                active[rid] = {'process': process, 'log': log, 'owned': owned_worker}
                unverified.pop(rid)
                queue.remove(rid); snapshot()
            except Exception as error:
                if rid in unverified:
                    failed[rid]={'error':str(error),'post_Popen_identity_failed':True,
                        'unverified_started_worker':unverified[rid]['evidence'],'automatic_retry':False}
                elif log is not None and rid not in active:
                    log.close()
                halt = 'FRESH_ADMISSION_OR_START_REQUIRES_ROOT_REVIEW: ' + str(error); break
        snapshot('STOPPED_ADMISSIONS_WAITING_STARTED_WORKERS' if halt else 'RUNNING')
        if halt and not active and not unverified:
            break
        time.sleep(3)
    if len(done) == 10 and not queue and not active and not unverified and not unverified_exits and not failed and halt is None:
        try:
            for rid, previous in done.items():
                require(complete(root, rid, plan, pin, authpin) == previous, 'completed opaque run identity changed before whole6 closure')
            from root10_common import verify_original, verify_files
            verify_original(evidence); verify_files(root, plan)
            validate_resource_admission(collect_telemetry(root, {}), {}, None)
        except Exception as error:
            halt = 'MINIMUM12_OPAQUE_PHYSICAL_CLOSURE_REQUIRES_ROOT_REVIEW: ' + str(error)
    from root10_common import verify_original
    _, _, original8, references = verify_original(evidence)
    success = len(done) == 10 and not queue and not active and not unverified and not unverified_exits and not failed and halt is None
    boundary = {'status': 'COMPLETE_ROOT10_ENVIRONMENT_RECOVERY_VALIDATION_TRAINING' if success else 'INCOMPLETE_REQUIRES_ROOT_REVIEW_NO_RETRY',
        'utc_epoch': time.time(), 'plan_sha256': pin, 'authorization_sha256': authpin,
        'actual_start_sha256': sha(root / 'ROOT10_ACTUAL_START.json'), 'done': done, 'queue': queue,
        'active': {}, 'failed': failed, 'admission_halt': halt, 'retained_root6_h96_references': references, 'preserved_original_complete8': original8,
        'unverified_started_workers': {r:v['evidence'] for r,v in unverified.items()},
        'unverified_worker_exits': unverified_exits,
        'new_training_count': 10, 'preserved_original_complete8_count': 8, 'h96_reference_count': 6, 'weights_copied': False,
        'all24_best_sha256': {**{rid: row['checkpoint_sha256'] for rid, row in done.items()},
                            **{rid: row['files']['best.pt']['sha256'] for rid,row in original8.items()},
                            **{rid+'_L96_H96_seed2026': row['physical']['files']['best.pt']['sha256'] for rid, row in references.items()}},
        'scores_released': False, 'test_accessed': False, 'automatic_retry': False,
        'VPNet6_numeric_qualification_then_future_selected_full_and_separate_puretest_still_required': True}
    exclusive(root / 'ROOT10_TRAINING_BOUNDARY.json', boundary); snapshot(boundary['status'])
    require(success, 'root10 incomplete; no automatic retry or resume')


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
