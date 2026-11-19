"""Future owned itsmixer18 one-attempt TRAIN+VAL controller; default PENDING."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from itsmixer5_common import (load_frozen, pinned, validate_initial_prestart, exclusive,
    atomic, pending, require, sha, metadata, process_identity, receipt_name, config_sha, safe_path, ALL5)


def complete(root,rid,plan,pin,authpin):
    from itsmixer5_completion import complete as check
    return check(root,rid,plan,pin,authpin)


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
    root, plan, recipe, documents, freeze_pin = load_frozen(pin, authpin)
    pre = pinned(root, prename, prepin); validate_initial_prestart(pre, plan, recipe, documents, pin, authpin, freeze_pin, time.time())
    claim = metadata(root / 'ITSMIXER5_START_CLAIM.json')
    require(claim['plan_sha256'] == pin and claim['authorization_sha256'] == authpin
            and claim['prestart_sha256'] == prepin and claim['prestart_name'] == prename
            and claim['automatic_retry'] is False, 'ROOT start claim differs')
    # The launcher writes PID/starttick evidence immediately after Popen. No scientific import occurs here.
    deadline = time.time() + 5
    while not (root / 'ITSMIXER5_LAUNCH_RECEIPT.json').exists() and time.time() < deadline:
        time.sleep(0.05)
    launch = metadata(root / 'ITSMIXER5_LAUNCH_RECEIPT.json'); identity = process_identity(os.getpid())
    require(launch['pid'] == os.getpid() and launch['start_ticks'] == identity['start_ticks']
            and launch['argv'] == identity['argv'] and launch['claim_sha256'] == sha(root / 'ITSMIXER5_START_CLAIM.json')
            and launch['plan_sha256'] == pin and launch['authorization_sha256'] == authpin
            and launch['prestart_sha256'] == prepin and launch['prestart_name'] == prename,
            'actual controller owned launch identity differs')
    validate_initial_prestart(pre, plan, recipe, documents, pin, authpin, freeze_pin, time.time())
    start = {'status': 'ACTUAL_ITSMIXER5_ONE_CONTROLLER_STARTED_NO_TEST', 'utc_epoch': time.time(),
        'pid': os.getpid(), 'start_ticks': identity['start_ticks'], 'argv': identity['argv'],
        'plan_sha256': pin, 'authorization_sha256': authpin, 'freeze_sha256': freeze_pin,
        'prestart_sha256': prepin, 'prestart_name': prename, 'claim_sha256': sha(root / 'ITSMIXER5_START_CLAIM.json'),
        'launch_sha256': sha(root / 'ITSMIXER5_LAUNCH_RECEIPT.json'), 'source_inventory': plan['files'],
        'automatic_retry': False, 'test_allowed': False}
    exclusive(root / 'ITSMIXER5_ACTUAL_START.json', start)
    (root / 'logs').mkdir(); (root / 'results').mkdir()
    from itsmixer5_resources import collect_telemetry, validate_resource_admission, PendingAdmission
    from independent_prestart_itsmixer5 import review
    queue = sorted(plan['runs']); active = {}; unverified = {}; unverified_exits = {}; done = {}; failed = {}; halt = None; resource_deferrals = []
    def snapshot(status='RUNNING'):
        atomic(root / 'ITSMIXER5_STATE.json', {'status': status, 'utc_epoch': time.time(), 'plan_sha256': pin,
            'authorization_sha256': authpin, 'actual_start_sha256': sha(root / 'ITSMIXER5_ACTUAL_START.json'),
            'done': done, 'active': {r: v['owned'] for r, v in active.items()}, 'queue': queue, 'failed': failed,
            'unverified_started_workers': {r:v['evidence'] for r,v in unverified.items()},
            'unverified_worker_exits': unverified_exits,
            'resource_deferrals':resource_deferrals,'admission_halt': halt, 'scores_released': False, 'test_accessed': False, 'automatic_retry': False})
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
        for rid in list(queue) if halt is None and not active and not unverified else []:
            # Scheduling only: one successful Popen must poll/drain before any
            # next capability/prestart/log/Popen. Resource ceilings unchanged.
            if active or unverified:break
            owned = {r: v['owned'] for r, v in active.items()}; configs = {r: plan['runs'][r] for r in active}
            try:
                telemetry = collect_telemetry(root, owned)
                validate_resource_admission(telemetry, configs, plan['runs'][rid])
            except PendingAdmission as error:
                resource_deferrals.append({'utc_epoch':time.time(),'run_id':rid,'reason':str(error),'diagnostic':getattr(error,'diagnostic',None)})
                resource_deferrals[:]=resource_deferrals[-32:]
                if not active:
                    halt = 'PENDING_RESOURCE_REQUIRES_ROOT_REVIEW: ' + str(error)
                    break
                continue
            except Exception as error:
                halt = 'UNOWNED_OR_INVALID_RESOURCE_REQUIRES_ROOT_REVIEW: ' + str(error); break
            log = None
            try:
                name, fresh = review(pin, authpin, rid, owned, configs)
                capability_name = receipt_name('ROOT_ITSMIXER5_RUN_CAPABILITY_')
                capability = {'status': 'ROOT_ITSMIXER5_ONE_RUN_CAPABILITY_AFTER_FRESH_INDEPENDENT_PRESTART',
                    'utc_epoch': time.time(), 'expires_after_seconds': 120, 'run_id': rid, 'plan_sha256': pin,
                    'run_config_sha256': config_sha(plan['runs'][rid]), 'authorization_sha256': authpin,
                    'freeze_sha256': freeze_pin, 'prestart_name': name, 'prestart_sha256': sha(root / name),
                    'controller_actual_start_sha256': sha(root / 'ITSMIXER5_ACTUAL_START.json'),
                    'controller_identity': {key: start[key] for key in ('pid', 'start_ticks', 'argv')},
                    'owned': owned, 'active_configs': configs, 'automatic_retry': False, 'test_allowed': False}
                exclusive(root / capability_name, capability)
                capability_pin = sha(root / capability_name)
                argv = ['/root/miniconda3/bin/python', '-u', str(root / 'itsmixer5_runtime.py'), '--execute-reviewed-run',
                    '--run-id', rid, '--plan-sha256', pin, '--authorization-sha256', authpin,
                    '--capability-sha256', capability_pin, '--capability-name', capability_name]
                log = (root / 'logs' / (rid + '.log')).open('x')
                process = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                # Keep every successful Popen handle before attempting owned
                # identity sampling. An unverified PID is never declared owned.
                birth_evidence={'status':'UNVERIFIED_STARTED_ITSMIXER5_WORKER_REQUIRES_ROOT_REVIEW','pid':process.pid,
                    'run_id':rid,'expected_argv':argv,'plan_sha256':pin,'authorization_sha256':authpin,
                    'capability_name':capability_name,'capability_sha256':capability_pin,'ownership_verified':False,'automatic_retry':False}
                # Durable Popen ledger is written before fallible /proc ownership.
                unverified[rid]={'process':process,'log':log,'evidence':birth_evidence}
                exclusive(root/('ITSMIXER5_POPEN_RETURNED_'+rid+'.json'),birth_evidence)
                owned_worker=capture_started_worker(unverified,rid,process,log,birth_evidence,process_identity,os.getpid())
                active[rid]={'process':process,'log':log,'owned':owned_worker}
                unverified.pop(rid)
                queue.remove(rid); snapshot()
            except PendingAdmission as error:
                require(rid not in unverified and rid not in active and log is None,'fresh review Pending must precede any Popen/log/capability')
                resource_deferrals.append({'utc_epoch':time.time(),'run_id':rid,'reason':str(error),'diagnostic':getattr(error,'diagnostic',None)})
                resource_deferrals[:]=resource_deferrals[-32:]
                snapshot();continue
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
    if len(done) == 5 and not queue and not active and not unverified and not unverified_exits and not failed and halt is None:
        try:
            for rid, previous in done.items():
                require(complete(root, rid, plan, pin, authpin) == previous, 'completed opaque run identity changed before whole6 closure')
            from itsmixer5_common import verify_references,verify_qualified_pair
            from itsmixer5_common import verify_files
            verify_references(documents['references']); verify_qualified_pair(root,plan,documents); verify_files(root, plan)
            from itsmixer5_common import gate_old_cohorts,global_census
            gate_old_cohorts();global_census()
            validate_resource_admission(collect_telemetry(root, {}), {}, None)
        except Exception as error:
            halt = 'MINIMUM12_OPAQUE_PHYSICAL_CLOSURE_REQUIRES_ROOT_REVIEW: ' + str(error)
    references = {row['run_id']:row for row in documents['itsmixer_references']['runs']}
    original5 = documents['recovery']['retained_original5_references']
    origin_ledger={rid:{'source_root':row['source_root'],'plan_sha256':row['plan_sha256'],
        'best_path':row['best_path'],'files':row['physical']['files'],'origin':'ORIGINAL_ITSMIXER6_H96'} for rid,row in references.items()}
    origin_ledger.update({rid:{'source_root':row['source_root'],'plan_sha256':row['plan_sha256'],
        'best_path':row['best_path'],'files':row['files'],'config':row['config'],'origin':'ORIGINAL_ITSMIXER18_RETAINED_FIVE'} for rid,row in original5.items()})
    retained8=documents['recovery']['retained_original8_references']
    origin_ledger.update({rid:{'source_root':row['source_root'],'plan_sha256':row['plan_sha256'],'best_path':row['best_path'],'files':row['files'],'config':row['config'],'origin':'ORIGINAL_S3_RETAINED_EIGHT'} for rid,row in retained8.items()})
    for rid,row in done.items():
        folder=root/'results'/rid
        files={name:{'sha256':sha(safe_path(folder,name)),'bytes':safe_path(folder,name).stat().st_size} for name in ALL5}
        origin_ledger[rid]={'source_root':str(root),'plan_sha256':pin,'best_path':str(folder/'best.pt'),
            'files':files,'config':plan['runs'][rid],'origin':'DISTINCT_ITSMIXER5_FIRST_TRAINING'}
    success = len(done) == 5 and not queue and not active and not unverified and not unverified_exits and not failed and halt is None
    boundary = {'status': 'COMPLETE_ITSMIXER5_VALIDATION_TRAINING' if success else 'INCOMPLETE_REQUIRES_ROOT_REVIEW_NO_RETRY',
        'utc_epoch': time.time(), 'plan_sha256': pin, 'authorization_sha256': authpin,
        'actual_start_sha256': sha(root / 'ITSMIXER5_ACTUAL_START.json'), 'done': done, 'queue': queue,
        'active': {}, 'failed': [] if not failed else [{'run_id':r,**row} for r,row in sorted(failed.items())], 'resource_deferrals':resource_deferrals,'admission_halt': halt, 'retained_ITSMixer_h96_references': references,'retained_original18_five_references':original5,'retained_original_s3_eight_references':retained8,'recovery_evidence_sha256':plan['recovery_evidence_sha256'],
        'unverified_started_workers': {r:v['evidence'] for r,v in unverified.items()},
        'unverified_worker_exits': unverified_exits,
        'all_workers_polled_and_drained':not active and not unverified,'verified_owned_mapping_only_not_physical_idle_when_incomplete':True,'new_training_count': 5, 'retained_previous_s3_count': 8, 'h96_reference_count': 6, 'weights_copied': False,
        'all24_best_sha256': {**{rid: row['checkpoint_sha256'] for rid, row in done.items()},
                            **{rid: row['physical']['files']['best.pt']['sha256'] for rid, row in references.items()},
                            **{rid:row['files']['best.pt']['sha256'] for rid,row in original5.items()},**{rid:row['files']['best.pt']['sha256'] for rid,row in retained8.items()}},
        'all_itsmixer_origin_ledger':origin_ledger,'retained_old18_count':5,'old5_retraining_allowed':False,
        'previous_source_only_isolated13_noattempt_proof_sha256':plan['previous_source_only_isolated13_noattempt_proof_sha256'],
        'previous_frozen13_noattempt_proof_sha256':plan['previous_frozen13_noattempt_proof_sha256'],
        'previous_started13_preframework_failure_proof_sha256':plan['previous_started13_preframework_failure_proof_sha256'],
        'previous_s3_eight_complete_five_no_scientific_claims_proof_sha256':plan['recovery_evidence_sha256'],
        'scores_released': False, 'test_accessed': False, 'automatic_retry': False,
        'ITSMixer18_numeric_qualification_then_future_selected_full_and_separate_puretest_still_required': True}
    exclusive(root / 'ITSMIXER5_TRAINING_BOUNDARY.json', boundary); snapshot(boundary['status'])
    require(success, 'itsmixer18 incomplete; no automatic retry or resume')


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
