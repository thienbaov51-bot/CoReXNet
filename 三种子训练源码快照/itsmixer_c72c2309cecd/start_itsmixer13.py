"""ROOT-explicit one controller start; default PENDING, no automatic retry."""
import argparse
import json
import subprocess
import time

from itsmixer13_common import (load_frozen, pinned, require, exclusive, pending,
    validate_initial_prestart, process_identity, sha)


def start(pin, authpin, prepin, prename):
    import re
    require(re.fullmatch(r'ACTUAL_INDEPENDENT_ITSMIXER13_PRESTART_[0-9a-f]{32}\.json', prename) is not None,
            'unsafe start prestart name')
    root, plan, recipe, documents, freeze_pin = load_frozen(pin, authpin)
    pre = pinned(root, prename, prepin)
    validate_initial_prestart(pre, plan, recipe, documents, pin, authpin, freeze_pin, time.time())
    from itsmixer13_resources import collect_telemetry, validate_resource_admission
    resource = collect_telemetry(root, {}); validate_resource_admission(resource, {}, None)
    require(not resource['gpu_compute_pids'], 'initial itsmixer18 start requires empty GPU compute')
    validate_initial_prestart(pre, plan, recipe, documents, pin, authpin, freeze_pin, time.time())
    for name in ('ITSMIXER13_ACTUAL_START.json', 'ITSMIXER13_STATE.json', 'ITSMIXER13_TRAINING_BOUNDARY.json', 'ITSMIXER13_LAUNCH_RECEIPT.json', 'logs', 'results'):
        require(not (root / name).exists(), 'never automatically resume/restart itsmixer18')
    claim = {'status': 'ROOT_EXPLICIT_ONE_ITSMIXER13_CONTROLLER_START_CLAIM', 'utc_epoch': time.time(),
        'plan_sha256': pin, 'authorization_sha256': authpin, 'prestart_sha256': prepin,
        'prestart_name': prename, 'freeze_sha256': freeze_pin, 'automatic_retry': False,
        'test_allowed': False, 'resource_telemetry': resource}
    exclusive(root / 'ITSMIXER13_START_CLAIM.json', claim)
    argv = ['/root/miniconda3/bin/python', '-u', str(root / 'itsmixer13_controller.py'), '--execute-reviewed-controller',
            '--plan-sha256', pin, '--authorization-sha256', authpin, '--prestart-sha256', prepin, '--prestart-name', prename]
    import os
    log=(root/'controller.log').open('x')
    process=subprocess.Popen(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    # Preserve the successful Popen handle and durable raw birth before /proc.
    birth={'status':'ACTUAL_ITSMIXER13_CONTROLLER_POPEN_RETURNED_UNVERIFIED','utc_epoch':time.time(),
        'pid':process.pid,'expected_argv':argv,'plan_sha256':pin,'authorization_sha256':authpin,
        'prestart_sha256':prepin,'prestart_name':prename,'claim_sha256':sha(root/'ITSMIXER13_START_CLAIM.json'),
        'ownership_verified':False,'automatic_retry':False,'test_allowed':False}
    try:
        exclusive(root/'ITSMIXER13_POPEN_RETURNED.json',birth)
        identity=process_identity(process.pid)
        require(identity['argv']==argv and identity['uid']==0 and identity['parent_pid']==os.getpid(),'actual launch owned argv/UID/parent differs')
        launch={'status':'ACTUAL_ITSMIXER13_ONE_CONTROLLER_LAUNCHED','utc_epoch':time.time(),
            'pid':process.pid,'start_ticks':identity['start_ticks'],'argv':argv,
            'plan_sha256':pin,'authorization_sha256':authpin,'prestart_sha256':prepin,'prestart_name':prename,
            'claim_sha256':sha(root/'ITSMIXER13_START_CLAIM.json'),'popen_returned_sha256':sha(root/'ITSMIXER13_POPEN_RETURNED.json'),
            'launcher_identity':process_identity(os.getpid()),'automatic_retry':False,'test_allowed':False}
        exclusive(root/'ITSMIXER13_LAUNCH_RECEIPT.json',launch)
    except Exception as error:
        # This handle is never declared owned, killed, retried or eligible for
        # COMPLETE. The controller cannot proceed without exact launch proof.
        observations=[]
        while True:
            code=process.poll();observations.append({'utc_epoch':time.time(),'returncode':code})
            if code is not None:break
            time.sleep(.1)
        exclusive(root/'ITSMIXER13_UNVERIFIED_CONTROLLER_EXIT.json',{**birth,'status':'FAILED_UNVERIFIED_CONTROLLER_POLLED_DRAINED_ROOT_REVIEW',
            'error':str(error),'returncode':code,'exit_observed_by_poll':True,'poll_observations':observations,
            'ownership_verified':False,'eligible_for_complete_receipt':False,'no_automatic_resume':True})
        raise
    finally:log.close()

    print(json.dumps({'status': launch['status'], 'pid': process.pid, 'automatic_retry': False}))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--start-verified-once', action='store_true')
    for name in ('plan-sha256', 'authorization-sha256', 'prestart-sha256', 'prestart-name'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if not args.start_verified_once or not all((args.plan_sha256, args.authorization_sha256, args.prestart_sha256, args.prestart_name)):
        print(json.dumps(pending())); return
    start(args.plan_sha256, args.authorization_sha256, args.prestart_sha256, args.prestart_name)


if __name__ == '__main__':
    main()
