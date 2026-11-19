"""ROOT-explicit one controller start; default PENDING, no automatic retry."""
import argparse
import json
import subprocess
import time

from root18_production_common import (load_frozen, pinned, require, exclusive, pending,
    validate_initial_prestart, process_identity, sha)


def start(pin, authpin, prepin, prename):
    import re
    require(re.fullmatch(r'ACTUAL_INDEPENDENT_ROOT18_PRESTART_REVIEW_[0-9a-f]{32}\.json', prename) is not None,
            'unsafe start prestart name')
    root, plan, recipe, documents, freeze_pin = load_frozen(pin, authpin)
    pre = pinned(root, prename, prepin)
    validate_initial_prestart(pre, plan, recipe, documents, pin, authpin, freeze_pin, time.time())
    from root18_resources import collect_telemetry, validate_resource_admission
    resource = collect_telemetry(root, {}); validate_resource_admission(resource, {}, None)
    require(not resource['gpu_compute_pids'], 'initial root18 start requires empty GPU compute')
    for name in ('ROOT18_ACTUAL_START.json', 'ROOT18_STATE.json', 'ROOT18_TRAINING_BOUNDARY.json', 'ROOT18_LAUNCH_RECEIPT.json', 'logs', 'results'):
        require(not (root / name).exists(), 'never automatically resume/restart root18')
    claim = {'status': 'ROOT_EXPLICIT_ONE_ROOT18_CONTROLLER_START_CLAIM', 'utc_epoch': time.time(),
        'plan_sha256': pin, 'authorization_sha256': authpin, 'prestart_sha256': prepin,
        'prestart_name': prename, 'freeze_sha256': freeze_pin, 'automatic_retry': False,
        'test_allowed': False, 'resource_telemetry': resource}
    exclusive(root / 'ROOT18_START_CLAIM.json', claim)
    argv = ['/root/miniconda3/bin/python', '-u', str(root / 'root18_controller.py'), '--execute-reviewed-controller',
            '--plan-sha256', pin, '--authorization-sha256', authpin, '--prestart-sha256', prepin, '--prestart-name', prename]
    with (root / 'controller.log').open('x') as log:
        process = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(process.pid)
    require(identity['argv'] == argv and identity['uid'] == 0, 'actual launch owned argv differs')
    launch = {'status': 'ACTUAL_ROOT18_ONE_CONTROLLER_LAUNCHED', 'utc_epoch': time.time(),
        'pid': process.pid, 'start_ticks': identity['start_ticks'], 'argv': argv,
        'plan_sha256': pin, 'authorization_sha256': authpin, 'prestart_sha256': prepin,
        'prestart_name': prename, 'claim_sha256': sha(root / 'ROOT18_START_CLAIM.json'),
        'automatic_retry': False, 'test_allowed': False}
    exclusive(root / 'ROOT18_LAUNCH_RECEIPT.json', launch)
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
