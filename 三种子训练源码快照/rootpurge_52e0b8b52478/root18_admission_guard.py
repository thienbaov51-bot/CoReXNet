"""New recovery-only metadata wrapper for the byte-identical original runtime."""
import json
import os
import re
import time
from pathlib import Path
from root10_common import (CHILD, RUN_IDS, require, safe_path, sha, metadata,
    config_sha, verify_files, load_frozen, pinned, validate_prestart, process_identity,
    no_other_producers, exclusive)


def validate_running_controller_state(state):
    require(state['status'] == 'RUNNING' and state['failed'] == {} and state['admission_halt'] is None
            and state['unverified_started_workers'] == {} and state['unverified_worker_exits'] == {},
            'recovery controller stopped admissions or has unverified worker identities')


def admit_run(root, rid, pin, authpin, cappin, capname):
    require(str(root) == CHILD and re.fullmatch(r'ROOT_ROOT10_RUN_CAPABILITY_[0-9a-f]{32}\.json', capname),
            'exact separate recovery child/capability required')
    root, plan, manifest, evidence, original_plan, docs, freeze_pin = load_frozen(pin, authpin)
    require(rid in plan['authorized_run_ids'] and rid not in plan['preserved_complete8'],
            'only the original untouched ten may train; old eight/H96 forbidden')
    no_other_producers()
    cap = pinned(root, capname, cappin)
    pre_name = cap['prestart_name']
    require(re.fullmatch(r'ACTUAL_INDEPENDENT_ROOT10_PRESTART_REVIEW_[0-9a-f]{32}\.json', pre_name), 'unsafe prestart receipt')
    pre = pinned(root, pre_name, cap['prestart_sha256'])
    now = time.time(); validate_prestart(pre, plan, pin, authpin, freeze_pin, now, rid, docs)
    require(cap['status'] == 'ROOT_EXPLICIT_ONE_RUN_ROOT10_CAPABILITY_AFTER_FRESH_INDEPENDENT_AUDIT'
            and cap['run_id'] == rid and cap['plan_sha256'] == pin and cap['authorization_sha256'] == authpin
            and cap['run_config_sha256'] == config_sha(plan['runs'][rid]) and cap['freeze_sha256'] == freeze_pin
            and cap['automatic_retry'] is False and cap['test_allowed'] is False
            and cap['expires_after_seconds'] == 120 and 0 <= now - cap['utc_epoch'] <= 120
            and pre['utc_epoch'] <= cap['utc_epoch'] <= pre['utc_epoch'] + 120,
            'fresh one-run recovery capability differs')
    start_path = safe_path(root, 'ROOT10_ACTUAL_START.json')
    require(sha(start_path) == cap['controller_actual_start_sha256'], 'capability/start binding differs')
    start = metadata(start_path); expected = cap['controller_identity']
    require(start['plan_sha256'] == pin and start['authorization_sha256'] == authpin
            and start['pid'] == expected['pid'] == os.getppid()
            and start['start_ticks'] == expected['start_ticks'] and start['argv'] == expected['argv']
            and pre['controller_identity']['actual_start_sha256'] == cap['controller_actual_start_sha256']
            and pre['owned'] == cap['owned'] and pre['active_configs'] == cap['active_configs'], 'strict recovery controller identity differs')
    controller = process_identity(os.getppid())
    require(controller['uid'] == 0 and controller['start_ticks'] == start['start_ticks']
            and controller['argv'] == start['argv'] and CHILD + '/root10_controller.py' in controller['argv'], 'controller PID was reused or malformed')
    deadline = time.time() + 5
    while True:
        state = metadata(safe_path(root, 'ROOT10_STATE.json'))
        require(state['plan_sha256'] == pin and state['authorization_sha256'] == authpin
                and state['actual_start_sha256'] == cap['controller_actual_start_sha256'], 'current recovery state pins differ')
        validate_running_controller_state(state)
        if rid in state['active']:
            break
        require(time.time() < deadline, 'worker ownership not published before kernel admission')
        time.sleep(.05)
    own = state['active'][rid]; worker = process_identity(os.getpid())
    require(own['pid'] == os.getpid() and own['start_ticks'] == worker['start_ticks']
            and own['argv'] == worker['argv'] and own['controller_pid'] == os.getppid() and worker['uid'] == 0,
            'current owned recovery worker identity differs')
    others = {key: value for key, value in state['active'].items() if key != rid}
    for key, value in others.items():
        require(key in plan['authorized_run_ids'] and value['controller_pid'] == os.getppid()
                and value['argv'][value['argv'].index('--plan-sha256') + 1] == pin
                and value['argv'][value['argv'].index('--authorization-sha256') + 1] == authpin,
                'another owned worker belongs to a different recovery authority')
    from root18_resources import collect_telemetry, validate_resource_admission
    validate_resource_admission(collect_telemetry(root, others), {key: plan['runs'][key] for key in others}, plan['runs'][rid])
    require(not safe_path(root, 'results/' + rid).exists(), 'one original untouched run may be attempted only once in new child')
    exclusive(root / ('ROOT10_RUN_CLAIM_' + rid + '.json'), {
        'status':'ACTUAL_ONE_UNATTEMPTED_ROOT10_OWNED_RUN_CLAIM','utc_epoch':time.time(),
        'run_id':rid,'plan_sha256':pin,'authorization_sha256':authpin,'capability_name':capname,
        'capability_sha256':cappin,'worker_identity':worker,'controller_actual_start_sha256':sha(start_path),
        'original18_plan_sha256':plan['original18_plan_sha256'],'recovery_evidence_sha256':plan['recovery_evidence_sha256'],
        'automatic_retry':False,'test_allowed':False})
    return plan
