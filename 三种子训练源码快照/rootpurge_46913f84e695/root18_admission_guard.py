"""New recovery-only metadata wrapper for the byte-identical original runtime."""
import json
import os
import re
import time
from pathlib import Path
from root9_common import (CHILD, RUN_IDS, require, safe_path, sha, metadata,
    config_sha, verify_files, load_frozen, pinned, validate_prestart, process_identity,
    no_other_producers, exclusive)


def validate_running_controller_state(state):
    require(state['status'] == 'RUNNING' and state['failed'] == {} and state['admission_halt'] is None
            and state['unverified_started_workers'] == {} and state['unverified_worker_exits'] == {},
            'recovery controller stopped admissions or has unverified worker identities')


def verify_live_authority(root,pin,authpin,cappin,capname,freezepin,pre_name,prepin,startpin):
    for name,expected in (('ROOT9_PLAN.json',pin),('ROOT_ROOT9_START_AUTHORIZATION.json',authpin),
        (capname,cappin),('ACTUAL_ROOT9_FREEZE_RECEIPT.json',freezepin),(pre_name,prepin),('ROOT9_ACTUAL_START.json',startpin)):
        require(sha(safe_path(root,name))==expected,'live capability/plan/authorization/freeze/prestart/start changed during stable wait: '+name)


def admit_run(root, rid, pin, authpin, cappin, capname):
    require(str(root) == CHILD and re.fullmatch(r'ROOT_ROOT9_RUN_CAPABILITY_[0-9a-f]{32}\.json', capname),
            'exact separate recovery child/capability required')
    root, plan, manifest, evidence, original_plan, docs, freeze_pin = load_frozen(pin, authpin)
    require(rid in plan['authorized_run_ids'] and rid not in plan['preserved_complete9'],
            'only the original untouched nine may train; retained nine/H96 forbidden')
    no_other_producers()
    cap = pinned(root, capname, cappin)
    pre_name = cap['prestart_name']
    require(re.fullmatch(r'ACTUAL_INDEPENDENT_ROOT9_PRESTART_REVIEW_[0-9a-f]{32}\.json', pre_name), 'unsafe prestart receipt')
    pre = pinned(root, pre_name, cap['prestart_sha256'])
    now = time.time(); validate_prestart(pre, plan, pin, authpin, freeze_pin, now, rid, docs)
    require(cap['status'] == 'ROOT_EXPLICIT_ONE_RUN_ROOT9_CAPABILITY_AFTER_FRESH_INDEPENDENT_AUDIT'
            and cap['run_id'] == rid and cap['plan_sha256'] == pin and cap['authorization_sha256'] == authpin
            and cap['run_config_sha256'] == config_sha(plan['runs'][rid]) and cap['freeze_sha256'] == freeze_pin
            and cap['automatic_retry'] is False and cap['test_allowed'] is False
            and cap['expires_after_seconds'] == 120 and 0 <= now - cap['utc_epoch'] <= 120
            and pre['utc_epoch'] <= cap['utc_epoch'] <= pre['utc_epoch'] + 120,
            'fresh one-run recovery capability differs')
    start_path = safe_path(root, 'ROOT9_ACTUAL_START.json')
    require(sha(start_path) == cap['controller_actual_start_sha256'], 'capability/start binding differs')
    start = metadata(start_path); expected = cap['controller_identity']
    require(start['plan_sha256'] == pin and start['authorization_sha256'] == authpin
            and start['pid'] == expected['pid'] == os.getppid()
            and start['start_ticks'] == expected['start_ticks'] and start['argv'] == expected['argv']
            and pre['controller_identity']['actual_start_sha256'] == cap['controller_actual_start_sha256']
            and pre['owned'] == cap['owned'] and pre['active_configs'] == cap['active_configs'], 'strict recovery controller identity differs')
    controller = process_identity(os.getppid())
    require(controller['uid'] == 0 and controller['start_ticks'] == start['start_ticks']
            and controller['argv'] == start['argv'] and CHILD + '/root9_controller.py' in controller['argv'], 'controller PID was reused or malformed')
    deadline = time.time() + 5
    while True:
        state = metadata(safe_path(root, 'ROOT9_STATE.json'))
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
    from root18_resources import collect_telemetry, validate_resource_admission, PendingAdmission
    # Same successful Popen and same capability: wait in stdlib before any
    # framework import, RNG construction, arrays or scientific attempt claim.
    deadline=min(cap['utc_epoch']+120,time.time()+5);stability_deferrals=[]
    while True:
        now=time.time();require(now<=cap['utc_epoch']+120,'same capability expired during stable-census wait')
        verify_live_authority(root,pin,authpin,cappin,capname,freeze_pin,pre_name,cap['prestart_sha256'],cap['controller_actual_start_sha256'])
        state=metadata(safe_path(root,'ROOT9_STATE.json'));validate_running_controller_state(state)
        require(state['plan_sha256']==pin and state['authorization_sha256']==authpin and state['actual_start_sha256']==cap['controller_actual_start_sha256'],'current state changed during stable-census wait')
        own=state['active'].get(rid);worker=process_identity(os.getpid())
        require(own is not None and own['pid']==os.getpid() and own['start_ticks']==worker['start_ticks'] and own['argv']==worker['argv']
            and own['controller_pid']==os.getppid() and worker['uid']==0,'exact self ownership during stable-census wait')
        others={key:value for key,value in state['active'].items() if key!=rid}
        for key,value in others.items():
            require(key in plan['authorized_run_ids'] and value['controller_pid']==os.getppid()
                and value['argv'][value['argv'].index('--plan-sha256')+1]==pin
                and value['argv'][value['argv'].index('--authorization-sha256')+1]==authpin,'other owned authority changed during wait')
        try:
            stability_telemetry=collect_telemetry(root,others)
            sampled_at=time.time()
            require(sampled_at<=deadline and 0<=sampled_at-cap['utc_epoch']<=120
                and 0<=sampled_at-pre['utc_epoch']<=120,'stable sample exceeded bounded wait or prestart/capability freshness')
            validate_resource_admission(stability_telemetry,{key:plan['runs'][key] for key in others},plan['runs'][rid]);break
        except PendingAdmission as error:
            stability_deferrals.append({'utc_epoch':time.time(),'diagnostic':getattr(error,'diagnostic',None),'reason':str(error)})
            require(time.time()<deadline,'stable owned census not achieved under same capability before deadline')
            time.sleep(.05)
    verify_live_authority(root,pin,authpin,cappin,capname,freeze_pin,pre_name,cap['prestart_sha256'],cap['controller_actual_start_sha256'])
    claimed_at=time.time()
    require(claimed_at<=deadline and 0<=claimed_at-cap['utc_epoch']<=120 and 0<=claimed_at-pre['utc_epoch']<=120,
        'claim boundary exceeded bounded wait or prestart/capability freshness')
    require(not safe_path(root, 'results/' + rid).exists(), 'one original untouched run may be attempted only once in new child')
    exclusive(root / ('ROOT9_RUN_CLAIM_' + rid + '.json'), {
        'status':'ACTUAL_ONE_UNATTEMPTED_ROOT9_OWNED_RUN_CLAIM','utc_epoch':claimed_at,
        'run_id':rid,'plan_sha256':pin,'authorization_sha256':authpin,'capability_name':capname,
        'capability_sha256':cappin,'worker_identity':worker,'controller_actual_start_sha256':sha(start_path),
        'original18_plan_sha256':plan['original18_plan_sha256'],'recovery_evidence_sha256':plan['recovery_evidence_sha256'],
        'automatic_retry':False,'test_allowed':False,'same_Popen_stability_deferrals':stability_deferrals,'fresh_stable_resource_telemetry':stability_telemetry})
    return plan
