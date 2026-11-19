"""Same Popen/capability bounded stdlib admission before framework or RNG."""
import os, re, time
from itsmixer13_common import *

def clean_state(state,pin,authpin,startpin):
    require(state['status']=='RUNNING' and state['failed']=={} and state['admission_halt'] is None
        and state['unverified_started_workers']=={} and state['unverified_worker_exits']=={}
        and state['plan_sha256']==pin and state['authorization_sha256']==authpin
        and state['actual_start_sha256']==startpin,'live controller halted, failed or has unverified identities')

def verify_live_authority(root,pin,authpin,cappin,capname,freezepin,prename,prepin,startpin):
    for name,h in (('ITSMIXER13_PLAN.json',pin),('ROOT_ITSMIXER13_START_AUTHORIZATION.json',authpin),
        (capname,cappin),('ACTUAL_ITSMIXER13_FREEZE_RECEIPT.json',freezepin),(prename,prepin),('ITSMIXER13_ACTUAL_START.json',startpin)):
        require(sha(safe_path(root,name))==h,'live admitted metadata bytes changed '+name)

def admit_run(run_id,plan_sha256,authorization_sha256,capability_sha256,capability_name):
    rid,pin,authpin,cappin,capname=run_id,plan_sha256,authorization_sha256,capability_sha256,capability_name
    root,plan,recipe,docs,freezepin=load_frozen(pin,authpin)
    require(rid in RUN_IDS and re.fullmatch(r'ROOT_ITSMIXER13_RUN_CAPABILITY_[0-9a-f]{32}\.json',capname),'exact one-run capability path')
    cap=pinned(root,capname,cappin);prename=cap['prestart_name']
    require(re.fullmatch(r'ACTUAL_INDEPENDENT_ITSMIXER13_PRESTART_[0-9a-f]{32}\.json',prename),'exact prestart path')
    pre=pinned(root,prename,cap['prestart_sha256'])
    require(pre['receipt_name']==prename and pre['authorization_sha256']==cap['authorization_sha256']==authpin
        and pre['freeze_sha256']==cap['freeze_sha256']==freezepin,'physical capability/prestart/freeze links')
    validate_evidence(plan,recipe,pin,docs,time.time(),pre,cap,rid)
    startpin=cap['controller_actual_start_sha256'];start=pinned(root,'ITSMIXER13_ACTUAL_START.json',startpin)
    expected=cap['controller_identity']
    require(start['pid']==expected['pid']==os.getppid() and start['start_ticks']==expected['start_ticks']
        and start['argv']==expected['argv'] and pre['controller_identity']['actual_start_sha256']==startpin
        and pre['candidate_run_id']==rid and pre['owned']==cap['owned'] and pre['active_configs']==cap['active_configs'],
        'same explicitly owned controller/capability authority')
    absolute_deadline=min(cap['utc_epoch']+120,pre['utc_epoch']+120)
    registration_deadline=min(absolute_deadline,time.time()+5)
    # No queue bypass. The controller publishes the exact self registration.
    while True:
        verify_live_authority(root,pin,authpin,cappin,capname,freezepin,prename,cap['prestart_sha256'],startpin)
        controller=verify_controller_identity(start,pin,authpin)
        require(controller['pid']==os.getppid(),'same parent throughout registration wait')
        state=metadata(root/'ITSMIXER13_STATE.json');clean_state(state,pin,authpin,startpin)
        require(time.time()<=registration_deadline,'owned worker registration exceeded same capability deadline')
        if rid in state['active']:break
        require(time.time()<registration_deadline,'owned worker registration exceeded same capability deadline')
        time.sleep(.05)
    from itsmixer13_resources import collect_telemetry,validate_resource_admission,validate_worker_argv,PendingAdmission
    def live_owned():
        controller=verify_controller_identity(start,pin,authpin)
        require(controller['pid']==os.getppid(),'controller PID/parent changed during bounded wait')
        state=metadata(root/'ITSMIXER13_STATE.json');clean_state(state,pin,authpin,startpin)
        me=process_identity(os.getpid());own=state['active'].get(rid)
        require(own is not None and own['pid']==me['pid']==os.getpid() and own['start_ticks']==me['start_ticks']
            and own['argv']==me['argv'] and own['controller_pid']==me['parent_pid']==controller['pid'] and me['uid']==0,
            'exact live self UID/PID/tick/argv/parent ownership')
        validate_worker_argv(me['argv'],rid)
        require(me['argv']==['/root/miniconda3/bin/python','-u',CHILD+'/itsmixer13_runtime.py','--execute-reviewed-run',
            '--run-id',rid,'--plan-sha256',pin,'--authorization-sha256',authpin,'--capability-sha256',cappin,'--capability-name',capname],
            'exact self full argv pins')
        others={r:o for r,o in state['active'].items() if r!=rid}
        for other,row in others.items():
            validate_worker_argv(row['argv'],other)
            require(row['controller_pid']==controller['pid'] and row['argv'][row['argv'].index('--plan-sha256')+1]==pin
                and row['argv'][row['argv'].index('--authorization-sha256')+1]==authpin,'other owned run authority changed')
        return me,others
    # Registration has its own maximum-five-second bound. Mandatory historical
    # physical proof is performed once, under the unchanged absolute cap/pre
    # 120-second expiry, before the maximum-five-second live resource window.
    # No RNG, framework, attempt claim or scientific folder exists yet.
    require(time.time()<=absolute_deadline,'absolute cap/pre authority expired before mandatory physical proof')
    verify_live_authority(root,pin,authpin,cappin,capname,freezepin,prename,cap['prestart_sha256'],startpin)
    live_owned()
    physical_proof_started=time.time()
    physical_proof=verify_old_cohort_physical()
    physical_proof_finished=time.time()
    require(physical_proof_finished<=absolute_deadline,'absolute cap/pre authority expired during mandatory physical proof')
    verify_live_authority(root,pin,authpin,cappin,capname,freezepin,prename,cap['prestart_sha256'],startpin)
    live_owned()
    require(time.time()<=absolute_deadline,'absolute cap/pre authority expired after mandatory physical proof')
    deadline=min(absolute_deadline,time.time()+5)
    stable_window_started=time.time()
    deferrals=[]
    while True:
        verify_live_authority(root,pin,authpin,cappin,capname,freezepin,prename,cap['prestart_sha256'],startpin)
        me,others=live_owned()
        try:
            global_census()
            telemetry=collect_telemetry(root,others)
            validate_resource_admission(telemetry,{r:plan['runs'][r] for r in others},plan['runs'][rid])
            require(time.time()<=deadline,'same Popen stable sample deadline/authority expired')
            verify_live_authority(root,pin,authpin,cappin,capname,freezepin,prename,cap['prestart_sha256'],startpin)
            me,others=live_owned();require(time.time()<=deadline,'claim boundary capability expiry')
            global_census();require(time.time()<=deadline,'final current process census exceeded capability deadline')
            break
        except PendingAdmission as error:
            deferrals.append({'utc_epoch':time.time(),'reason':str(error),'diagnostic':getattr(error,'diagnostic',None)})
            require(time.time()<deadline,'strict owned GPU membership/capacity not stable within same Popen wait')
            time.sleep(.05)
    require(not safe_path(root,'results/'+rid).exists(),'run directory cannot be resumed or retried')
    claim_path=root/('ITSMIXER13_RUN_ATTEMPT_'+rid+'.json')
    claim_utc=time.time()
    require(claim_utc<=deadline and claim_utc<=absolute_deadline,'final noattempt check exceeded same Popen deadline/authority')
    exclusive(claim_path,{'status':'ACTUAL_ITSMIXER13_ONE_OWNED_RUN_ATTEMPT_CLAIM','utc_epoch':claim_utc,
        'run_id':rid,'plan_sha256':pin,'authorization_sha256':authpin,'capability_sha256':cappin,'capability_name':capname,
        'pid':me['pid'],'start_ticks':me['start_ticks'],'argv':me['argv'],'parent_pid':me['parent_pid'],
        'controller_actual_start_sha256':startpin,'prestart_name':prename,'prestart_sha256':cap['prestart_sha256'],
        'root_pre_score_policy_sha256':ROOT_POLICY_SHA,'automatic_retry':False,'test_allowed':False,
        'same_Popen_stability_deferrals':deferrals,'fresh_stable_resource_telemetry':telemetry,
        'absolute_cap_pre_deadline_utc_epoch':absolute_deadline,'registration_deadline_utc_epoch':registration_deadline,
        'mandatory_physical_proof_started_utc_epoch':physical_proof_started,
        'mandatory_physical_proof_finished_utc_epoch':physical_proof_finished,
        'mandatory_physical_proof_duration_seconds':physical_proof_finished-physical_proof_started,
        'mandatory_physical_proof':physical_proof,'stable_window_started_utc_epoch':stable_window_started,
        'stable_window_deadline_utc_epoch':deadline,'stable_window_max_seconds':5,
        'absolute_cap_pre_expiry_extended':False})
    return plan['runs'][rid]
