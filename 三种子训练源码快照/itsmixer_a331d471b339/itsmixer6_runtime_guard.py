"""Same Popen/capability bounded stdlib admission before framework or RNG."""
import os, re, time
from itsmixer6_common import *

def clean_state(state,pin,authpin,startpin):
    require(state['status']=='RUNNING' and state['failed']=={} and state['admission_halt'] is None
        and state['unverified_started_workers']=={} and state['unverified_worker_exits']=={}
        and state['plan_sha256']==pin and state['authorization_sha256']==authpin
        and state['actual_start_sha256']==startpin,'live controller halted, failed or has unverified identities')

def verify_live_authority(root,pin,authpin,cappin,capname,freezepin,prename,prepin,startpin):
    for name,h in (('ITSMIXER6_PLAN.json',pin),('ROOT_ITSMIXER6_START_AUTHORIZATION.json',authpin),
        (capname,cappin),('ACTUAL_ITSMIXER6_FREEZE_RECEIPT.json',freezepin),(prename,prepin),('ITSMIXER6_ACTUAL_START.json',startpin)):
        require(sha(safe_path(root,name))==h,'live admitted metadata bytes changed '+name)

def admit_run(run_id,plan_sha256,authorization_sha256,capability_sha256,capability_name):
    rid,pin,authpin,cappin,capname=run_id,plan_sha256,authorization_sha256,capability_sha256,capability_name
    root,plan,recipe,docs,freezepin=load_frozen(pin,authpin)
    require(rid in RUN_IDS and re.fullmatch(r'ROOT_ITSMIXER6_RUN_CAPABILITY_[0-9a-f]{32}\.json',capname),'exact one-run capability path')
    cap=pinned(root,capname,cappin);prename=cap['prestart_name']
    require(re.fullmatch(r'ACTUAL_INDEPENDENT_ITSMIXER6_PRESTART_[0-9a-f]{32}\.json',prename),'exact prestart path')
    pre=pinned(root,prename,cap['prestart_sha256'])
    require(pre['receipt_name']==prename and pre['authorization_sha256']==cap['authorization_sha256']==authpin
        and pre['freeze_sha256']==cap['freeze_sha256']==freezepin,'physical capability/prestart/freeze links')
    validate_evidence(plan,recipe,pin,docs,time.time(),pre,cap,rid)
    startpin=cap['controller_actual_start_sha256'];start=pinned(root,'ITSMIXER6_ACTUAL_START.json',startpin)
    expected=cap['controller_identity']
    require(start['pid']==expected['pid']==os.getppid() and start['start_ticks']==expected['start_ticks']
        and start['argv']==expected['argv'] and pre['controller_identity']['actual_start_sha256']==startpin
        and pre['candidate_run_id']==rid and pre['owned']==cap['owned'] and pre['active_configs']==cap['active_configs'],
        'same explicitly owned controller/capability authority')
    deadline=min(cap['utc_epoch']+120,pre['utc_epoch']+120,time.time()+5)
    # No queue bypass. The controller publishes the exact self registration.
    while True:
        verify_live_authority(root,pin,authpin,cappin,capname,freezepin,prename,cap['prestart_sha256'],startpin)
        controller=verify_controller_identity(start,pin,authpin)
        require(controller['pid']==os.getppid(),'same parent throughout registration wait')
        state=metadata(root/'ITSMIXER6_STATE.json');clean_state(state,pin,authpin,startpin)
        if rid in state['active']:break
        require(time.time()<deadline,'owned worker registration exceeded same capability deadline')
        time.sleep(.05)
    from itsmixer6_resources import collect_telemetry,validate_resource_admission,validate_worker_argv,PendingAdmission
    def live_owned():
        controller=verify_controller_identity(start,pin,authpin)
        require(controller['pid']==os.getppid(),'controller PID/parent changed during bounded wait')
        state=metadata(root/'ITSMIXER6_STATE.json');clean_state(state,pin,authpin,startpin)
        me=process_identity(os.getpid());own=state['active'].get(rid)
        require(own is not None and own['pid']==me['pid']==os.getpid() and own['start_ticks']==me['start_ticks']
            and own['argv']==me['argv'] and own['controller_pid']==me['parent_pid']==controller['pid'] and me['uid']==0,
            'exact live self UID/PID/tick/argv/parent ownership')
        validate_worker_argv(me['argv'],rid)
        require(me['argv']==['/root/miniconda3/bin/python','-u',CHILD+'/itsmixer6_runtime.py','--execute-reviewed-run',
            '--run-id',rid,'--plan-sha256',pin,'--authorization-sha256',authpin,'--capability-sha256',cappin,'--capability-name',capname],
            'exact self full argv pins')
        others={r:o for r,o in state['active'].items() if r!=rid}
        for other,row in others.items():
            validate_worker_argv(row['argv'],other)
            require(row['controller_pid']==controller['pid'] and row['argv'][row['argv'].index('--plan-sha256')+1]==pin
                and row['argv'][row['argv'].index('--authorization-sha256')+1]==authpin,'other owned run authority changed')
        return me,others
    # Registered self ownership must precede the old-cohort global census.
    # Old physical verification remains mandatory before any claim or framework.
    live_owned()
    gate_old_cohorts()
    deferrals=[]
    while True:
        verify_live_authority(root,pin,authpin,cappin,capname,freezepin,prename,cap['prestart_sha256'],startpin)
        me,others=live_owned();global_census()
        try:
            telemetry=collect_telemetry(root,others)
            validate_resource_admission(telemetry,{r:plan['runs'][r] for r in others},plan['runs'][rid])
            require(time.time()<=deadline,'same Popen stable sample deadline/authority expired');break
        except PendingAdmission as error:
            deferrals.append({'utc_epoch':time.time(),'reason':str(error),'diagnostic':getattr(error,'diagnostic',None)})
            require(time.time()<deadline,'strict owned GPU membership/capacity not stable within same Popen wait')
            time.sleep(.05)
    verify_live_authority(root,pin,authpin,cappin,capname,freezepin,prename,cap['prestart_sha256'],startpin)
    me,others=live_owned();require(time.time()<=deadline,'claim boundary capability expiry')
    global_census();require(time.time()<=deadline,'final current process census exceeded capability deadline')
    require(not safe_path(root,'results/'+rid).exists(),'run directory cannot be resumed or retried')
    claim_path=root/('ITSMIXER6_RUN_ATTEMPT_'+rid+'.json')
    exclusive(claim_path,{'status':'ACTUAL_ITSMIXER6_ONE_OWNED_RUN_ATTEMPT_CLAIM','utc_epoch':time.time(),
        'run_id':rid,'plan_sha256':pin,'authorization_sha256':authpin,'capability_sha256':cappin,'capability_name':capname,
        'pid':me['pid'],'start_ticks':me['start_ticks'],'argv':me['argv'],'parent_pid':me['parent_pid'],
        'controller_actual_start_sha256':startpin,'prestart_name':prename,'prestart_sha256':cap['prestart_sha256'],
        'root_pre_score_policy_sha256':ROOT_POLICY_SHA,'automatic_retry':False,'test_allowed':False,
        'same_Popen_stability_deferrals':deferrals,'fresh_stable_resource_telemetry':telemetry})
    return plan['runs'][rid]
