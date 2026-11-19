"""Pure opaque completion verifier. Numeric payloads are hashed, never decoded."""
import hashlib,re
from itsmixer13_common import *

def complete(root,rid,plan,pin,authpin):
    folder=safe_path(root,'results/'+rid)
    require(set(p.name for p in folder.iterdir())==ALL5,'completed run must have exactly native all5')
    receipt=metadata(folder/'COMPLETE_RECEIPT.json');cfg=plan['runs'][rid]
    require(receipt['status']=='COMPLETE_VALIDATION_ONLY' and receipt['run_id']==rid and receipt['plan_sha256']==pin
        and receipt['root_authorization_sha256']==authpin and receipt['test_accessed'] is False
        and receipt['scores_released'] is False and receipt['automatic_retry'] is False,'native completion oneauthority')
    require(receipt['config_sha256']==hashlib.sha256(encoded(cfg)).hexdigest(),'exact scientific+measured native config bytes')
    require(receipt['recovery_evidence_sha256']==plan['recovery_evidence_sha256']
        and receipt['previous_source_only_isolated13_noattempt_proof_sha256']==plan['previous_source_only_isolated13_noattempt_proof_sha256']
        and receipt['previous_frozen13_noattempt_proof_sha256']==plan['previous_frozen13_noattempt_proof_sha256']
        and receipt['previous_started13_preframework_failure_proof_sha256']==plan['previous_started13_preframework_failure_proof_sha256']
        and receipt['original_resource_projection_only'] is True,
        'distinct original18 resource projection and retained ancestry identity')
    for k in ('train_windows_available','train_windows_used_per_epoch','train_windows_dropped_per_epoch','validation_windows','validation_elements'):
        require(receipt[k]==cfg[k],'native scorefree scientific accounting '+k)
    require(type(receipt['epochs']) is int and receipt['epochs']==cfg['epochs'] and type(receipt['steps']) is int
        and receipt['steps']==cfg['expected_optimizer_steps_at_full_epoch_budget']
        and type(receipt['best_epoch']) is int and 1<=receipt['best_epoch']<=cfg['epochs']
        and receipt['best_step']==receipt['best_epoch']*cfg['train_optimizer_steps_per_epoch']
        and receipt['checkpoint_criterion_name']==cfg['checkpoint_criterion']
        and receipt['root_pre_score_policy_sha256']==ROOT_POLICY_SHA and receipt['actual_synthetic_qa_sha256']==plan['qa_sha256']
        and receipt['source_inventory_sha256']==hashlib.sha256(encoded(plan['files'])).hexdigest(),'full author budget/source/native criterion')
    attempt_path=root/('ITSMIXER13_RUN_ATTEMPT_'+rid+'.json');attempt=metadata(attempt_path)
    require(sha(attempt_path)==receipt['run_attempt_claim_sha256'] and attempt['status']=='ACTUAL_ITSMIXER13_ONE_OWNED_RUN_ATTEMPT_CLAIM'
        and attempt['run_id']==rid and attempt['plan_sha256']==pin and attempt['authorization_sha256']==authpin
        and attempt['capability_sha256']==receipt['run_capability_sha256'] and attempt['automatic_retry'] is False
        and attempt['test_allowed'] is False and attempt['root_pre_score_policy_sha256']==ROOT_POLICY_SHA,'native exclusive owned attempt')
    capname=attempt['capability_name'];require(re.fullmatch(r'ROOT_ITSMIXER13_RUN_CAPABILITY_[0-9a-f]{32}\.json',capname),'completion capability name')
    cap=pinned(root,capname,receipt['run_capability_sha256'])
    require(cap['run_id']==rid and cap['plan_sha256']==pin and cap['authorization_sha256']==authpin
        and cap['run_config_sha256']==config_sha(cfg) and cap['controller_actual_start_sha256']==attempt['controller_actual_start_sha256']
        and cap['prestart_name']==attempt['prestart_name']==receipt['prestart_receipt_name']
        and cap['prestart_sha256']==attempt['prestart_sha256']==receipt['prestart_receipt_sha256'],'native historical capability/prestart links')
    pre=pinned(root,cap['prestart_name'],cap['prestart_sha256'])
    root_live,plan_live,recipe,docs,freezepin=load_frozen(pin,authpin)
    validate_evidence(plan_live,recipe,pin,docs,attempt['utc_epoch'],pre,cap,rid)
    require(root_live==root and cap['freeze_sha256']==pre['freeze_sha256']==freezepin
        and pre['authorization_sha256']==authpin,'native historical fresh authority')
    start=pinned(root,'ITSMIXER13_ACTUAL_START.json',attempt['controller_actual_start_sha256'])
    require(attempt['parent_pid']==start['pid']==cap['controller_identity']['pid']
        and attempt['argv']==['/root/miniconda3/bin/python','-u',CHILD+'/itsmixer13_runtime.py','--execute-reviewed-run',
            '--run-id',rid,'--plan-sha256',pin,'--authorization-sha256',authpin,'--capability-sha256',receipt['run_capability_sha256'],'--capability-name',capname]
        and cap['utc_epoch']<=attempt['utc_epoch']<=cap['utc_epoch']+120,'native exact launch and fresh scientific claim')
    for name,key in (('best.pt','checkpoint_sha256'),('config.json','config_sha256'),('history.json','history_sha256'),('VALIDATION_RESULT.json','validation_result_sha256')):
        require(sha(safe_path(folder,name))==receipt[key],'opaque all4 completion hash '+name)
    return {**receipt,'complete_receipt_sha256':sha(folder/'COMPLETE_RECEIPT.json')}
