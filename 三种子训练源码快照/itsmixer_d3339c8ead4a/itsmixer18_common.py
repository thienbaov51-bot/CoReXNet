"""Reviewed-production building blocks; importing performs no physical I/O."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
from itsmixer18_contract import *

DOCUMENTS={'replacement':('ROOT_VPNET6_REPLACEMENT_RECORD.json','replacement_sha256'),
    'closure':('CANONICAL12_WHOLE_PHYSICAL_CLOSURE.json','canonical_closure_sha256'),
    'references':('ROOTPURGE_H96_REFERENCES.json','retained_references_sha256'),
    'qa':('ACTUAL_ITSMIXER18_SYNTHETIC_CUDA_RECEIPT.json','qa_sha256'),
    'root_review':('ROOT_ITSMIXER18_SOURCE_RESOURCE_REVIEW.json','root_source_resource_review_sha256'),
    'independent_review':('INDEPENDENT_ITSMIXER18_SOURCE_RESOURCE_REVIEW.json','independent_source_resource_review_sha256'),
    'policy':(POLICY_NAME,'root_pre_score_policy_sha256'),
    'prior_vp_native':('VPNET6_PRIOR_NATIVE_PHYSICAL_CLOSURE.json','prior_vpnet6_native_closure_sha256'),
    'prior_vp_whole':('VPNET6_PRIOR_WHOLE12_PHYSICAL_CLOSURE.json','prior_vpnet6_whole12_closure_sha256')}
DOCUMENTS.update(QUALIFICATION_DOCUMENTS)


def pending():
    return {'status':'PENDING_ITSMIXER18_EXPLICIT_ROOT_REVIEWED_SOURCE_ACTUAL_QA_FREEZE_PRESTART_REQUIRED',
        'training_started':False,'test_accessed':False,'remote_action_performed':False,'automatic_retry':False}


def exclusive(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    temporary.replace(path)


def receipt_name(prefix):return prefix+uuid.uuid4().hex+'.json'


def local_root():
    require(os.name=='posix' and os.geteuid()==0 and Path(__file__).absolute().parent==Path(CHILD),
        'PENDING exact ROOT Linux ITSMixer18 child required')
    root=Path(CHILD);require(root.resolve()==root and all(not p.is_symlink() for p in (root,*root.parents)), 'child symlink forbidden')
    return root


def pinned(root,name,expected):
    path=safe_path(root,name);require(sha(path)==expected,'physical metadata SHA differs: '+name)
    return metadata(path)


def verify_references(references):
    rows=references['runs'];expected={'RootPurgeLinear_'+ds for ds in DATASETS}
    require(type(rows) is list and len(rows)==6 and {row['run_id'] for row in rows}==expected,
        'retained RootPurge native LIST6 identity differs')
    for row in rows:
        rid=row['run_id'];folder=Path(row['source_root'])/'results'/rid
        require(row['source_root']==ORIGINAL and row['plan_sha256']==ORIGINAL_PLAN_SHA
            and row['best_path']==ORIGINAL+'/results/'+rid+'/best.pt'
            and set(row['physical']['files'])=={'best.pt','config.json','history.json','VALIDATION_RESULT.json','COMPLETE_RECEIPT.json'},
            'retained RootPurge original all5 physical paths differ')
        for name,identity in row['physical']['files'].items():
            path=folder/name
            require(type(identity['bytes']) is int and identity['bytes']>0
                and all(not p.is_symlink() for p in (path,*path.parents))
                and sha(path)==identity['sha256'] and path.stat().st_size==identity['bytes'],
                'retained RootPurge all5 physical byte SHA/size differs')
        # The four numerical artifacts remain opaque. Only this separate
        # score-free receipt is decoded to close its original physical links.
        receipt=metadata(folder/'COMPLETE_RECEIPT.json')
        require(receipt['status']=='COMPLETE_VALIDATION_ONLY' and receipt['run_id']==rid
            and receipt['plan_sha256']==ORIGINAL_PLAN_SHA and receipt['test_accessed'] is False,
            'retained RootPurge original completion identity differs')
        for field,name in (('checkpoint_sha256','best.pt'),('config_sha256','config.json'),
                ('history_sha256','history.json'),('validation_result_sha256','VALIDATION_RESULT.json')):
            require(receipt[field]==row['physical']['files'][name]['sha256'],
                'retained RootPurge all5 completion byte links differ')


def source_template(manifest,recipe,runs):
    protocol=metadata(Path(__file__).absolute().parent/'ITSMIXER_REMAINING18_SCIENCE_PROTOCOL_PENDING.json')
    return {'status':'ROOT_FROZEN_ITSMIXER18_VALIDATION_ONLY','child_root':CHILD,'original_root':ORIGINAL,
        'original_plan_sha256':ORIGINAL_PLAN_SHA,'recipe_sha256':RECIPE_SHA,'adapter_package_sha256':ADAPTER_PACKAGE_SHA,
        'protocol_sha256':PROTOCOL_SHA,'seed':2026,'length':96,'horizons':[192,336,720],
        'new_training_count':18,'retained_h96_count':6,'H96_retraining_allowed':False,
        'test_allowed':False,'automatic_retry':False,
        'all_validation_windows':True,'TRAIN_drop_last_author_budget':True,'all_train_windows_consumed_each_epoch':False,
        'weights_copied':False,'RootPurge_retraining_allowed':False,'resource_policy':RESOURCE_POLICY,
        'runs':runs,'data_metadata':protocol['native_data_metadata'],'files':manifest['files'],
        'replacement_sha256':REPLACEMENT_SHA,'canonical_closure_sha256':CANONICAL_CLOSURE_SHA,
        'retained_references_sha256':RETAINED_REFERENCES_SHA,'source_manifest_sha256':manifest.get('manifest_sha256'),
        'root_pre_score_policy_sha256':ROOT_POLICY_SHA,'prior_vpnet6_native_closure_sha256':PRIOR_VP_NATIVE_SHA,
        'prior_vpnet6_whole12_closure_sha256':PRIOR_VP_WHOLE12_SHA,
        **{field:manifest[field] for _,field in QUALIFICATION_DOCUMENTS.values()}}


def verify_manifest(pin):
    root=local_root();manifest=pinned(root,'ITSMIXER18_SOURCE_MANIFEST.json',pin)
    require(manifest['status']=='REVIEWED_ITSMIXER18_QUALIFIED_SOURCE_DEPLOYMENT_ONLY_NO_SCIENTIFIC_FREEZE'
        and manifest['child_root']==CHILD and manifest['recipe_sha256']==RECIPE_SHA
        and manifest['adapter_package_sha256']==ADAPTER_PACKAGE_SHA and manifest['protocol_sha256']==PROTOCOL_SHA
        and manifest['replacement_sha256']==REPLACEMENT_SHA and manifest['canonical_closure_sha256']==CANONICAL_CLOSURE_SHA
        and manifest['retained_references_sha256']==RETAINED_REFERENCES_SHA
        and manifest['qualified_pair_required'] is True and manifest['training_started'] is False
        and manifest['test_allowed'] is False,'actual qualified reviewed source manifest differs')
    expected=set(SOURCE_FILES)|{'data/'+ds+'_trainval.npz' for ds in DATASETS}|{'data/'+ds+'_known_calendar.npy' for ds in DATASETS}
    require(set(manifest['files'])==expected,'source/actual-qualification/prefix inventory exact membership differs')
    documents={key:pinned(root,name,manifest[field]) for key,(name,field) in QUALIFICATION_DOCUMENTS.items()}
    # These metadata/opaque old physical gates precede every future framework import.
    verify_qualified_pair(root,manifest,documents)
    for name,pin_file in manifest['files'].items():require(sha(safe_path(root,name))==pin_file,'deployed source/prefix/qualification SHA differs: '+name)
    recipe=pinned(root,'ITSMIXER_REMAINING18_CONFIG_RECIPE_PENDING.json',RECIPE_SHA)
    protocol=pinned(root,'ITSMIXER_REMAINING18_SCIENCE_PROTOCOL_PENDING.json',PROTOCOL_SHA)
    science=pinned(root,'ITSMIXER_REMAINING18_SCIENCE_SOURCE_AUTHOR_MANIFEST.json',ADAPTER_PACKAGE_SHA)
    require(science['snapshot_sha256']==SCIENCE_SNAPSHOT_SHA,'exact corresponding-horizon science snapshot')
    for name,leaf in science['files'].items():require(manifest['files'][name]==leaf,'unchanged exact scientific/provenance leaf '+name)
    require(set(recipe['runs'])==RUN_IDS and recipe['runs']==protocol['runs'],'all18 source recipe identity')
    for rid,cfg in recipe['runs'].items():validate_config(cfg,protocol['native_data_metadata'][cfg['dataset']])
    historic=pinned(root,'HISTORICAL_NATIVE_WHOLE_PREFIX_26973449_PROOF.json',HISTORICAL_NATIVE_PREFIX_PROOF_SHA)
    require(historic['prefix_bitwise_elements_checked']==26973449
        and sum(m['prefix_rows']*m['channels'] for m in protocol['native_data_metadata'].values())==26973449,
        'historical full-prefix proof identity only; fresh actual proof still required')
    documents.update({key:pinned(root,name,manifest[field]) for key,(name,field) in DOCUMENTS.items() if key in ('replacement','closure','references')})
    validate_replacement(documents['replacement'],documents['closure'],documents['references']);verify_references(documents['references'])
    for role,status in (('root','PASS_ROOT_ITSMIXER18_QUALIFIED_LOCAL_PRODUCTION_SOURCE_REVIEW'),
            ('independent','PASS_INDEPENDENT_ITSMIXER18_QUALIFIED_LOCAL_PRODUCTION_SOURCE_REVIEW')):
        proof=pinned(root,manifest[role+'_source_review_name'],manifest[role+'_source_review_sha256'])
        require(proof['status']==status and proof['snapshot']==manifest['source_snapshot']
            and proof['actual_gpu_or_arrays_or_scores_read'] is False,'actual qualified source-only review authority differs')
    require(manifest['source_snapshot']['files']==manifest['files']
        and manifest['source_snapshot']['operator_sha256']==manifest['operator_sha256'], 'reviewed qualified source snapshot differs')
    return root,manifest,recipe,documents


def verify_files(root,plan):
    require(str(root)==CHILD and root.resolve()==root and Path(__file__).absolute().parent==root,'active frozen child code identity differs')
    recipe=pinned(root,'ITSMIXER_REMAINING18_CONFIG_RECIPE_PENDING.json',RECIPE_SHA);validate_plan(plan,recipe)
    require(set(plan['files'])==set(SOURCE_FILES)|{'data/'+ds+'_trainval.npz' for ds in DATASETS}|{'data/'+ds+'_known_calendar.npy' for ds in DATASETS},'frozen exact source file membership differs')
    for name,pin in plan['files'].items():require(sha(safe_path(root,name))==pin,'frozen source/prefix SHA differs: '+name)


def load_frozen(pin,authpin):
    root=local_root();plan=pinned(root,'ITSMIXER18_PLAN.json',pin);verify_files(root,plan)
    _,manifest,recipe,_=verify_manifest(plan['source_manifest_sha256'])
    require(manifest['files']==plan['files'],'frozen/source manifest inventory differs')
    documents={key:pinned(root,name,plan[field]) for key,(name,field) in DOCUMENTS.items()}
    documents['auth']=pinned(root,'ROOT_ITSMIXER18_START_AUTHORIZATION.json',authpin)
    documents['original_plan']=pinned(Path(ORIGINAL),'MINIMUM12_PLAN.json',ORIGINAL_PLAN_SHA)
    freeze=safe_path(root,'ACTUAL_ITSMIXER18_FREEZE_RECEIPT.json');documents['freeze']=metadata(freeze)
    require(documents['freeze']['authorization_sha256']==authpin,'actual freeze authorization differs')
    validate_evidence(plan,recipe,pin,documents,time.time())
    probe=pinned(root,'ITSMIXER18_PROBE_CLAIM.json',documents['qa']['probe_claim_sha256'])
    require(probe['status']=='ACTUAL_ITSMIXER18_ONE_PROBE_ATTEMPT_CLAIM'
        and probe['manifest_sha256']==plan['source_manifest_sha256'] and probe['automatic_retry'] is False
        and probe['training_started'] is False and probe['utc_epoch']<=documents['qa']['utc_epoch'],
        'actual once synthetic claim and immutable source chronology')
    # Physical algorithm source pins are rechecked before NumPy/Torch imports
    # in prestart or worker. This reads library source bytes, never scientific values.
    first=documents['qa']['rows'][0]
    for row in documents['qa']['rows']:
        require(row['framework_source_paths']==first['framework_source_paths']
            and row['framework_source_sha256']==first['framework_source_sha256']
            and row['torch_version']==first['torch_version'],'actual six-case framework environment differs')
    for key,name in first['framework_source_paths'].items():
        path=Path(name)
        require(path.resolve()==path and all(not entry.is_symlink() for entry in (path,*path.parents))
            and sha(path)==first['framework_source_sha256'][key],'actual admitted framework source changed')
    verify_references(documents['references']);verify_qualified_pair(root,plan,documents)
    return root,plan,recipe,documents,sha(freeze)


def process_identity(pid):
    from itsmixer18_resources import _process_identity,_nvidia_query
    process=_process_identity(pid,sorted(set(_nvidia_query('pid','compute-apps'))));require(process['alive'],'owned process exited')
    return {'pid':pid,**{key:process[key] for key in ('uid','start_ticks','argv','parent_pid')}}


def verify_physical5(folder,files,rid,pin):
    require(set(files)==ALL5,'historical all5 exact membership')
    for name,row in files.items():
        path=safe_path(folder,name)
        require(type(row['bytes']) is int and row['bytes']>0 and path.stat().st_size==row['bytes']
            and sha(path)==row['sha256'],'historical all5 fresh SHA/bytes identity')
    receipt=metadata(folder/'COMPLETE_RECEIPT.json')
    require(receipt['status']=='COMPLETE_VALIDATION_ONLY' and receipt['run_id']==rid
        and receipt['plan_sha256']==pin and receipt['test_accessed'] is False,'historical score-free completion identity')
    for field,name in (('checkpoint_sha256','best.pt'),('config_sha256','config.json'),('history_sha256','history.json'),('validation_result_sha256','VALIDATION_RESULT.json')):
        require(receipt[field]==files[name]['sha256'],'historical complete receipt all4 byte links')
    return receipt


def verify_prior_vpnet6():
    root=Path(__file__).absolute().parent;prior=Path(ORIGINAL)/'vpnet_minimum6'
    proof=pinned(root,'VPNET6_PRIOR_NATIVE_PHYSICAL_CLOSURE.json',PRIOR_VP_NATIVE_SHA)
    pinned(root,'VPNET6_PRIOR_WHOLE12_PHYSICAL_CLOSURE.json',PRIOR_VP_WHOLE12_SHA)
    ids={'VPNet_'+d+'_H96' for d in DATASETS}
    require(proof['status']=='COMPLETE_VPNET6_OPAQUE_PHYSICAL_CLOSURE_NO_SCORES' and proof['child_root']==str(prior)
        and proof['plan_sha256']==PRIOR_VP_PLAN_SHA and set(proof['runs'])==ids and proof['scores_decoded'] is False
        and proof['test_accessed'] is False and proof['training_continuation_authorized'] is False,'prior VP6 whole physical provenance')
    plan=pinned(prior,'VPNET6_PLAN.json',PRIOR_VP_PLAN_SHA)
    state=metadata(prior/'VPNET6_STATE.json');boundary=pinned(prior,'VPNET6_TRAINING_BOUNDARY.json',proof['boundary_sha256'])
    require(state['status']==boundary['status']=='COMPLETE_VPNET6_VALIDATION_TRAINING'
        and set(state['done'])==set(boundary['done'])==ids and state['queue']==boundary['queue']==[]
        and state['active']==boundary['active']=={} and not state['failed'] and not boundary['failed']
        and not state['unverified_started_workers'] and not state['unverified_worker_exits']
        and state['admission_halt'] is boundary['admission_halt'] is None,'prior VP6 complete drained boundary')
    for name,h in plan['files'].items():require(sha(safe_path(prior,name))==h,'prior VP6 frozen science/source/prefix identity')
    for rid,row in proof['runs'].items():
        receipt=verify_physical5(prior/'results'/rid,row['files'],rid,PRIOR_VP_PLAN_SHA)
        require(receipt=={k:v for k,v in row['completion'].items() if k!='complete_receipt_sha256'},'prior VP6 complete metadata byte chain')
    return {'prior_vpnet6_whole_physical_closed':True,'qualification_required':False,'new_scores_accessed':False}


PRODUCER_PATHS=frozenset(ORIGINAL+'/'+child+('/' if child else '')+name for child,names in (
    ('',('minimum12_controller.py','trainval_runtime.py')),
    ('fact_freq_corrected4',('corrected4_controller.py','trainval_runtime.py')),
    ('rootpurge_retain18',('root18_controller.py','root18_trainval_runtime_pending.py')),
    ('rootpurge_recovery10',('root10_controller.py','root18_trainval_runtime_pending.py')),
    ('rootpurge_recovery9',('root9_controller.py','root18_trainval_runtime_pending.py')),
    ('vpnet_minimum6',('vpnet6_controller.py','vpnet6_runtime.py')),
    ('vpnet18_after_selection',('vpnet18_controller.py','vpnet18_runtime.py')),
    ('vpnet_remaining18',('vpnet18_controller.py','vpnet18_runtime.py')),
    ('itsmixer_minimum6',('itsmixer6_controller.py','itsmixer6_runtime.py')),
    ('itsmixer_remaining18',('itsmixer18_controller.py','itsmixer18_runtime.py'))) for name in names)


def current_authorized_helper():
    """Only this exact live PID may be the pinned probe/prestart/start helper."""
    names={'synthetic_cuda18.py','independent_prestart_itsmixer18.py','start_itsmixer18.py'}
    if not sys.argv or sys.argv[0] not in {CHILD+'/'+n for n in names}:return None
    root=Path(CHILD);name=Path(sys.argv[0]).name;args=sys.argv
    require(type(args) is list and all(type(x) is str and x for x in args),'current helper full argv vector')
    if name=='synthetic_cuda18.py':
        require(len(args)==4 and args[1:3]==['--probe-once','--manifest-sha256']
            and re.fullmatch('[0-9a-f]{64}',args[3]),'exact current once-probe CLI')
        record=pinned(root,'ITSMIXER18_SOURCE_MANIFEST.json',args[3])
        require(record['status']=='REVIEWED_ITSMIXER18_QUALIFIED_SOURCE_DEPLOYMENT_ONLY_NO_SCIENTIFIC_FREEZE'
            and record['child_root']==CHILD and record['qualified_pair_required'] is True
            and record['training_started'] is record['test_allowed'] is False,'current probe exact qualified manifest authority')
    else:
        expected='--review-start-once' if name=='independent_prestart_itsmixer18.py' else '--start-verified-once'
        require(len(args)==(6 if name=='independent_prestart_itsmixer18.py' else 10)
            and args[1:3]==[expected,'--plan-sha256'] and args[4]=='--authorization-sha256'
            and all(re.fullmatch('[0-9a-f]{64}',args[i]) for i in (3,5)), 'exact current frozen helper CLI')
        record=pinned(root,'ITSMIXER18_PLAN.json',args[3])
        auth=pinned(root,'ROOT_ITSMIXER18_START_AUTHORIZATION.json',args[5])
        require(record['status']=='ROOT_FROZEN_ITSMIXER18_VALIDATION_ONLY' and record['child_root']==CHILD
            and auth['status']=='ROOT_AUTHORIZED_ITSMIXER18_SUBJECT_TO_FRESH_INDEPENDENT_PRESTART'
            and auth['plan_sha256']==args[3] and auth['files']==record['files']
            and auth['automatic_retry'] is auth['test_allowed'] is False and auth['controller_one_attempt_allowed'] is True,
            'current helper exact frozen ROOT authority')
        if name=='start_itsmixer18.py':
            require(args[6]=='--prestart-name' and args[8]=='--prestart-sha256'
                and re.fullmatch('ACTUAL_INDEPENDENT_ITSMIXER18_PRESTART_[0-9a-f]{32}\\.json',args[7])
                and re.fullmatch('[0-9a-f]{64}',args[9]),'exact current verified-start prestart CLI')
            pre=pinned(root,args[7],args[9])
            require(pre['status']=='PASS_ACTUAL_INDEPENDENT_ITSMIXER18_PREFIX_RESOURCE_PRESTART'
                and pre['plan_sha256']==args[3] and pre['authorization_sha256']==args[5]
                and pre['files']==record['files'],'current start same pinned native prestart')
    require(record['protocol_sha256']==PROTOCOL_SHA and record['recipe_sha256']==RECIPE_SHA
        and record['root_pre_score_policy_sha256']==ROOT_POLICY_SHA
        and sha(safe_path(root,name))==record['files'][name], 'current helper byte-pinned scientific/source authority')
    row=process_identity(os.getpid())
    require(row['pid']==os.getpid() and row['uid']==0 and type(row['start_ticks']) is int and row['start_ticks']>0
        and row['argv']==['/root/miniconda3/bin/python','-u',*args], 'current helper exact live UID/PID/ticks/fullargv')
    return row


def global_census():
    """Raw argv census of all cohorts, allowing only this exact owned authority."""
    ours={CHILD+'/itsmixer18_controller.py',CHILD+'/itsmixer18_runtime.py'}
    expected={};root=Path(CHILD)
    if (root/'ITSMIXER18_ACTUAL_START.json').exists():
        start=metadata(root/'ITSMIXER18_ACTUAL_START.json')
        identity=verify_controller_identity(start,start['plan_sha256'],start['authorization_sha256'])
        expected[identity['pid']]=identity
        if (root/'ITSMIXER18_STATE.json').exists():
            state=metadata(root/'ITSMIXER18_STATE.json')
            require(state['plan_sha256']==start['plan_sha256'] and state['authorization_sha256']==start['authorization_sha256']
                and state['actual_start_sha256']==sha(root/'ITSMIXER18_ACTUAL_START.json'),'global census current owned authority')
            for rid,row in state['active'].items():
                from itsmixer18_resources import validate_worker_argv
                validate_worker_argv(row['argv'],rid)
                require(row['controller_pid']==identity['pid'] and row['argv'][row['argv'].index('--plan-sha256')+1]==start['plan_sha256']
                    and row['argv'][row['argv'].index('--authorization-sha256')+1]==start['authorization_sha256'],'global owned worker authority')
                expected[row['pid']]=row
    helper=current_authorized_helper()
    if helper is not None:expected[helper['pid']]=helper;ours.add(sys.argv[0])
    forbidden={x.encode() for x in PRODUCER_PATHS};prefix=ORIGINAL.encode()+b'/';hits=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try:raw=(proc/'cmdline').read_bytes()
        except (FileNotFoundError,ProcessLookupError):continue
        paths={x for x in raw.split(b'\0') if x and (x in forbidden or (x.startswith(prefix) and x.endswith(b'.py')))}
        if not paths:continue
        require(paths<={x.encode() for x in ours},'other scientific controller/runtime blocks joint execution')
        pid=int(proc.name);require(pid in expected,'unverified or unowned ITSMixer controller/runtime blocks admission')
        current=process_identity(pid);row=expected[pid]
        require(current['uid']==0 and current['start_ticks']==row['start_ticks'] and current['argv']==row['argv'],
            'global owned PID reused or argv changed')
        if 'controller_pid' in row:require(current['parent_pid']==row['controller_pid'],'global owned worker parent changed')
        hits.append(pid)
    return sorted(hits)


def gate_old_cohorts():
    """All old16 and failed VP6 physical identities; no score/history decode."""
    for folder,planname,pin,statename,boundname,status in (
      (Path(ORIGINAL),'MINIMUM12_PLAN.json',ORIGINAL_PLAN_SHA,'MINIMUM12_STATE.json','MINIMUM12_TRAINING_BOUNDARY.json','COMPLETE_VALIDATION_TRAINING'),
      (Path(CORRECTED),'CORRECTED4_PLAN.json',CORRECTED_PLAN_SHA,'CORRECTED4_STATE.json','CORRECTED4_TRAINING_BOUNDARY.json','COMPLETE_CORRECTED4_VALIDATION_TRAINING')):
        plan=pinned(folder,planname,pin);state=metadata(safe_path(folder,statename));bound=metadata(safe_path(folder,boundname))
        require(bound['status']==status and bound['plan_sha256']==pin and set(bound['done'])==set(plan['runs'])
          and not bound['failed'] and bound['test_accessed'] is False and not state['active'] and not state['queue']
          and not state['failed'] and set(state['done'])==set(plan['runs']),'historic cohort complete clean boundary')
        for name,h in plan['files'].items():require(sha(safe_path(folder,name))==h,'historic full frozen source/prefix identity')
    closure=pinned(Path(__file__).absolute().parent,'CANONICAL12_WHOLE_PHYSICAL_CLOSURE.json',CANONICAL_CLOSURE_SHA)
    for key,folder in (('original12',Path(ORIGINAL)),('corrected4',Path(CORRECTED))):
        for rid,row in closure[key]['runs'].items():verify_physical5(folder/'results'/rid,row['files'],rid,closure[key]['plan_sha256'])
    verify_prior_vpnet6()
    root=local_root();manifest=metadata(root/'ITSMIXER18_SOURCE_MANIFEST.json')
    docs={key:pinned(root,name,manifest[field]) for key,(name,field) in QUALIFICATION_DOCUMENTS.items()}
    verify_qualified_pair(root,manifest,docs);global_census()
    return {'status':'ACTUAL_OLD16_PRIOR_VP6_AND_QUALIFIED_ITSMIXER6_CLOSED_GLOBAL_PRODUCERS_IDLE','original_plan_sha256':ORIGINAL_PLAN_SHA,
      'corrected_plan_sha256':CORRECTED_PLAN_SHA,'prior_vpnet6_plan_sha256':PRIOR_VP_PLAN_SHA,
      'old_worker_or_controller_pids':[],'new_scores_accessed':False}


def gpu_idle_except(allowed_pid=None):
    from itsmixer18_resources import _nvidia_query
    pids=sorted(set(_nvidia_query('pid','compute-apps')));used=_nvidia_query('memory.used','gpu')
    require(len(used)==1 and (not pids or pids==[allowed_pid]) and used[0]<=22500,'PENDING unknown GPU compute/cap/device')
    return {'gpu_compute_pids':pids,'gpu_used_mib':used[0]}


def verify_controller_identity(start,pin,authpin):
    require(start['status']=='ACTUAL_ITSMIXER18_ONE_CONTROLLER_STARTED_NO_TEST'
        and start['plan_sha256']==pin and start['authorization_sha256']==authpin
        and start['automatic_retry'] is False and start['test_allowed'] is False,'owned controller actual attempt differs')
    identity=process_identity(start['pid'])
    require(identity['uid']==0 and identity['start_ticks']==start['start_ticks'] and identity['argv']==start['argv']
        and identity['argv']==['/root/miniconda3/bin/python','-u',CHILD+'/itsmixer18_controller.py','--execute-reviewed-controller','--plan-sha256',pin,'--authorization-sha256',authpin,'--prestart-sha256',start['prestart_sha256'],'--prestart-name',start['prestart_name']],'controller UID/starttick/argv differs')
    return identity


def validate_initial_prestart(pre,plan,recipe,documents,pin,authpin,freezepin,now):
    require(pre['authorization_sha256']==authpin and pre['freeze_sha256']==freezepin
        and pre['candidate_run_id'] is None and pre['controller_identity'] is None
        and not pre['owned'] and not pre['active_configs'],'initial GPU-idle independent prestart differs')
    from itsmixer18_resources import validate_resource_admission
    validate_resource_admission(pre['resource_telemetry'],{},None)
    require(not pre['resource_telemetry']['gpu_compute_pids'],'initial GPU must be empty')
    rid=sorted(RUN_IDS)[0]
    virtual={'status':'ROOT_ITSMIXER18_ONE_RUN_CAPABILITY_AFTER_FRESH_INDEPENDENT_PRESTART','run_id':rid,'plan_sha256':pin,
        'run_config_sha256':config_sha(plan['runs'][rid]),'automatic_retry':False,'test_allowed':False,
        'utc_epoch':pre['utc_epoch'],'expires_after_seconds':120}
    validate_evidence(plan,recipe,pin,documents,now,pre,virtual,rid)

def verify_qualified_pair(root,record,documents):
    from itsmixer18_qualification import validate_documents,verify_opaque_audit_evidence,verify_h96_physical
    documents['references']=pinned(root,'ROOTPURGE_H96_REFERENCES.json',RETAINED_REFERENCES_SHA)
    documents['original_plan']=pinned(Path(ORIGINAL),'MINIMUM12_PLAN.json',ORIGINAL_PLAN_SHA)
    fields={field:record[field] for _,field in QUALIFICATION_DOCUMENTS.values()}
    validate_documents(documents,fields)
    peer=pinned(root,'INDEPENDENT_ITSMIXER18_SCIENCE_SOURCE_REVIEW.json',INDEPENDENT_SCIENCE_REVIEW_SHA)
    require(peer['status']=='PASS_INDEPENDENT_ITSMIXER_REMAINING18_SCIENCE_SOURCE_REVIEW'
        and peer['source_manifest_sha256']==ADAPTER_PACKAGE_SHA and peer['snapshot']['snapshot_sha256']==SCIENCE_SNAPSHOT_SHA
        and peer['new_scores_accessed'] is False and peer['test_accessed'] is False,'different exact corresponding-horizon scientific peer review')
    verify_opaque_audit_evidence(root,record['files'],documents)
    verify_h96_physical(root,documents,metadata,safe_path)
