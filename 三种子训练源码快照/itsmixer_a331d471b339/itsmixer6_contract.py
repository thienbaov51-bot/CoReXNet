"""Inert ITSMixer6 production metadata contract; no scientific data decoders."""
import hashlib, json, math, re
from pathlib import Path, PurePosixPath
from itsmixer6_protocol import DATASETS, require, validate_config, encoded

ORIGINAL='/root/autodl-tmp/paper5/models2026_pilot_20261011'
CORRECTED=ORIGINAL+'/fact_freq_corrected4'
CHILD=ORIGINAL+'/itsmixer_minimum6'
ORIGINAL_PLAN_SHA='16a9dacee118f30b966489e0e17ad90274cfd56828ceaffb0c612b6fca9144ed'
CORRECTED_PLAN_SHA='5bc7c609686333cd1fcab53cba39497e6e7f490244581db716c1ddefc111cc5d'
ADAPTER_PACKAGE_SHA='e8fe24e644b72be8b90c8df70b3f113d7480b76c34e126e1036933d434bfc100'
SCIENCE_SNAPSHOT_SHA='a178c9760b5d57c0c80f3a02add9cc01074ec22cf99cbcc02d44528e63996c7e'
ROOT_SCIENCE_REVIEW_SHA='ae654b0feb337a4a051cf27c9bfcc1832b038f282cec77262c5660d8098b77ba'
INDEPENDENT_SCIENCE_REVIEW_SHA='0ba5ecaaa0c3080703d2231408e48c5cfdf3b8470b4101eacd68181c048e1e29'
HISTORICAL_NATIVE_PREFIX_PROOF_SHA='8a89bfca4ea85f52c0006a750dd5757547f038eff4ea86ce749ea8386e05ea4f'
PROTOCOL_SHA='239f52087edfd9890df9485e8e1c396f2bba30f6361ecb6af862f82903a81ee8'
RECIPE_SHA='20fe4ad2a0db147127056e24298bb874e3e1c882d35b22259c41a46aff7a4683'
KERNEL_SHA='183a32d00c3177f078adf89fdc40fd26d01f16fda14779a74c26dbb510a3413a'
MODEL_SHA='17e2b015e82ac6384d3d34e0f686573c68a4a4e72a75edf1d9ae21f5c1ce565f'
ROOT_POLICY_SHA='81b4abb79f6130cf5bf215a61091d808e3a7fef6cb3b041731b0f4e9a4a9408c'
POLICY_NAME='ROOT_ITSMIXER_PROSPECTIVE_REPLACEMENT_PRE_SCORE_PROTOCOL_ADOPTION.json'
REPLACEMENT_SHA='605f030f852fbf7b04fdae610c560e937440b8af8b692463b5a144f1735b6e04'
RETAINED_REFERENCES_SHA='06fa1a3584f45f8cdc11c9c75d140f4fc3f1bf2950d00a7b11576f4c573fa46d'
CANONICAL_CLOSURE_SHA='b7417ae45c588c32522b6030928e5fb381cab4ae5da09abc9b5a46c23bda7c1a'
PRIOR_VP_NATIVE_SHA='51c2ee3e7a80d7f822832a2d71bd3590681051dcbb786e49f23c5ab88a4effdd'
PRIOR_VP_WHOLE12_SHA='8469a096281207c81de6c8291ed8f1569ed32d62577bb1a820a2397433a5350f'
PRIOR_VP_PLAN_SHA='a6d71010852d1aa1b8801db862fe60addcd264712dd7bcb8e9918cfc4dab35d1'
RUN_IDS=frozenset('ITSMixer_'+d+'_H96' for d in DATASETS)
RESOURCE_POLICY={'max_workers':8,'gpu_budget_mib':22500,'active_disk_cap_bytes':4<<30,
 'actual_disk_floor_bytes':1536<<20,'unknown_gpu_process_blocks_admission':True,
 'owned_membership_change_only_defers':True,'joint_other_scientific_cohort_allowed':False}
MEASURED=frozenset(('parameter_count','peak_gpu_bytes','peak_gpu_reserved_bytes','reserved_gpu_mib','reserved_disk_bytes'))
SOURCE_FILES=('ITSMixer_ECL96_author_exact.py','itsmixer6_science_kernel.py','itsmixer6_protocol.py',
 'ITSMIXER_MINIMUM6_CONFIG_RECIPE_PENDING.json','ITSMIXER_MINIMUM6_SCIENCE_PROTOCOL_PENDING.json',
 'ITSMIXER6_SCIENCE_SOURCE_PACKAGE_MANIFEST_PENDING.json','FUTURE_NATIVE_ITSMIXER6_OUTPUT_CONTRACT_PENDING.json',
 'AUTHOR_MODEL_EXACT_AST_PROOF.json','SCIENTIFIC_SOURCE_ADAPTATION_PROOF_PENDING.json','LICENSE.ITSMixer',
 'ROOT_ITSMIXER6_SCIENCE_SOURCE_REVIEW.json','INDEPENDENT_ITSMIXER6_SCIENCE_SOURCE_REVIEW.json',
 'HISTORICAL_NATIVE_WHOLE_PREFIX_26973449_PROOF.json',
 'ITSMIXER6_PRODUCTION_NATIVE_OUTPUT_CONTRACT_PENDING.json',
 'itsmixer6_contract.py','itsmixer6_retained_contract.py','itsmixer6_common.py','itsmixer6_resources.py',
 'itsmixer6_runtime_guard.py','itsmixer6_completion.py','itsmixer6_runtime.py','itsmixer6_controller.py','synthetic_cuda6.py',
 'independent_prestart_itsmixer6.py','start_itsmixer6.py',POLICY_NAME,'ROOT_VPNET6_REPLACEMENT_RECORD.json',
 'ROOTPURGE_H96_REFERENCES.json','CANONICAL12_WHOLE_PHYSICAL_CLOSURE.json',
 'VPNET6_PRIOR_NATIVE_PHYSICAL_CLOSURE.json','VPNET6_PRIOR_WHOLE12_PHYSICAL_CLOSURE.json')
ALL5=frozenset(('best.pt','config.json','history.json','VALIDATION_RESULT.json','COMPLETE_RECEIPT.json'))

def safe_path(root,name):
    n=PurePosixPath(name)
    require(type(name) is str and n.parts and n.as_posix()==name and not n.is_absolute()
        and '..' not in n.parts and '\\' not in name and ':' not in name,'unsafe production relative path')
    p=Path(root)/name
    require(all(not q.is_symlink() for q in (p,*p.parents)),'production symlink forbidden')
    return p

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def metadata(path):
    from itsmixer6_retained_contract import metadata_bytes
    path=Path(path)
    require(path.suffix=='.json' and path.name not in ('history.json','VALIDATION_RESULT.json')
        and 'NUMERIC_AUDIT' not in path.name,'numeric result/history decoding forbidden in production metadata')
    if path.name==POLICY_NAME:
        # The fixed pre-score document contains a "metrics" list naming MSE
        # and MAE. Keep it opaque rather than relaxing the score-key decoder.
        require(sha(path)==ROOT_POLICY_SHA,'exact byte-only ROOT pre-score policy')
        return {'status':'OPAQUE_BYTE_PINNED_ROOT_PRE_SCORE_POLICY','sha256':ROOT_POLICY_SHA}
    return metadata_bytes(path.read_bytes())

def config_sha(config):return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_replacement(record,closure,references):
    # The historic FACT->VPNet record is retained Root6 provenance only.
    # ITSMixer authority is distinct ROOT_POLICY_SHA, not VP qualification.
    from itsmixer6_retained_contract import validate_retained_metadata
    validate_retained_metadata(record,closure,references)

def validate_plan(plan,recipe):
    require(plan['status']=='ROOT_FROZEN_ITSMIXER6_VALIDATION_ONLY' and plan['child_root']==CHILD
      and plan['original_root']==ORIGINAL and plan['original_plan_sha256']==ORIGINAL_PLAN_SHA
      and plan['adapter_package_sha256']==ADAPTER_PACKAGE_SHA and plan['protocol_sha256']==PROTOCOL_SHA
      and plan['recipe_sha256']==RECIPE_SHA and plan['root_pre_score_policy_sha256']==ROOT_POLICY_SHA
      and plan['replacement_sha256']==REPLACEMENT_SHA and plan['canonical_closure_sha256']==CANONICAL_CLOSURE_SHA
      and plan['retained_references_sha256']==RETAINED_REFERENCES_SHA
      and plan['prior_vpnet6_native_closure_sha256']==PRIOR_VP_NATIVE_SHA
      and plan['prior_vpnet6_whole12_closure_sha256']==PRIOR_VP_WHOLE12_SHA
      and type(plan['seed']) is int and plan['seed']==2026 and plan['length']==plan['horizon']==96
      and plan['test_allowed'] is False and plan['automatic_retry'] is False
      and plan['all_validation_windows'] is True and plan['TRAIN_drop_last_author_budget'] is True
      and plan['all_train_windows_consumed_each_epoch'] is False and plan['weights_copied'] is False
      and plan['RootPurge_retraining_allowed'] is False and plan['resource_policy']==RESOURCE_POLICY
      and set(plan['runs'])==RUN_IDS and set(plan['data_metadata'])==set(DATASETS),'immutable six-case production contract')
    require(recipe['root_pre_score_policy_sha256']==ROOT_POLICY_SHA and recipe['parent_plan_sha256']==ORIGINAL_PLAN_SHA
      and recipe['model_source_sha256']==MODEL_SHA and set(recipe['runs'])==RUN_IDS,'exact scientific recipe lineage')
    required=set(SOURCE_FILES)|{'data/'+d+'_trainval.npz' for d in DATASETS}|{'data/'+d+'_known_calendar.npy' for d in DATASETS}
    require(set(plan['files'])==required,'exact full source/prefix/calendar inventory')
    for name,pin in (('ITSMixer_ECL96_author_exact.py',MODEL_SHA),('itsmixer6_science_kernel.py',KERNEL_SHA),
      ('ITSMIXER_MINIMUM6_SCIENCE_PROTOCOL_PENDING.json',PROTOCOL_SHA),('ITSMIXER_MINIMUM6_CONFIG_RECIPE_PENDING.json',RECIPE_SHA),
      ('ITSMIXER6_SCIENCE_SOURCE_PACKAGE_MANIFEST_PENDING.json',ADAPTER_PACKAGE_SHA),(POLICY_NAME,ROOT_POLICY_SHA)):
        require(plan['files'][name]==pin,'unchanged scientific source/policy '+name)
    for rid,cfg in plan['runs'].items():
        require(set(cfg)==set(recipe['runs'][rid])|MEASURED,'only actual measured resource config fields')
        scientific={k:v for k,v in cfg.items() if k not in MEASURED}
        require(scientific==recipe['runs'][rid],'scientific recipe cannot change')
        meta=plan['data_metadata'][cfg['dataset']];validate_config(scientific,meta)
        for key in MEASURED:require(type(cfg[key]) is int and cfg[key]>0,'positive typed measured field '+key)
        require(cfg['reserved_gpu_mib']<=22500 and (64<<20)<=cfg['reserved_disk_bytes']<=(4<<30),'measured reserve limits')
        require(plan['files']['data/'+cfg['dataset']+'_trainval.npz']==cfg['prefix_sha256']==meta['prefix_sha256']
            and plan['files']['data/'+cfg['dataset']+'_known_calendar.npy']==cfg['calendar_sha256']==meta['calendar_sha256'],
            'native full prefix/calendar source SHA binding')

def review_snapshot(plan):
    return {'child_root':CHILD,'files':plan['files'],'recipe_sha256':RECIPE_SHA,'protocol_sha256':PROTOCOL_SHA,
     'adapter_package_sha256':ADAPTER_PACKAGE_SHA,'root_pre_score_policy_sha256':ROOT_POLICY_SHA,'qa_sha256':plan['qa_sha256'],
     'replacement_sha256':REPLACEMENT_SHA,'canonical_closure_sha256':CANONICAL_CLOSURE_SHA,
     'retained_references_sha256':RETAINED_REFERENCES_SHA,'prior_vpnet6_native_closure_sha256':PRIOR_VP_NATIVE_SHA,
     'prior_vpnet6_whole12_closure_sha256':PRIOR_VP_WHOLE12_SHA}

def validate_qa(qa,plan):
    require(qa['status']=='PASS_ALL6_ACTUAL_ORIGINAL_BATCH_ITSMIXER_SYNTHETIC_CUDA_ONLY'
      and qa['source_inventory']==plan['files'] and qa['recipe_sha256']==RECIPE_SHA
      and qa['manifest_sha256']==plan['source_manifest_sha256'] and qa['root_pre_score_policy_sha256']==ROOT_POLICY_SHA
      and qa['measurement_array_read'] is False and qa['research_scores_read'] is False
      and qa['training_started'] is False and qa['automatic_retry'] is False
      and len(qa['rows'])==6 and {r['run_id'] for r in qa['rows']}==RUN_IDS
      and re.fullmatch('[0-9a-f]{64}',qa['probe_claim_sha256'])
      and qa['probe_source_sha256']==plan['files']['synthetic_cuda6.py'],'actual original B256 six CUDA probes')
    old=qa['old_cohort_admission']
    require(old['status']=='ACTUAL_OLD16_COMPLETE_PRIOR_VP6_CLOSED_GLOBAL_PRODUCERS_IDLE'
       and not old['old_worker_or_controller_pids'] and not old['gpu_compute_pids']
       and old['new_scores_accessed'] is False,'fresh old/cohort global idle synthetic boundary')
    for row in qa['rows']:
        cfg=plan['runs'][row['run_id']]
        require(row['status']=='PASS' and row['config']==cfg and row['config_sha256']==config_sha(cfg)
         and row['optimizer_steps']==2 and row['precision']=='FP32_TF32' and row['synthetic_only'] is True
         and row['measurement_array_read'] is False and row['fullprefix_shape']==[cfg['prefix_rows'],cfg['channels']]
         and row['loss_policy']=='EXACT_AUTHOR_0.5_MSE_PLUS_0.5_L1_STANDARDIZED'
         and row['checkpoint_criterion']==cfg['checkpoint_criterion'] and row['batch_size']==256
         and row['calendar_consumed'] is False and row['validation_tail_windows']==cfg['validation_tail_windows'],
         'actual fixed math/batch/fullprefix/tail probe')
        for key in ('peak_gpu_bytes','peak_gpu_reserved_bytes','parameter_count'):
            require(type(row[key]) is int and row[key]>0 and row[key]==cfg[key],'actual measured resource '+key)
        require(row['torch_version'] and row['numpy_version'] and row['device_name']
          and set(row['framework_source_sha256'])==set(row['framework_source_paths'])=={'AdamW','OneCycleLR','LRScheduler'},'actual framework identity')
        for key,path in row['framework_source_paths'].items():
            require(re.fullmatch('[0-9a-f]{64}',row['framework_source_sha256'][key])
                and path.startswith('/root/miniconda3/lib/') and PurePosixPath(path).name==('adamw.py' if key=='AdamW' else 'lr_scheduler.py'),
                'actual admitted framework source path/hash')
        require(row['optimizer_parameter_groups_verified'] is True and row['onecycle_initial_last_epoch']==0
            and math.isclose(row['initial_learning_rate'],.00004,rel_tol=1e-12) and row['initial_betas']==[.95,.999]
            and row['weight_decay']==.01 and row['eps']==1e-8 and row['amsgrad'] is False
            and row['after_epoch1_learning_rate']==.001 and row['after_epoch2_learning_rate']==.001
            and row['type3_scheduler_step_calls']==0,'actual AdamW/OneCycle constructor and type3')
        gpu=max(1200,(max(row['peak_gpu_bytes'],row['peak_gpu_reserved_bytes'])+(1<<20)-1)//(1<<20)+512)
        disk=max(64<<20,row['parameter_count']*12+(16<<20))
        require(cfg['reserved_gpu_mib']==row['reserved_gpu_mib']==gpu and cfg['reserved_disk_bytes']==row['reserved_disk_bytes']==disk,'full original budget reserve formula')

def validate_evidence(plan,recipe,pin,documents,now,pre=None,capability=None,rid=None):
    validate_plan(plan,recipe);validate_replacement(documents['replacement'],documents['closure'],documents['references'])
    qa,auth,freeze=documents['qa'],documents['auth'],documents['freeze'];validate_qa(qa,plan)
    snapshot=review_snapshot(plan)
    for role in ('root','independent'):
        review=documents[role+'_review']
        require(review['status']=='PASS_'+role.upper()+'_ITSMIXER6_SOURCE_RESOURCE_REVIEW' and review['snapshot']==snapshot
            and review['new_scores_accessed'] is False and review['test_accessed'] is False,'independent current actual source/resource review')
    fields=('qa_sha256','replacement_sha256','canonical_closure_sha256','retained_references_sha256',
        'root_pre_score_policy_sha256','prior_vpnet6_native_closure_sha256','prior_vpnet6_whole12_closure_sha256',
        'root_source_resource_review_sha256','independent_source_resource_review_sha256')
    require(auth['status']=='ROOT_AUTHORIZED_ITSMIXER6_SUBJECT_TO_FRESH_INDEPENDENT_PRESTART'
        and auth['plan_sha256']==pin and auth['files']==plan['files'] and auth['run_ids']==sorted(RUN_IDS)
        and auth['controller_one_attempt_allowed'] is True and auth['automatic_retry'] is False and auth['test_allowed'] is False
        and all(auth[k]==plan[k] for k in fields),'distinct explicit ROOT scientific authorization')
    require(freeze['status']=='ACTUAL_ITSMIXER6_SOURCE_RESOURCE_PLAN_FROZEN_NO_START' and freeze['plan_sha256']==pin
        and freeze['qa_sha256']==plan['qa_sha256'] and freeze['training_started'] is False and freeze['scores_accessed'] is False
        and qa['utc_epoch']<=freeze['utc_epoch'],'actual freeze/QA chronology')
    if pre is None and capability is None:return
    require(pre is not None and capability is not None and rid in RUN_IDS,'complete fresh perrun authority')
    require(pre['status']=='PASS_ACTUAL_INDEPENDENT_ITSMIXER6_PREFIX_RESOURCE_PRESTART' and pre['plan_sha256']==pin
        and pre['files']==plan['files'] and pre['qa_sha256']==plan['qa_sha256'] and pre['root_pre_score_policy_sha256']==ROOT_POLICY_SHA
        and pre['auditor_source_sha256']==plan['files']['independent_prestart_itsmixer6.py']
        and pre['test_scores_accessed'] is False and pre['new_validation_scores_accessed'] is False
        and pre['training_started_by_auditor'] is False and pre['expires_after_seconds']==120
        and 0<=now-pre['utc_epoch']<=120 and freeze['utc_epoch']<=pre['utc_epoch'],'fresh actual bitwise prefix prestart')
    require(set(pre['datasets'])==set(DATASETS) and pre['prefix_bitwise_elements_checked']==sum(m['prefix_rows']*m['channels'] for m in plan['data_metadata'].values()),'all full prefix elements checked')
    for ds,row in pre['datasets'].items():
        m=plan['data_metadata'][ds]
        require(row=={k:m[k] for k in ('prefix_sha256','scaler_mean_sha256','scaler_std_sha256','prefix_rows','channels')}|{'prefix_bitwise_elements':m['prefix_rows']*m['channels']},'exact prestart original split/scaler')
    require(capability['status']=='ROOT_ITSMIXER6_ONE_RUN_CAPABILITY_AFTER_FRESH_INDEPENDENT_PRESTART'
        and capability['run_id']==rid and capability['plan_sha256']==pin and capability['run_config_sha256']==config_sha(plan['runs'][rid])
        and capability['automatic_retry'] is False and capability['test_allowed'] is False
        and capability['expires_after_seconds']==120 and 0<=now-capability['utc_epoch']<=120
        and pre['utc_epoch']<=capability['utc_epoch']<=pre['utc_epoch']+120,'one fresh capability identity')
