"""Inert ITSMixer18 production metadata contract; no scientific data decoders."""
import hashlib, json, math, re
from pathlib import Path, PurePosixPath
from itsmixer18_protocol import DATASETS, HORIZONS, VARIANTS, require, validate_config, encoded
from itsmixer5_qualification import QUALIFICATION_DOCUMENTS, QUALIFICATION_FILES, validate_documents


ORIGINAL='/root/autodl-tmp/paper5/models2026_pilot_20261011'
CORRECTED=ORIGINAL+'/fact_freq_corrected4'
CHILD=ORIGINAL+'/itsmixer_recovery5_serial'
ORIGINAL_PLAN_SHA='16a9dacee118f30b966489e0e17ad90274cfd56828ceaffb0c612b6fca9144ed'
CORRECTED_PLAN_SHA='5bc7c609686333cd1fcab53cba39497e6e7f490244581db716c1ddefc111cc5d'
ADAPTER_PACKAGE_SHA='06ffd9dab47faf099fc8f4229992f00f8f229132033cc0eb87baa2ad87020008'
SCIENCE_SNAPSHOT_SHA='cacf0f74440f2e1d0f367aeba67b0745c9309b6c42f9d5d64a780c6b23bb449a'
INDEPENDENT_SCIENCE_REVIEW_SHA='d9d5d7ff7db8a8792f223039827bd59107262d67f43b9f418c04313dda0aa686'
HISTORICAL_NATIVE_PREFIX_PROOF_SHA='8a89bfca4ea85f52c0006a750dd5757547f038eff4ea86ce749ea8386e05ea4f'
PROTOCOL_SHA='d225fd4c412e8a31502f673b53483284227cb9ad6e5a6defdca8aae96c54aed1'
RECIPE_SHA='8aaf21f06ada391cd21e9670b87c62c1cfd4f280a06895f4a0fc621673e649f2'
KERNEL_SHA='b5005325b1fe10ec4ba3547e8c45201fe91e72864846423a6166f21606c5975e'
ROOT_POLICY_SHA='3f42cd092488a9de60c487cc5523c0bd380248c9a3aa5956faf821bba643ede6'
POLICY_NAME='ROOT_ITSMIXER_REMAINING18_PRE_SCORE_HORIZON_BUDGET_ADOPTION_20261012.json'
REPLACEMENT_SHA='605f030f852fbf7b04fdae610c560e937440b8af8b692463b5a144f1735b6e04'
RETAINED_REFERENCES_SHA='06fa1a3584f45f8cdc11c9c75d140f4fc3f1bf2950d00a7b11576f4c573fa46d'
CANONICAL_CLOSURE_SHA='b7417ae45c588c32522b6030928e5fb381cab4ae5da09abc9b5a46c23bda7c1a'
PRIOR_VP_NATIVE_SHA='51c2ee3e7a80d7f822832a2d71bd3590681051dcbb786e49f23c5ab88a4effdd'
PRIOR_VP_WHOLE12_SHA='8469a096281207c81de6c8291ed8f1569ed32d62577bb1a820a2397433a5350f'
PRIOR_VP_PLAN_SHA='a6d71010852d1aa1b8801db862fe60addcd264712dd7bcb8e9918cfc4dab35d1'
ALL18_RUN_IDS=frozenset('ITSMixer_'+d+'_H'+str(h) for d in DATASETS for h in HORIZONS)
RETAINED_OLD5_IDS=frozenset(('ITSMixer_AEMO5_H192', 'ITSMixer_AEMO5_H336', 'ITSMixer_AEMO5_H720', 'ITSMixer_Electricity_H192', 'ITSMixer_Electricity_H336'))
RETAINED_S3_EIGHT_IDS=frozenset(('ITSMixer_Electricity_H720','ITSMixer_PEMS03_H192','ITSMixer_PEMS03_H720','ITSMixer_PEMS04_H192','ITSMixer_PEMS04_H336','ITSMixer_PEMS04_H720','ITSMixer_PEMS08_H192','ITSMixer_PEMS08_H336'))
RUN_IDS=ALL18_RUN_IDS-RETAINED_OLD5_IDS-RETAINED_S3_EIGHT_IDS
RESOURCE_POLICY={'max_workers':1,'gpu_budget_mib':22500,'active_disk_cap_bytes':4<<30,
 'actual_disk_floor_bytes':1536<<20,'unknown_gpu_process_blocks_admission':True,
 'owned_membership_change_only_defers':True,'joint_other_scientific_cohort_allowed':False}
MEASURED=frozenset(('parameter_count','peak_gpu_bytes','peak_gpu_reserved_bytes','reserved_gpu_mib','reserved_disk_bytes'))
STATIC_SOURCE_FILES=('AUTHOR_EXACT_VARIANT_AND_KERNEL_ADAPTATION_PROOF.json', 'AUTHOR_MODEL_EXACT_AST_PROOF.json', 'AUTHOR_PRE_DELIVERY_MANIFEST_4FCF_SUPERSEDED_PROOF_NEWLINE_ONLY.json', 'AUTHOR_PRE_DELIVERY_MANIFEST_CA83_SUPERSEDED_JSON_ORDER_ONLY.json', 'AUTHOR_REMAINING18_PRIVATE_SOURCE_QA.json', 'finalize_remaining18_source_only.py', 'FUTURE_NATIVE_ITSMIXER6_OUTPUT_CONTRACT_PENDING.json', 'H96_SCIENCE_ORIGINAL_PACKAGE_MANIFEST.json', 'H96_TO_REMAINING18_EXACT_SCIENCE_KERNEL.diff', 'itsmixer18_protocol.py', 'itsmixer18_runtime_pending.py', 'itsmixer18_science_kernel.py', 'itsmixer6_protocol.py', 'itsmixer6_science_kernel.py', 'ITSMixer_ECL192_author_exact.py', 'ITSMixer_ECL336_author_exact.py', 'ITSMixer_ECL720_author_exact.py', 'ITSMixer_ECL96_author_exact.py', 'ITSMIXER_MINIMUM6_CONFIG_RECIPE_PENDING.json', 'ITSMIXER_MINIMUM6_SCIENCE_PROTOCOL_PENDING.json', 'ITSMIXER_REMAINING18_CONFIG_RECIPE_PENDING.json', 'ITSMIXER_REMAINING18_SCIENCE_PROTOCOL_PENDING.json', 'LICENSE.ITSMixer', 'prepare_remaining18_source_only.py', 'PRIOR_NEGATIVE_ATTEMPT_OPAQUE_PROVENANCE_PENDING.json', 'provenance/Experiments__electricity-192__seq-len-512.py.GITHUB_BLOB.json', 'provenance/Experiments__electricity-336__seq-len-512.py.GITHUB_BLOB.json', 'provenance/Experiments__electricity-720__seq-len-512.py.GITHUB_BLOB.json', 'qa_remaining18_source_only.py', 'README.md', 'ROOT_H96_PRE_SCORE_POLICY_ORIGINAL.json', 'ROOT_ITSMIXER_REMAINING18_PRE_SCORE_HORIZON_BUDGET_ADOPTION_20261012.json', 'SCIENTIFIC_SOURCE_ADAPTATION_PROOF_PENDING.json', 'SUPERSEDED_COMMON15_PRE_CODE_SOURCE_PROPOSAL_ONLY.json', 'ITSMIXER_REMAINING18_SCIENCE_SOURCE_AUTHOR_MANIFEST.json', 'ROOT_VPNET6_REPLACEMENT_RECORD.json', 'ROOTPURGE_H96_REFERENCES.json', 'CANONICAL12_WHOLE_PHYSICAL_CLOSURE.json', 'VPNET6_PRIOR_NATIVE_PHYSICAL_CLOSURE.json', 'VPNET6_PRIOR_WHOLE12_PHYSICAL_CLOSURE.json', 'HISTORICAL_NATIVE_WHOLE_PREFIX_26973449_PROOF.json', 'INDEPENDENT_ITSMIXER18_SCIENCE_SOURCE_REVIEW.json', 'ITSMIXER5_PRODUCTION_NATIVE_OUTPUT_CONTRACT_PENDING.json', 'itsmixer5_contract.py', 'itsmixer5_qualification.py', 'itsmixer5_retained_contract.py', 'itsmixer5_common.py', 'itsmixer5_resources.py', 'itsmixer5_runtime_guard.py', 'itsmixer5_completion.py', 'itsmixer5_runtime.py', 'itsmixer5_controller.py', 'project_original_cuda18.py', 'independent_prestart_itsmixer5.py', 'start_itsmixer5.py')
STATIC_SOURCE_FILES=STATIC_SOURCE_FILES+('itsmixer5_upstream.py','collect_itsmixer_s3_recovery5_rootgrid3.py','ORIGINAL_ACTUAL_CUDA18_RESOURCE_RECEIPT.json')
STATIC_SOURCE_FILES=STATIC_SOURCE_FILES+('itsmixer13_isolated_ancestry.py', 'PREVIOUS_FROZEN13_NO_SCIENTIFIC_ATTEMPTS_OPAQUE_PROOF.json', 'PREVIOUS_FROZEN13_PRESTART_REJECTION_RECORD.json', 'PREVIOUS_FROZEN13_SOURCE_MANIFEST.json', 'PREVIOUS_FROZEN13_PLAN.json', 'PREVIOUS_FROZEN13_START_AUTHORIZATION.json', 'PREVIOUS_FROZEN13_FREEZE_RECEIPT.json', 'PREVIOUS_FROZEN13_RESOURCE_PROJECTION.json')
STATIC_SOURCE_FILES=STATIC_SOURCE_FILES+('PREVIOUS_ISOLATED13_SOURCE_MANIFEST.json', 'PREVIOUS_ISOLATED13_DEPLOYMENT_RECEIPT.json', 'PREVIOUS_ISOLATED13_PREFIX_COPY_RECEIPT.json')
STATIC_SOURCE_FILES=STATIC_SOURCE_FILES+('PREVIOUS_ISOLATED13_NO_SCIENTIFIC_ATTEMPTS_OPAQUE_PROOF.json',)
STATIC_SOURCE_FILES=STATIC_SOURCE_FILES+('PREVIOUS_STARTED13_PREFRAMEWORK_FAILURE_PROOF.json','collect_itsmixer_s2_failed13.py','ITSMIXER_S2_FAILED13_NATIVE_PINS.json')
STATIC_SOURCE_FILES=STATIC_SOURCE_FILES+('collect_itsmixer18_recovery13.py',)
SOURCE_FILES=STATIC_SOURCE_FILES+QUALIFICATION_FILES
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
    from itsmixer5_retained_contract import metadata_bytes
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
    from itsmixer5_retained_contract import validate_retained_metadata
    validate_retained_metadata(record,closure,references)


def validate_plan(plan,recipe):
    require(plan['status']=='ROOT_FROZEN_ITSMIXER5_VALIDATION_ONLY' and plan['child_root']==CHILD
      and plan['original_root']==ORIGINAL and plan['original_plan_sha256']==ORIGINAL_PLAN_SHA
      and plan['adapter_package_sha256']==ADAPTER_PACKAGE_SHA and plan['protocol_sha256']==PROTOCOL_SHA
      and plan['recipe_sha256']==RECIPE_SHA and plan['root_pre_score_policy_sha256']==ROOT_POLICY_SHA
      and plan['replacement_sha256']==REPLACEMENT_SHA and plan['canonical_closure_sha256']==CANONICAL_CLOSURE_SHA
      and plan['retained_references_sha256']==RETAINED_REFERENCES_SHA
      and plan['prior_vpnet6_native_closure_sha256']==PRIOR_VP_NATIVE_SHA
      and plan['prior_vpnet6_whole12_closure_sha256']==PRIOR_VP_WHOLE12_SHA
      and type(plan['seed']) is int and plan['seed']==2026 and type(plan['length']) is int and plan['length']==96
      and plan['horizons']==[192,336,720] and all(type(h) is int for h in plan['horizons'])
      and plan['new_training_count']==5 and plan['retained_previous_s3_count']==8 and plan['retained_h96_count']==6 and plan['retained_old18_count']==5
      and plan['test_allowed'] is False and plan['automatic_retry'] is False
      and plan['all_validation_windows'] is True and plan['TRAIN_drop_last_author_budget'] is True
      and plan['all_train_windows_consumed_each_epoch'] is False and plan['weights_copied'] is False
      and plan['H96_retraining_allowed'] is False and plan['RootPurge_retraining_allowed'] is False
      and plan['resource_policy']==RESOURCE_POLICY and set(plan['runs'])==RUN_IDS
      and set(plan['data_metadata'])==set(DATASETS),'immutable qualified18 + retained6 production contract')
    require(re.fullmatch('[0-9a-f]{64}',plan['recovery_evidence_sha256']) is not None,
        'exact distinct ROOT-reviewed actual opaque recovery evidence required')
    require(plan['previous_s3_eight_complete_five_no_scientific_claims_proof_sha256']==plan['recovery_evidence_sha256'],
        'explicit genuine S3 retained8/no-science5 ancestry in frozen source')
    require(plan['previous_frozen13_noattempt_proof_sha256']==
        '11a26285c936af5c5dae3aea8b97988b5f000339015e272892b9e9762e9d27f0' and plan['files']['PREVIOUS_FROZEN13_NO_SCIENTIFIC_ATTEMPTS_OPAQUE_PROOF.json']==
        plan['previous_frozen13_noattempt_proof_sha256'],'previous frozen13 physical noattempt proof identity')
    require(plan['previous_source_only_isolated13_noattempt_proof_sha256']=='4762841c460d48ee0a20330e43df731fb5bb8dda84a79a006f4ed2d77b7855d1' and plan['files']['PREVIOUS_ISOLATED13_NO_SCIENTIFIC_ATTEMPTS_OPAQUE_PROOF.json']==plan['previous_source_only_isolated13_noattempt_proof_sha256'],'actual unused80 noattempt physical identity')
    require(plan['previous_started13_preframework_failure_proof_sha256']=='11d420fb57555ce738cd9a60dd41e09afef749add996dbc8f575ccb150d54b74' and plan['files']['PREVIOUS_STARTED13_PREFRAMEWORK_FAILURE_PROOF.json']==plan['previous_started13_preframework_failure_proof_sha256'],'genuine prior one-Popen/zero-scientific-claim ancestry')
    require(recipe['root_pre_score_policy_sha256']==ROOT_POLICY_SHA and recipe['parent_plan_sha256']==ORIGINAL_PLAN_SHA
      and recipe['original_h96_science_sha256']=='e8fe24e644b72be8b90c8df70b3f113d7480b76c34e126e1036933d434bfc100'
      and set(recipe['runs'])==ALL18_RUN_IDS,'exact per-H scientific recipe lineage')
    required=set(SOURCE_FILES)|{'data/'+d+'_trainval.npz' for d in DATASETS}|{'data/'+d+'_known_calendar.npy' for d in DATASETS}
    require(set(plan['files'])==required,'exact complete source/qualification/prefix/calendar inventory')
    for name,pin in (('itsmixer18_science_kernel.py',KERNEL_SHA),
      ('ITSMIXER_REMAINING18_SCIENCE_PROTOCOL_PENDING.json',PROTOCOL_SHA),('ITSMIXER_REMAINING18_CONFIG_RECIPE_PENDING.json',RECIPE_SHA),
      ('ITSMIXER_REMAINING18_SCIENCE_SOURCE_AUTHOR_MANIFEST.json',ADAPTER_PACKAGE_SHA),(POLICY_NAME,ROOT_POLICY_SHA),
      ('INDEPENDENT_ITSMIXER18_SCIENCE_SOURCE_REVIEW.json',INDEPENDENT_SCIENCE_REVIEW_SHA)):
        require(plan['files'][name]==pin,'unchanged corresponding scientific source/policy '+name)
    for v in VARIANTS.values():require(plan['files'][v['source_file']]==v['source_sha256'],'exact author per-H model byte pins')
    for _,(name,field) in QUALIFICATION_DOCUMENTS.items():
        require(re.fullmatch('[0-9a-f]{64}',plan[field]) and plan['files'][name]==plan[field],'actual qualified metadata/source pin '+field)
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
     'previous_s3_eight_complete_five_no_scientific_claims_proof_sha256':plan['recovery_evidence_sha256'],
     'previous_source_only_isolated13_noattempt_proof_sha256':plan['previous_source_only_isolated13_noattempt_proof_sha256'],
     'previous_frozen13_noattempt_proof_sha256':plan['previous_frozen13_noattempt_proof_sha256'],
     'adapter_package_sha256':ADAPTER_PACKAGE_SHA,'root_pre_score_policy_sha256':ROOT_POLICY_SHA,'qa_sha256':plan['qa_sha256'],'recovery_evidence_sha256':plan['recovery_evidence_sha256'],
     'replacement_sha256':REPLACEMENT_SHA,'canonical_closure_sha256':CANONICAL_CLOSURE_SHA,
     'retained_references_sha256':RETAINED_REFERENCES_SHA,'prior_vpnet6_native_closure_sha256':PRIOR_VP_NATIVE_SHA,
     'prior_vpnet6_whole12_closure_sha256':PRIOR_VP_WHOLE12_SHA,
     **{field:plan[field] for _,field in QUALIFICATION_DOCUMENTS.values()}}

def validate_qa(qa,plan):
    from itsmixer5_upstream import validate_projection
    validate_projection(qa,plan)

def validate_evidence(plan,recipe,pin,documents,now,pre=None,capability=None,rid=None):
    validate_plan(plan,recipe);validate_replacement(documents['replacement'],documents['closure'],documents['references'])
    import collect_itsmixer_s3_recovery5_rootgrid3 as collector
    collector.verify_evidence_shape(documents['recovery'])
    require(set(documents['recovery']['runs'])==RUN_IDS and set(documents['recovery']['retained_original5_references'])==RETAINED_OLD5_IDS and set(documents['recovery']['retained_original8_references'])==RETAINED_S3_EIGHT_IDS,
        'exact original5/S3eight preserved and serial five first scientific attempts')
    require(documents['recovery']['runs']==plan['runs'],'projected config must equal unchanged original frozen13')
    validate_documents(documents,{field:plan[field] for _,field in QUALIFICATION_DOCUMENTS.values()})
    qa,auth,freeze=documents['qa'],documents['auth'],documents['freeze'];validate_qa(qa,plan)
    snapshot=review_snapshot(plan)
    for role in ('root','independent'):
        review=documents[role+'_review']
        require(review['status']=='PASS_'+role.upper()+'_ITSMIXER5_SOURCE_RESOURCE_REVIEW' and review['snapshot']==snapshot
            and review['new_scores_accessed'] is False and review['test_accessed'] is False,'independent current actual source/resource review')
    fields=('previous_s3_eight_complete_five_no_scientific_claims_proof_sha256','previous_source_only_isolated13_noattempt_proof_sha256','previous_frozen13_noattempt_proof_sha256','recovery_evidence_sha256','qa_sha256','replacement_sha256','canonical_closure_sha256','retained_references_sha256',
        'root_pre_score_policy_sha256','prior_vpnet6_native_closure_sha256','prior_vpnet6_whole12_closure_sha256',
        'root_source_resource_review_sha256','independent_source_resource_review_sha256',*tuple(field for _,field in QUALIFICATION_DOCUMENTS.values()))
    require(auth['status']=='ROOT_AUTHORIZED_ITSMIXER5_SUBJECT_TO_FRESH_INDEPENDENT_PRESTART'
        and auth['plan_sha256']==pin and auth['files']==plan['files'] and auth['run_ids']==sorted(RUN_IDS)
        and auth['controller_one_attempt_allowed'] is True and auth['automatic_retry'] is False and auth['test_allowed'] is False
        and all(auth[k]==plan[k] for k in fields),'distinct explicit ROOT scientific authorization')
    require(freeze['status']=='ACTUAL_ITSMIXER5_SOURCE_RESOURCE_PLAN_FROZEN_NO_START' and freeze['plan_sha256']==pin
        and freeze['qa_sha256']==plan['qa_sha256'] and freeze['training_started'] is False and freeze['scores_accessed'] is False
        and qa['utc_epoch']<=freeze['utc_epoch'],'actual freeze/QA chronology')
    if pre is None and capability is None:return
    require(pre is not None and capability is not None and rid in RUN_IDS,'complete fresh perrun authority')
    require(pre['status']=='PASS_ACTUAL_INDEPENDENT_ITSMIXER5_PREFIX_RESOURCE_PRESTART' and pre['plan_sha256']==pin
        and pre['files']==plan['files'] and pre['qa_sha256']==plan['qa_sha256'] and pre['root_pre_score_policy_sha256']==ROOT_POLICY_SHA
        and pre['auditor_source_sha256']==plan['files']['independent_prestart_itsmixer5.py']
        and pre['test_scores_accessed'] is False and pre['new_validation_scores_accessed'] is False
        and pre['training_started_by_auditor'] is False and pre['expires_after_seconds']==120
        and 0<=now-pre['utc_epoch']<=120 and freeze['utc_epoch']<=pre['utc_epoch'],'fresh actual bitwise prefix prestart')
    require(set(pre['datasets'])==set(DATASETS) and pre['prefix_bitwise_elements_checked']==sum(m['prefix_rows']*m['channels'] for m in plan['data_metadata'].values()),'all full prefix elements checked')
    for ds,row in pre['datasets'].items():
        m=plan['data_metadata'][ds]
        require(row=={k:m[k] for k in ('prefix_sha256','scaler_mean_sha256','scaler_std_sha256','prefix_rows','channels')}|{'prefix_bitwise_elements':m['prefix_rows']*m['channels']},'exact prestart original split/scaler')
    require(capability['status']=='ROOT_ITSMIXER5_ONE_RUN_CAPABILITY_AFTER_FRESH_INDEPENDENT_PRESTART'
        and capability['run_id']==rid and capability['plan_sha256']==pin and capability['run_config_sha256']==config_sha(plan['runs'][rid])
        and capability['automatic_retry'] is False and capability['test_allowed'] is False
        and capability['expires_after_seconds']==120 and 0<=now-capability['utc_epoch']<=120
        and pre['utc_epoch']<=capability['utc_epoch']<=pre['utc_epoch']+120,'one fresh capability identity')
