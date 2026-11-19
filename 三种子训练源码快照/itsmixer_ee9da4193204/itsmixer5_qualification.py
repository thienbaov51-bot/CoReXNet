"""Strict prospective ROOT/ITSMixer pair + retained H96 metadata contract.

No numeric report is decoded. Qualification headers must be made by ROOT only
after complete independent numeric audits; source-only fixtures are no authority.
"""
import hashlib
import json
import math
import re
from pathlib import Path
from itsmixer18_protocol import DATASETS, require, typed_equal

PARENT='/root/autodl-tmp/paper5/models2026_pilot_20261011'
CHILD=PARENT+'/itsmixer_remaining18'
MINIMUM=PARENT+'/itsmixer_minimum6'
MINIMUM_PLAN_SHA='e39994c13b78949f918eb1eeb611f2310f40fba769cebf3e19003b4eaab8f479'
MINIMUM_AUTH_SHA='2d9cd543aeea83bf86b8fb09ef63ff9ac78e5e381ece193ae216f7b5154a5fbc'
MINIMUM_START_SHA='1360ae9200925da1a8c651cb58c59ea9de3c8b10a0e5b8a70159c988647b5cb3'
ORIGINAL_PLAN_SHA='16a9dacee118f30b966489e0e17ad90274cfd56828ceaffb0c612b6fca9144ed'
ORIGINAL_CLOSURE_SHA='b7417ae45c588c32522b6030928e5fb381cab4ae5da09abc9b5a46c23bda7c1a'
ROOT6_REFERENCES_SHA='06fa1a3584f45f8cdc11c9c75d140f4fc3f1bf2950d00a7b11576f4c573fa46d'
SCIENCE_MANIFEST_SHA='06ffd9dab47faf099fc8f4229992f00f8f229132033cc0eb87baa2ad87020008'
SCIENCE_SNAPSHOT_SHA='cacf0f74440f2e1d0f367aeba67b0745c9309b6c42f9d5d64a780c6b23bb449a'
POLICY_SHA='3f42cd092488a9de60c487cc5523c0bd380248c9a3aa5956faf821bba643ede6'
PEER_SCIENCE_SHA='d9d5d7ff7db8a8792f223039827bd59107262d67f43b9f418c04313dda0aa686'
PAIR_STATUS='ROOT_SELECTED_ROOTPURGE_ITSMIXER_PAIR_AFTER_COMPLETE_MINIMUM12_NUMERIC_AUDIT'
GATE_RULE='OURS_STRICTLY_LOWER_COMMON_STD_MSE_AND_MAE_EACH_DATASET_H96'
H96_IDS=frozenset('ITSMixer_'+d+'_H96' for d in DATASETS)
ALL5=frozenset(('best.pt','config.json','history.json','VALIDATION_RESULT.json','COMPLETE_RECEIPT.json'))
QUALIFICATION_DOCUMENTS={
 'pair':('ROOT_ITSMIXER18_PAIR_QUALIFICATION.json','pair_qualification_sha256'),
 'root_qualification':('ROOT_NUMERIC_QUALIFICATION.json','root_numeric_qualification_sha256'),
 'itsmixer_qualification':('ITSMIXER6_NUMERIC_QUALIFICATION.json','itsmixer6_numeric_qualification_sha256'),
 'itsmixer_references':('ITSMIXER6_H96_REFERENCES.json','retained_itsmixer_h96_references_sha256'),
 'itsmixer_native_closure':('ITSMIXER6_NATIVE_PHYSICAL_CLOSURE.json','itsmixer6_native_physical_closure_sha256'),
 'itsmixer_whole_closure':('ITSMIXER6_WHOLE12_PHYSICAL_CLOSURE.json','itsmixer6_whole12_physical_closure_sha256'),
 'root18_science':('ROOT_ITSMIXER18_SCIENCE_SOURCE_REVIEW.json','root_remaining18_science_review_sha256'),
}
QUALIFICATION_OPAQUE_FILES=(
 'qualification_evidence/ROOT_NUMERIC_AUDIT.json',
 'qualification_evidence/INDEPENDENT_ROOT_NUMERIC_AUDIT.json',
 'qualification_evidence/ROOT_DOWNLOAD_RECEIPT.json',
 'qualification_evidence/ITSMIXER6_NUMERIC_AUDIT.json',
 'qualification_evidence/INDEPENDENT_ITSMIXER6_NUMERIC_AUDIT.json',
 'qualification_evidence/ITSMIXER6_DOWNLOAD_RECEIPT.json',
)
QUALIFICATION_FILES=tuple(n for n,_ in QUALIFICATION_DOCUMENTS.values())+QUALIFICATION_OPAQUE_FILES


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for body in iter(lambda:stream.read(1<<20),b''):h.update(body)
    return h.hexdigest()


def digest(value):return hashlib.sha256((json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()).hexdigest()


def valid_sha(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def validate_header(header,model,native_pin,closure_pin):
    expected='ROOT_'+('ROOTPURGE' if model=='RootPurgeLinear' else 'ITSMIXER6')+'_MINIMUM12_NUMERIC_QUALIFICATION_AFTER_COMPLETE_PHYSICAL_AUDIT'
    require(header['status']==expected and header['model']==model and header['native_plan_sha256']==native_pin
        and header['physical_closure_sha256']==closure_pin and header['gate_rule']==GATE_RULE
        and header['identity_audit_pass'] is True and header['independent_identity_audit_pass'] is True
        and header['strict12_gate_pass'] is True and type(header['strict12_pass_count']) is int
        and header['strict12_pass_count']==12 and type(header['strict12_total_count']) is int
        and header['strict12_total_count']==12 and header['test_accessed'] is False
        and header['new_training_authorized_by_header'] is False,
        'actual complete numeric identity PASS and distinct strict12 gate PASS header required')
    for key in ('numeric_audit_sha256','independent_numeric_audit_sha256','download_receipt_sha256'):
        require(valid_sha(header[key]),'actual numeric/download audit byte pin '+key)
    require(type(header['utc_epoch']) in (int,float) and math.isfinite(header['utc_epoch']) and header['utc_epoch']>0,'actual ROOT qualification chronology')


def validate_documents(documents,pins):
    pair=documents['pair'];native=documents['itsmixer_native_closure'];whole=documents['itsmixer_whole_closure']
    refs=documents['itsmixer_references'];root_header=documents['root_qualification'];it_header=documents['itsmixer_qualification']
    require(pair['status']==PAIR_STATUS and pair['models']==['RootPurgeLinear','ITSMixer']
        and pair['child_root']==CHILD and pair['original_plan_sha256']==ORIGINAL_PLAN_SHA
        and pair['itsmixer6_plan_sha256']==MINIMUM_PLAN_SHA
        and pair['remaining18_science_manifest_sha256']==SCIENCE_MANIFEST_SHA
        and pair['remaining18_science_snapshot_sha256']==SCIENCE_SNAPSHOT_SHA
        and pair['root_remaining18_pre_score_policy_sha256']==POLICY_SHA
        and pair['full_gate_rule']==GATE_RULE and pair['new_training_count']==18 and pair['retained_h96_count']==6
        and pair['h96_retraining_allowed'] is False and pair['weights_copied'] is False
        and pair['test_allowed'] is False and pair['automatic_retry'] is False
        and pair['source_deployment_authorized_subject_to_actual_source_resource_reviews'] is True,
        'distinct actual ROOT-selected qualified RootPurge/ITSMixer pair, not VP or source preparation')
    for key in ('root_numeric_qualification_sha256','itsmixer6_numeric_qualification_sha256','retained_itsmixer_h96_references_sha256','itsmixer6_native_physical_closure_sha256','itsmixer6_whole12_physical_closure_sha256','root_remaining18_science_review_sha256'):
        require(valid_sha(pins[key]) and pair[key]==pins[key],'selected pair exact raw metadata link '+key)
    validate_header(root_header,'RootPurgeLinear',ORIGINAL_PLAN_SHA,ORIGINAL_CLOSURE_SHA)
    validate_header(it_header,'ITSMixer',MINIMUM_PLAN_SHA,pins['itsmixer6_whole12_physical_closure_sha256'])
    require(type(pair['utc_epoch']) in (int,float) and math.isfinite(pair['utc_epoch'])
        and pair['utc_epoch']>=max(root_header['utc_epoch'],it_header['utc_epoch']), 'pair issued after both complete numeric headers')
    require(native['status']=='COMPLETE_ITSMIXER6_OPAQUE_PHYSICAL_CLOSURE_NO_SCORES'
        and native['child_root']==MINIMUM and native['plan_sha256']==MINIMUM_PLAN_SHA
        and native['authorization_sha256']==MINIMUM_AUTH_SHA and native['actual_start_sha256']==MINIMUM_START_SHA
        and type(native['run_count']) is int and native['run_count']==6 and set(native['runs'])==H96_IDS
        and native['controller_and_workers_ended'] is True and native['scores_decoded'] is False
        and native['test_accessed'] is False and native['pair_qualified_by_this_receipt'] is False
        and native['training_continuation_authorized'] is False,'actual native six complete physical metadata, not a gate')
    require(whole['status']=='COMPLETE_ITSMIXER6_PLUS_RETAINED_ROOT6_WHOLE_OPAQUE_PHYSICAL_CLOSURE'
        and whole['child_root']==MINIMUM and whole['plan_sha256']==MINIMUM_PLAN_SHA
        and whole['new_run_count']==whole['retained_h96_count']==6 and whole['canonical_count']==12
        and whole['physical_file_count']==60 and whole['controller_and_workers_ended'] is True
        and whole['gpu_compute_pids']==whole['global_producer_pids']==[]
        and whole['scores_decoded'] is False and whole['test_accessed'] is False,
        'whole12 all60 SHA/size closure precedes numeric qualification')
    expected_root={'RootPurgeLinear_'+d for d in DATASETS}
    require(set(whole['runs'])==H96_IDS|expected_root and whole['original_plan_sha256']==ORIGINAL_PLAN_SHA
        and whole['original_closure_sha256']==ORIGINAL_CLOSURE_SHA
        and whole['retained_references_sha256']==ROOT6_REFERENCES_SHA,'qualified whole12 exact retained Root6 identity')
    root_refs=documents['references'];original_plan=documents['original_plan']
    require(type(root_refs['runs']) is list and len(root_refs['runs'])==6
        and {r['run_id'] for r in root_refs['runs']}==expected_root,'same original retained Root native LIST6')
    for original in root_refs['runs']:
        rid=original['run_id'];row=whole['runs'][rid]
        require(row['run_id']==rid and row['source_root']==PARENT and row['plan_sha256']==ORIGINAL_PLAN_SHA
            and row['config']==original_plan['runs'][rid] and row['files']==original['physical']['files']
            and row['completion_receipt']['status']=='COMPLETE_VALIDATION_ONLY'
            and row['completion_receipt']['run_id']==rid and row['completion_receipt']['plan_sha256']==ORIGINAL_PLAN_SHA
            and row['completion_receipt']['test_accessed'] is False,'whole retained Root6 same immutable original all5/config')
        for field,name in (('checkpoint_sha256','best.pt'),('config_sha256','config.json'),('history_sha256','history.json'),('validation_result_sha256','VALIDATION_RESULT.json')):
            require(row['completion_receipt'][field]==row['files'][name]['sha256'],'whole retained Root6 receipt identity '+name)
    require(native['plan']==whole['native_plan'] and native['plan']['files']==whole['source_inventory']
        and native['boundary_sha256']==whole['pins']['boundary']
        and native['actual_start_sha256']==whole['pins']['start']
        and native['completed_utc_epoch']==whole['native_boundary']['utc_epoch'],
        'native/whole source plan, boundary/start and chronology equality')
    require(refs['status']=='ROOT_RETAINED_ITSMIXER6_H96_PHYSICAL_REFERENCES_AFTER_COMPLETE_NUMERIC_AUDIT'
        and refs['child_root']==MINIMUM and refs['plan_sha256']==MINIMUM_PLAN_SHA
        and refs['authorization_sha256']==MINIMUM_AUTH_SHA and refs['actual_start_sha256']==MINIMUM_START_SHA
        and refs['native_physical_closure_sha256']==pins['itsmixer6_native_physical_closure_sha256']
        and refs['whole12_physical_closure_sha256']==pins['itsmixer6_whole12_physical_closure_sha256']
        and refs['h96_retraining_allowed'] is False and refs['weights_copied'] is False
        and type(refs['runs']) is list and len(refs['runs'])==6 and {r['run_id'] for r in refs['runs']}==H96_IDS,
        'actual retained H96 native LIST6 all5 reference set')
    for row in refs['runs']:
        rid=row['run_id'];prior=native['runs'][rid];w=whole['runs'][rid]
        require(row['source_root']==MINIMUM and row['plan_sha256']==MINIMUM_PLAN_SHA
            and row['best_path']==MINIMUM+'/results/'+rid+'/best.pt'
            and row['config']==native['plan']['runs'][rid]==w['config']
            and row['physical']['files']==prior['files']==w['files'] and set(prior['files'])==ALL5
            and prior['completion']==whole['native_boundary']['done'][rid],
            'retained H96 exact config / complete native all5 identities')
        completion=prior['completion']
        require(completion['complete_receipt_sha256']==prior['files']['COMPLETE_RECEIPT.json']['sha256']
            and {k:v for k,v in completion.items() if k!='complete_receipt_sha256'}==w['completion_receipt'],
            'native/whole all5 completion receipt bytes chain')
        for name,identity in prior['files'].items():
            require(set(identity)=={'sha256','bytes'} and valid_sha(identity['sha256'])
                and type(identity['bytes']) is int and identity['bytes']>0,'typed opaque all5 physical identity '+name)
    science=documents['root18_science']
    require(science['status']=='PASS_ROOT_ITSMIXER_REMAINING18_SCIENCE_SOURCE_REVIEW'
        and science['source_manifest_sha256']==SCIENCE_MANIFEST_SHA
        and science['snapshot']['snapshot_sha256']==SCIENCE_SNAPSHOT_SHA
        and science['new_scores_accessed'] is False and science['test_accessed'] is False,
        'different ROOT exact future18 scientific source review')
    return pair


def verify_opaque_audit_evidence(root,files,documents):
    """Only hash report/download raw bytes. Numeric reports never enter decoder."""
    mapping={
      'qualification_evidence/ROOT_NUMERIC_AUDIT.json':documents['root_qualification']['numeric_audit_sha256'],
      'qualification_evidence/INDEPENDENT_ROOT_NUMERIC_AUDIT.json':documents['root_qualification']['independent_numeric_audit_sha256'],
      'qualification_evidence/ROOT_DOWNLOAD_RECEIPT.json':documents['root_qualification']['download_receipt_sha256'],
      'qualification_evidence/ITSMIXER6_NUMERIC_AUDIT.json':documents['itsmixer_qualification']['numeric_audit_sha256'],
      'qualification_evidence/INDEPENDENT_ITSMIXER6_NUMERIC_AUDIT.json':documents['itsmixer_qualification']['independent_numeric_audit_sha256'],
      'qualification_evidence/ITSMIXER6_DOWNLOAD_RECEIPT.json':documents['itsmixer_qualification']['download_receipt_sha256'],
    }
    require(set(mapping)==set(QUALIFICATION_OPAQUE_FILES),'all six exact opaque qualification evidence files')
    for name,pin in mapping.items():
        path=Path(root)/name
        require(all(not p.is_symlink() for p in (path,*path.parents)) and path.is_file()
            and path.stat().st_size>0 and sha(path)==files[name]==pin,'opaque numeric/audit/download evidence exact physical bytes '+name)


def verify_h96_physical(root,documents,read_metadata,safe):
    """Rehash all5 unchanged checkpoints/config/history/result before framework."""
    refs=documents['itsmixer_references'];native=documents['itsmixer_native_closure']
    prior=Path(MINIMUM);plan_path=safe(prior,'ITSMIXER6_PLAN.json')
    require(sha(plan_path)==MINIMUM_PLAN_SHA and read_metadata(plan_path)==native['plan'], 'retained actual native frozen plan bytes')
    require(sha(safe(prior,'ROOT_ITSMIXER6_START_AUTHORIZATION.json'))==MINIMUM_AUTH_SHA
        and sha(safe(prior,'ITSMIXER6_ACTUAL_START.json'))==MINIMUM_START_SHA,
        'retained native one first-start metadata bytes')
    state=read_metadata(safe(prior,'ITSMIXER6_STATE.json'))
    boundary_path=safe(prior,'ITSMIXER6_TRAINING_BOUNDARY.json')
    require(sha(boundary_path)==native['boundary_sha256'],'native six actual complete boundary raw SHA')
    boundary=read_metadata(boundary_path)
    require(state['status']==boundary['status']=='COMPLETE_ITSMIXER6_VALIDATION_TRAINING'
        and set(state['done'])==set(boundary['done'])==H96_IDS and state['done']==boundary['done']
        and not state['queue'] and not state['active'] and not state['failed'] and state['admission_halt'] is None
        and not state['unverified_started_workers'] and not state['unverified_worker_exits']
        and not boundary['queue'] and not boundary['active'] and not boundary['failed'] and boundary['admission_halt'] is None
        and not boundary['unverified_started_workers'] and not boundary['unverified_worker_exits']
        and boundary['all_workers_polled_and_drained'] is True,
        'retained six producer complete/drained, no partial references')
    for name,pin in native['plan']['files'].items():
        require(sha(safe(prior,name))==pin,'retained H96 frozen source/prefix identity '+name)
    for row in refs['runs']:
        rid=row['run_id'];folder=safe(prior,'results/'+rid);files=row['physical']['files']
        require(set(p.name for p in folder.iterdir())==ALL5,'retained H96 native all5 no extras')
        for name,identity in files.items():
            path=safe(folder,name)
            require(path.stat().st_size==identity['bytes'] and sha(path)==identity['sha256'],'retained H96 fresh all5 opaque bytes '+name)
        receipt=read_metadata(folder/'COMPLETE_RECEIPT.json');expected=native['runs'][rid]['completion']
        require(receipt=={k:v for k,v in expected.items() if k!='complete_receipt_sha256'}
            and receipt['status']=='COMPLETE_VALIDATION_ONLY' and receipt['run_id']==rid
            and receipt['plan_sha256']==MINIMUM_PLAN_SHA and receipt['root_authorization_sha256']==MINIMUM_AUTH_SHA
            and receipt['test_accessed'] is False and receipt['scores_released'] is False
            and receipt['config_sha256']==digest(row['config'])
            and receipt['epochs']==15 and receipt['steps']==row['config']['expected_optimizer_steps_at_full_15_epoch_budget'],
            'retained H96 score-free receipt full config / budget / authority')
        for field,name in (('checkpoint_sha256','best.pt'),('config_sha256','config.json'),('history_sha256','history.json'),('validation_result_sha256','VALIDATION_RESULT.json')):
            require(receipt[field]==files[name]['sha256'],'retained H96 all5 native receipt link '+name)
