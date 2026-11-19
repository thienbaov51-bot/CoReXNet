"""Isolated stdlib historical replay; imports perform no I/O.

The original opaque collector and its no-framework assertion stay unchanged.
This verifier reads only metadata/source bytes and opaque artifact SHA/size.
It has no live-process exemptions, arrays, scores, framework or GPU API.
"""
import hashlib, json, os, subprocess, sys
from pathlib import Path

PARENT='/root/autodl-tmp/paper5/models2026_pilot_20261011'
PREVIOUS_CHILD=PARENT+'/itsmixer_recovery13'
PREVIOUS_SOURCE_ONLY_CHILD=PREVIOUS_CHILD+'_isolated'
CHILD=PREVIOUS_CHILD+'_isolated3'
PYTHON='/root/miniconda3/bin/python'
COLLECTOR_SHA='9e55132e383eee11a83fcea06f8f8ea8e015acc211c49b0a47482eb7cebefb80'
PROOF_SHA='11a26285c936af5c5dae3aea8b97988b5f000339015e272892b9e9762e9d27f0'
PROOF_NAME='PREVIOUS_FROZEN13_NO_SCIENTIFIC_ATTEMPTS_OPAQUE_PROOF.json'
SOURCE_ONLY_PROOF_SHA='4762841c460d48ee0a20330e43df731fb5bb8dda84a79a006f4ed2d77b7855d1'
SOURCE_ONLY_PROOF_NAME='PREVIOUS_ISOLATED13_NO_SCIENTIFIC_ATTEMPTS_OPAQUE_PROOF.json'
SOURCE_ONLY_ALIASES={'ITSMIXER13_SOURCE_MANIFEST.json':'PREVIOUS_ISOLATED13_SOURCE_MANIFEST.json',
 'ACTUAL_ITSMIXER13_DEPLOYMENT_RECEIPT.json':'PREVIOUS_ISOLATED13_DEPLOYMENT_RECEIPT.json',
 'ACTUAL_ITSMIXER13_PREFIX_COPY_RECEIPT.json':'PREVIOUS_ISOLATED13_PREFIX_COPY_RECEIPT.json'}
ALIASES={
 'ITSMIXER13_SOURCE_MANIFEST.json':'PREVIOUS_FROZEN13_SOURCE_MANIFEST.json',
 'ITSMIXER13_PLAN.json':'PREVIOUS_FROZEN13_PLAN.json',
 'ROOT_ITSMIXER13_START_AUTHORIZATION.json':'PREVIOUS_FROZEN13_START_AUTHORIZATION.json',
 'ACTUAL_ITSMIXER13_FREEZE_RECEIPT.json':'PREVIOUS_FROZEN13_FREEZE_RECEIPT.json',
 'ACTUAL_ITSMIXER13_ORIGINAL_CUDA18_RESOURCE_PROJECTION.json':'PREVIOUS_FROZEN13_RESOURCE_PROJECTION.json'}
STATUS='PASS_CLEAN_STDLIB_ORIGINAL11_AND_PREVIOUS_ENVIRONMENT_ZERO_SCIENTIFIC_CLAIMS_PHYSICAL_REPLAY'
TIMEOUT_SECONDS=30

STARTED_CHILD=PREVIOUS_CHILD+'_isolated2'
STARTED_PROOF_NAME='PREVIOUS_STARTED13_PREFRAMEWORK_FAILURE_PROOF.json'
STARTED_PROOF_SHA='11d420fb57555ce738cd9a60dd41e09afef749add996dbc8f575ccb150d54b74'
STARTED_COLLECTOR_NAME='collect_itsmixer_s2_failed13.py'
STARTED_COLLECTOR_SHA='6379934a1e32196df73c8aed6060ecd593e064f353d1e9d572ea55abee0d94a1'

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    p=Path(path);require(p.is_absolute() and all(not q.is_symlink() for q in (p,*p.parents)),'isolated physical symlink')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def metadata(path):
    from collect_itsmixer18_recovery13 import metadata_bytes
    return metadata_bytes(Path(path).read_bytes())

def _assert_previous_absence(root):
    # Native producer filenames, not an assumption based only on GPU idleness.
    absent=('ITSMIXER13_START_CLAIM.json','ITSMIXER13_ACTUAL_START.json',
        'ITSMIXER13_LAUNCH_RECEIPT.json','ITSMIXER13_POPEN_RETURNED.json',
        'ITSMIXER13_UNVERIFIED_CONTROLLER_EXIT.json','ITSMIXER13_STATE.json',
        'ITSMIXER13_TRAINING_BOUNDARY.json','controller.log','logs','results')
    for name in absent:require(not (root/name).exists() and not (root/name).is_symlink(),'previous frozen13 acquired scientific/start footprint: '+name)
    for pattern in ('ROOT_ITSMIXER13_RUN_CAPABILITY_*.json','ITSMIXER13_RUN_ATTEMPT_*.json',
        'ITSMIXER13_POPEN_RETURNED_*.json','ACTUAL_INDEPENDENT_ITSMIXER13_PRESTART_*.json'):
        require(not list(root.glob(pattern)),'previous frozen13 acquired capability/attempt/prestart footprint: '+pattern)

def verify_previous_frozen13(package_root):
    """Clean-process-only physical replay of the genuine failed environment."""
    require('numpy' not in sys.modules and 'torch' not in sys.modules,'previous native replay requires clean stdlib caller')
    root=Path(package_root);old=Path(PREVIOUS_CHILD)
    require(sha(root/PROOF_NAME)==PROOF_SHA,'actual preserved previous frozen13 noattempt proof bytes')
    proof=metadata(root/PROOF_NAME)
    require(proof['status']=='ACTUAL_FROZEN13_PRESTART_REJECTED_NO_CONTROLLER_OR_RUN_FOOTPRINTS'
        and proof['root']==PREVIOUS_CHILD and proof['footprints']=={}
        and proof['arrays_or_scores_read'] is False and set(proof['files'])==set(ALIASES),'genuine previous native noattempt proof schema')
    for name,alias in ALIASES.items():
        identity=proof['files'][name]
        require(type(identity['bytes']) is int and identity['bytes']>0
            and (root/alias).stat().st_size==(old/name).stat().st_size==identity['bytes']
            and sha(root/alias)==sha(old/name)==identity['sha256'],'previous frozen native header changed: '+name)
    _assert_previous_absence(old)
    # Pin every module/data byte before importing the frozen native validator.
    # The legacy validator preserves all actual source/QA/auth/freeze links;
    # it imports no NumPy/Torch in this clean process and reads no score body.
    plan=metadata(root/ALIASES['ITSMIXER13_PLAN.json'])
    for name,pin in plan['files'].items():require(sha(old/name)==pin,'previous frozen source/prefix changed: '+name)
    require(plan['child_root']==PREVIOUS_CHILD and plan['new_training_count']==13,'previous frozen native plan identity')
    require('itsmixer13_common' not in sys.modules,'native13 validator namespace must be clean')
    sys.path.insert(0,PREVIOUS_CHILD)
    try:
        import itsmixer13_common as native
        require(sha(Path(native.__file__))==plan['files']['itsmixer13_common.py'],'exact frozen previous native common module')
        _,native_plan,_,_,freeze_pin=native.load_frozen(proof['files']['ITSMIXER13_PLAN.json']['sha256'],
            proof['files']['ROOT_ITSMIXER13_START_AUTHORIZATION.json']['sha256'])
        require(native_plan==plan and freeze_pin==proof['files']['ACTUAL_ITSMIXER13_FREEZE_RECEIPT.json']['sha256'],
            'previous frozen native chain differs')
    finally:sys.path.pop(0)
    _assert_previous_absence(old)
    for name,alias in ALIASES.items():
        require(sha(root/alias)==sha(old/name)==proof['files'][name]['sha256'],'previous frozen header changed during clean replay')
    require('numpy' not in sys.modules and 'torch' not in sys.modules,'opaque replay imported framework')
    return {'previous_frozen13_noattempt_proof_sha256':PROOF_SHA,
        'previous_frozen13_plan_sha256':proof['files']['ITSMIXER13_PLAN.json']['sha256'],
        'previous_frozen13_authorization_sha256':proof['files']['ROOT_ITSMIXER13_START_AUTHORIZATION.json']['sha256'],
        'previous_frozen13_freeze_sha256':proof['files']['ACTUAL_ITSMIXER13_FREEZE_RECEIPT.json']['sha256'],
        'previous_frozen13_source_manifest_sha256':proof['files']['ITSMIXER13_SOURCE_MANIFEST.json']['sha256']}

def verify_previous_source_only13(package_root):
    require('numpy' not in sys.modules and 'torch' not in sys.modules,'unused80 physical replay requires clean stdlib caller')
    root=Path(package_root);old=Path(PREVIOUS_SOURCE_ONLY_CHILD)
    require(sha(root/SOURCE_ONLY_PROOF_NAME)==SOURCE_ONLY_PROOF_SHA,'genuine unused80 noattempt proof bytes')
    proof=metadata(root/SOURCE_ONLY_PROOF_NAME)
    require(proof['status']=='ACTUAL_ISOLATED13_80_DEPLOYED_ONLY_NO_QA_FREEZE_PRESTART_OR_SCIENTIFIC_FOOTPRINTS'
        and proof['root']==PREVIOUS_SOURCE_ONLY_CHILD and proof['footprints']=={} and proof['arrays_or_scores_read'] is False
        and set(proof['files'])==set(SOURCE_ONLY_ALIASES),'unused80 genuine native proof schema')
    for name,alias in SOURCE_ONLY_ALIASES.items():
        row=proof['files'][name]
        require(type(row['bytes']) is int and row['bytes']>0 and (root/alias).stat().st_size==(old/name).stat().st_size==row['bytes']
            and sha(root/alias)==sha(old/name)==row['sha256'],'unused80 native metadata byte identity: '+name)
    _assert_previous_absence(old)
    for name in ('ITSMIXER13_PROJECTION_CLAIM.json','ACTUAL_ITSMIXER13_ORIGINAL_CUDA18_RESOURCE_PROJECTION.json',
        'ITSMIXER13_PLAN.json','ROOT_ITSMIXER13_START_AUTHORIZATION.json','ACTUAL_ITSMIXER13_FREEZE_RECEIPT.json'):
        require(not (old/name).exists() and not (old/name).is_symlink(),'unused80 acquired projection/freeze authority: '+name)
    manifest=metadata(root/SOURCE_ONLY_ALIASES['ITSMIXER13_SOURCE_MANIFEST.json'])
    deployed=metadata(root/SOURCE_ONLY_ALIASES['ACTUAL_ITSMIXER13_DEPLOYMENT_RECEIPT.json'])
    prefix=metadata(root/SOURCE_ONLY_ALIASES['ACTUAL_ITSMIXER13_PREFIX_COPY_RECEIPT.json'])
    require(manifest['child_root']==PREVIOUS_SOURCE_ONLY_CHILD
        and manifest['status']=='REVIEWED_ITSMIXER13_QUALIFIED_SOURCE_DEPLOYMENT_ONLY_NO_SCIENTIFIC_FREEZE'
        and manifest['training_started'] is manifest['test_allowed'] is False
        and deployed['status']=='ACTUAL_ITSMIXER13_SOURCE_PREFIX_SHA_DEPLOYED_NO_QA_NO_TRAIN'
        and deployed['manifest_sha256']==proof['files']['ITSMIXER13_SOURCE_MANIFEST.json']['sha256']
        and deployed['files']==manifest['files'] and deployed['scores_accessed'] is deployed['gpu_qa_started'] is deployed['training_started'] is False
        and prefix['status']=='ACTUAL_ITSMIXER13_PREFIX_BYTECOPY_SHA_PASS_SOURCE_PENDING_UPLOAD'
        and prefix['files']=={n:v for n,v in manifest['files'].items() if n.startswith('data/')}
        and prefix['measurement_arrays_decoded'] is prefix['scores_accessed'] is prefix['training_started'] is False
        and prefix['utc_epoch']<=deployed['utc_epoch']<=proof['utc_epoch'],'unused80 source-only deployment/prefix chain')
    for name,pin in manifest['files'].items():require(sha(old/name)==pin,'unused80 deployed frozen SOURCE/prefix changed: '+name)
    for name,field in (('root_source_review_name','root_source_review_sha256'),('independent_source_review_name','independent_source_review_sha256')):
        require(sha(old/manifest[name])==manifest[field],'unused80 source review byte identity')
    _assert_previous_absence(old)
    for name,alias in SOURCE_ONLY_ALIASES.items():
        require(sha(root/alias)==sha(old/name)==proof['files'][name]['sha256'],'unused80 native metadata changed during replay')
    return {'previous_source_only_isolated13_noattempt_proof_sha256':SOURCE_ONLY_PROOF_SHA,
        'previous_source_only_isolated13_source_manifest_sha256':proof['files']['ITSMIXER13_SOURCE_MANIFEST.json']['sha256'],
        'previous_source_only_isolated13_deployment_receipt_sha256':proof['files']['ACTUAL_ITSMIXER13_DEPLOYMENT_RECEIPT.json']['sha256']}


def verify_previous_started13(package_root):
    require('numpy' not in sys.modules and 'torch' not in sys.modules,'failed-S2 replay requires clean stdlib caller')
    root=Path(package_root)
    require(sha(root/STARTED_PROOF_NAME)==STARTED_PROOF_SHA and sha(root/STARTED_COLLECTOR_NAME)==STARTED_COLLECTOR_SHA,
        'genuine failed-S2 proof/collector byte identities')
    import collect_itsmixer_s2_failed13 as historical
    require(sha(Path(historical.__file__))==STARTED_COLLECTOR_SHA,'exact failed-S2 collector SOURCE module')
    evidence=historical.decode((root/STARTED_PROOF_NAME).read_bytes())
    historical.verify_evidence_physical(evidence)
    require(evidence['child_root']==STARTED_CHILD and evidence['first_Popen_run_id']=='ITSMixer_Electricity_H720'
        and len(evidence['thirteen_scientific_claim_absence'])==13 and len(evidence['twelve_no_Popen_absence'])==12,
        'one prior environment Popen is preserved, not called unlaunched')
    require(sha(root/STARTED_PROOF_NAME)==STARTED_PROOF_SHA and sha(root/STARTED_COLLECTOR_NAME)==STARTED_COLLECTOR_SHA,
        'failed-S2 proof/collector changed during physical replay')
    return started_output(evidence)

def started_output(evidence):
    return {'previous_started13_preframework_failure_proof_sha256':STARTED_PROOF_SHA,
        'previous_started13_plan_sha256':evidence['plan_sha256'],
        'previous_started13_authorization_sha256':evidence['authorization_sha256'],
        'previous_started13_actual_start_sha256':evidence['actual_start_sha256'],
        'previous_started13_boundary_sha256':evidence['original_boundary_sha256'],
        'previous_started13_one_environment_Popen_preserved':True,
        'previous_started13_zero_scientific_claims_verified':True,
        'previous_started13_twelve_unlaunched_verified':True}

# This string is part of the byte-pinned production SOURCE. -I prevents parent
# PYTHONPATH/site user configuration and inherited sys.modules from entering it.
READER_SCRIPT=r'''
import hashlib,json,os,sys
from pathlib import Path
assert len(sys.argv)==5 and sys.argv[1]=='/root/autodl-tmp/paper5/models2026_pilot_20261011/itsmixer_recovery13_isolated3'
root=Path(sys.argv[1]);evidence_pin,helper_pin,collector_pin=sys.argv[2:]
assert os.geteuid()==0 and 'numpy' not in sys.modules and 'torch' not in sys.modules
def sha(p):
 assert p.is_absolute() and all(not q.is_symlink() for q in (p,*p.parents))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
assert sha(root/'itsmixer13_isolated_ancestry.py')==helper_pin
assert sha(root/'collect_itsmixer18_recovery13.py')==collector_pin=='9e55132e383eee11a83fcea06f8f8ea8e015acc211c49b0a47482eb7cebefb80'
assert sha(root/'ITSMIXER13_RECOVERY_EVIDENCE.json')==evidence_pin
sys.path.insert(0,str(root))
import collect_itsmixer18_recovery13 as collector
evidence=collector.metadata_bytes((root/'ITSMIXER13_RECOVERY_EVIDENCE.json').read_bytes())
collector.validate_evidence_shape(evidence)
collector.verify_evidence_physical(evidence)
from itsmixer13_isolated_ancestry import verify_previous_frozen13,verify_previous_source_only13,verify_previous_started13,STATUS
prior=verify_previous_frozen13(root)
prior.update(verify_previous_source_only13(root))
prior.update(verify_previous_started13(root))
assert sha(root/'itsmixer13_isolated_ancestry.py')==helper_pin
assert sha(root/'collect_itsmixer18_recovery13.py')==collector_pin
assert sha(root/'ITSMIXER13_RECOVERY_EVIDENCE.json')==evidence_pin
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
print(json.dumps({'status':STATUS,'child_root':str(root),'original_evidence_child_root':evidence['child_root'],
 'collector_source_sha256':collector_pin,'isolated_verifier_source_sha256':helper_pin,
 'recovery_evidence_sha256':evidence_pin,'retained_original5_ids':sorted(evidence['retained_original5_references']),
 'retained_h96_ids':sorted(row['run_id'] for row in evidence['retained_itsmixer_h96_references']),
 'remaining13_sorted':evidence['remaining13_sorted'],'previous_frozen13_native_chain_replayed':True,
 'arrays_or_scores_read':False,'framework_imported':False,'global_process_or_GPU_sample_performed':False,
 'scientific_training_started':False,**prior},sort_keys=True,allow_nan=False))
'''

def expected_output(root,evidence,evidence_pin,helper_pin):
    proof=metadata(root/PROOF_NAME)
    unused=metadata(root/SOURCE_ONLY_PROOF_NAME)
    return {**started_output(metadata(root/STARTED_PROOF_NAME)),'status':STATUS,'child_root':CHILD,'original_evidence_child_root':PREVIOUS_CHILD,
        'collector_source_sha256':COLLECTOR_SHA,'isolated_verifier_source_sha256':helper_pin,
        'recovery_evidence_sha256':evidence_pin,'retained_original5_ids':sorted(evidence['retained_original5_references']),
        'retained_h96_ids':sorted(row['run_id'] for row in evidence['retained_itsmixer_h96_references']),
        'remaining13_sorted':evidence['remaining13_sorted'],'previous_frozen13_native_chain_replayed':True,
        'arrays_or_scores_read':False,'framework_imported':False,'global_process_or_GPU_sample_performed':False,
        'scientific_training_started':False,'previous_frozen13_noattempt_proof_sha256':PROOF_SHA,
        'previous_frozen13_plan_sha256':proof['files']['ITSMIXER13_PLAN.json']['sha256'],
        'previous_frozen13_authorization_sha256':proof['files']['ROOT_ITSMIXER13_START_AUTHORIZATION.json']['sha256'],
        'previous_frozen13_freeze_sha256':proof['files']['ACTUAL_ITSMIXER13_FREEZE_RECEIPT.json']['sha256'],
        'previous_frozen13_source_manifest_sha256':proof['files']['ITSMIXER13_SOURCE_MANIFEST.json']['sha256'],
        'previous_source_only_isolated13_noattempt_proof_sha256':SOURCE_ONLY_PROOF_SHA,
        'previous_source_only_isolated13_source_manifest_sha256':unused['files']['ITSMIXER13_SOURCE_MANIFEST.json']['sha256'],
        'previous_source_only_isolated13_deployment_receipt_sha256':unused['files']['ACTUAL_ITSMIXER13_DEPLOYMENT_RECEIPT.json']['sha256']}

def validate_reader_output(completed,expected):
    from collect_itsmixer18_recovery13 import metadata_bytes
    require(completed.returncode==0 and completed.stderr==b'','clean ancestry reader exit/stderr failure')
    output=metadata_bytes(completed.stdout)
    require(output==expected and set(output)==set(expected),'clean ancestry reader exact score-free output identity')
    require(all(type(output[k]) is bool for k in ('previous_frozen13_native_chain_replayed',
        'arrays_or_scores_read','framework_imported','global_process_or_GPU_sample_performed','scientific_training_started',
        'previous_started13_one_environment_Popen_preserved','previous_started13_zero_scientific_claims_verified','previous_started13_twelve_unlaunched_verified')),
        'clean ancestry reader strict boolean truth')
    return output

def verify_isolated_ancestry(evidence):
    root=Path(__file__).absolute().parent
    require(os.name=='posix' and os.geteuid()==0 and str(root)==CHILD and sys.executable==PYTHON,
        'fixed isolated ROOT Python execution identity')
    manifest=metadata(root/'ITSMIXER13_SOURCE_MANIFEST.json')
    helper_pin=manifest['files']['itsmixer13_isolated_ancestry.py']
    evidence_pin=manifest['recovery_evidence_sha256']
    require(manifest['child_root']==CHILD and sha(root/'itsmixer13_isolated_ancestry.py')==helper_pin
        and sha(root/'collect_itsmixer18_recovery13.py')==COLLECTOR_SHA
        and sha(root/PROOF_NAME)==PROOF_SHA and sha(root/'ITSMIXER13_RECOVERY_EVIDENCE.json')==evidence_pin,
        'isolated reader physical SOURCE/evidence/proof identity')
    require(metadata(root/'ITSMIXER13_RECOVERY_EVIDENCE.json')==evidence,'caller evidence differs from byte-pinned original evidence')
    require(sha(root/SOURCE_ONLY_PROOF_NAME)==SOURCE_ONLY_PROOF_SHA,'before child unused80 proof bytes')
    require(sha(root/STARTED_PROOF_NAME)==STARTED_PROOF_SHA and sha(root/STARTED_COLLECTOR_NAME)==STARTED_COLLECTOR_SHA,'before child failed-S2 proof/collector pins')
    expected=expected_output(root,evidence,evidence_pin,helper_pin)
    completed=subprocess.run([PYTHON,'-I','-c',READER_SCRIPT,CHILD,evidence_pin,helper_pin,COLLECTOR_SHA],
        stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        timeout=TIMEOUT_SECONDS,check=False)
    output=validate_reader_output(completed,expected)
    require(sha(root/STARTED_PROOF_NAME)==STARTED_PROOF_SHA and sha(root/STARTED_COLLECTOR_NAME)==STARTED_COLLECTOR_SHA,'after child failed-S2 proof/collector pins')
    require(sha(root/'itsmixer13_isolated_ancestry.py')==helper_pin
        and sha(root/'collect_itsmixer18_recovery13.py')==COLLECTOR_SHA
        and sha(root/PROOF_NAME)==PROOF_SHA and sha(root/'ITSMIXER13_RECOVERY_EVIDENCE.json')==evidence_pin,
        'isolated reader SOURCE/evidence/proof changed during child replay')
    require(sha(root/SOURCE_ONLY_PROOF_NAME)==SOURCE_ONLY_PROOF_SHA,'after child unused80 proof bytes')
    return output

if __name__=='__main__':
    print(json.dumps({'status':'PENDING_CLEAN_STDLIB_ANCESTRY_REPLAY_REQUIRES_NATIVE_AUTHORIZED_CALLER','input_IO':False}))
