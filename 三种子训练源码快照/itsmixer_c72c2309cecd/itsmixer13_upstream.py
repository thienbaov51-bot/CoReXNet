"""Exact opaque old18 ancestry and original actual CUDA18 projection contract.

Importing performs no filesystem, process, array, framework or score I/O.
"""
import hashlib, json, re
from pathlib import Path

ORIGINAL='/root/autodl-tmp/paper5/models2026_pilot_20261011/itsmixer_remaining18'
CHILD='/root/autodl-tmp/paper5/models2026_pilot_20261011/itsmixer_recovery13_isolated3'
ORIGINAL_EVIDENCE_CHILD='/root/autodl-tmp/paper5/models2026_pilot_20261011/itsmixer_recovery13'
PLAN_SHA='3f62ec743c9e0af9d427e80abbbc8e7353eedc93958369939e8caea0fac2d8fd'
AUTH_SHA='ed1c7d23c26ece0450e618829488275cfd52d4f762d29226bf16403ba19fe376'
START_SHA='dcb81015c66c8ef9c0ce4a25bf8983230768b68e29c1d49ad486f41f9ef77a30'
SOURCE_PACKAGE_SHA='53435186004f7c55c6b58d59be03cfe27771d553c2ccafdb3f40fd8b47f95cb3'
QA_SHA='06d419caf4dc8b6d10bc437639e12e879cd349fdc469ebc3f6859b3fe214b792'
SOURCE_MANIFEST_SHA='9ba34ed9d3d87381f34d5a2993b6ab3d0d39a5ae61a514d00877d8a9de7c62f5'
OLD5=frozenset(('ITSMixer_AEMO5_H192','ITSMixer_AEMO5_H336','ITSMixer_AEMO5_H720','ITSMixer_Electricity_H192','ITSMixer_Electricity_H336'))
ALL18=frozenset('ITSMixer_'+d+'_H'+str(h) for d in ('Electricity','PEMS03','PEMS04','PEMS08','Solar','AEMO5') for h in (192,336,720))
NEW13=ALL18-OLD5

def require(condition,message):
    if not condition:raise ValueError(message)

def sha(path):
    path=Path(path);require(all(not p.is_symlink() for p in (path,*path.parents)), 'opaque upstream symlink')
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()

def original_qa():
    # Resource-only actual rows: no validation/test payload is decoded here.
    p=Path(__file__).absolute().parent/'ORIGINAL_ACTUAL_CUDA18_RESOURCE_RECEIPT.json'
    require(sha(p)==QA_SHA,'exact original actual CUDA18 resource receipt bytes')
    result=json.loads(p.read_bytes());require(result['status']=='PASS_ALL18_ACTUAL_ORIGINAL_BATCH_ITSMIXER_SYNTHETIC_CUDA_ONLY','original CUDA18 receipt status')
    require(len(result['rows'])==18 and {r['run_id'] for r in result['rows']}==ALL18
        and result['measurement_array_read'] is result['research_scores_read'] is result['training_started'] is False,
        'original resource-only full18 measurement identity')
    return result

def verify_original_recovery(evidence):
    from itsmixer13_isolated_ancestry import verify_isolated_ancestry
    verify_isolated_ancestry(evidence)
    require(evidence['original_root']==ORIGINAL and evidence['child_root']==ORIGINAL_EVIDENCE_CHILD
        and evidence['original_plan_sha256']==PLAN_SHA and evidence['original_authorization_sha256']==AUTH_SHA
        and evidence['original_actual_start_sha256']==START_SHA
        and set(evidence['retained_original5_references'])==OLD5 and evidence['remaining13_sorted']==sorted(NEW13)
        and set(evidence['runs'])==NEW13 and evidence['original_cuda18_qa_sha256']==QA_SHA
        and evidence['original_source_manifest_sha256']==SOURCE_MANIFEST_SHA,'exact old five and never-attempted thirteen ancestry')
    return evidence

def validate_projection(qa,plan):
    baseline=original_qa();rows={r['run_id']:r for r in baseline['rows']}
    require(qa['status']=='PASS_ITSMIXER13_EXACT_ORIGINAL_CUDA18_RESOURCE_METADATA_PROJECTION_NO_NEW_CUDA'
        and qa['child_root']==CHILD and qa['original_root']==ORIGINAL and qa['original_cuda18_qa_sha256']==QA_SHA
        and qa['original_plan_sha256']==PLAN_SHA and qa['original_source_manifest_sha256']==SOURCE_MANIFEST_SHA
        and qa['source_inventory']==plan['files'] and qa['manifest_sha256']==plan['source_manifest_sha256']
        and qa['recovery_evidence_sha256']==plan['recovery_evidence_sha256']
        and qa['previous_source_only_isolated13_noattempt_proof_sha256']==plan['previous_source_only_isolated13_noattempt_proof_sha256']
        and qa['previous_frozen13_noattempt_proof_sha256']==plan['previous_frozen13_noattempt_proof_sha256']
        and qa['previous_started13_preframework_failure_proof_sha256']==plan['previous_started13_preframework_failure_proof_sha256']
        and qa['new_CUDA_probe_performed'] is qa['measurement_array_read'] is qa['research_scores_read'] is qa['training_started'] is qa['automatic_retry'] is False
        and qa['original_measurement_reused_exactly'] is True and qa['rows']==[rows[r] for r in sorted(NEW13)],
        'exact original18 resource-only subset13 projection; never claim new CUDA')
    require(qa['rows'] and set(plan['runs'])==NEW13,'exact projected thirteen plan')
    for row in qa['rows']:
        cfg=plan['runs'][row['run_id']]
        require(row['config']==cfg and row['optimizer_steps']==2 and row['batch_size']==256
            and row['precision']=='FP32_TF32' and row['fullprefix_shape']==[cfg['prefix_rows'],cfg['channels']],
            'original measured B256 full-resident scientific config unchanged')
        gpu=max(1200,(max(row['peak_gpu_bytes'],row['peak_gpu_reserved_bytes'])+(1<<20)-1)//(1<<20)+512)
        disk=max(64<<20,row['parameter_count']*12+(16<<20))
        require(cfg['reserved_gpu_mib']==row['reserved_gpu_mib']==gpu and cfg['reserved_disk_bytes']==row['reserved_disk_bytes']==disk,
            'unchanged measured reservation formulas')
    require(re.fullmatch('[0-9a-f]{64}',qa['projection_claim_sha256']) is not None,'exact exclusive metadata-projection claim SHA')

def verify_framework_projection(qa):
    first=qa['rows'][0]
    for row in qa['rows']:
        require(row['framework_source_paths']==first['framework_source_paths'] and row['framework_source_sha256']==first['framework_source_sha256']
            and row['torch_version']==first['torch_version'] and row['numpy_version']==first['numpy_version'],
            'original exact common physical framework identity')
    for key,path in first['framework_source_paths'].items():
        p=Path(path);require(p.is_absolute() and p.resolve()==p and path.startswith('/root/miniconda3/lib/')
            and sha(p)==first['framework_source_sha256'][key],'fresh physical framework source hash differs from original actual measurement')
