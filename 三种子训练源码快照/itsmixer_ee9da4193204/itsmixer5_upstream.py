"""Serial5 score-free recovery admission and original measured-resource projection."""
import hashlib,json,re,subprocess,sys
from pathlib import Path

PARENT='/root/autodl-tmp/paper5/models2026_pilot_20261011'
ORIGINAL=PARENT+'/itsmixer_recovery13_isolated3'
CHILD=PARENT+'/itsmixer_recovery5_serial'
PLAN_SHA='4a882c6df5eb4f81bc26b8f2a2c9b439c952e274ccd5a47dd421c5d9182f0ced'
AUTH_SHA='ebd9ad7b77196452aa9d6415193b22ce5f06ae5eca8aed69cfeb61b4c1794319'
START_SHA='a3dbe40fed6ef10592757501611601575f6759e62f8be8376f5479a6c2994f88'
SOURCE_PACKAGE_SHA='259016e677ef5877c90581e2511a16ec0083c9b97c2d5263442742e6aa2628fc'
SOURCE_MANIFEST_SHA='d488309f47bd4ed5b747499201fa62412a5d2397ca34f7a292bcf46802b27c82'
COLLECTOR_SHA='610cda40637055de934369e50fdd9ce4ecfec6b8b0cf3bae838680388eec7fc7'
QA_SHA='06d419caf4dc8b6d10bc437639e12e879cd349fdc469ebc3f6859b3fe214b792'
RESOURCE_ROOT=PARENT+'/itsmixer_remaining18'
RESOURCE_PLAN_SHA='3f62ec743c9e0af9d427e80abbbc8e7353eedc93958369939e8caea0fac2d8fd'
RESOURCE_SOURCE_MANIFEST_SHA='9ba34ed9d3d87381f34d5a2993b6ab3d0d39a5ae61a514d00877d8a9de7c62f5'
NEW5=frozenset(('ITSMixer_PEMS03_H336','ITSMixer_PEMS08_H720','ITSMixer_Solar_H192','ITSMixer_Solar_H336','ITSMixer_Solar_H720'))
ALL18=frozenset('ITSMixer_'+d+'_H'+str(h) for d in ('Electricity','PEMS03','PEMS04','PEMS08','Solar','AEMO5') for h in (192,336,720))

def require(value,message):
    if not value:raise ValueError(message)
def sha(path):
    path=Path(path);require(all(not part.is_symlink() for part in (path,*path.parents)),'opaque upstream symlink')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()
def original_qa():
    path=Path(__file__).absolute().parent/'ORIGINAL_ACTUAL_CUDA18_RESOURCE_RECEIPT.json'
    require(sha(path)==QA_SHA,'exact original measured CUDA18 receipt')
    result=json.loads(path.read_bytes())
    require(result['status']=='PASS_ALL18_ACTUAL_ORIGINAL_BATCH_ITSMIXER_SYNTHETIC_CUDA_ONLY'
        and len(result['rows'])==18 and {row['run_id'] for row in result['rows']}==ALL18
        and result['measurement_array_read'] is result['research_scores_read'] is result['training_started'] is False,'original resource-only actual evidence')
    return result

def verify_original_recovery(evidence):
    require(type(COLLECTOR_SHA) is str and type(SOURCE_MANIFEST_SHA) is str,'PENDING exact independently reviewed serial5 collector SOURCE pins')
    root=Path(__file__).absolute().parent;source=root/'collect_itsmixer_s3_recovery5_rootgrid3.py'
    require(sha(source)==COLLECTOR_SHA,'current collector SOURCE byte pin before import')
    import collect_itsmixer_s3_recovery5_rootgrid3 as collector
    require(sha(Path(collector.__file__))==COLLECTOR_SHA,'exact standalone recovery5 verifier')
    collector.verify_evidence_shape(evidence)
    require(evidence['original_root']==ORIGINAL and evidence['child_root']==CHILD
        and evidence['original_plan_sha256']==PLAN_SHA and evidence['original_authorization_sha256']==AUTH_SHA
        and evidence['original_actual_start_sha256']==START_SHA and evidence['original_source_manifest_sha256']==SOURCE_MANIFEST_SHA
        and evidence['original_cuda18_qa_sha256']==QA_SHA and evidence['remaining5_sorted']==sorted(NEW5)
        and set(evidence['runs'])==NEW5 and len(evidence['retained_original8_references'])==8,'genuine S3 eight preserved + five no scientific claims')
    # Replays must remain clean even if called after a completed worker imported
    # the framework. The historical source validator is never imported into the
    # new worker's live module namespace.
    proof=root/'ITSMIXER5_RECOVERY_EVIDENCE.json';pin=sha(proof)
    code="import hashlib,json,sys;from pathlib import Path;r=Path(sys.argv[1]);sys.path.insert(0,str(r));import collect_itsmixer_s3_recovery5_rootgrid3 as c;assert c.sha((r/'collect_itsmixer_s3_recovery5_rootgrid3.py').read_bytes())==sys.argv[2];p=r/'ITSMIXER5_RECOVERY_EVIDENCE.json';assert c.sha(p.read_bytes())==sys.argv[3];v=c.decode(p.read_bytes());c.verify_evidence_shape(v);c.verify_evidence_physical(v);assert c.sha(p.read_bytes())==sys.argv[3];assert 'numpy' not in sys.modules and 'torch' not in sys.modules;print(json.dumps({'status':'PASS_CLEAN_SERIAL5_HISTORICAL_OPAQUE_REPLAY','evidence_sha256':sys.argv[3]}))"
    result=subprocess.run([sys.executable,'-I','-c',code,str(root),COLLECTOR_SHA,pin],capture_output=True,timeout=120,check=False)
    require(result.returncode==0 and not result.stderr,'clean historical opaque replay failed')
    require(json.loads(result.stdout)=={'status':'PASS_CLEAN_SERIAL5_HISTORICAL_OPAQUE_REPLAY','evidence_sha256':pin}
        and sha(source)==COLLECTOR_SHA and sha(proof)==pin,'historical replay exact response and immutable bytes')
    return evidence

def validate_projection(qa,plan):
    rows={row['run_id']:row for row in original_qa()['rows']}
    require(qa['status']=='PASS_ITSMIXER5_EXACT_ORIGINAL_CUDA18_RESOURCE_METADATA_PROJECTION_NO_NEW_CUDA'
        and qa['child_root']==CHILD and qa['original_root']==RESOURCE_ROOT and qa['original_cuda18_qa_sha256']==QA_SHA
        and qa['original_plan_sha256']==RESOURCE_PLAN_SHA and qa['original_source_manifest_sha256']==RESOURCE_SOURCE_MANIFEST_SHA
        and qa['source_inventory']==plan['files'] and qa['manifest_sha256']==plan['source_manifest_sha256']
        and qa['recovery_evidence_sha256']==plan['recovery_evidence_sha256']
        and qa['previous_s3_eight_complete_five_no_scientific_claims_proof_sha256']==plan['recovery_evidence_sha256']
        and all(qa[key]==plan[key] for key in ('previous_frozen13_noattempt_proof_sha256','previous_source_only_isolated13_noattempt_proof_sha256','previous_started13_preframework_failure_proof_sha256'))
        and qa['new_CUDA_probe_performed'] is qa['measurement_array_read'] is qa['research_scores_read'] is qa['training_started'] is qa['automatic_retry'] is False
        and qa['original_measurement_reused_exactly'] is True and qa['rows']==[rows[rid] for rid in sorted(NEW5)]
        and set(plan['runs'])==NEW5,'exact original18 measured subset5 projection; no new CUDA authority')
    for row in qa['rows']:
        cfg=plan['runs'][row['run_id']]
        require(row['config']==cfg and row['optimizer_steps']==2 and row['batch_size']==256
            and row['precision']=='FP32_TF32' and row['fullprefix_shape']==[cfg['prefix_rows'],cfg['channels']],'unchanged measured full-resident B256 config')
        gpu=max(1200,(max(row['peak_gpu_bytes'],row['peak_gpu_reserved_bytes'])+(1<<20)-1)//(1<<20)+512)
        disk=max(64<<20,row['parameter_count']*12+(16<<20))
        require(cfg['reserved_gpu_mib']==row['reserved_gpu_mib']==gpu and cfg['reserved_disk_bytes']==row['reserved_disk_bytes']==disk,'unchanged measured reserve formulas')
    require(re.fullmatch('[0-9a-f]{64}',qa['projection_claim_sha256']) is not None,'exclusive once projection claim SHA')

def verify_framework_projection(qa):
    first=qa['rows'][0]
    for row in qa['rows']:
        require(row['framework_source_paths']==first['framework_source_paths'] and row['framework_source_sha256']==first['framework_source_sha256']
            and row['torch_version']==first['torch_version'] and row['numpy_version']==first['numpy_version'],'same measured library environment')
    for key,path in first['framework_source_paths'].items():
        source=Path(path);require(source.is_absolute() and source.resolve()==source and path.startswith('/root/miniconda3/lib/')
            and sha(source)==first['framework_source_sha256'][key],'fresh framework source bytes only')
