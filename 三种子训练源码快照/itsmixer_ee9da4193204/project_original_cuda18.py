"""One resource metadata projection; no framework, arrays, model or GPU probe."""
import argparse,json,time
from itsmixer5_common import *
from itsmixer5_upstream import original_qa,verify_framework_projection

def project_once(pin):
    root,manifest,recipe,documents=verify_manifest(pin)
    require(not (root/'ITSMIXER5_PROJECTION_CLAIM.json').exists()
        and not (root/'ACTUAL_ITSMIXER5_ORIGINAL_CUDA18_RESOURCE_PROJECTION.json').exists(), 'never repeat original-resource metadata projection')
    old=gate_old_cohorts();gpu=gpu_idle_except()
    require(not gpu['gpu_compute_pids'],'fresh idle before metadata projection')
    baseline=original_qa();allrows={r['run_id']:r for r in baseline['rows']}
    selected=[allrows[r] for r in sorted(RUN_IDS)]
    verify_framework_projection({'rows':selected})
    claim={'status':'ACTUAL_ITSMIXER5_ONE_RESOURCE_METADATA_PROJECTION_CLAIM_NO_CUDA','utc_epoch':time.time(),
        'manifest_sha256':pin,'source_inventory':manifest['files'],'recovery_evidence_sha256':manifest['recovery_evidence_sha256'],
        'previous_s3_eight_complete_five_no_scientific_claims_proof_sha256':manifest['recovery_evidence_sha256'],
        'previous_source_only_isolated13_noattempt_proof_sha256':manifest['previous_source_only_isolated13_noattempt_proof_sha256'],
        'previous_frozen13_noattempt_proof_sha256':manifest['previous_frozen13_noattempt_proof_sha256'],'previous_started13_preframework_failure_proof_sha256':manifest['previous_started13_preframework_failure_proof_sha256'],
        'new_CUDA_probe_performed':False,'training_started':False,'automatic_retry':False,'test_accessed':False}
    exclusive(root/'ITSMIXER5_PROJECTION_CLAIM.json',claim)
    result={'status':'PASS_ITSMIXER5_EXACT_ORIGINAL_CUDA18_RESOURCE_METADATA_PROJECTION_NO_NEW_CUDA','utc_epoch':time.time(),
        'child_root':CHILD,'original_root':ORIGINAL+'/itsmixer_remaining18','original_plan_sha256':'3f62ec743c9e0af9d427e80abbbc8e7353eedc93958369939e8caea0fac2d8fd',
        'original_cuda18_qa_sha256':'06d419caf4dc8b6d10bc437639e12e879cd349fdc469ebc3f6859b3fe214b792',
        'original_source_manifest_sha256':'9ba34ed9d3d87381f34d5a2993b6ab3d0d39a5ae61a514d00877d8a9de7c62f5',
        'manifest_sha256':pin,'source_inventory':manifest['files'],'recovery_evidence_sha256':manifest['recovery_evidence_sha256'],
        'previous_s3_eight_complete_five_no_scientific_claims_proof_sha256':manifest['recovery_evidence_sha256'],
        'previous_source_only_isolated13_noattempt_proof_sha256':manifest['previous_source_only_isolated13_noattempt_proof_sha256'],
        'previous_frozen13_noattempt_proof_sha256':manifest['previous_frozen13_noattempt_proof_sha256'],'previous_started13_preframework_failure_proof_sha256':manifest['previous_started13_preframework_failure_proof_sha256'],
        'rows':selected,'original_measurement_reused_exactly':True,'projection_claim_sha256':sha(root/'ITSMIXER5_PROJECTION_CLAIM.json'),
        'old_cohort_admission':old,'fresh_gpu_idle':gpu,'new_CUDA_probe_performed':False,'measurement_array_read':False,
        'research_scores_read':False,'training_started':False,'automatic_retry':False,'test_accessed':False}
    plan=source_template({**manifest,'manifest_sha256':pin},recipe,{r['run_id']:r['config'] for r in selected})
    validate_plan(plan,recipe);validate_qa(result,plan)
    exclusive(root/'ACTUAL_ITSMIXER5_ORIGINAL_CUDA18_RESOURCE_PROJECTION.json',result)
    return {'status':result['status'],'receipt_name':'ACTUAL_ITSMIXER5_ORIGINAL_CUDA18_RESOURCE_PROJECTION.json','projection_claim_name':'ITSMIXER5_PROJECTION_CLAIM.json','new_CUDA_probe_performed':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--project-once',action='store_true');p.add_argument('--manifest-sha256');a=p.parse_args()
    print(json.dumps(project_once(a.manifest_sha256) if a.project_once and a.manifest_sha256 else pending()))
if __name__=='__main__':main()
