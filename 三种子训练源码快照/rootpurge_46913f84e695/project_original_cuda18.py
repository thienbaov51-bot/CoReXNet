"""Explicit future projection of real frozen CUDA18 measurements; no new kernel."""
import argparse
import json
import time
from pathlib import Path
from root9_common import (verify_manifest, no_other_producers, ORIGINAL18,
    pinned, exclusive, pending, require, sha, MODEL_SHA, KERNEL_SHA)


def project(manifest_pin):
    root, manifest, evidence, original_plan = verify_manifest(manifest_pin)
    no_other_producers()
    from root18_resources import collect_telemetry, validate_resource_admission
    resource = collect_telemetry(root, {})
    validate_resource_admission(resource, {}, None)
    require(resource['gpu_compute_pids'] == [], 'projection requires idle physical GPU; no joint admission')
    original = pinned(Path(ORIGINAL18), 'ACTUAL_ROOT18_SYNTHETIC_CUDA_RECEIPT.json', original_plan['qa_sha256'])
    require(original['status'] == 'PASS_ALL18_ACTUAL_ORIGINAL_BATCH_SYNTHETIC_CUDA_ONLY'
            and original['source_inventory'] == original_plan['files']
            and original['measurement_array_read'] is False and original['research_scores_read'] is False,
            'actual original CUDA18 measurement provenance differs')
    by_run = {row['run_id']:row for row in original['rows']}
    require(len(by_run) == len(original['rows']) == 18, 'all18 original measured rows required')
    rows = [by_run[rid] for rid in manifest['authorized_run_ids']]
    for row in rows:
        require(row['status'] == 'PASS' and row['config'] == original_plan['runs'][row['run_id']]
                and row['optimizer_steps'] == 2 and row['config']['batch_size'] == 64,
                'measured original fullprefix/batch/resource row differs')
    claim = {'status':'ROOT_EXPLICIT_METADATA_ONLY_ORIGINAL_CUDA18_PROJECTION_CLAIM',
        'utc_epoch':time.time(),'manifest_sha256':manifest_pin,'automatic_retry':False,
        'new_kernel_probe_performed':False,'training_started':False,
        'original_cuda18_sha256':original_plan['qa_sha256'],'resource_telemetry':resource}
    exclusive(root / 'ROOT9_PROBE_CLAIM.json',claim)
    receipt = {'status':'ACTUAL_CUDA18_MEASURED_RESOURCE_PROJECTION_TO_UNTOUCHED9_NO_NEW_KERNEL',
        'utc_epoch':time.time(),'manifest_sha256':manifest_pin,'source_inventory':manifest['files'],
        'recovery_evidence_sha256':manifest['recovery_evidence_sha256'],
        'probe_claim_sha256':sha(root / 'ROOT9_PROBE_CLAIM.json'),
        'original_cuda18_sha256':original_plan['qa_sha256'],
        'original_scientific_runtime_sha256':KERNEL_SHA,'original_scientific_model_sha256':MODEL_SHA,
        'rows':rows,'resource_telemetry':resource,'original_measured_rows_reused':True,
        'new_kernel_probe_performed':False,'scientific_math_changed':False,'measurement_array_read':False,
        'research_scores_read':False,'automatic_retry':False,'training_started':False}
    exclusive(root / 'ACTUAL_ROOT9_ORIGINAL_CUDA18_RESOURCE_PROJECTION.json',receipt)
    print(json.dumps({'status':receipt['status'],'measured_original_rows_referenced':len(rows),
        'new_kernel_probe_performed':False,'qa_sha256':sha(root / 'ACTUAL_ROOT9_ORIGINAL_CUDA18_RESOURCE_PROJECTION.json')}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project-original-cuda18-after-root-review',action='store_true')
    parser.add_argument('--manifest-sha256');args=parser.parse_args()
    if not args.project_original_cuda18_after_root_review or not args.manifest_sha256:
        print(json.dumps(pending()));return
    project(args.manifest_sha256)


if __name__=='__main__':main()
