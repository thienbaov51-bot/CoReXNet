"""Future explicit Linux ROOT deploy/freeze operator, source preparation only.

No SSH transport, training, probing, framework or score decoder. Default no I/O.
ROOT transfers this reviewed source package and review receipts into an external
staging directory before an explicit deploy. No existing child is overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from root9_common import (CHILD, ORIGINAL18, ORIGINAL18_PLAN_SHA, MODEL_SHA,
    KERNEL_SHA, RECIPE_SHA, SOURCE_FILES, require, metadata, pinned, sha,
    safe_path, verify_original, exclusive, pending, validate_plan, validate_static)


def digest(value):
    return hashlib.sha256(json.dumps(value, indent=2, allow_nan=False).encode()).hexdigest()


def require_root():
    require(sys.platform == 'linux' and os.geteuid() == 0, 'explicit future Linux ROOT required')


def deploy(package_pin, root_review_pin, independent_review_pin, evidence_path, evidence_pin):
    require_root(); stage = Path(__file__).absolute().parent
    require(str(stage) != CHILD and stage.resolve() == stage, 'deploy from a separate reviewed external source staging directory')
    package = pinned(stage, 'ROOT9_SOURCE_AUTHOR_PACKAGE_MANIFEST.json', package_pin)
    require(package['status'] == 'SOURCE_ONLY_DISTINCT_ROOT9_WRAPPER_READY_NO_ACTUAL_AUTHORITY'
            and package['child_root'] == CHILD and package['original18_plan_sha256'] == ORIGINAL18_PLAN_SHA,
            'reviewed source-only recovery package identity differs')
    for name, sourcepin in package['source_files'].items():
        require(sha(safe_path(stage,name)) == sourcepin, 'local reviewed source package bytes differ')
    require(set(package['source_files']) == {*SOURCE_FILES,'operate_root9_pending.py'}, 'exact reviewed source member set required')
    require(package['source_files']['RootPurge_classes_exact.py'] == MODEL_SHA
            and package['source_files']['root18_trainval_runtime_pending.py'] == KERNEL_SHA
            and package['source_files']['FUTURE_ROOT18_SOURCE_RECIPE_MAP_PENDING.json'] == RECIPE_SHA,
            'original scientific bytes must remain unchanged')
    evidence_file = Path(evidence_path)
    expected_evidence_parent = Path(ORIGINAL18).parent / 'rootpurge_recovery9_evidence'
    require(evidence_file.name == 'ACTUAL_ROOT_CHAIN_RECOVERY9_EVIDENCE.json'
            and evidence_file.parent.parent == expected_evidence_parent
            and evidence_file.parent.name.startswith('ACTUAL_ROOT_CHAIN_RECOVERY9_EVIDENCE_')
            and sha(evidence_file) == evidence_pin, 'exact external ROOT collection receipt required')
    evidence = metadata(evidence_file); original_plan, queue, _, _ = verify_original(evidence)
    files = {name:package['source_files'][name] for name in SOURCE_FILES}
    files.update({name:pin for name,pin in original_plan['files'].items() if name.startswith('data/')})
    snapshot = {'files':files,'recovery_evidence_sha256':evidence_pin,'original18_plan_sha256':ORIGINAL18_PLAN_SHA,
        'child_root':CHILD,'authorized_run_ids':sorted(queue)}
    for name,pin,status in (('ROOT_ROOT9_LOCAL_SOURCE_REVIEW.json',root_review_pin,'PASS_ROOT_ROOT9_LOCAL_SOURCE_REVIEW'),
                ('INDEPENDENT_ROOT9_LOCAL_SOURCE_REVIEW.json',independent_review_pin,'PASS_INDEPENDENT_ROOT9_LOCAL_SOURCE_REVIEW')):
        review = pinned(stage,name,pin)
        require(review['status'] == status and review['snapshot'] == snapshot
                and review['source_package_sha256'] == package_pin and review['actual_scores_read'] is False
                and review['actual_arrays_read'] is False and review['scientific_math_changed'] is False,
                'new exact local ROOT and independent source reviews required')
    require(package['source_files']['collect_root_chain_recovery9.py'] == evidence['collector_source_sha256'],
            'recovery evidence collector version differs')
    root = Path(CHILD)
    require(not root.exists() and all(not part.is_symlink() for part in (root,*root.parents)), 'distinct recovery child must be absent')
    root.mkdir()
    for name in package['source_files']:
        target = safe_path(root,name)
        with safe_path(stage,name).open('rb') as source, target.open('xb') as output:
            shutil.copyfileobj(source,output)
    for name in files:
        if name.startswith('data/'):
            target = safe_path(root,name);target.parent.mkdir(exist_ok=True)
            with safe_path(Path(ORIGINAL18),name).open('rb') as source, target.open('xb') as output:
                shutil.copyfileobj(source,output)
            require(sha(target) == files[name], 'actual copied original prefix/calendar SHA differs')
    for name,pin in (('ROOT_ROOT9_LOCAL_SOURCE_REVIEW.json',root_review_pin),
                     ('INDEPENDENT_ROOT9_LOCAL_SOURCE_REVIEW.json',independent_review_pin)):
        with safe_path(stage,name).open('rb') as source, safe_path(root,name).open('xb') as output:
            shutil.copyfileobj(source,output)
    with evidence_file.open('rb') as source, (root/'ROOT_CHAIN_RECOVERY9_EVIDENCE.json').open('xb') as output:
        shutil.copyfileobj(source,output)
    manifest = {'status':'REVIEWED_DISTINCT_ROOT9_ENVIRONMENT_RECOVERY_SOURCE_DEPLOYMENT_ONLY',
        'utc_epoch':time.time(),'child_root':CHILD,'original18_root':ORIGINAL18,'original18_plan_sha256':ORIGINAL18_PLAN_SHA,
        'files':files,'authorized_run_ids':sorted(queue),'data_metadata':original_plan['data_metadata'],
        'recovery_evidence_sha256':evidence_pin,'root_source_review_sha256':root_review_pin,
        'independent_source_review_sha256':independent_review_pin,'source_package_sha256':package_pin,
        'operator_source_sha256':package['source_files']['operate_root9_pending.py'],
        'scientific_math_changed':False,'training_started':False,'test_allowed':False,'weights_copied':False}
    exclusive(root/'ROOT9_SOURCE_MANIFEST.json',manifest)
    exclusive(root/'ACTUAL_ROOT9_DEPLOYMENT_RECEIPT.json',{'status':'ACTUAL_ROOT9_DISTINCT_SOURCE_PREFIX_COPY_NO_START',
        'utc_epoch':time.time(),'source_manifest_sha256':sha(root/'ROOT9_SOURCE_MANIFEST.json'),
        'source_package_sha256':package_pin,'copied_prefix_calendar_files':{name:pin for name,pin in files.items() if name.startswith('data/')},
        'new_scores_accessed':False,'arrays_opened':False,'training_started':False,'weights_copied':False})
    print(json.dumps({'status':manifest['status'],'child_root':CHILD,
        'source_manifest_sha256':sha(root/'ROOT9_SOURCE_MANIFEST.json'),'training_started':False}))


def freeze(manifest_pin, projection_pin, root_review_pin, independent_review_pin):
    require_root()
    from root9_common import verify_manifest
    root,manifest,evidence,original_plan = verify_manifest(manifest_pin)
    require(Path(__file__).absolute().parent == root, 'freeze uses the deployed reviewed recovery operator')
    qa = pinned(root,'ACTUAL_ROOT9_ORIGINAL_CUDA18_RESOURCE_PROJECTION.json',projection_pin)
    claimpin=sha(root/'ROOT9_PROBE_CLAIM.json');claim=pinned(root,'ROOT9_PROBE_CLAIM.json',claimpin)
    root_review=pinned(root,'ROOT_ROOT9_SOURCE_RESOURCE_REVIEW.json',root_review_pin)
    peer_review=pinned(root,'INDEPENDENT_ROOT9_SOURCE_RESOURCE_REVIEW.json',independent_review_pin)
    for name in ('ROOT9_PLAN.json','ROOT_ROOT9_START_AUTHORIZATION.json','ACTUAL_ROOT9_FREEZE_RECEIPT.json',
                 'ROOT9_ACTUAL_START.json','ROOT9_START_CLAIM.json','ROOT9_STATE.json','results','logs'):
        require(not (root/name).exists(), 'exclusive recovery freeze; no resume or retry')
    plan={'status':'ROOT_FROZEN_ROOT9_ENVIRONMENT_ONLY_RECOVERY_VALIDATION_ONLY','child_root':CHILD,
        'original18_root':ORIGINAL18,'original18_plan_sha256':ORIGINAL18_PLAN_SHA,
        'runs':original_plan['runs'],'authorized_run_ids':manifest['authorized_run_ids'],
        'preserved_complete9':evidence['original_complete9'],'data_metadata':original_plan['data_metadata'],'files':manifest['files'],
        'source_manifest_sha256':manifest_pin,'recovery_evidence_sha256':manifest['recovery_evidence_sha256'],
        'qa_sha256':projection_pin,'probe_claim_sha256':claimpin,'original_cuda18_sha256':original_plan['qa_sha256'],
        'root_source_resource_review_sha256':root_review_pin,'independent_source_resource_review_sha256':independent_review_pin,
        'scientific_runtime_sha256':KERNEL_SHA,'scientific_model_sha256':MODEL_SHA,'all18_recipe_sha256':RECIPE_SHA,
        'test_allowed':False,'automatic_retry':False,'weights_copied':False,'scientific_math_changed':False,
        'retained_completed9_retraining_allowed':False,'h96_retraining_allowed':False,'two_model_pair_selected':False,
        'resource_policy':{'max_workers':8,'gpu_budget_mib':22500,'active_disk_cap_bytes':4<<30,
            'actual_disk_floor_bytes':1536<<20,'gpu_count':1,'unknown_gpu_process_blocks_admission':True}}
    pin=digest(plan)
    auth={'status':'ROOT_AUTHORIZED_DISTINCT_UNATTEMPTED_ROOT9_AFTER_NEW_SOURCE_RESOURCE_REVIEWS',
        'utc_epoch':time.time(),'plan_sha256':pin,'files':plan['files'],'run_ids':plan['authorized_run_ids'],
        'controller_one_attempt_allowed':True,'automatic_retry':False,'new_test_allowed':False,
        'retained_completed9_retraining_allowed':False,'joint_gpu_authorization_allowed':False,
        **{key:plan[key] for key in ('source_manifest_sha256','recovery_evidence_sha256','qa_sha256','probe_claim_sha256',
            'root_source_resource_review_sha256','independent_source_resource_review_sha256')}}
    authpin=digest(auth)
    freeze_receipt={'status':'ACTUAL_ROOT9_SOURCE_RESOURCE_PLAN_FROZEN_NO_START','utc_epoch':time.time(),
        'plan_sha256':pin,'authorization_sha256':authpin,'qa_sha256':projection_pin,
        'training_started':False,'scores_accessed':False,'original_cuda18_sha256':original_plan['qa_sha256'],
        'new_kernel_probe_performed':False,'scientific_math_changed':False}
    original_qa=pinned(Path(ORIGINAL18),'ACTUAL_ROOT18_SYNTHETIC_CUDA_RECEIPT.json',original_plan['qa_sha256'])
    validate_static(plan,pin,auth,qa,claim,root_review,peer_review,freeze_receipt,manifest,evidence,original_plan,original_qa)
    from root9_common import no_other_producers
    from root18_resources import collect_telemetry,validate_resource_admission
    no_other_producers();resource=collect_telemetry(root,{})
    validate_resource_admission(resource,{},None);require(not resource['gpu_compute_pids'],'freeze requires physical GPU idle')
    exclusive(root/'ROOT9_PLAN.json',plan)
    exclusive(root/'ROOT_ROOT9_START_AUTHORIZATION.json',auth)
    exclusive(root/'ACTUAL_ROOT9_FREEZE_RECEIPT.json',freeze_receipt)
    print(json.dumps({'status':freeze_receipt['status'],'plan_sha256':pin,'authorization_sha256':authpin,
        'training_started':False,'new_kernel_probe_performed':False}))


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group();group.add_argument('--deploy-after-reviewed',action='store_true')
    group.add_argument('--freeze-after-reviewed',action='store_true')
    for name in ('package-sha256','root-local-review-sha256','independent-local-review-sha256',
                 'evidence-path','evidence-sha256','manifest-sha256','projection-sha256',
                 'root-resource-review-sha256','independent-resource-review-sha256'):
        parser.add_argument('--'+name)
    args=parser.parse_args()
    if args.deploy_after_reviewed:
        require(all((args.package_sha256,args.root_local_review_sha256,args.independent_local_review_sha256,
            args.evidence_path,args.evidence_sha256)),'all exact deployment proof pins required')
        deploy(args.package_sha256,args.root_local_review_sha256,args.independent_local_review_sha256,args.evidence_path,args.evidence_sha256)
    elif args.freeze_after_reviewed:
        require(all((args.manifest_sha256,args.projection_sha256,args.root_resource_review_sha256,args.independent_resource_review_sha256)),
            'all exact actual resource/projection review pins required')
        freeze(args.manifest_sha256,args.projection_sha256,args.root_resource_review_sha256,args.independent_resource_review_sha256)
    else:
        print(json.dumps(pending()))


if __name__=='__main__':main()
