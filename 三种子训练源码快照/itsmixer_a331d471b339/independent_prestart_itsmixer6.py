"""Score-free actual six-prefix auditor; explicit invocation only, default PENDING."""
import argparse
import hashlib
import json
import os
import time
from itsmixer6_common import *


def review(pin,authpin,run_id=None,owned=None,active_configs=None):
    root,plan,recipe,documents,freeze_pin=load_frozen(pin,authpin)
    old=gate_old_cohorts();owned={} if owned is None else owned;active_configs={} if active_configs is None else active_configs
    from itsmixer6_resources import collect_telemetry,validate_resource_admission
    candidate=plan['runs'][run_id] if run_id is not None else None
    telemetry=collect_telemetry(root,owned);validate_resource_admission(telemetry,active_configs,candidate)
    identity=None
    if run_id is None:require(not telemetry['gpu_compute_pids'],'initial audit GPU must be empty')
    else:
        start_path=safe_path(root,'ITSMIXER6_ACTUAL_START.json');identity=verify_controller_identity(metadata(start_path),pin,authpin)
        require(identity['pid']==os.getpid(),'only the explicitly owned controller audits queued runs')
        identity['actual_start_sha256']=sha(start_path)
    # Every actual authority/source/whole16/reference/resource gate precedes
    # prefix value decoding. No TEST, history or validation result is opened.
    import numpy as np
    datasets={};total=0
    for ds in sorted(DATASETS):
        data=plan['data_metadata'][ds]
        with np.load(root/'data'/(ds+'_trainval.npz'),allow_pickle=False) as copied, \
                np.load(Path(ORIGINAL)/'data'/(ds+'_trainval.npz'),allow_pickle=False) as original:
            require(set(copied.files)==set(original.files)=={'data','split_counts'},'TRAIN+VAL prefix fields differ')
            new,old_values=copied['data'],original['data'];bounds=copied['split_counts']
            require(new.dtype==old_values.dtype==np.dtype('float32') and new.shape==(data['prefix_rows'],data['channels'])
                and bounds.tolist()==data['bounds'][:2] and np.array_equal(bounds,original['split_counts'])
                and np.array_equal(new.view(np.uint32),old_values.view(np.uint32)) and np.isfinite(new).all(),
                'actual full prefix bitwise/finite split identity differs')
            mean=new[:bounds[0]].mean(0,dtype=np.float64).astype(np.float32)
            scale=new[:bounds[0]].std(0,dtype=np.float64).astype(np.float32);scale[scale<1e-6]=1
            require(hashlib.sha256(mean.tobytes()).hexdigest()==data['scaler_mean_sha256']
                and hashlib.sha256(scale.tobytes()).hexdigest()==data['scaler_std_sha256'],'TRAIN-only scaler SHA differs')
            total+=new.size
        datasets[ds]={key:data[key] for key in ('prefix_sha256','scaler_mean_sha256','scaler_std_sha256','prefix_rows','channels')}
        datasets[ds]['prefix_bitwise_elements']=data['prefix_rows']*data['channels']
        del new,old_values,bounds,mean,scale
    telemetry=collect_telemetry(root,owned);validate_resource_admission(telemetry,active_configs,candidate)
    gate_old_cohorts();verify_references(documents['references'])
    name=receipt_name('ACTUAL_INDEPENDENT_ITSMIXER6_PRESTART_');now=time.time()
    pre={'status':'PASS_ACTUAL_INDEPENDENT_ITSMIXER6_PREFIX_RESOURCE_PRESTART','receipt_name':name,
        'utc_epoch':now,'expires_after_seconds':120,'plan_sha256':pin,'authorization_sha256':authpin,'freeze_sha256':freeze_pin,
        'files':plan['files'],'qa_sha256':plan['qa_sha256'],'replacement_sha256':plan['replacement_sha256'],'root_pre_score_policy_sha256':ROOT_POLICY_SHA,
        'auditor_source_sha256':plan['files']['independent_prestart_itsmixer6.py'],'datasets':datasets,
        'prefix_bitwise_elements_checked':total,'test_scores_accessed':False,'new_validation_scores_accessed':False,
        'training_started_by_auditor':False,'old_cohort_admission':old,'resource_telemetry':telemetry,
        'owned':owned,'active_configs':active_configs,'candidate_run_id':run_id,'controller_identity':identity,'automatic_retry':False}
    rid=run_id or sorted(RUN_IDS)[0]
    virtual={'status':'ROOT_ITSMIXER6_ONE_RUN_CAPABILITY_AFTER_FRESH_INDEPENDENT_PRESTART','run_id':rid,'plan_sha256':pin,
        'run_config_sha256':config_sha(plan['runs'][rid]),'automatic_retry':False,'test_allowed':False,
        'utc_epoch':now,'expires_after_seconds':120}
    validate_evidence(plan,recipe,pin,documents,now,pre,virtual,rid)
    exclusive(root/name,pre)
    return name,pre


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--review-start-once',action='store_true')
    parser.add_argument('--plan-sha256');parser.add_argument('--authorization-sha256');args=parser.parse_args()
    if not args.review_start_once or not args.plan_sha256 or not args.authorization_sha256:
        print(json.dumps(pending()));return
    name,pre=review(args.plan_sha256,args.authorization_sha256)
    print(json.dumps({'status':pre['status'],'receipt_name':name}))


if __name__=='__main__':main()
