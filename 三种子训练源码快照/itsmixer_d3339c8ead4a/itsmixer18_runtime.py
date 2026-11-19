"""One admitted future ITSMixer18 worker; default PENDING, no imports of arrays."""
import argparse, hashlib, importlib.util, json, traceback
from pathlib import Path
from itsmixer18_common import *
from itsmixer18_runtime_guard import admit_run

def run(capability):
    config=admit_run(**capability)
    root=Path(CHILD);rid=capability['run_id'];folder=safe_path(root,'results/'+rid)
    folder.mkdir() # exclusive scientific attempt has already been durably claimed
    try:
        # All SOURCE, prefix/scaler metadata, ownership, capability and stable
        # resource gates and exclusive attempt claim precede these imports.
        import numpy as np
        import torch
        from itsmixer18_science_kernel import configure_framework,build_model,ResidentWindows,trainval_science_core
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one actual CUDA GPU')
        configure_framework(torch,np)
        with np.load(safe_path(root,'data/'+config['dataset']+'_trainval.npz'),allow_pickle=False) as archive:
            require(set(archive.files)=={'data','split_counts'},'full prefix native two fields')
            values=archive['data'];bounds=archive['split_counts']
            require(values.dtype==np.dtype('float32') and values.shape==(config['prefix_rows'],config['channels'])
                and bounds.tolist()==[config['train_end'],config['validation_end']] and np.isfinite(values).all(),'native full finite TRAIN/VAL prefix')
            mean=values[:config['train_end']].mean(0,dtype=np.float64).astype(np.float32)
            scale=values[:config['train_end']].std(0,dtype=np.float64).astype(np.float32);scale[scale<1e-6]=1
            require(hashlib.sha256(mean.tobytes()).hexdigest()==config['scaler_mean_sha256']
                and hashlib.sha256(scale.tobytes()).hexdigest()==config['scaler_std_sha256'],'TRAIN-only original scalar identity')
            standardized=np.asarray((values-mean)/scale,dtype=np.float32)
        tensor=torch.from_numpy(standardized).to('cuda');del values,bounds,mean,scale,standardized
        spec=importlib.util.spec_from_file_location('itsmixer_exact_owned',root/config['model_source_file'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model=build_model(module,config);resident=ResidentWindows(torch,tensor,config)
        outcome=trainval_science_core(torch,model,resident,config)
        with (folder/'best.pt').open('xb') as f:torch.save(outcome['best_state'],f)
        with (folder/'config.json').open('xb') as f:f.write(encoded(config))
        exclusive(folder/'history.json',outcome['history'])
        validation={**outcome['validation'],'status':'VALIDATION_ONLY','run_id':rid,'model':'ITSMixer',
            'dataset':config['dataset'],'plan_sha256':capability['plan_sha256'],'seed':2026,'length':96,'horizon':config['horizon'],
            'epochs':outcome['epochs'],'steps':outcome['steps'],'best_epoch':outcome['best_epoch'],'best_step':outcome['best_step'],
            'train_windows_available':config['train_windows_available'],'train_windows_used_per_epoch':config['train_windows_used_per_epoch'],
            'train_windows_dropped_per_epoch':config['train_windows_dropped_per_epoch'],
            'validation_windows':config['validation_windows'],'validation_elements':config['validation_elements'],
            'criterion':config['criterion'],'checkpoint_criterion_name':config['checkpoint_criterion'],
            'root_pre_score_policy_sha256':ROOT_POLICY_SHA,'test_accessed':False,'automatic_retry':False}
        exclusive(folder/'VALIDATION_RESULT.json',validation)
        attempt=root/('ITSMIXER18_RUN_ATTEMPT_'+rid+'.json');claim=metadata(attempt)
        cap=pinned(root,capability['capability_name'],capability['capability_sha256'])
        plan=pinned(root,'ITSMIXER18_PLAN.json',capability['plan_sha256'])
        receipt={'status':'COMPLETE_VALIDATION_ONLY','run_id':rid,'plan_sha256':capability['plan_sha256'],
            'root_authorization_sha256':capability['authorization_sha256'],'run_capability_sha256':capability['capability_sha256'],
            'run_attempt_claim_sha256':sha(attempt),'prestart_receipt_sha256':claim['prestart_sha256'],'prestart_receipt_name':claim['prestart_name'],
            'source_inventory_sha256':hashlib.sha256(encoded(plan['files'])).hexdigest(),'root_pre_score_policy_sha256':ROOT_POLICY_SHA,
            'epochs':outcome['epochs'],'steps':outcome['steps'],'best_epoch':outcome['best_epoch'],'best_step':outcome['best_step'],
            'train_windows_available':config['train_windows_available'],'train_windows_used_per_epoch':config['train_windows_used_per_epoch'],
            'train_windows_dropped_per_epoch':config['train_windows_dropped_per_epoch'],'validation_windows':config['validation_windows'],
            'validation_elements':config['validation_elements'],'checkpoint_criterion_name':config['checkpoint_criterion'],
            'actual_synthetic_qa_sha256':plan['qa_sha256'],'test_accessed':False,'scores_released':False,'automatic_retry':False}
        for key,name in (('checkpoint_sha256','best.pt'),('config_sha256','config.json'),('history_sha256','history.json'),('validation_result_sha256','VALIDATION_RESULT.json')):
            receipt[key]=sha(safe_path(folder,name))
        exclusive(folder/'COMPLETE_RECEIPT.json',receipt)
    except Exception as error:
        if not (folder/'FAILED.json').exists():exclusive(folder/'FAILED.json',{'error':str(error),'traceback':traceback.format_exc(),'automatic_retry':False})
        raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute-reviewed-run',action='store_true')
    for n in ('run-id','plan-sha256','authorization-sha256','capability-sha256','capability-name'):parser.add_argument('--'+n)
    args=parser.parse_args()
    if not args.execute_reviewed_run or not all((args.run_id,args.plan_sha256,args.authorization_sha256,args.capability_sha256,args.capability_name)):
        print(json.dumps(pending()));return
    run({n:getattr(args,n) for n in ('run_id','plan_sha256','authorization_sha256','capability_sha256','capability_name')})
if __name__=='__main__':main()
