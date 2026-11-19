"""Future ROOT once-only fullresident B256 synthetic CUDA QA; default PENDING."""
import argparse,gc,importlib.util,inspect,json,os,shutil,time,traceback
from pathlib import Path
from itsmixer18_common import *

def probe(pin):
    root,manifest,recipe,docs=verify_manifest(pin)
    old=gate_old_cohorts();old.update(gpu_idle_except())
    require(shutil.disk_usage(root).free>=1536<<20,'actual disk floor before synthetic import')
    exclusive(root/'ITSMIXER18_PROBE_CLAIM.json',{'status':'ACTUAL_ITSMIXER18_ONE_PROBE_ATTEMPT_CLAIM',
        'utc_epoch':time.time(),'manifest_sha256':pin,'pid_identity':process_identity(os.getpid()),
        'old_cohort_admission':old,'automatic_retry':False,'training_started':False})
    # The once claim and all source/oldphysical/globalidle gates precede imports.
    import numpy as np
    import torch
    from itsmixer18_science_kernel import configure_framework,build_model,build_optimizer_and_scheduler,compound_loss,predict,ResidentWindows
    from itsmixer18_protocol import type3_lr_after_epoch
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one actual CUDA device')
    # Exact corresponding model is imported once per admitted H case below.
    classes={'AdamW':torch.optim.AdamW,'OneCycleLR':torch.optim.lr_scheduler.OneCycleLR,'LRScheduler':torch.optim.lr_scheduler.LRScheduler}
    paths={k:str(Path(inspect.getsourcefile(v)).resolve()) for k,v in classes.items()};hashes={k:sha(Path(v)) for k,v in paths.items()}
    rows=[]
    for rid,original in sorted(recipe['runs'].items()):
        cfg=dict(original);row={'run_id':rid,'synthetic_only':True,'measurement_array_read':False}
        try:
            gate_old_cohorts();gpu_idle_except(os.getpid());require(shutil.disk_usage(root).free>=1536<<20,'physical disk floor')
            configure_framework(torch,np);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            spec=importlib.util.spec_from_file_location('itsmixer_exact_synthetic_'+rid,root/cfg['model_source_file'])
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            model=build_model(module,cfg)
            prefix=torch.zeros((cfg['prefix_rows'],cfg['channels']),device='cuda',dtype=torch.float32)
            resident=ResidentWindows(torch,prefix,cfg)
            optimizer,scheduler=build_optimizer_and_scheduler(torch,model,cfg)
            # Actual unchanged B256 gather/optimizer buffers are exercised twice.
            starts=torch.arange(96,96+256,dtype=torch.int64)
            model.train()
            for _ in range(2):
                optimizer.zero_grad();x,y=resident.batch(starts);forecast=predict(model,x,cfg['channels'],cfg['horizon'])
                loss=compound_loss(torch.nn.functional,forecast,y);require(bool(torch.isfinite(loss)),'finite synthetic composite')
                loss.backward();optimizer.step()
            model.eval()
            with torch.inference_mode():
                forecast=predict(model,x,cfg['channels'],cfg['horizon']);require(bool(torch.isfinite(forecast).all()),'finite full forecast')
                error=forecast.float()-y
                sums=torch.stack((error.square().sum(dtype=torch.float64),error.abs().sum(dtype=torch.float64)))
                require(bool(torch.isfinite(sums).all()),'finite pooled FP64 synthetic reduction')
                tail=cfg['validation_tail_windows'];tail_forecast=None
                if tail:tail_forecast=predict(model,x[:tail],cfg['channels'],cfg['horizon']);require(bool(torch.isfinite(tail_forecast).all()),'finite exact fullVAL tail')
            for epoch in (1,2):
                for group in optimizer.param_groups:group['lr']=type3_lr_after_epoch(epoch,cfg['horizon'])
                require(group['betas']==(.95,.999) and scheduler.last_epoch==0,'type3 cannot step OneCycle/beta')
            torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
            count=sum(p.numel() for p in model.parameters());gpu=max(1200,(max(peak,reserved)+(1<<20)-1)//(1<<20)+512)
            disk=max(64<<20,count*12+(16<<20));require(gpu<=22500 and disk<=4<<30,'original B256 capacity cannot be shrunk')
            cfg.update(parameter_count=count,peak_gpu_bytes=peak,peak_gpu_reserved_bytes=reserved,reserved_gpu_mib=gpu,reserved_disk_bytes=disk)
            row.update(status='PASS',config=cfg,config_sha256=config_sha(cfg),optimizer_steps=2,precision='FP32_TF32',
                fullprefix_shape=[cfg['prefix_rows'],cfg['channels']],peak_gpu_bytes=peak,peak_gpu_reserved_bytes=reserved,parameter_count=count,
                reserved_gpu_mib=gpu,reserved_disk_bytes=disk,torch_version=torch.__version__,numpy_version=np.__version__,device_name=torch.cuda.get_device_name(),
                loss_policy='EXACT_AUTHOR_0.5_MSE_PLUS_0.5_L1_STANDARDIZED',checkpoint_criterion=cfg['checkpoint_criterion'],
                batch_size=256,calendar_consumed=False,validation_tail_windows=tail,framework_source_paths=paths,framework_source_sha256=hashes,
                optimizer_parameter_groups_verified=True,onecycle_initial_last_epoch=0,initial_learning_rate=.00004,initial_betas=[.95,.999],
                weight_decay=.01,eps=1e-8,amsgrad=False,after_epoch1_learning_rate=.001,after_epoch2_learning_rate=.001,type3_scheduler_step_calls=0)
            del model,prefix,resident,optimizer,scheduler,starts,x,y,forecast,loss,error,sums,tail_forecast
            gc.collect();torch.cuda.empty_cache()
        except Exception as error:
            row.update(status='FAIL',error=str(error),traceback=traceback.format_exc());rows.append(row);break
        rows.append(row)
    receipt={'status':'PASS_ALL18_ACTUAL_ORIGINAL_BATCH_ITSMIXER_SYNTHETIC_CUDA_ONLY' if len(rows)==18 and all(r['status']=='PASS' for r in rows)
        else 'FAILED_ONE_ATTEMPT_ITSMIXER18_SYNTHETIC_QA_REQUIRES_ROOT_REVIEW','utc_epoch':time.time(),'rows':rows,
        'source_inventory':manifest['files'],'manifest_sha256':pin,'recipe_sha256':RECIPE_SHA,'root_pre_score_policy_sha256':ROOT_POLICY_SHA,
        'probe_claim_sha256':sha(root/'ITSMIXER18_PROBE_CLAIM.json'),'probe_source_sha256':manifest['files']['synthetic_cuda18.py'],
        'old_cohort_admission':old,'measurement_array_read':False,'research_scores_read':False,'training_started':False,'automatic_retry':False}
    exclusive(root/'ACTUAL_ITSMIXER18_SYNTHETIC_CUDA_RECEIPT.json',receipt)
    require(receipt['status']=='PASS_ALL18_ACTUAL_ORIGINAL_BATCH_ITSMIXER_SYNTHETIC_CUDA_ONLY','one probe failed; ROOT review required, no retry')
    print(json.dumps({'status':receipt['status'],'row_count':len(rows)}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--probe-once',action='store_true');p.add_argument('--manifest-sha256');a=p.parse_args()
    if not a.probe_once or not a.manifest_sha256:print(json.dumps(pending()));return
    probe(a.manifest_sha256)
if __name__=='__main__':main()
