"""Serial official-core synthetic resource admission for the6 H96 pilots.

No NPZ/data-array/metric/test-result/checkpoint file is read. Random batches and
a synthetic resident prefix reproduce the declared dimensions/batches. State
serialization occurs in memory only; failures never alter the frozen recipe.
"""
import argparse
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from secure_paths import safe_path, safe_tree

MODELS=('OURS','DLinear','SparseTSF','ModernTCN','MixLinear','PhaseFormer')
CORE_SHA='dd389b075e7295999de49fb9acc0d6cf8541602e89e1782ae6a8bbcb5fffadc0'
ADAPTER_SHA='7373e173d7a7466be92494761f738382063c47380d5a50531093a09e6f9dc44a'
REUSE_SHA='d374ff9769bc68ede336a4909d281c1eb07e1298717559a07fe36b9c3c711f83'


def require(flag,message):
    if not flag:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with safe_path(path).open('rb') as s:
        for b in iter(lambda:s.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def metadata(path):
    p=safe_path(path)
    require(p.suffix=='.json' and p.name not in ('metrics.json','history.json','test_result.json','validation_result.json'),'metadata/config inputs only')
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(path,value):
    p=safe_path(path);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('xb') as s:s.write((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    os.replace(tmp,p)


def module(path,label):
    name='_pilot_probe_'+label+'_'+sha(path)[:12]
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj;spec.loader.exec_module(obj);return obj


def gpu_processes():
    q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    require(q.returncode==0,'GPU ownership query must succeed')
    return {int(x.split(',')[0]) for x in q.stdout.splitlines() if x.strip()}


def reservations(train_reserved,val_reserved,tensor_bytes,best_bytes,resume_bytes):
    reserve=lambda b:max(1200,int(math.ceil(((b/(1<<20))*1.25+600)/100))*100)
    peak=max(train_reserved,val_reserved)
    return dict(reserve_gpu_mib=reserve(peak),validation_reserve_gpu_mib=reserve(val_reserved),
                reserve_disk_bytes=max(8*tensor_bytes,2*resume_bytes+2*best_bytes)+(64<<20),
                GPU_formula='max(train allocator reserved, validation allocator reserved with optimizer)*1.25 +600MiB; ceil100MiB; floor1200MiB',
                disk_formula='max(8*state tensor bytes, 2*serialized resume +2*serialized best)+64MiB')


def prefix_shape(config,meta):
    d=meta.get('datasets',meta.get('data',meta)).get(config['dataset'],{}) if meta else {}
    bounds=config.get('bounds',d.get('bounds'))
    train_end=config.get('train_end',bounds[0] if bounds else None)
    prefix_rows=config.get('trainval_rows',bounds[1] if bounds else None)
    require(isinstance(train_end,int) and isinstance(prefix_rows,int) and 815<=train_end and train_end+720<=prefix_rows,'declare train_end and trainval_rows/bounds metadata')
    if d.get('channels') is not None:require(d['channels']==config['channels'],'declared metadata/config C')
    return train_end,prefix_rows


def candidates(configs,meta):
    paths=sorted(safe_tree(configs).glob('*.json'));require(len(paths)==120,'exactly120 draft configs required')
    rows={};grid=set()
    for path in paths:
        c=metadata(path);rid=c['run_id']
        require(rid not in rows and path.stem==rid,'unique filename/run identity')
        require(c['model'] in MODELS and c['seed']==2026 and c['length']==96 and c['horizon'] in (96,192,336,720),'fixed model/L96/H96/seed')
        require(c['amp'] is (c['model']=='OURS') and c.get('evaluate_test') is False,'fixed precision/no test')
        require(c.get('original_epochs',c['epochs'])==c['epochs'] and c.get('original_patience',c['patience'])==c['patience'],'original budget unchanged')
        for k in ('channels','batch_size','test_batch_size','epochs','patience'):require(isinstance(c[k],int) and c[k]>0,'positive scientific '+k)
        require(c.get('loss','MSE').upper() in ('MSE','MAE') and c.get('optimizer','Adam') in ('Adam','AdamW'),'original optimizer/objective')
        require(c.get('additional_loss') in (None,False),'no unapproved auxiliary loss')
        end,n=prefix_shape(c,meta);grid.add((c['dataset'],c['model'],c['horizon']))
        rows[rid]=dict(config=c,config_sha256=sha(path),config_bytes=path.stat().st_size,train_end=end,trainval_rows=n)
    datasets=sorted({c['config']['dataset'] for c in rows.values()})
    require(len(datasets)==5 and grid=={(d,m,h) for d in datasets for m in MODELS for h in (96,192,336,720)},'complete5 by6 by4 candidates')
    for ds in datasets:
        require(len({(r['config']['channels'],r['train_end'],r['trainval_rows']) for r in rows.values() if r['config']['dataset']==ds})==1,'same dataset C/prefix bounds for all6')
    return rows,datasets


class SyntheticValidation:
    def __init__(self,torch,resident,channels,batch,horizon):
        self.values=resident;self.channels=channels;self.horizon=horizon
        self.indices={'val':torch.arange(96,96+2*batch,device='cuda')};self.torch=torch

    def batch(self,indices):
        B=len(indices);C=self.channels;t=self.torch
        return t.randn(B,96,C,device='cuda'),t.randn(B,self.horizon,C,device='cuda')


def probe_row(root,core,adapter,declared,steps):
    torch=core.torch;c=dict(declared['config']);B,C,EB,H=c['batch_size'],c['channels'],c['test_batch_size'],c['horizon']
    require(not(gpu_processes()-{os.getpid()}),'other GPU work present; serial isolated probe required')
    gc.collect();torch.cuda.empty_cache();core.seed_all(2026);torch.cuda.reset_peak_memory_stats()
    # This is a random/synthetic allocation, never a loaded dataset.
    resident=torch.zeros(declared['trainval_rows'],C,device='cuda',dtype=torch.float32)
    construction=dict(c,frozen_model_sources_root=str(root),model_reuse_manifest_path=str(root/'MODEL_REUSE_MANIFEST.json'),dependency_root=str(root/'dependencies'),device='cuda')
    constructor=getattr(adapter,'make_model',None) or getattr(adapter,'build_model',None)
    require(callable(constructor),'official adapter constructor unavailable')
    model=constructor(construction).cuda();optimizer=core.make_optimizer(model,c)
    scheduler,per_batch=core.make_scheduler(optimizer,c,math.ceil((declared['train_end']-96-H+1)/B))
    model.train();x=torch.randn(B,96,C,device='cuda');y=torch.randn(B,H,C,device='cuda');origins=torch.arange(96,96+B,device='cuda');elapsed=[]
    for _ in range(steps):
        begin=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['amp']):
            pred=core.predict(model,x,origins);require(tuple(pred.shape)==(B,H,C) and torch.isfinite(pred).all().item(),'synthetic prediction shape/finite')
            diff=pred.float()-y;loss=diff.abs().mean() if c.get('loss','MSE').upper()=='MAE' else diff.square().mean()
            require(not hasattr(model,'auxiliary_loss') and not hasattr(model,'supervised_auxiliary_loss'),'unapproved training hook')
        loss.backward();require(all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),'nonfinite synthetic gradient')
        if c.get('grad_clip') is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip'])
        optimizer.step()
        if per_batch:scheduler.step()
        torch.cuda.synchronize();elapsed.append(time.perf_counter()-begin)
    if not per_batch:scheduler.step()
    train_peak=torch.cuda.max_memory_allocated();train_reserved=torch.cuda.max_memory_reserved()
    # Actual runner validates while its optimizer, moments and final training
    # tensors/gradients remain resident. Do not free them before this measurement.
    torch.cuda.reset_peak_memory_stats();windows=SyntheticValidation(torch,resident,C,EB,H)
    val=core.evaluate(model,windows,'val',EB,c['amp']);torch.cuda.synchronize()
    require(all(math.isfinite(val[k]) for k in ('mse','mae')) and val['windows']==2*EB and val['elements']==2*EB*H*C,'synthetic validation finite/counts')
    val_peak=torch.cuda.max_memory_allocated();val_reserved=torch.cuda.max_memory_reserved()
    tensors=model.state_dict();tensor_bytes=sum(t.numel()*t.element_size() for t in tensors.values())
    best={k:v.detach().cpu().clone() for k,v in tensors.items()}
    buf=io.BytesIO();torch.save(best,buf);best_bytes=buf.tell();del buf
    shuffle=torch.Generator(device='cuda');shuffle.manual_seed(2026+12345)
    state=dict(plan_sha256='synthetic_resource_only',config=c,model=tensors,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),shuffle_rng=shuffle.get_state(),python_rng=core.random.getstate(),numpy_rng=core.np.random.get_state(),best_val=0.,best_epoch=1,wait=0,next_epoch=1,history=[dict(epoch=1,synthetic_only=True)])
    buf=io.BytesIO();torch.save(state,buf);resume_bytes=buf.tell();del buf
    row=dict(status='PASS_SYNTHETIC_RESOURCE',model=c['model'],dataset=c['dataset'],channels=C,length=96,horizon=H,seed=2026,batch_size=B,validation_batch_size=EB,amp=c['amp'],epochs=c['epochs'],patience=c['patience'],parameters=sum(p.numel() for p in model.parameters()),trainval_rows=declared['trainval_rows'],synthetic_resident_prefix_bytes=resident.numel()*resident.element_size(),state_dict_tensor_bytes=tensor_bytes,train_peak_gpu_bytes=train_peak,train_allocator_reserved_bytes=train_reserved,validation_peak_gpu_with_optimizer_bytes=val_peak,validation_allocator_reserved_with_optimizer_bytes=val_reserved,synthetic_optimizer_step_seconds=elapsed,synthetic_optimizer_steps=steps,best_serialized_bytes=best_bytes,training_state_serialized_bytes=resume_bytes,serialized_states_written_to_disk=False,validation_metrics_printed_or_saved=False,**reservations(train_reserved,val_reserved,tensor_bytes,best_bytes,resume_bytes))
    require(row['reserve_gpu_mib']<=22500,'declared recipe exceeds single GPU admission; no batch fallback')
    del model,optimizer,scheduler,resident,x,y,origins,pred,diff,loss,windows,tensors,best,state,shuffle,val
    gc.collect();torch.cuda.empty_cache();return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--configs');p.add_argument('--output');p.add_argument('--metadata');p.add_argument('--metadata-sha256');p.add_argument('--steps',type=int,choices=[1,2],default=2)
    p.add_argument('--core-sha256',default=CORE_SHA);p.add_argument('--adapter-sha256',default=ADAPTER_SHA);p.add_argument('--reuse-manifest-sha256',default=REUSE_SHA);a=p.parse_args()
    if not(a.root and a.configs and a.output):print(json.dumps(dict(status='PENDING_EXPLICIT_SYNTHETIC_PROBE_ARGUMENTS',research_training=False)));return
    root=safe_tree(a.root).resolve();out=safe_path(a.output).resolve();require(not out.exists(),'resource receipt cannot be overwritten');out.parent.mkdir(parents=True,exist_ok=True)
    for name,expected in [('core_utils.py',a.core_sha256),('models_adapter.py',a.adapter_sha256),('MODEL_REUSE_MANIFEST.json',a.reuse_manifest_sha256)]:require(sha(root/name)==expected,'actual original source pin '+name)
    meta=metadata(a.metadata) if a.metadata else {}
    if a.metadata_sha256:require(a.metadata and sha(a.metadata)==a.metadata_sha256,'metadata identity')
    rows,datasets=candidates(a.configs,meta)
    require(not gpu_processes(),'GPU must be idle before the resource-only process initializes CUDA')
    core=module(root/'core_utils.py','core');torch=core.initialize_tensor_runtime(root/'dependencies')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one actual CUDA GPU required')
    adapter=module(root/'models_adapter.py','adapter')
    report=dict(status='RUNNING_SYNTHETIC_RESOURCE_ONLY',run_count=120,completed_count=0,failed_count=0,datasets=datasets,models=list(MODELS),source_sha256={n:sha(root/n) for n in ('core_utils.py','models_adapter.py','MODEL_REUSE_MANIFEST.json')},probe_source_sha256=sha(__file__),config_sha256={rid:r['config_sha256'] for rid,r in rows.items()},metadata_sha256=sha(a.metadata) if a.metadata else None,runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory),real_data_arrays_read=False,real_metrics_or_test_scores_read=False,scientific_training_executed=False,synthetic_optimizer_steps_only=True,model_registry_or_scientific_configs_modified=False,states_serialized_in_memory_only=True,created_utc_epoch=time.time(),records={})
    write(out,report)
    for rid,declared in rows.items():
        try:row=probe_row(root,core,adapter,declared,a.steps)
        except Exception as exc:
            row=dict(status='FAILED_SYNTHETIC_RESOURCE',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),model=declared['config']['model'],dataset=declared['config']['dataset'],science_recipe_unchanged=True,automatic_retry_or_batch_change=False)
            report['failed_count']+=1;gc.collect();torch.cuda.empty_cache()
        report['records'][rid]=dict(row,run_id=rid,config_sha256=declared['config_sha256']);report['completed_count']+=1;write(out,report)
        print(json.dumps(dict(run_id=rid,status=row['status'],reserve_gpu_mib=row.get('reserve_gpu_mib'),parameters=row.get('parameters'),scores_saved=False)),flush=True)
    report['status']='PASS120_SYNTHETIC_RESOURCE_ADMISSION' if not report['failed_count'] else 'FAILED_SYNTHETIC_RESOURCE_ADMISSION'
    report['completed_utc_epoch']=time.time();report['all120_original_batch_and_precision_checked']=True
    write(out,report)
    print(json.dumps(dict(status=report['status'],completed=120,failed=report['failed_count'],output_sha256=sha(out))))
    require(report['failed_count']==0,'failed resource configurations cannot receive research admission')


if __name__=='__main__':main()
