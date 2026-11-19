"""Resource-only synthetic probe. Does not read measurement data or scores."""
import importlib.util,json,time,gc,hashlib,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
def load(name,file):
    sp=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert torch.cuda.is_available()
torch.set_num_threads(2);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
fact=load('fact_fixed','FACT_branch_corrected.py');purge=load('purge_exact','RootPurge_classes_exact.py')
results=[]
for cfg in json.loads((ROOT/'SYNTHETIC12_CONFIGS.json').read_text()):
    rid=cfg['model']+'_'+cfg['dataset'];row={'run_id':rid,'config':cfg,'synthetic_only':True,'measurement_array_read':False}
    try:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();C=cfg['channels'];B=cfg['batch_size'];args=SimpleNamespace(**cfg)
        resident=torch.zeros((cfg['prefix_rows'],C),device='cuda')
        model=(fact.Model(args) if cfg['model']=='FACT' else purge.RootPurgeModel(args,purge.BaseLinearModel)).cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
        x=torch.randn(B,96,C,device='cuda');y=torch.randn_like(x)
        mark=torch.zeros(B,96,cfg.get('calendar_mark_channels',1),device='cuda')
        model.train()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            if cfg['model']=='FACT':pred=model(x,mark,None,None);loss=(pred-y).square().mean()
            else:loss=model(x,y)['loss']
            assert torch.isfinite(loss);loss.backward();optimizer.step()
        model.eval()
        with torch.inference_mode():
            pred=model(x,mark,None,None) if cfg['model']=='FACT' else model(x)
            assert pred.shape==y.shape and torch.isfinite(pred).all()
        torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();row.update(status='PASS',peak_gpu_bytes=peak,parameter_count=sum(p.numel() for p in model.parameters()),reserve_gpu_mib=max(1200,(peak+(1<<20)-1)//(1<<20)+512))
        assert row['reserve_gpu_mib']<=22500,'original recipe exceeds available GPU budget'
        del model,optimizer,x,y,mark,pred,loss,resident;gc.collect();torch.cuda.empty_cache()
    except Exception as e:
        row.update(status='FAIL',error=str(e),traceback=traceback.format_exc());results.append(row);break
    results.append(row)
    (ROOT/'SYNTHETIC12_PROGRESS.json').write_text(json.dumps({'done':len(results),'total':12,'rows':[{k:v for k,v in r.items() if k not in ['config']} for r in results]},indent=2))
receipt={'status':'PASS_ALL12_SYNTHETIC_CUDA_ONLY' if len(results)==12 and all(r['status']=='PASS' for r in results) else 'FAIL_SYNTHETIC_RESOURCE','utc_epoch':time.time(),'rows':results,'source_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')},'data_or_research_scores_accessed':False,'research_started':False}
with (ROOT/'ACTUAL_SYNTHETIC12_CUDA_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps({'status':receipt['status'],'rows':len(results)}))
