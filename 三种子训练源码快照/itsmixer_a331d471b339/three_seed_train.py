import argparse,hashlib,json,math,os,time,traceback,importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
import random
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,p)
def require(v,m):
    if not v:raise ValueError(m)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def args_and_config():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root',required=True);a=p.parse_args()
    c=json.loads(Path(a.config).read_text('utf-8-sig'));out=Path(a.output).resolve()
    require(c['seed'] in (2027,2028),'explicit new seed required');require(out.is_relative_to(Path('/root/autodl-tmp/paper5/three_seed_20261012/results').resolve()),'new output scope');require(not out.exists(),'exclusive new scientific attempt, no automatic retry');out.mkdir(parents=True)
    return a,c,out
def verify_local():
    info=json.loads((Path(__file__).parent/'FORK_SOURCE_PINS.json').read_text('utf-8'))
    for name,h in info.items():require(sha(Path(__file__).parent/name)==h,'source changed '+name)
def record_failure(out,e):write(out/'FAILED.json',{'error':repr(e),'traceback':traceback.format_exc(),'automatic_retry':False})
def main():
    a,c,folder=args_and_config();verify_local();rid=c['run_id']
    import numpy as np
    import torch
    import three_seed_itsmixer6_science_kernel as kernel
    kernel.configure_framework(torch,np,c['seed'])
    prefix=Path(a.data_root)/(c['dataset']+'_trainval.npz');require(sha(prefix)==c['prefix_sha256'],'same prefix bytes')
    with np.load(prefix,allow_pickle=False) as z:
        require(set(z.files)=={'data','split_counts'},'native prefix fields');values=z['data'];bounds=z['split_counts']
        require(values.dtype==np.dtype('float32') and values.shape==(c['prefix_rows'],c['channels']) and bounds.tolist()==[c['train_end'],c['validation_end']] and np.isfinite(values).all(),'original full prefix')
        mean=values[:c['train_end']].mean(0,dtype=np.float64).astype(np.float32);scale=values[:c['train_end']].std(0,dtype=np.float64).astype(np.float32);scale[scale<1e-6]=1
        require(hashlib.sha256(mean.tobytes()).hexdigest()==c['scaler_mean_sha256'] and hashlib.sha256(scale.tobytes()).hexdigest()==c['scaler_std_sha256'],'sameTRAINscalers');standardized=np.asarray((values-mean)/scale,dtype=np.float32)
    tensor=torch.from_numpy(standardized).to('cuda');del values,bounds,mean,scale,standardized
    author=module('itsmixer_exact_owned',Path(__file__).parent/c['model_source_file']);model=kernel.build_model(author,c);resident=kernel.ResidentWindows(torch,tensor,c)
    write(folder/'config.json',c)
    try:
        started=time.time();outcome=kernel.trainval_science_core(torch,model,resident,c)
        with (folder/'best.pt').open('xb') as f:torch.save(outcome['best_state'],f)
        write(folder/'history.json',outcome['history']);verify_local();require(sha(prefix)==c['prefix_sha256'],'sameprefix aftertraining')
        metrics=dict(config=c,dataset=c['dataset'],model=c['model'],seed=c['seed'],best_epoch=outcome['best_epoch'],best_step=outcome['best_step'],epochs_run=outcome['epochs'],steps=outcome['steps'],validation=outcome['validation'],parameters=sum(p.numel() for p in model.parameters()),seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=sha(folder/'best.pt'),data_sha256=c['prefix_sha256'],test_accessed=False,checkpoint_criterion=c['checkpoint_criterion'])
        write(folder/'metrics.json',metrics);write(folder/'COMPLETE_RECEIPT.json',dict(status='COMPLETE_VALIDATION_ONLY',run_id=rid,config_sha256=sha(folder/'config.json'),checkpoint_sha256=sha(folder/'best.pt'),history_sha256=sha(folder/'history.json'),metrics_sha256=sha(folder/'metrics.json'),test_accessed=False))
    except Exception as exc:record_failure(folder,exc);raise
if __name__=='__main__':main()
