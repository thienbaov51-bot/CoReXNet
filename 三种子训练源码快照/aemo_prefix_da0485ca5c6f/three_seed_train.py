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
class PrefixWindows:
    """Exact old train/scaler/window arithmetic, with no test indices/data."""
    def __init__(self, path, metadata, core, device, train_cap=None, validation_cap=None, horizon=96):
        np,torch=core.np,core.torch
        with np.load(path,allow_pickle=False) as z:
            require(set(z.files)=={'data','split_counts'},'prefix payload keys')
            a=np.asarray(z['data'],dtype=np.float32)
            split=list(map(int,z['split_counts']))
        require(len(split)==2 and split==metadata['bounds'][:2],'trainval prefix has exactly two boundaries')
        require(a.ndim==2 and a.shape==(split[1],metadata['channels']) and np.isfinite(a).all(),'prefix shape/finite')
        mean=a[:split[0]].mean(axis=0,dtype=np.float64).astype(np.float32)
        std=a[:split[0]].std(axis=0,dtype=np.float64).astype(np.float32);std[std<1e-6]=1.0
        self.values=torch.as_tensor((a-mean)/std,device=device)
        self.channels,self.length,self.horizon=a.shape[1],96,horizon
        self.mean,self.std,self.bounds=mean,std,metadata['bounds']
        self.xoff=torch.arange(-96,0,device=device);self.yoff=torch.arange(horizon,device=device)
        self.indices={};self.full_counts={}
        for label,start,end,cap in [('train',96,split[0],train_cap),('val',split[0],split[1],validation_cap)]:
            ix=torch.arange(start,end-horizon+1,device=device);require(len(ix)>0,'empty '+label+' windows')
            self.full_counts[label]=len(ix)
            if cap is not None:
                require(isinstance(cap,int) and cap>0,'invalid optional prefix cap')
                if len(ix)>cap:ix=ix[torch.linspace(0,len(ix)-1,cap,device=device).long()]
            self.indices[label]=ix

    def batch(self, indices):
        return self.values[indices[:,None]+self.xoff],self.values[indices[:,None]+self.yoff]

def validation_only(model, windows, core, batch_size, amp=False, split='val'):
    require(split=='val' and set(windows.indices)=={'train','val'},'test/evaluator split is prohibited')
    # Call the byte-preserved evaluator only with the val-only window object.
    return core.evaluate(model,windows,'val',batch_size,amp)

import core_utils as core
import models_adapter as adapter
def main():
    a,c,folder=args_and_config();verify_local();rid=c['run_id'];plan_sha=sha(a.config);device='cuda';root=Path(__file__).parent
    torch=core.initialize_tensor_runtime(Path('/root/autodl-tmp/paper5/new5_pilot_20261011/dependencies'));require(torch.cuda.is_available(),'production CUDA required');core.seed_all(c['seed']);torch.cuda.reset_peak_memory_stats();write(folder/'config.json',c)
    data_path=Path(a.data_root)/(c['dataset']+'_trainval.npz');require(sha(data_path)==c['expected_data_sha256'],'prefix data SHA')
    meta={'bounds':c['bounds'],'channels':c['channels']};windows=PrefixWindows(data_path,meta,core,device,None,None,c['horizon'])
    cfg=dict(c,dependency_root='/root/autodl-tmp/paper5/new5_pilot_20261011/dependencies',device=device);model=adapter.make_model(cfg).to(device)
    optimizer=core.make_optimizer(model,c)
    scheduler,per_batch=core.make_scheduler(optimizer,c,math.ceil(len(windows.indices['train'])/c['batch_size']))
    shuffle=torch.Generator(device=device);shuffle.manual_seed(c['seed']+12345)
    best_val,best_epoch,wait,next_epoch=float('inf'),-1,0,0;history=[];best_state=None
    state_path=folder/'training_state.pt'
    if state_path.exists():
        state=torch.load(state_path,map_location=device,weights_only=False)
        require(state['plan_sha256']==plan_sha and state['config']==c,'same-recipe resume identity')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if device=='cuda':torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        shuffle.set_state(state['shuffle_rng'].cpu());core.random.setstate(state['python_rng']);core.np.random.set_state(state['numpy_rng'])
        best_val,best_epoch,wait,next_epoch,history=state['best_val'],state['best_epoch'],state['wait'],state['next_epoch'],state['history']
        best_state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    started=time.time()
    try:
        for epoch in range(next_epoch,c['epochs']):
            began=time.time();model.train();ix=windows.indices['train'];order=ix[torch.randperm(len(ix),device=device,generator=shuffle)]
            sums=torch.zeros(2,device=device,dtype=torch.float64)
            for chunk in order.split(c['batch_size']):
                x,y=windows.batch(chunk);optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device,dtype=torch.bfloat16,enabled=c.get('amp',False)):
                    pred=core.predict(model,x,chunk);require(pred.shape==y.shape,'training shape')
                    diff=pred.float()-y;loss=diff.abs().mean() if c.get('loss','MSE').upper()=='MAE' else diff.square().mean()
                    require(not hasattr(model,'auxiliary_loss') and not hasattr(model,'supervised_auxiliary_loss'),'undeclared auxiliary/supervised hook prohibited')
                loss.backward()
                if c.get('grad_clip') is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip'])
                optimizer.step()
                if per_batch:scheduler.step()
                sums[0]+=loss.detach().double()*len(chunk);sums[1]+=len(chunk)
            if not per_batch:scheduler.step()
            val=validation_only(model,windows,core,c['test_batch_size'],c.get('amp',False));v=sums.cpu().numpy()
            require(core.np.isfinite(v).all() and all(math.isfinite(val[k]) for k in ('mse','mae')),'nonfinite train/validation')
            history.append(dict(epoch=epoch+1,train_loss=float(v[0]/v[1]),val_mse=val['mse'],val_mae=val['mae'],lr=scheduler.get_last_lr()[0],seconds=time.time()-began))
            write(folder/'history.json',history)
            if val['mse']<best_val-1e-10:
                best_val,best_epoch,wait=val['mse'],epoch+1,0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save(best_state,folder/'best.pt.tmp');os.replace(folder/'best.pt.tmp',folder/'best.pt')
            else:wait+=1
            state=dict(plan_sha256=plan_sha,config=c,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None,shuffle_rng=shuffle.get_state(),python_rng=core.random.getstate(),numpy_rng=core.np.random.get_state(),best_val=best_val,best_epoch=best_epoch,wait=wait,next_epoch=epoch+1,history=history)
            torch.save(state,folder/'training_state.pt.tmp');os.replace(folder/'training_state.pt.tmp',state_path)
            print(json.dumps(dict(run_id=rid,epoch=epoch+1,seconds=history[-1]['seconds'],scores_printed=False)),flush=True)
            if wait>=c['patience']:break
        require(best_state is not None,'no finite validation checkpoint')
        model.load_state_dict(best_state,strict=True)
        val=validation_only(model,windows,core,c['test_batch_size'],c.get('amp',False))
        verify_local();require(sha(data_path)==c['expected_data_sha256'],'data changed while training')
        metrics=dict(config=c,dataset=c['dataset'],model=c['model'],seed=c['seed'],best_epoch=best_epoch,best_val_mse=best_val,epochs_run=len(history),parameters=sum(p.numel() for p in model.parameters()),seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),bounds=c['bounds'],channels=c['channels'],validation=val,training_windows=len(windows.indices['train']),validation_windows=len(windows.indices['val']),data_sha256=c['expected_data_sha256'],checkpoint_sha256=sha(folder/'best.pt'),test_accessed=False)
        write(folder/'metrics.json',metrics);write(folder/'COMPLETE_RECEIPT.json',dict(status='COMPLETE_VALIDATION_ONLY',run_id=rid,config_sha256=sha(folder/'config.json'),checkpoint_sha256=sha(folder/'best.pt'),history_sha256=sha(folder/'history.json'),metrics_sha256=sha(folder/'metrics.json'),test_accessed=False))
    except Exception as exc:record_failure(folder,exc);raise
if __name__=='__main__':main()
