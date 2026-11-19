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
class Windows:
    def __init__(self,c):
        self.L=c['seq_len'];self.H=c['pred_len']
        with np.load(DATA_ROOT/f"{c['dataset']}_trainval.npz",allow_pickle=False) as z:
            assert set(z.files)=={'data','split_counts'};a=z['data'];b=list(map(int,z['split_counts']))
        assert len(b)==2 and len(a)==b[1] and a.shape[1]==c['channels']
        m=a[:b[0]].mean(0,dtype=np.float64).astype(np.float32);s=a[:b[0]].std(0,dtype=np.float64).astype(np.float32);s[s<1e-6]=1
        self.values=torch.as_tensor((a-m)/s,device='cuda');self.bounds=b;self.C=a.shape[1]
        self.marks=torch.as_tensor(np.load(DATA_ROOT/f"{c['dataset']}_known_calendar.npy",allow_pickle=False),device='cuda')
        self.indices={'train':torch.arange(self.L,b[0]-self.H+1,device='cuda'),'val':torch.arange(b[0],b[1]-self.H+1,device='cuda')}
        self.xoff=torch.arange(-self.L,0,device='cuda');self.yoff=torch.arange(self.H,device='cuda')
    def batch(self,ix):return self.values[ix[:,None]+self.xoff],self.values[ix[:,None]+self.yoff]

def predict(model,c,w,x,ix):
    return model(x)

def val(model,c,w):
    model.eval();s=torch.zeros(3,device='cuda',dtype=torch.float64);chs=torch.zeros(w.C,device='cuda',dtype=torch.float64)
    with torch.inference_mode():
        for ix in w.indices['val'].split(c['batch_size']):
            x,y=w.batch(ix);p=predict(model,c,w,x,ix);assert p.shape==y.shape and torch.isfinite(p).all()
            d=p.float()-y;s[0]+=d.square().sum(dtype=torch.float64);s[1]+=d.abs().sum(dtype=torch.float64);s[2]+=d.numel();chs+=d.square().sum((0,1),dtype=torch.float64)
    v=s.cpu().tolist();return {'mse':v[0]/v[2],'mae':v[1]/v[2],'elements':int(v[2]),'windows':len(w.indices['val']),'channel_mse':(chs/(len(w.indices['val'])*c['pred_len'])).cpu().tolist()}

import numpy as np
import torch
ROOT=Path(__file__).parent
DATA_ROOT=None
def main():
    global DATA_ROOT
    a,c,folder=args_and_config();DATA_ROOT=Path(a.data_root);verify_local();rid=c['run_id'];expected=sha(a.config);plan={'files':json.loads((ROOT/'FORK_SOURCE_PINS.json').read_text())}
    require(c['model']=='RootPurgeLinear','selected model');require(sha(DATA_ROOT/(c['dataset']+'_trainval.npz'))==c['expected_prefix_sha256'],'actual prefix SHA');require(sha(DATA_ROOT/(c['dataset']+'_known_calendar.npy'))==c['expected_calendar_sha256'],'known calendar SHA')
    write(folder/'config.json',c);random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    try:
        w=Windows(c);args=SimpleNamespace(**c)
        assert c['model']=='RootPurgeLinear'
        m=module('purge_exact',ROOT/'RootPurge_classes_exact.py');model=m.RootPurgeModel(args,m.BaseLinearModel).cuda()
        opt=torch.optim.Adam(model.parameters(),lr=c['lr']);rng=torch.Generator(device='cuda');rng.manual_seed(c['seed']+12345)
        history=[];best=float('inf');wait=0;step=0;epoch=0;best_step=0;best_epoch=0;began=time.time()
        max_epochs=c.get('epochs',100000);max_steps=c.get('train_steps',10**12)
        while epoch<max_epochs and step<max_steps:
            model.train();ix=w.indices['train'];order=ix[torch.randperm(len(ix),generator=rng,device='cuda')]
            for chunk in order.split(c['batch_size']):
                x,y=w.batch(chunk);opt.zero_grad(set_to_none=True)
                loss=model(x,y)['loss']
                assert torch.isfinite(loss);loss.backward();opt.step();step+=1
                if c['model']=='RootPurgeLinear' and step%50==0:
                    score=val(model,c,w);history.append({'epoch':epoch+1,'step':step,'validation':score})
                    if score['mse']<best:best=score['mse'];best_step=step;best_epoch=epoch+1;torch.save(model.state_dict(),folder/'best.pt')
                    write(folder/'history.json',history);write(folder/'PROGRESS.json',{'epoch':epoch+1,'step':step,'score_access_released':False});model.train()
                if step>=max_steps:break
            epoch+=1
            for group in opt.param_groups:group['lr']=c['lr']*(1 if epoch<3 else 0.9**(epoch-3))
        model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True));score=val(model,c,w);verify_local()
        result={'status':'COMPLETE_VALIDATION_ONLY','run_id':rid,'config':c,'plan_sha256':expected,'checkpoint_sha256':sha(folder/'best.pt'),'source_inventory':plan['files'],'history_sha256':sha(folder/'history.json'),'config_sha256':sha(folder/'config.json'),'best_epoch':best_epoch,'best_step':best_step,'epochs':epoch,'steps':step,'train_windows':len(w.indices['train']),'validation':score,'test_windows_constructed':False,'test_scores_accessed':False,'seconds':time.time()-began}
        write(folder/'VALIDATION_RESULT.json',result)
        write(folder/'COMPLETE_RECEIPT.json',{'status':'COMPLETE_VALIDATION_ONLY','run_id':rid,'plan_sha256':expected,'checkpoint_sha256':result['checkpoint_sha256'],'config_sha256':result['config_sha256'],'history_sha256':result['history_sha256'],'validation_result_sha256':sha(folder/'VALIDATION_RESULT.json'),'test_accessed':False})
        write(folder/'metrics.json',dict(result,model=c['model'],dataset=c['dataset'],seed=c['seed'],best_val_mse=score['mse'],epochs_run=epoch,parameters=sum(p.numel() for p in model.parameters()),data_sha256=c['expected_prefix_sha256']))
    except Exception as exc:record_failure(folder,exc);raise
if __name__=='__main__':main()
