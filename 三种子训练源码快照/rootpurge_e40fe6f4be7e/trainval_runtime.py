"""Prospective H96 baseline screening. Prefix arrays only; no test entry point."""
import argparse,hashlib,importlib.util,json,time,random,os,math,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));os.replace(tmp,p)
def module(name,file):
    sp=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def verify(plan):
    assert str(ROOT)=='/root/autodl-tmp/paper5/models2026_pilot_20261011'
    for p,h in plan['files'].items():assert sha(ROOT/p)==h,p
    assert plan['seed']==2026 and plan['length']==96 and plan['horizon']==96 and plan['test_allowed'] is False
class Windows:
    def __init__(self,c):
        with np.load(ROOT/'data'/f"{c['dataset']}_trainval.npz",allow_pickle=False) as z:
            assert set(z.files)=={'data','split_counts'};a=z['data'];b=list(map(int,z['split_counts']))
        assert len(b)==2 and len(a)==b[1] and a.shape[1]==c['channels']
        m=a[:b[0]].mean(0,dtype=np.float64).astype(np.float32);s=a[:b[0]].std(0,dtype=np.float64).astype(np.float32);s[s<1e-6]=1
        self.values=torch.as_tensor((a-m)/s,device='cuda');self.bounds=b;self.C=a.shape[1]
        self.marks=torch.as_tensor(np.load(ROOT/'data'/f"{c['dataset']}_known_calendar.npy",allow_pickle=False),device='cuda')
        self.indices={'train':torch.arange(96,b[0]-95,device='cuda'),'val':torch.arange(b[0],b[1]-95,device='cuda')}
        self.xoff=torch.arange(-96,0,device='cuda');self.yoff=torch.arange(96,device='cuda')
    def batch(self,ix):return self.values[ix[:,None]+self.xoff],self.values[ix[:,None]+self.yoff]
def predict(model,c,w,x,ix):
    if c['model']=='FACT':return model(x,w.marks[ix[:,None]+w.xoff],None,None)
    return model(x)
def val(model,c,w):
    model.eval();s=torch.zeros(3,device='cuda',dtype=torch.float64);chs=torch.zeros(w.C,device='cuda',dtype=torch.float64)
    with torch.inference_mode():
        for ix in w.indices['val'].split(c['batch_size']):
            x,y=w.batch(ix);p=predict(model,c,w,x,ix);assert p.shape==y.shape and torch.isfinite(p).all()
            d=p.float()-y;s[0]+=d.square().sum(dtype=torch.float64);s[1]+=d.abs().sum(dtype=torch.float64);s[2]+=d.numel();chs+=d.square().sum((0,1),dtype=torch.float64)
    v=s.cpu().tolist();return {'mse':v[0]/v[2],'mae':v[1]/v[2],'elements':int(v[2]),'windows':len(w.indices['val']),'channel_mse':(chs/(len(w.indices['val'])*96)).cpu().tolist()}
def train(rid,expected):
    assert sha(ROOT/'MINIMUM12_PLAN.json')==expected;plan=json.loads((ROOT/'MINIMUM12_PLAN.json').read_text());verify(plan);c=plan['runs'][rid]
    folder=ROOT/'results'/rid;assert not folder.exists(),'never automatically retrain';folder.mkdir(parents=True)
    write(folder/'config.json',c);random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    try:
        w=Windows(c);args=SimpleNamespace(**c)
        if c['model']=='FACT':model=module('fact_fixed','FACT_branch_corrected.py').Model(args).cuda()
        else:
            m=module('purge_exact','RootPurge_classes_exact.py');model=m.RootPurgeModel(args,m.BaseLinearModel).cuda()
        opt=torch.optim.Adam(model.parameters(),lr=c['lr']);rng=torch.Generator(device='cuda');rng.manual_seed(2026+12345)
        history=[];best=float('inf');wait=0;step=0;epoch=0;best_step=0;best_epoch=0;began=time.time()
        max_epochs=c.get('epochs',100000);max_steps=c.get('train_steps',10**12)
        while epoch<max_epochs and step<max_steps:
            model.train();ix=w.indices['train'];order=ix[torch.randperm(len(ix),generator=rng,device='cuda')]
            for chunk in order.split(c['batch_size']):
                x,y=w.batch(chunk);opt.zero_grad(set_to_none=True)
                if c['model']=='RootPurgeLinear':loss=model(x,y)['loss']
                else:loss=(predict(model,c,w,x,chunk)-y).square().mean()
                assert torch.isfinite(loss);loss.backward();opt.step();step+=1
                if c['model']=='RootPurgeLinear' and step%50==0:
                    score=val(model,c,w);history.append({'epoch':epoch+1,'step':step,'validation':score})
                    if score['mse']<best:best=score['mse'];best_step=step;best_epoch=epoch+1;torch.save(model.state_dict(),folder/'best.pt')
                    write(folder/'history.json',history);write(folder/'PROGRESS.json',{'epoch':epoch+1,'step':step,'score_access_released':False});model.train()
                if step>=max_steps:break
            epoch+=1
            if c['model']=='FACT':
                score=val(model,c,w);history.append({'epoch':epoch,'step':step,'validation':score})
                if score['mse']<best-1e-10:best=score['mse'];best_step=step;best_epoch=epoch;wait=0;torch.save(model.state_dict(),folder/'best.pt')
                else:wait+=1
                for group in opt.param_groups:group['lr']=c['lr']*0.5**max(0,epoch-1)
                write(folder/'history.json',history);write(folder/'PROGRESS.json',{'epoch':epoch,'step':step,'score_access_released':False})
                if wait>=c['patience']:break
            else:
                for group in opt.param_groups:group['lr']=c['lr']*(1 if epoch<3 else 0.9**(epoch-3))
        model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True));score=val(model,c,w);verify(plan)
        result={'status':'COMPLETE_VALIDATION_ONLY','run_id':rid,'config':c,'plan_sha256':expected,'checkpoint_sha256':sha(folder/'best.pt'),'source_inventory':plan['files'],'history_sha256':sha(folder/'history.json'),'config_sha256':sha(folder/'config.json'),'best_epoch':best_epoch,'best_step':best_step,'epochs':epoch,'steps':step,'train_windows':len(w.indices['train']),'validation':score,'test_windows_constructed':False,'test_scores_accessed':False,'seconds':time.time()-began}
        write(folder/'VALIDATION_RESULT.json',result)
        write(folder/'COMPLETE_RECEIPT.json',{'status':'COMPLETE_VALIDATION_ONLY','run_id':rid,'plan_sha256':expected,'checkpoint_sha256':result['checkpoint_sha256'],'config_sha256':result['config_sha256'],'history_sha256':result['history_sha256'],'validation_result_sha256':sha(folder/'VALIDATION_RESULT.json'),'test_accessed':False})
    except Exception as exc:
        write(folder/'FAILED.json',{'error':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False});raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--plan-sha256',required=True);a=p.parse_args();train(a.run_id,a.plan_sha256)
