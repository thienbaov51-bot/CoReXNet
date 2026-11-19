"""GPU-resident, leakage-controlled single-seed forecast experiment runner."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    # Single-seed exploration; CUDA reduction reproducibility limits are documented.

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Windows:
    def __init__(self, path, length, horizon, device, pilot=False):
        z = np.load(path, allow_pickle=False)
        a = np.asarray(z['data'], dtype=np.float32)
        bounds = list(map(int, z['split_counts']))
        if len(bounds) != 3 or bounds[-1] != len(a):
            raise ValueError('split_counts must contain absolute train/val/test end offsets')
        if not np.isfinite(a).all():
            raise ValueError('non-finite data: resolve preprocessing before any training')
        mean = a[:bounds[0]].mean(axis=0, dtype=np.float64).astype(np.float32)
        std = a[:bounds[0]].std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        self.values = torch.as_tensor((a - mean) / std, device=device)
        self.mean, self.std = mean, std
        self.channels = a.shape[1]
        self.length, self.horizon, self.bounds = length, horizon, bounds
        self.xoff = torch.arange(-length, 0, device=device)
        self.yoff = torch.arange(horizon, device=device)
        self.indices = {}
        for name, start, end in [('train',length,bounds[0]),('val',bounds[0],bounds[1]),('test',bounds[1],bounds[2])]:
            ix = torch.arange(start, end-horizon+1, device=device)
            if len(ix) == 0: raise ValueError(f'empty {name} split')
            # Pilot uses deterministic evenly-spaced windows, never test windows.
            cap = 12000 if name == 'train' else 4096
            if pilot and name != 'test' and len(ix) > cap:
                ix = ix[torch.linspace(0,len(ix)-1,cap,device=device).long()]
            self.indices[name] = ix

    def batch(self, indices):
        return self.values[indices[:,None]+self.xoff], self.values[indices[:,None]+self.yoff]

@torch.inference_mode()
def evaluate(model, windows, split, batch_size, amp):
    model.eval()
    totals = torch.zeros(3,device='cuda',dtype=torch.float64)
    channel_se = torch.zeros(windows.channels,device='cuda',dtype=torch.float64)
    indices = windows.indices[split]
    for chunk in indices.split(batch_size):
        x,y = windows.batch(chunk)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            pred = model(x)
        difference = pred.float()-y
        totals[0] += difference.square().sum(dtype=torch.float64)
        totals[1] += difference.abs().sum(dtype=torch.float64)
        totals[2] += difference.numel()
        channel_se += difference.square().sum((0,1),dtype=torch.float64)
    totals = totals.cpu().numpy()
    channel_mse = (channel_se / (len(indices)*windows.horizon)).cpu().tolist()
    return {'mse':float(totals[0]/totals[2]),'mae':float(totals[1]/totals[2]),
            'elements':int(totals[2]),'windows':len(indices),'channel_mse':channel_mse}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--data-root', default='/root/autodl-tmp/paper5/data')
    args=p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    output = Path(args.output); output.mkdir(parents=True,exist_ok=True)
    if (output/'metrics.json').exists():
        existing=json.loads((output/'metrics.json').read_text())
        if existing['config']!=config: raise ValueError('run_id collision: completed config differs')
        print('already complete',output,flush=True); return
    write_json(output/'config.json',config)
    snapshot=output/'code_snapshot'; snapshot.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path,snapshot/path.name)
    source_hashes={p.name:digest(p) for p in snapshot.glob('*.py')}
    started=time.time()
    seed_all(config.get('seed',2026))
    length,horizon = config.get('length',96),config.get('horizon',96)
    pilot = config.get('phase','full')=='pilot'
    dataset = Windows(Path(args.data_root)/(config['dataset']+'.npz'),length,horizon,'cuda',pilot)
    from models import create_model
    model = create_model(config,dataset.channels).cuda()
    batch_size=config.get('batch_size',256 if dataset.channels<=64 else 128)
    if batch_size<1: raise ValueError('invalid batch size')
    epochs=config.get('epochs',8 if pilot else 30)
    patience=config.get('patience',epochs if pilot else 7)
    amp=config.get('amp',True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('lr',1e-3),weight_decay=config.get('weight_decay',1e-4))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=config.get('lr',1e-3)*0.05)
    shuffle_generator=torch.Generator(device='cuda'); shuffle_generator.manual_seed(config.get('seed',2026)+12345)
    best_val=float('inf'); best_epoch=-1; best_state=None; wait=0
    training_history=[]
    next_epoch=0
    if (output/'training_state.pt').exists():
        state=torch.load(output/'training_state.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng'].cpu()); torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        shuffle_generator.set_state(state['shuffle_rng'].cpu())
        best_val=state['best_val']; best_epoch=state['best_epoch']; wait=state['wait']
        next_epoch=state['next_epoch']; training_history=state['history']
        best_state=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
        print('RESUME',next_epoch,flush=True)
    try:
        for epoch in range(next_epoch,epochs):
            e_start=time.time(); model.train()
            ix=dataset.indices['train']
            order=ix[torch.randperm(len(ix),device='cuda',generator=shuffle_generator)]
            train_sum=torch.zeros(2,device='cuda',dtype=torch.float64)
            for chunk in order.split(batch_size):
                x,y=dataset.batch(chunk)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                    pred=model(x)
                    loss=(pred.float()-y).square().mean()
                    if hasattr(model,'auxiliary_loss'):
                        loss=loss+model.auxiliary_loss()
                    if hasattr(model,'supervised_auxiliary_loss'):
                        loss=loss+model.supervised_auxiliary_loss(y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                train_sum[0]+=loss.detach().double()*len(chunk); train_sum[1]+=len(chunk)
            scheduler.step()
            val=evaluate(model,dataset,'val',batch_size*2,amp)
            train_sum=train_sum.cpu().numpy()
            if not np.isfinite(train_sum).all() or not np.isfinite(val['mse']):
                raise FloatingPointError('non-finite epoch training/validation loss')
            row={'epoch':epoch+1,'train_loss':float(train_sum[0]/train_sum[1]),
                 'val_mse':val['mse'],'val_mae':val['mae'],'lr':scheduler.get_last_lr()[0],
                 'seconds':time.time()-e_start}
            training_history.append(row)
            print(json.dumps(row),flush=True)
            write_json(output/'history.json',training_history)
            if val['mse']<best_val-1e-10:
                best_val=val['mse']; best_epoch=epoch+1; wait=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save(best_state,output/'best.pt.tmp'); (output/'best.pt.tmp').replace(output/'best.pt')
            else: wait+=1
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                   'shuffle_rng':shuffle_generator.get_state(),'best_val':best_val,'best_epoch':best_epoch,
                   'wait':wait,'next_epoch':epoch+1,'history':training_history}
            torch.save(state,output/'training_state.pt.tmp')
            (output/'training_state.pt.tmp').replace(output/'training_state.pt')
            if wait>=patience: break
        model.load_state_dict(best_state)
        torch.save(best_state,output/'best.pt')
        metrics={'config':config,'dataset':config['dataset'],'model':config['model'],
                 'seed':config.get('seed',2026),'best_epoch':best_epoch,'best_val_mse':best_val,
                 'epochs_run':len(training_history),'parameters':sum(p.numel() for p in model.parameters()),
                 'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                 'seconds':time.time()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                 'bounds':dataset.bounds,'channels':dataset.channels,'torch':torch.__version__,
                 'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
                 'code_sha256':source_hashes,
                 'data_sha256':digest(Path(args.data_root)/(config['dataset']+'.npz')),
                 'validation':evaluate(model,dataset,'val',batch_size*2,amp)}
        if not pilot and config.get('evaluate_test',True):
            metrics['test']=evaluate(model,dataset,'test',batch_size*2,amp)
        metrics['seconds']=time.time()-started
        write_json(output/'metrics.json',metrics)
        print('COMPLETE',output,'val',metrics['validation']['mse'],
              'test',metrics.get('test',{}).get('mse'),flush=True)
    except Exception as e:
        write_json(output/'failure.json',{'error':repr(e),'config':config,'seconds':time.time()-started})
        raise

if __name__=='__main__': main()
