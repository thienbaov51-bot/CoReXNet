"""Fixed six-model L96/H96 validation-only pilot; no test entry point.

The old byte-pinned core supplies the unchanged seed, optimizer, scheduler,
predict and metric arithmetic. Only a train+validation prefix is deployable.
ROOT prepares/authorizes the plan; this module never registers/deploys models.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
from secure_paths import safe_path, safe_tree

MODELS = ('OURS','DLinear','SparseTSF','ModernTCN','MixLinear','PhaseFormer')
CAMPAIGN = 'PILOT20261011_NEW5'
REMOTE_ROOT = '/root/autodl-tmp/paper5/new5_pilot_20261011'
HORIZONS=(96,192,336,720)
HEADER = ('run_id','dataset','model','seed','length','horizon','channels','plan_sha256',
          'config_sha256','checkpoint_sha256','data_sha256','source_sha256',
          'training_windows','validation_windows','validation_elements','parameters','epochs_run','best_epoch')


def require(flag, message):
    if not flag: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with safe_path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(safe_path(path).read_text(encoding='utf-8-sig'))


def write(path, value, exclusive=False):
    path=safe_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    if exclusive:
        with path.open('xb') as stream:stream.write(raw)
    else:
        tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
        with tmp.open('xb') as stream:stream.write(raw)
        os.replace(tmp,path)


def inside(root, relative):
    root=safe_path(root).resolve();p=Path(relative)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in str(relative),'unsafe relative path')
    out=safe_path(root/p);require(out.is_relative_to(root),'path escapes new5 root')
    return out


def verify_sources(root, plan):
    require(plan.get('source_inventory'),'source inventory missing')
    for name,row in plan['source_inventory'].items():
        p=inside(root,name);h=row if isinstance(row,str) else row['sha256']
        require(p.is_file() and not p.is_symlink() and sha(p)==h,'source identity differs: '+name)
        if isinstance(row,dict) and 'bytes' in row:require(p.stat().st_size==row['bytes'],'source byte count differs')


def validate_config(c, rid, plan):
    require(re.fullmatch(r'[A-Za-z0-9_.-]+',rid) and c['run_id']==rid,'run identity')
    require(c['model'] in MODELS and c['dataset'] in plan['datasets'],'model/data panel')
    require(c['seed']==2026 and c['length']==96 and c['horizon'] in HORIZONS,'pilot seed/L/H')
    require(c.get('evaluate_test') is False and c.get('initialization')=='from_scratch_no_old_checkpoint','no test/pretrained initialization')
    require(c.get('original_epochs',c['epochs'])==c['epochs'] and c.get('original_patience',c['patience'])==c['patience'],'original full-budget epoch/patience preserved')
    require(c.get('amp',False) is (c['model']=='OURS'),'fixed OURS BF16/comparators FP32')
    require(c.get('loss','MSE').upper() in ('MSE','MAE') and c.get('optimizer','Adam') in ('Adam','AdamW'),'frozen objective/optimizer')
    for key in ('epochs','patience','batch_size','test_batch_size','reserve_gpu_mib','reserve_disk_bytes'):
        require(isinstance(c.get(key),int) and c[key]>0,'positive locked '+key)
    require(c.get('resource_qa_pass') is True,'actual recipe/resource admission required')
    d=plan['data'][c['dataset']]
    require(c['channels']==d['channels'] and c['expected_data_sha256']==d['sha256'],'payload/channel pin')
    for key,val in [('seq_len',96),('pred_len',c['horizon']),('enc_in',c['channels'])]:
        require(c.get(key,val)==val and c.get('official_config',{}).get(key,val)==val,'shape aliases')
    require(c.get('train_window_cap') is None and c.get('validation_window_cap') is None,'full-window pilot without caps')
    require(c.get('additional_loss') in (None,False),'pilot panel has no additional supervised/auxiliary training')


def load_plan(root, expected_sha, source_check=True):
    root=safe_path(root).resolve();require(str(root)==REMOTE_ROOT,'exact new5 root only');path=root/'PILOT_PLAN.json'
    require(expected_sha and sha(path)==expected_sha,'explicit immutable PILOT_PLAN SHA required')
    p=read(path)
    require(p.get('campaign')==CAMPAIGN and Path(p['cohort_root']).resolve()==root,'new pilot namespace/root')
    require(root.name not in ('add3data_20261010','add5_20261010','main20261009','ADD3DATA_L96_SINGLE_SEED_20261010','ADD5_L96_SINGLE_SEED_20261010','MAIN_L96_SINGLE_SEED_20261009'),'old frozen roots cannot host a new pilot')
    require(tuple(p['models'])==MODELS and len(p['datasets'])==len(set(p['datasets']))==5,'fixed six by five panel')
    require(p['length']==96 and p['seed']==2026 and p['horizons']==list(HORIZONS),'L/H/seed')
    require(p['expected_count']==len(p['runs'])==120,'complete fixed120')
    require(p.get('selection_policy')=='BATCH5_H96_OBSERVATION_FINAL_EACH_FOUR_H_6_OF_8' and p.get('required_selected_count')==1,'H96 observation-only; final fourH six-of-eight contract')
    require(p.get('test_access_allowed') is False and p.get('automatic_full_or_test') is False,'pilot-only boundary')
    require('train_window_cap' in p and 'validation_window_cap' in p and p['train_window_cap'] is None and p['validation_window_cap'] is None,'explicit full train/val window policy')
    require(p.get('budget_policy')=='ORIGINAL_TEMPLATE_EPOCHS_PATIENCE_AND_FULL_WINDOWS','original full-budget policy')
    require(p.get('memory_budget_mib',22500)<=22500 and p.get('disk_budget_bytes',4*(1<<30))>0,'resource admission')
    grid=set()
    for ds,d in p['data'].items():
        require(ds in p['datasets'] and re.fullmatch(r'[A-Za-z0-9_.-]+',ds),'safe dataset ID')
        require(d.get('path',f'data/{ds}_trainval.npz')==f'data/{ds}_trainval.npz','prefix-only data filename')
        require(d['channels']>0 and len(d['bounds'])==3 and 96+max(HORIZONS)-1<=d['bounds'][0] and d['bounds'][0]+max(HORIZONS)<=d['bounds'][1]<d['bounds'][2],'legal prefix metadata')
    require(set(p['data'])==set(p['datasets']),'data panel complete')
    for rid,e in p['runs'].items():
        validate_config(e['config'],rid,p);require(e.get('config_sha256'),'submitted config pin')
        grid.add((e['config']['dataset'],e['config']['model'],e['config']['horizon']))
    require(grid==set(itertools.product(p['datasets'],MODELS,HORIZONS)),'complete matched grid')
    require(p.get('frozen_runtime_relative','core_utils.py') in p['source_inventory'] and p.get('models_adapter_relative','models_adapter.py') in p['source_inventory'],'original runtime/adapter closure')
    if source_check:
        verify_sources(root,p)
        require('runtime.py' in p['source_inventory'] and sha(__file__)==sha(root/'runtime.py'),'executing runtime is the pinned pilot source')
    # Reject an accidentally deployed complete/raw test payload.
    require({x.name for x in (root/'data').glob('*.npz')}=={ds+'_trainval.npz' for ds in p['datasets']},'test/full/extra NPZ must not be deployed to pilot')
    for ds,d in p['data'].items():require(sha(root/'data'/f'{ds}_trainval.npz')==d['sha256'],'prefix data bytes differ')
    return p


def authorize(root, plan_sha, auth_sha, plan):
    path=Path(root)/'PILOT_AUTHORIZATION.json'
    require(auth_sha and sha(path)==auth_sha,'explicit authorization SHA')
    a=read(path)
    require(a['status']=='ROOT_AUTHORIZED_VALIDATION_ONLY' and a['plan_sha256']==plan_sha,'pilot authorization')
    require(Path(a['cohort_root']).resolve()==Path(root).resolve() and a['no_test'] is True,'authorization namespace/no-test')
    require(1<=a['workers']<=8 and a['memory_budget_mib']==plan.get('memory_budget_mib',22500),'authorization resources')
    return a


def frozen_core(root, plan):
    path=inside(root,plan.get('frozen_runtime_relative','core_utils.py'))
    name='_pilot_original_core_'+sha(path)[:12]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def model_adapter(root, plan):
    path=inside(root,plan.get('models_adapter_relative','models_adapter.py'))
    name='_pilot_original_adapter_'+sha(path)[:12]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


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


def complete_metadata(root, rid, plan, plan_sha, physical_best=True):
    folder=Path(root)/'results'/rid;entry=plan['runs'][rid]
    path=folder/'validation_receipt.json'
    if not path.exists():return None
    r=read(path);c=entry['config']
    require(r['status']=='COMPLETE_VALIDATION_ONLY' and r['run_id']==rid and r['plan_sha256']==plan_sha,'completed validation receipt')
    require(r['test_values_accessed'] is False and r['test_indices_constructed'] is False,'no-test receipt')
    require(read(folder/'config.json')==c and r['config_sha256']==sha(folder/'config.json'),'actual config identity')
    require(r['dataset']==c['dataset'] and r['model']==c['model'] and r['data_sha256']==c['expected_data_sha256'],'run/data identity')
    for name,row in plan['source_inventory'].items():
        h=row if isinstance(row,str) else row['sha256']
        require(r['source_sha256'][name]==h,'source receipt closure')
    require(set(r['source_sha256'])==set(plan['source_inventory']),'source set closure')
    for name,key,sizekey in [('validation_result.json','validation_result_sha256','validation_result_bytes'),('metrics.json','training_metrics_sha256',None),('history.json','history_sha256',None)]:
        require(sha(folder/name)==r[key],'completed evidence bytes '+name)
        if sizekey:require((folder/name).stat().st_size==r[sizekey],'result byte count')
    if physical_best:require(sha(folder/'best.pt')==r['checkpoint_sha256'] and (folder/'best.pt').stat().st_size==r['checkpoint_bytes'],'best raw bytes')
    b=plan['data'][c['dataset']]['bounds'];C=c['channels']
    require(r['training_windows']==b[0]-96-c['horizon']+1 and r['validation_windows']==b[1]-b[0]-c['horizon']+1,'complete prefix window counts')
    require(r['validation_elements']==r['validation_windows']*c['horizon']*C and 0<r['best_epoch']<=r['epochs_run']<=c['epochs'],'elements/epoch count')
    require(r['seed']==2026 and r['length']==96 and r['horizon']==c['horizon'],'seed/dimension receipt')
    return r


def finish_completed_cleanup(root, rid, plan, plan_sha):
    """Resource cleanup is permitted only after complete best/header closure."""
    receipt=complete_metadata(root,rid,plan,plan_sha)
    require(receipt is not None,'cannot clean active/failed/incomplete resume state')
    folder=Path(root)/'results'/rid;state=folder/'training_state.pt';path=folder/'RESUME_CLEANUP_RECEIPT.json'
    if state.exists():
        if path.exists():
            cleanup=read(path);require(cleanup['run_id']==rid and cleanup['plan_sha256']==plan_sha and cleanup['removed_sha256']==sha(state) and cleanup['removed_bytes']==state.stat().st_size,'interrupted cleanup state identity')
        else:
            cleanup=dict(status='DISPOSABLE_RESUME_REMOVED_AFTER_VALIDATION_COMPLETE',run_id=rid,plan_sha256=plan_sha,removed_sha256=sha(state),removed_bytes=state.stat().st_size,preserved_best_sha256=receipt['checkpoint_sha256'])
            write(path,cleanup,exclusive=True)
        state.unlink()
    require(path.exists() and not state.exists() and not(folder/'training_state.pt.tmp').exists(),'completed cleanup missing or active temporary state')
    cleanup=read(path)
    require(cleanup['status']=='DISPOSABLE_RESUME_REMOVED_AFTER_VALIDATION_COMPLETE' and cleanup['run_id']==rid and cleanup['plan_sha256']==plan_sha and cleanup['preserved_best_sha256']==receipt['checkpoint_sha256'],'completed cleanup receipt')
    return True


def completed_clean(root, rid, plan, plan_sha):
    if complete_metadata(root,rid,plan,plan_sha) is None:return False
    folder=Path(root)/'results'/rid
    if (folder/'training_state.pt').exists() or (folder/'training_state.pt.tmp').exists() or not(folder/'RESUME_CLEANUP_RECEIPT.json').exists():return False
    c=read(folder/'RESUME_CLEANUP_RECEIPT.json')
    return c['run_id']==rid and c['plan_sha256']==plan_sha and c['preserved_best_sha256']==sha(folder/'best.pt')


def train(root, plan, plan_sha, rid, synthetic_fixture=False, device='cuda', extension_authorization_sha256=None):
    root=Path(root).resolve();require(rid in plan['runs'],'declared run ID required')
    require(synthetic_fixture is plan.get('synthetic_fixture_only',False),'synthetic/research scope mismatch')
    require(synthetic_fixture or device=='cuda','research must use approved CUDA runtime')
    verify_sources(root,plan);c=dict(plan['runs'][rid]['config']);validate_config(c,rid,plan)
    if not synthetic_fixture and c['horizon']!=96:
        import batch_common
        batch_common.authorize_extension(root,plan,plan_sha,extension_authorization_sha256,physical_check=False)
    folder=root/'results'/rid;folder.mkdir(parents=True,exist_ok=True)
    if complete_metadata(root,rid,plan,plan_sha):
        finish_completed_cleanup(root,rid,plan,plan_sha);return
    require(not(folder/'FAILED_VALIDATION_TRAINING.json').exists(),'failed run is sealed; no automatic retry')
    core=frozen_core(root,plan);torch=core.initialize_tensor_runtime(root/'dependencies')
    require(synthetic_fixture or torch.cuda.is_available(),'CUDA unavailable')
    core.seed_all(2026)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    write(folder/'config.json',c)
    snapshot=folder/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ('runtime.py',plan.get('frozen_runtime_relative','core_utils.py'),plan.get('models_adapter_relative','models_adapter.py')):
        src=inside(root,name);dst=snapshot/Path(name).name
        if dst.exists():require(sha(dst)==sha(src),'resume snapshot collision')
        else:dst.write_bytes(src.read_bytes())
    data_path=root/'data'/f"{c['dataset']}_trainval.npz"
    require(sha(data_path)==c['expected_data_sha256'],'trainval byte pin')
    windows=PrefixWindows(data_path,plan['data'][c['dataset']],core,device,plan['train_window_cap'],plan['validation_window_cap'],c['horizon'])
    cfg=dict(c,dependency_root=str(root/'dependencies'),device=device)
    model=model_adapter(root,plan).make_model(cfg).to(device)
    optimizer=core.make_optimizer(model,c)
    scheduler,per_batch=core.make_scheduler(optimizer,c,math.ceil(len(windows.indices['train'])/c['batch_size']))
    shuffle=torch.Generator(device=device);shuffle.manual_seed(2026+12345)
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
        verify_sources(root,plan);require(sha(data_path)==c['expected_data_sha256'],'data changed while training')
        source={k:(v if isinstance(v,str) else v['sha256']) for k,v in plan['source_inventory'].items()}
        header=dict(run_id=rid,dataset=c['dataset'],model=c['model'],seed=2026,length=96,horizon=c['horizon'],channels=c['channels'],plan_sha256=plan_sha,config_sha256=sha(folder/'config.json'),checkpoint_sha256=sha(folder/'best.pt'),data_sha256=c['expected_data_sha256'],source_sha256=source,training_windows=len(windows.indices['train']),validation_windows=len(windows.indices['val']),validation_elements=val['elements'],parameters=sum(p.numel() for p in model.parameters()),epochs_run=len(history),best_epoch=best_epoch,test_values_accessed=False,test_indices_constructed=False)
        write(folder/'metrics.json',dict(header,config=c,validation=val,seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,synthetic_fixture_only=synthetic_fixture),exclusive=True)
        body=dict(header,status='COMPLETE_VALIDATION_ONLY',validation=val);write(folder/'validation_result.json',body,exclusive=True)
        receipt=dict(header,status='COMPLETE_VALIDATION_ONLY',checkpoint_bytes=(folder/'best.pt').stat().st_size,training_metrics_sha256=sha(folder/'metrics.json'),history_sha256=sha(folder/'history.json'),validation_result_sha256=sha(folder/'validation_result.json'),validation_result_bytes=(folder/'validation_result.json').stat().st_size)
        write(folder/'validation_receipt.json',receipt,exclusive=True)
        complete_metadata(root,rid,plan,plan_sha)
        finish_completed_cleanup(root,rid,plan,plan_sha)
    except Exception as exc:
        write(folder/'FAILED_VALIDATION_TRAINING.json',dict(run_id=rid,plan_sha256=plan_sha,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),automatic_retry=False,science_config_changed=False,test_values_accessed=False),exclusive=True)
        raise


def dataset_gate(rows):
    require(set(rows)==set(MODELS),'dataset six complete')
    ours=rows['OURS'];comparisons={}
    for m in MODELS[1:]:
        other=rows[m];gain={}
        for k in ('mse','mae'):
            require(math.isfinite(ours[k]) and ours[k]>=0 and math.isfinite(other[k]) and other[k]>=0,'finite nonnegative metric')
            gain[k]=dict(ours=ours[k],baseline=other[k],raw_gain=other[k]-ours[k],relative_gain_percent=100*(other[k]-ours[k])/other[k] if other[k]>0 else None,strict_win=ours[k]<other[k])
        comparisons[m]=gain
    qualifies=all(v[k]['strict_win'] for v in comparisons.values() for k in ('mse','mae'))
    strict_metric_wins=sum(all(comparisons[m][k]['strict_win'] for m in MODELS[1:]) for k in ('mse','mae'))
    return dict(eligible_for_full_review=True,H96_observation_only=True,
                strict_metric_wins=strict_metric_wins,total_metric_groups=2,
                both_H96_metrics_strict_best=qualifies,
                early_stop_on_H96_loss_allowed=False,root_explicit_extension_required=True,
                criterion='H96 observation only; final fourH validation requires at least six of eight strict metric wins',
                comparisons=comparisons,all_six_results=rows)


def audit_complete(*args, **kwargs):
    raise ValueError('use audit_validation.py after explicit phase or whole120 physical closure')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--plan-sha256');p.add_argument('--authorization-sha256');p.add_argument('--config');p.add_argument('--action',choices=['metadata','train','audit'],default='metadata');p.add_argument('--root-released-complete-validation',action='store_true');p.add_argument('--extension-authorization-sha256');p.add_argument('--prestart-manifest-sha256')
    a=p.parse_args()
    if not a.root or not a.plan_sha256:print(json.dumps(dict(status='PENDING_EXPLICIT_PILOT_PLAN',test_entrypoint_exists=False)));return
    root=safe_path(a.root).resolve();plan=load_plan(root,a.plan_sha256);authorize(root,a.plan_sha256,a.authorization_sha256,plan)
    if a.action=='metadata':
        missing=[rid for rid in plan['runs'] if complete_metadata(root,rid,plan,a.plan_sha256) is None]
        print(json.dumps(dict(status='ALL120_METADATA_PASS' if not missing else 'PENDING',missing=missing,scores_decoded=False)));return
    if a.action=='audit':
        require(a.root_released_complete_validation,'explicit complete validation release required');r=audit_complete(root,plan,a.plan_sha256)
        print(json.dumps(dict(status=r['status'],eligible_dataset_count=r['eligible_dataset_count'],test_values_accessed=False)));return
    import batch_common
    batch_common.prestart(root,plan,a.plan_sha256,a.authorization_sha256,a.prestart_manifest_sha256)
    safe_tree(root/'results')
    require(a.config,'explicit submitted config required');submitted=safe_path(a.config).resolve()
    require(submitted.parent==(root/'running').resolve(),'train config must belong to owned running registry')
    c=read(submitted);rid=c['run_id'];require(rid in plan['runs'] and sha(submitted)==plan['runs'][rid]['config_sha256'] and c==plan['runs'][rid]['config'],'submitted config')
    try:train(root,plan,a.plan_sha256,rid,extension_authorization_sha256=a.extension_authorization_sha256)
    except Exception as exc:
        path=root/'results'/rid/'FAILED_VALIDATION_TRAINING.json'
        if not path.exists():
            write(path,dict(run_id=rid,plan_sha256=a.plan_sha256,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),automatic_retry=False,science_config_changed=False,test_values_accessed=False),exclusive=True)
        raise


if __name__=='__main__':main()
