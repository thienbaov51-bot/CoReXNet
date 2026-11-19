"""Prospective common standardized PUREtest inference; default no IO.

Only after final native producer contracts, all48 physical closure, true pair,
actual B64/B256 synthetic resources, freeze/prestart and an exclusive native
attempt may this source import NumPy/Torch or read scientific arrays/weights.
This source has no optimizer, training, loss-selection, retry or score release.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

from once48_identity import *
from once48_native_contract import Admission,validate_producer_contract

def file_sha(path):return physical(path)['sha256']

def exclusive(path,value):
    with safe(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)

def verified_admission(root,bindings):
    from once48_contract import validate_admission
    docs={};pins={}
    for name,binding in bindings.items():
        body=safe(binding['path']).read_bytes();require(sha_bytes(body)==binding['sha256'],'raw metadata binding SHA changed')
        docs[name]=metadata_bytes(body);pins[name]=binding['sha256']
    admitted=validate_admission(docs,pins)
    for name,pin in docs['source_manifest']['files'].items():require(file_sha(safe(root)/relative(name))==pin,'frozen runtime source changed')
    return admitted

MODEL_SOURCES={
    96:('ITSMixer_ECL96_author_exact.py','17e2b015e82ac6384d3d34e0f686573c68a4a4e72a75edf1d9ae21f5c1ce565f'),
    192:('ITSMixer_ECL192_author_exact.py','cad8ec381697edc070dd263b764f64be3854263249c127efad245c9ed7d5707d'),
    336:('ITSMixer_ECL336_author_exact.py','1c0bc5d9e563db2883a8beb756b1a59c3add943fdde48d863f631eee1778b643'),
    720:('ITSMixer_ECL720_author_exact.py','c2db425241bb538d56f487b128bf0a674c5f81334b2574b287c81c561aebbac5')}
KERNEL_SOURCES={96:('itsmixer6_science_kernel.py','183a32d00c3177f078adf89fdc40fd26d01f16fda14779a74c26dbb510a3413a'),
    192:('itsmixer18_science_kernel.py','b5005325b1fe10ec4ba3547e8c45201fe91e72864846423a6166f21606c5975e'),
    336:('itsmixer18_science_kernel.py','b5005325b1fe10ec4ba3547e8c45201fe91e72864846423a6166f21606c5975e'),
    720:('itsmixer18_science_kernel.py','b5005325b1fe10ec4ba3547e8c45201fe91e72864846423a6166f21606c5975e')}

def safe(path):
    path=Path(path).absolute()
    require(all(not p.is_symlink() and not (hasattr(p,'is_junction') and p.is_junction()) for p in (path,*path.parents)),
        'leaf/ancestor symlink/junction forbidden')
    return path

def physical(path):
    path=safe(path);before=path.stat();require(path.is_file() and before.st_size>0,'regular physical file')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    after=path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'physical artifact changed during hash')
    return {'sha256':digest.hexdigest(),'bytes':before.st_size}

def verify_physical(admitted):
    require(type(admitted) is Admission,'validated whole48 admission')
    validate_producer_contract(admitted.documents['producer_native_schema'])
    for row in admitted.runs.values():
        base=safe(row['source_root'])/'results'/row['origin_run_id']
        for leaf,identity in row['files'].items():require(physical(base/leaf)==identity,'all240 native physical artifacts before framework')
        receipt=metadata_bytes((base/'COMPLETE_RECEIPT.json').read_bytes())
        require(receipt['status']=='COMPLETE_VALIDATION_ONLY' and receipt['run_id']==row['origin_run_id']
            and receipt['plan_sha256']==row['origin_plan_sha256'] and receipt['test_accessed'] is False,
            'score-free native physical completion links')
        for leaf,field in CORE.items():require(receipt[field]==row['files'][leaf]['sha256'],'native all4 complete byte links')
        require(metadata_bytes((base/'config.json').read_bytes())==row['config'],'exact native trained configuration')
        for name,pin in row['native_source_inventory'].items():
            require(physical(safe(row['source_root'])/relative(name))['sha256']==pin,'native full source/prefix/calendar identity')
    for row in admitted.documents['data_admission']['datasets'].values():
        for prefix in ('full_data','prefix','known_calendar'):
            require(physical(row[prefix+'_path'])['sha256']==row[prefix+'_sha256'],'original full-array/prefix/calendar SHA')
    for name,pin in admitted.documents['source_manifest']['files'].items():
        require(file_sha(safe(ONCE48)/relative(name))==pin,'frozen once48 source changed')
    if 'actual_synthetic_test48_qa' in admitted.documents:
        qa=admitted.documents['actual_synthetic_test48_qa']
        for name,path in qa['framework_source_paths'].items():
            require(file_sha(path)==qa['framework_source_sha256'][name],'actual measured framework source changed before inference')

def validate_attempt(attempt,rid,admitted):
    require(attempt['status']=='ONE_FROZEN_TEST_ATTEMPT_RESERVED'
        and attempt['run_id']==rid and type(attempt['test_per_run_count']) is int and attempt['test_per_run_count']==1
        and attempt['plan_sha256']==admitted.pins['test48_plan']
        and attempt['authorization_sha256']==admitted.pins['test48_authorization']
        and attempt['all48_train_physical_sha256']==admitted.pins['all48_train_physical']
        and attempt['freeze_sha256']==admitted.pins['native_pretest_freeze']
        and attempt['automatic_retry_allowed'] is attempt['training_called'] is attempt['optimizer_created'] is False,
        'exclusive once attempt before first framework/array read; no training/optimizer/retry')

def native_header(row,admitted):
    """Existing prepared192 native receipt fields; no numerical content."""
    view=config_view(row['config'])
    return {'run_id':row['run_id'],'dataset':row['dataset'],'model':row['model'],'length':96,'horizon':row['horizon'],
        'seed':2026,'channels':view['channels'],'freeze_sha256':admitted.pins['native_pretest_freeze'],
        'plan_sha256':admitted.pins['test48_plan'],'checkpoint_sha256':row['files']['best.pt']['sha256'],
        'config_sha256':row['files']['config.json']['sha256'],'source_sha256':row['source_inventory'],
        'origin_source_sha256':row['native_source_inventory'],'authorization_sha256':admitted.pins['test48_authorization'],
        'all48_train_physical_sha256':admitted.pins['all48_train_physical'],
        'full_data_sha256':row['full_data_sha256'],'prefix_sha256':row['prefix_sha256'],
        'calendar_sha256':row['calendar_sha256'],'scaler_mean_sha256':row['scaler_mean_sha256'],
        'scaler_std_sha256':row['scaler_std_sha256'],'full_calendar_sha256':row.get('full_calendar_sha256'),
        'origin_plan_sha256':row['origin_plan_sha256'],'origin_run_id':row['origin_run_id'],
        'test_windows':row['test_windows'],'test_elements':row['test_elements'],'test_per_run_count':1,
        'automatic_retry_allowed':False,'training_called':False,'optimizer_created':False,
        'training_allowed':False,'optimizer_allowed':False}

class FullTestWindows:
    def __init__(self,np,torch,config,data):
        view=config_view(config);self.C=view['channels'];self.H=view['pred_len'];self.L=96
        with np.load(data['full_data_path'],allow_pickle=False) as archive:
            require(set(archive.files)==({'data','split_counts','train_mean','train_std','metadata_json'} if view['dataset'] in ('Electricity','PEMS03','PEMS04','PEMS08') else {'data','split_counts'}),'original full NPZ schema')
            values=archive['data'];bounds=archive['split_counts'].tolist()
        with np.load(data['prefix_path'],allow_pickle=False) as archive:
            require(set(archive.files)=={'data','split_counts'},'frozen prefix NPZ schema')
            prefix=archive['data'];pbounds=archive['split_counts'].tolist()
        require(values.dtype==prefix.dtype==np.float32 and values.shape==(data['rows'],self.C)
            and prefix.shape==(bounds[1],self.C) and bounds==data['bounds'] and pbounds==bounds[:2]
            and np.isfinite(values).all() and np.array_equal(values[:bounds[1]].view(np.uint32),prefix.view(np.uint32)),
            'actual original full-prefix bit identity/shape/splits before windows')
        mean=values[:bounds[0]].mean(0,dtype=np.float64).astype(np.float32)
        std=values[:bounds[0]].std(0,dtype=np.float64).astype(np.float32);std[std<1e-6]=1
        require(sha_bytes(mean.tobytes())==data['scaler_mean_sha256'] and sha_bytes(std.tobytes())==data['scaler_std_sha256'],
            'bit-identical original TRAIN-only scaler, no TEST statistic fit')
        require(bounds[1]>=96,'legal TEST context')
        self.values=torch.as_tensor((values-mean)/std,device='cuda')
        self.indices=torch.arange(bounds[1],bounds[2]-self.H+1,device='cuda')
        self.xoff=torch.arange(-96,0,device='cuda');self.yoff=torch.arange(self.H,device='cuda')
        require(len(self.indices)==data['rows']-bounds[1]-self.H+1>0,'full chronological uncapped TEST suffix')
    def batch(self,ix):return self.values[ix[:,None]+self.xoff],self.values[ix[:,None]+self.yoff]

def load_module(root,name,pin):
    path=safe(root/name);require(physical(path)['sha256']==pin,'exact pinned scientific source')
    spec=importlib.util.spec_from_file_location('once_test_'+name.replace('.','_'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def build_model(root,config,torch,np):
    view=config_view(config)
    if view['model']=='RootPurgeLinear':
        module=load_module(root,'RootPurge_classes_exact.py',ROOT_MODEL_SHA)
        random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=False
        model=module.RootPurgeModel(SimpleNamespace(**config),module.BaseLinearModel).float().cuda()
        return model,None
    name,pin=MODEL_SOURCES[view['pred_len']];module=load_module(root,name,pin)
    name,pin=KERNEL_SOURCES[view['pred_len']];kernel=load_module(root,name,pin)
    kernel.configure_framework(torch,np)
    return kernel.build_model(module,config),kernel

def predict(model,kernel,inputs,view):
    if view['model']=='RootPurgeLinear':return model(inputs) # original forward_test, alpha training loss is not applied
    if view['pred_len']==96:return kernel.predict(model,inputs,view['channels'])
    return kernel.predict(model,inputs,view['channels'],view['pred_len'])

def evaluate(torch,model,kernel,windows,config):
    view=config_view(config);model.eval()
    sums=torch.zeros(3,device='cuda',dtype=torch.float64)
    channels=torch.zeros(windows.C,device='cuda',dtype=torch.float64)
    with torch.inference_mode():
        for ix in windows.indices.split(view['batch_size']):
            x,target=windows.batch(ix);forecast=predict(model,kernel,x,view)
            require(forecast.shape==target.shape and bool(torch.isfinite(forecast).all()),'native forecast complete horizon/channel finite shape')
            error=forecast.float()-target
            sums[0]+=error.square().sum(dtype=torch.float64);sums[1]+=error.abs().sum(dtype=torch.float64)
            sums[2]+=error.numel();channels+=error.square().sum((0,1),dtype=torch.float64)
    sse,sae,ne=sums.cpu().tolist();n=len(windows.indices)
    require(int(ne)==n*windows.H*windows.C,'pooled complete elements')
    result={'mse':sse/ne,'mae':sae/ne,'squared_error_sum':sse,'absolute_error_sum':sae,
        'channel_mse':(channels/(n*windows.H)).cpu().tolist(),'windows':n,'elements':int(ne),
        'metric_space':'COMMON_EXTERNAL_TRAIN_STANDARDIZED','masking':False,'test_drop_last':False,'test_shuffle':False}
    require(all(math.isfinite(result[k]) and result[k]>=0 for k in ('mse','mae')),'finite native standardized pooled metrics')
    return result

def execute_inference(root,rid,admitted,attempt,claim_path):
    # This first gate is hard PENDING until a different reviewed source binds
    # the true native IT18 contract/adapter. No placeholder can reach scientific IO.
    require(type(admitted) is Admission and rid in RUN_IDS,'complete48 admission/run')
    validate_producer_contract(admitted.documents['producer_native_schema'])
    validate_attempt(attempt,rid,admitted)
    require(metadata_bytes(safe(claim_path).read_bytes())==attempt,'persisted exclusive native once attempt before scientific IO')
    verify_physical(admitted)
    from once48_production_common import pinned,validate_live_authority
    cap=pinned(root,attempt['capability_name'],attempt['run_capability_sha256'])
    validate_live_authority(root,admitted,cap,attempt['capability_name'],attempt['run_capability_sha256'])
    import numpy as np
    import torch
    torch.set_num_threads(2)
    row=admitted.runs[rid];config=row['config'];root=safe(root)
    model,kernel=build_model(root,config,torch,np)
    model.load_state_dict(torch.load(row['checkpoint_path'],map_location='cuda',weights_only=True),strict=True)
    windows=FullTestWindows(np,torch,config,admitted.documents['data_admission']['datasets'][row['dataset']])
    result=evaluate(torch,model,kernel,windows,config)
    require(result['windows']==row['test_windows'] and result['elements']==row['test_elements'],'actual whole test windows/elements')
    verify_physical(admitted)
    return result # caller writes sealed native result; never print or release a partial score

def evaluate_frozen_once(root,admitted,rid,cap):
    """Exclusive native attempt is persisted before framework, arrays or weights."""
    require(cap['run_id']==rid and cap['plan_sha256']==admitted.pins['test48_plan'],'owned capability exact run')
    folder=safe(root)/'results'/rid;folder.mkdir(exist_ok=False)
    header=native_header(admitted.runs[rid],admitted)
    header.update(run_capability_sha256=cap['_capability_sha256'],capability_name=cap['_capability_name'],
        controller_actual_start_sha256=cap['controller_actual_start_sha256'])
    attempt={**header,'status':'ONE_FROZEN_TEST_ATTEMPT_RESERVED','utc_epoch':int(time.time())}
    claim_path=folder/'TEST_ATTEMPT_RECEIPT.json';exclusive(claim_path,attempt)
    try:
        result=execute_inference(root,rid,admitted,attempt,claim_path)
        exclusive(folder/'TEST_RESULT.json',{**header,'status':'COMPLETE_ONCE_FROZEN_TEST','test':result})
        receipt={**header,'status':'COMPLETE_ONCE_FROZEN_TEST','utc_epoch':int(time.time()),
            'test_result_sha256':file_sha(folder/'TEST_RESULT.json'),'attempt_receipt_sha256':file_sha(claim_path)}
        exclusive(folder/'TEST_RECEIPT.json',receipt)
        return {'status':'COMPLETE_ONCE_FROZEN_TEST','run_id':rid,'scores_printed':False}
    except BaseException as error:
        exclusive(folder/'TEST_FAILURE.json',{'status':'FAILED_ONE_RESERVED_TEST_ATTEMPT_NO_RETRY','run_id':rid,
            'attempt_receipt_sha256':file_sha(claim_path),'error_type':type(error).__name__,'automatic_retry_allowed':False})
        raise

def main():
    argparse.ArgumentParser().parse_args()
    print(json.dumps({'status':PENDING,'actual_test_started':False,'arrays_or_framework_or_scores_read':False}))
if __name__=='__main__':main()
