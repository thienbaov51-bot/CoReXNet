"""AEMO5 selected original24 checkpoints: synthetic admission then one pure test.
All mutations are confined to a new child. No optimizer, fit, retry or checkpoint copy.
"""
import argparse,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
PARENT=Path('/root/autodl-tmp/paper5/new5_pilot_20261011')
CHILD=PARENT/'selected_test24/AEMO5'
PLAN_SHA='144f12323755d34d173f39b2d581b288690c864ad38233257a739d1422b6f69d'
AUTH_SHA='7b4fcb6a33fccf1be1bd7f90924b8d1c0fd8eca030a07f6a2b4640ae240c3329'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def safe():
    for p in [CHILD,*CHILD.parents]:assert not p.is_symlink()
    assert Path(__file__).resolve()==CHILD/'actual_test24.py'
def context():
    safe();sys.path.insert(0,str(PARENT));import runtime as rt
    plan=rt.load_plan(PARENT,PLAN_SHA);rt.authorize(PARENT,PLAN_SHA,AUTH_SHA,plan)
    a=read(CHILD/'ROOT_SELECTED24_TEST_AUTHORIZATION.json')
    assert a['selected_dataset']=='AEMO5' and a['plan_sha256']==PLAN_SHA and a['test_per_run_count']==1 and not a['retraining_allowed'] and not a['recipe_changes_allowed']
    parentproof=read(CHILD/'ROOT_FRESH_ALL120_PHYSICAL_VERIFICATION_BEFORE_SELECTED_TEST.json')
    inventory={rid:p for rid,p in parentproof['closure']['physical_inventory'].items() if p['dataset']=='AEMO5'}
    assert sorted(inventory)==a['selected_run_ids'] and len(inventory)==24
    for rid,p in inventory.items():
        folder=PARENT/'results'/rid
        assert sha(folder/'best.pt')==p['checkpoint_sha256'] and sha(folder/'config.json')==p['config_sha256']
    full=read(CHILD/'FULL_DATA_PREPARATION_RECEIPT.json')
    assert sha(CHILD/'AEMO5_full.npz')==full['full_sha256'] and full['shape']==[87648,5] and full['bounds']==[61353,70118,87648]
    assert full['original_prefix_float32_bitwise_equal'] and full['train_scaler_bitwise_equal']
    return rt,plan,inventory,full
def core_model(rt,plan,config,checkpoint=None):
    core=rt.frozen_core(PARENT,plan);torch=core.initialize_tensor_runtime(PARENT/'dependencies')
    assert torch.cuda.is_available();core.seed_all(2026);torch.cuda.reset_peak_memory_stats()
    model=rt.model_adapter(PARENT,plan).make_model(dict(config,dependency_root=str(PARENT/'dependencies'),device='cuda')).cuda()
    if checkpoint:model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True),strict=True)
    return core,torch,model
def probe():
    rt,plan,inventory,full=context();import numpy as np
    path=CHILD/'synthetic_full_resident.npz';assert not path.exists()
    rng=np.random.default_rng(2026)
    with path.open('xb') as f:np.savez_compressed(f,data=rng.standard_normal((87648,5)).astype(np.float32),split_counts=np.array(full['bounds'],dtype=np.int64))
    records={}
    try:
        for rid,p in inventory.items():
            config=read(PARENT/'results'/rid/'config.json');core,torch,model=core_model(rt,plan,config)
            view=core.Windows(path,96,p['horizon'],'cuda',False)
            view.indices={'test':view.indices['test'][:config['test_batch_size']]}
            core.evaluate(model,view,'test',config['test_batch_size'],config.get('amp',False))
            torch.cuda.synchronize();peak=int(torch.cuda.max_memory_allocated())
            records[rid]={'status':'PASS_SYNTHETIC_FULL_RESIDENT_ORIGINAL_TEST_BATCH','peak_allocated_bytes':peak,'reserve_gpu_mib':max(1200,math.ceil(peak/(1<<20))+512),'test_batch_size':config['test_batch_size'],'amp':config['amp'],'full_rows':87648,'horizon':p['horizon'],'config_sha256':p['config_sha256'],'actual_scores_read':False}
            del model,view;torch.cuda.empty_cache()
    finally:
        if path.exists():path.unlink()
    assert len(records)==24 and sum(sorted([r['reserve_gpu_mib'] for r in records.values()],reverse=True)[:8])<=22500
    save(CHILD/'ACTUAL_FULL_RESIDENT24_CUDA_QA.json',{'status':'PASS_ACTUAL24_SYNTHETIC_FULL_RESIDENT_ORIGINAL_BATCH_CUDA','records':records,'scientific_test_started':False,'real_test_scores_read':False,'utc_epoch':time.time()})
    print(json.dumps({'status':'PASS_ACTUAL24_SYNTHETIC_FULL_RESIDENT_ORIGINAL_BATCH_CUDA','count':24}))
def freeze():
    rt,plan,inventory,full=context();qa=read(CHILD/'ACTUAL_FULL_RESIDENT24_CUDA_QA.json')
    assert qa['status']=='PASS_ACTUAL24_SYNTHETIC_FULL_RESIDENT_ORIGINAL_BATCH_CUDA' and set(qa['records'])==set(inventory)
    review=read(CHILD/'INDEPENDENT_ACTUAL_FULL_DATA_SOURCE_RESOURCE_REVIEW.json');assert review['status']=='PASS_INDEPENDENT_ACTUAL_AEMO_FULL_DATA_SOURCE_RESOURCE_REVIEW' and not review['test_scores_read']
    files={p.name:sha(p) for p in CHILD.iterdir() if p.is_file() and p.name not in ['PRETEST_FREEZE.json']}
    save(CHILD/'PRETEST_FREEZE.json',{'status':'ALL24_PRETEST_FROZEN_AFTER_VALIDATION_GATE_PASS','parent_plan_sha256':PLAN_SHA,'selected_dataset':'AEMO5','physical_inventory':inventory,'child_files_sha256':files,'parent_source_inventory':plan['source_inventory'],'full_data':full,'test_per_run_count':1,'automatic_retry_allowed':False,'utc_epoch':time.time()})
    print(json.dumps({'status':'ALL24_PRETEST_FROZEN_AFTER_VALIDATION_GATE_PASS','freeze_sha256':sha(CHILD/'PRETEST_FREEZE.json')}))
def verified_freeze(pin):
    assert sha(CHILD/'PRETEST_FREEZE.json')==pin
    f=read(CHILD/'PRETEST_FREEZE.json')
    for name,h in f['child_files_sha256'].items():assert sha(CHILD/name)==h
    rt,plan,inv,full=context();assert inv==f['physical_inventory'] and full==f['full_data']
    return rt,plan,inv,f
def test_one(rid,pin):
    rt,plan,inventory,freeze=verified_freeze(pin);assert rid in inventory
    p=inventory[rid];folder=CHILD/'results'/rid;folder.mkdir(parents=True,exist_ok=True)
    assert not any(folder.iterdir()),'an existing attempt blocks any evaluation'
    attempt={'status':'ONE_FROZEN_TEST_ATTEMPT_RESERVED','run_id':rid,'freeze_sha256':pin,'checkpoint_sha256':p['checkpoint_sha256'],'automatic_retry_allowed':False,'training_called':False,'optimizer_created':False,'utc_epoch':time.time()}
    save(folder/'TEST_ATTEMPT_RECEIPT.json',attempt)
    try:
        config=read(PARENT/'results'/rid/'config.json');core,torch,model=core_model(rt,plan,config,PARENT/'results'/rid/'best.pt')
        view=core.Windows(CHILD/'AEMO5_full.npz',96,p['horizon'],'cuda',False);view.indices={'test':view.indices['test']}
        values=core.evaluate(model,view,'test',config['test_batch_size'],config['amp'])
        assert values['windows']==87648-70118-p['horizon']+1 and values['elements']==values['windows']*p['horizon']*5
        assert all(math.isfinite(values[k]) and values[k]>=0 for k in ['mse','mae'])
        assert math.isclose(sum(values['channel_mse'])/5,values['mse'],rel_tol=1e-10,abs_tol=1e-12)
        assert sha(PARENT/'results'/rid/'best.pt')==p['checkpoint_sha256'];rt.verify_sources(PARENT,plan)
        header={'status':'COMPLETE_ONCE_FROZEN_TEST','run_id':rid,'dataset':'AEMO5','model':p['model'],'length':96,'horizon':p['horizon'],'seed':2026,'channels':5,'freeze_sha256':pin,'parent_plan_sha256':PLAN_SHA,'checkpoint_sha256':p['checkpoint_sha256'],'config_sha256':p['config_sha256'],'source_sha256':p['source_sha256'],'full_data_sha256':freeze['full_data']['full_sha256'],'prefix_sha256':p['data_sha256'],'test_windows':values['windows'],'test_elements':values['elements'],'test_per_run_count':1,'training_called':False,'optimizer_created':False}
        save(folder/'TEST_RESULT.json',dict(header,test=values))
        save(folder/'TEST_RECEIPT.json',dict(header,test_result_sha256=sha(folder/'TEST_RESULT.json'),attempt_receipt_sha256=sha(folder/'TEST_ATTEMPT_RECEIPT.json')))
        print(json.dumps({'status':'COMPLETE_ONCE_FROZEN_TEST','run_id':rid,'scores_printed':False}))
    except Exception as e:
        save(folder/'TEST_FAILURE.json',{'status':'FAILED_ONCE_TEST_NO_RETRY','run_id':rid,'error':repr(e),'utc_epoch':time.time()});raise
def controller(pin):
    rt,plan,inventory,freeze=verified_freeze(pin)
    assert not (CHILD/'CONTROLLER_START.json').exists()
    save(CHILD/'CONTROLLER_START.json',{'status':'ACTUAL_ONCE24_TEST_CONTROLLER_STARTED','pid':os.getpid(),'freeze_sha256':pin,'utc_epoch':time.time()})
    queue=list(inventory);active={};finished=[];failed=[]
    qa=read(CHILD/'ACTUAL_FULL_RESIDENT24_CUDA_QA.json')['records']
    while queue or active:
        for rid,(proc,log) in list(active.items()):
            code=proc.poll()
            if code is not None:
                log.close();(finished if code==0 else failed).append(rid);del active[rid]
        if failed:break
        while queue and len(active)<8:
            import shutil
            assert shutil.disk_usage(CHILD).free>=1536*(1<<20)
            rid=queue[0];reserved=sum(qa[x]['reserve_gpu_mib'] for x in active)+qa[rid]['reserve_gpu_mib']
            if reserved>22500:break
            memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
            if float(memory)<qa[rid]['reserve_gpu_mib']:break
            queue.pop(0);log=(CHILD/'logs'/f'{rid}.log').open('x')
            proc=subprocess.Popen([sys.executable,'-u',str(CHILD/'actual_test24.py'),'--test-one',rid,'--freeze-sha256',pin],stdout=log,stderr=subprocess.STDOUT);active[rid]=(proc,log)
        state={'status':'RUNNING_ONCE_TEST24','done':len(finished),'active':{rid:p.pid for rid,(p,l) in active.items()},'queued':len(queue),'failed':failed,'freeze_sha256':pin,'utc_epoch':time.time(),'scores_read':False}
        tmp=CHILD/'STATE.tmp';tmp.write_text(json.dumps(state));os.replace(tmp,CHILD/'STATE.json');time.sleep(2)
    for proc,log in active.values():proc.wait();log.close()
    save(CHILD/'TEST_BOUNDARY.json',{'status':'COMPLETE_ONCE_TEST24' if not failed and len(finished)==24 else 'FAILED_ONCE_TEST24_NO_RETRY','done':len(finished),'failed':failed,'freeze_sha256':pin,'utc_epoch':time.time(),'scores_read':False})
def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');p.add_argument('--freeze',action='store_true');p.add_argument('--controller',action='store_true');p.add_argument('--test-one');p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.probe:probe()
    elif a.freeze:freeze()
    elif a.controller:controller(a.freeze_sha256)
    elif a.test_one:test_one(a.test_one,a.freeze_sha256)
    else:print('PENDING_NO_TEST_OR_DATA_READ')
if __name__=='__main__':main()
