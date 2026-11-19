"""One frozen new-seed PUREtest via each original scientific forward/evaluator."""
import argparse,hashlib,importlib.util,json,math,os,random,sys,time
from pathlib import Path
from types import SimpleNamespace
import test_common as admin
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
sys.dont_write_bytecode=True

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def initialize_family(config,science):
    """Construct the original frozen model only, with no optimizer or training."""
    for path in reversed(science['python_paths']):sys.path.insert(0,path)
    source=Path(science['source_dir']);sys.path.insert(0,str(source))
    import numpy as np
    import torch
    torch.set_num_threads(2)
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    admin.require(torch.cuda.is_available(),'actual production CUDA required')
    family=science['family'];kernel=None
    if family in ('native_ours','native_classical'):
        runner=module('frozen_original_runner',science['runner_path']);runner.seed_all(seed)
        factory=module('frozen_original_factory',science['factory_module_path'])
        model=factory.create_model(config,science['channels']).cuda()
        evaluator=runner
    elif family in ('adapter_core','aemo_prefix'):
        core=module('frozen_original_core',science['core_module_path']);core.initialize_tensor_runtime(science.get('dependency_root'));core.seed_all(seed)
        effective=core.effective_config(config)
        if science.get('dependency_root'):effective['dependency_root']=science['dependency_root']
        model=core.make_model(effective).cuda();evaluator=core
    elif family=='rootpurge':
        exact=module('frozen_original_rootpurge',science['factory_module_path'])
        model=exact.RootPurgeModel(SimpleNamespace(**config),exact.BaseLinearModel).float().cuda()
        evaluator=module('root_it_exact_test_kernels',Path(__file__).parent/'root_it_original_puretest_kernels.py')
    elif family=='itsmixer':
        kernel=module('frozen_original_itsmixer_kernel',science['factory_module_path']);kernel.configure_framework(torch,np,seed)
        author=module('frozen_original_itsmixer_author',science['model_source_path'])
        model=kernel.build_model(author,config)
        evaluator=module('root_it_exact_test_kernels',Path(__file__).parent/'root_it_original_puretest_kernels.py')
    else:raise ValueError('undeclared scientific family')
    return dict(np=np,torch=torch,model=model,evaluator=evaluator,kernel=kernel)

def full_windows(runtime,config,science):
    """Use the original window parser and external TRAIN-only standardization."""
    np,torch=runtime['np'],runtime['torch'];evaluator=runtime['evaluator']
    family=science['family']
    if family in ('rootpurge','itsmixer'):
        windows=evaluator.FullTestWindows(np,torch,config,science)
        admin.require(len(windows.indices)==science['test_windows'],'original full Root/IT chronological suffix')
    else:
        windows=evaluator.Windows(Path(science['full_data_path']),96,science['horizon'],'cuda',False)
        admin.require(windows.bounds==science['bounds'] and windows.channels==science['channels'] and len(windows.values)==science['rows'],'original full-array shape/split identity')
        admin.require(hashlib.sha256(windows.mean.tobytes()).hexdigest()==science['scaler_mean_sha256'] and hashlib.sha256(windows.std.tobytes()).hexdigest()==science['scaler_std_sha256'],'bit-identical original TRAIN scaler')
        admin.require(len(windows.indices['test'])==science['test_windows'],'all original chronological TEST windows')
    return windows

def evaluate_original(runtime,windows,config,science):
    if science['family'] in ('rootpurge','itsmixer'):
        return runtime['evaluator'].evaluate(runtime['torch'],runtime['model'],runtime['kernel'],windows,config)
    return runtime['evaluator'].evaluate(runtime['model'],windows,'test',science['evaluation_batch_size'],science['amp'])

def synthetic_windows(runtime,science):
    """No real data. Same full residency plus exact original batch and tail."""
    torch=runtime['torch'];C,H,rows=science['channels'],science['horizon'],science['rows'];bounds=science['bounds'];E=science['evaluation_batch_size']
    tail=science['test_windows']%E;count=E+tail
    class SyntheticFullWindows:
        def __init__(self):
            self.values=torch.randn(rows,C,device='cuda',dtype=torch.float32)
            self.length,self.L=96,96;self.horizon,self.H=H,H;self.channels,self.C=C,C
            self.xoff=torch.arange(-96,0,device='cuda');self.yoff=torch.arange(H,device='cuda')
            # Preserve all native full-array index residency before selecting
            # the two synthetic batches. Root/IT original test keeps onlyTEST.
            alltest=torch.arange(bounds[1],bounds[2]-H+1,device='cuda')
            if science['family'] in ('rootpurge','itsmixer'):
                self.full_test_indices=alltest;self.indices=alltest[:count]
            else:
                self.full_test_indices=alltest
                self.indices={'train':torch.arange(96,bounds[0]-H+1,device='cuda'),'val':torch.arange(bounds[0],bounds[1]-H+1,device='cuda'),'test':alltest[:count]}
            admin.require(len(alltest)>=count,'declared original suffix contains full batch plus actual partial tail')
        def batch(self,ix):return self.values[ix[:,None]+self.xoff],self.values[ix[:,None]+self.yoff]
    return SyntheticFullWindows()

def fixed_curve(runtime,windows,config,science):
    case=science['fixed_qualitative'];torch=runtime['torch'];model=runtime['model'];model.eval()
    origin,channel=case['origin'],case['channel'];H=science['horizon']
    admin.require(origin==42053 and channel==117 and science['bounds'][1]<=origin and origin+H<=science['rows'],'fixed original Solar qualitative window')
    ix=torch.tensor([origin],device='cuda',dtype=torch.long)
    with torch.inference_mode():
        inputs,truth=windows.batch(ix)
        if science['family'] in ('rootpurge','itsmixer'):
            view=runtime['evaluator'].config_view(config)
            forecast=runtime['evaluator'].predict(model,runtime['kernel'],inputs,view)
        else:
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=science['amp']):
                forecast=runtime['evaluator'].predict(model,inputs,ix) if hasattr(runtime['evaluator'],'predict') else model(inputs)
        admin.require(tuple(forecast.shape)==tuple(truth.shape) and bool(torch.isfinite(forecast).all()),'fixed full-horizon qualitative forecast shape')
    return dict(dataset='Solar',origin=origin,channel=channel,length=96,horizon=H,seed=config['seed'],metric_space='COMMON_EXTERNAL_TRAIN_STANDARDIZED',history=inputs[0,:,channel].float().cpu().tolist(),ground_truth=truth[0,:,channel].float().cpu().tolist(),prediction=forecast[0,:,channel].float().cpu().tolist(),window_reselected=False)

def execute(root,job_id,checkpoint,output):
    root=Path(root).resolve();claim=admin.read(admin.owned(root,'attempts/'+job_id+'/TEST_CLAIM.json'))
    admin.require(claim['job_id']==job_id,'owned ROOT-dispatched once claim')
    value=admin.plan(root,claim['plan_sha256']);job=admin.selected(value,job_id)
    admin.require(Path(checkpoint).resolve()==admin.owned(root,job['cache_relative']).resolve() and Path(output).resolve()==admin.owned(root,job['output_relative']).resolve(),'literal exact owned cache/output paths')
    admin.verify_pins(job)
    config_path=Path(job['config_path']);admin.require(admin.sha(config_path)==job['config_sha256'],'frozen original trained seed config')
    config=admin.read(config_path);admin.require(config['seed']==job['seed'] and job['seed'] in (2027,2028),'only added new seed')
    admin.require(admin.sha(checkpoint)==job['best_sha256'] and Path(checkpoint).stat().st_size==job['best_bytes'],'same selected best before first scientific IO')
    science=job['science'];admin.require(admin.sha(science['full_data_path'])==science['full_data_sha256'],'exact original full data bytes')
    admin.require(science['family'] in ('native_ours','native_classical','adapter_core','aemo_prefix','rootpurge','itsmixer'),'declared original family')
    out=Path(output);admin.require(not out.exists(),'exclusive puretest output');out.mkdir(parents=True)
    runtime=initialize_family(config,science);torch=runtime['torch'];model=runtime['model']
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True),strict=True)
    windows=full_windows(runtime,config,science)
    result=evaluate_original(runtime,windows,config,science)
    admin.require(result['windows']==science['test_windows'] and result['elements']==science['test_elements'],'complete original TEST window/element counts')
    admin.require(all(math.isfinite(result[k]) and result[k]>=0 for k in ('mse','mae')),'finite original pooled test metrics')
    header=dict(job_id=job_id,seed=job['seed'],dataset=job['dataset'],model=job['model'],horizon=job['horizon'],length=96,plan_sha256=claim['plan_sha256'],config_sha256=job['config_sha256'],checkpoint_sha256=job['best_sha256'],source_data_config_sha256=job['pins'],evaluation_batch_size=science['evaluation_batch_size'],amp=science['amp'],no_optimizer=True,training_executed=False,checkpoint_selection_executed=False,automatic_retry=False,test_per_run_count=1)
    admin.write(out/'TEST_RESULT.json',{**header,'status':'COMPLETE_ONCE_TEST_SEALED_PENDING_ALL426','test':result},exclusive=True)
    optional={}
    if 'fixed_qualitative' in science:
        admin.write(out/'QUALITATIVE_CURVE.json',{**header,**fixed_curve(runtime,windows,config,science)},exclusive=True)
        optional['QUALITATIVE_CURVE.json']=admin.sha(out/'QUALITATIVE_CURVE.json')
    admin.verify_pins(job);admin.require(admin.sha(checkpoint)==job['best_sha256'],'same checkpoint after original pure inference')
    receipt={**header,'status':'COMPLETE_ONCE_TEST_FILES_PENDING_ALL426_RELEASE','windows':result['windows'],'elements':result['elements'],'test_result_sha256':admin.sha(out/'TEST_RESULT.json'),'additional_artifact_sha256':optional,'utc_epoch':time.time(),'scientific_scores_printed':False}
    admin.write(out/'COMPLETE_TEST_RECEIPT.json',receipt,exclusive=True)
    print(json.dumps(dict(status=receipt['status'],job_id=job_id,scores_printed=False)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--job-id');p.add_argument('--checkpoint');p.add_argument('--output');a=p.parse_args()
    if not(a.root and a.job_id and a.checkpoint and a.output):print('PENDING_COMPLETE_ALL474_TRAINING_PHYSICAL_CLOSURE_AND_ROOT_AUTHORITY_NO_ACTION');return
    execute(a.root,a.job_id,a.checkpoint,a.output)
if __name__=='__main__':main()
