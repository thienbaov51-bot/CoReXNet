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
import shutil
import core_utils as cu
def main():
    a,config,output=args_and_config();verify_local();c=cu.effective_config(config);rid=c['run_id'];root=Path('/root/autodl-tmp/paper5/add5_20261010');a.plan_sha256=sha(a.config)
    entry={'source_sha256':json.loads((Path(__file__).parent/'FORK_SOURCE_PINS.json').read_text())}
    bounds_by_dataset={'AEMO5': [61353, 70118, 87648], 'Electricity': [18412, 21044, 26304], 'PEMS03': [15724, 20966, 26208], 'PEMS04': [10195, 13593, 16992], 'PEMS08': [10713, 14284, 17856], 'Solar': [36792, 42048, 52560]}
    plan={'data_root':a.data_root,'data':{c['dataset']:{'bounds':bounds_by_dataset[c['dataset']],'channels':c['channels']}}}
    cu.write_json(output/'config.json',config);snapshot=output/'code_snapshot';snapshot.mkdir()
    for name in entry['source_sha256']:shutil.copyfile(Path(__file__).parent/name,snapshot/name)
    data_path = Path(plan['data_root']) / (c['dataset'] + '.npz')
    if cu.sha(data_path) != c['expected_data_sha256']:
        raise ValueError('prepared data bytes differ from the frozen source')
    torch = cu.initialize_tensor_runtime(root / 'dependencies')
    if not torch.cuda.is_available():
        raise RuntimeError('authorized research training requires the production CUDA runtime')
    cu.seed_all(c['seed'])
    torch.cuda.reset_peak_memory_stats()
    windows = cu.Windows(data_path, 96, c['horizon'], 'cuda', False)
    ds = plan['data'][c['dataset']]
    if windows.bounds != ds['bounds'] or windows.channels != c['channels']:
        raise ValueError('prepared dimensions/splits mismatch')
    c['dependency_root'] = str(root / 'dependencies')
    model = cu.make_model(c).cuda()
    optimizer = cu.make_optimizer(model, c)
    scheduler, per_batch_schedule = cu.make_scheduler(
        optimizer, c, math.ceil(len(windows.indices['train']) / c['batch_size']))
    shuffle = torch.Generator(device='cuda')
    shuffle.manual_seed(c['seed'] + 12345)
    best_val, best_epoch, wait, next_epoch = float('inf'), -1, 0, 0
    history, best_state = [], None
    resumed = False
    state_path = output / 'training_state.pt'
    if state_path.exists():
        state = torch.load(state_path, map_location='cuda', weights_only=False)
        if state['plan_sha256'] != a.plan_sha256 or state['config'] != config or state['source_sha256'] != entry['source_sha256']:
            raise ValueError('resume state belongs to another recipe/cohort/source')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        shuffle.set_state(state['shuffle_rng'].cpu())
        cu.random.setstate(state['python_rng'])
        cu.np.random.set_state(state['numpy_rng'])
        best_val, best_epoch, wait = state['best_val'], state['best_epoch'], state['wait']
        next_epoch, history = state['next_epoch'], state['history']
        best_state = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
        resumed = True
    started = time.time()
    try:
        for epoch in range(next_epoch, c['epochs']):
            epoch_started = time.time()
            model.train()
            ix = windows.indices['train']
            order = ix[torch.randperm(len(ix), device='cuda', generator=shuffle)]
            train_sum = torch.zeros(2, device='cuda', dtype=torch.float64)
            for chunk in order.split(c['batch_size']):
                x, y = windows.batch(chunk)
                optimizer.zero_grad(set_to_none=True)
                pred = cu.predict(model, x, chunk)
                if pred.shape != y.shape:
                    raise ValueError('training prediction shape mismatch')
                difference = pred.float() - y
                loss = difference.abs().mean() if c.get('loss', 'MSE') == 'MAE' else difference.square().mean()
                if hasattr(model, 'auxiliary_loss'):
                    if not c.get('additional_loss'):
                        raise ValueError('official auxiliary loss must be explicitly frozen in the recipe')
                    loss = loss + model.auxiliary_loss()
                # Any official architecture auxiliary term must be declared in
                # its recipe. This campaign never invokes supervised new hooks.
                loss.backward()
                if c.get('grad_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'])
                optimizer.step()
                if per_batch_schedule:
                    scheduler.step()
                train_sum[0] += loss.detach().double() * len(chunk)
                train_sum[1] += len(chunk)
            if not per_batch_schedule:
                scheduler.step()
            validation = cu.evaluate(model, windows, 'val', c['test_batch_size'], False)
            values = train_sum.cpu().numpy()
            if not cu.np.isfinite(values).all() or not math.isfinite(validation['mse']):
                raise FloatingPointError('non-finite train/validation objective')
            row = {'epoch': epoch + 1, 'train_loss': float(values[0] / values[1]),
                   'val_mse': validation['mse'], 'val_mae': validation['mae'],
                   'lr': scheduler.get_last_lr()[0], 'seconds': time.time() - epoch_started}
            history.append(row)
            cu.write_json(output / 'history.json', history)
            if validation['mse'] < best_val - 1e-10:
                best_val, best_epoch, wait = validation['mse'], epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, output / 'best.pt.tmp')
                (output / 'best.pt.tmp').replace(output / 'best.pt')
            else:
                wait += 1
            state = {'plan_sha256': a.plan_sha256, 'config': config, 'source_sha256': entry['source_sha256'],
                     'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state(), 'shuffle_rng': shuffle.get_state(),
                     'python_rng': cu.random.getstate(), 'numpy_rng': cu.np.random.get_state(),
                     'best_val': best_val, 'best_epoch': best_epoch, 'wait': wait,
                     'next_epoch': epoch + 1, 'history': history}
            torch.save(state, output / 'training_state.pt.tmp')
            (output / 'training_state.pt.tmp').replace(state_path)
            print(json.dumps({'run_id': rid, 'epoch': epoch + 1, 'seconds': row['seconds']}), flush=True)
            if wait >= c['patience']:
                break
        if best_state is None:
            raise RuntimeError('no finite validation checkpoint')
        model.load_state_dict(best_state, strict=True)
        verify_local()
        if cu.sha(data_path) != c['expected_data_sha256']:
            raise ValueError('data changed during training')
        metrics = {'config': config, 'dataset': c['dataset'], 'model': c['model'],
                   'seed': c['seed'], 'best_epoch': best_epoch, 'best_val_mse': best_val,
                   'epochs_run': len(history), 'parameters': sum(p.numel() for p in model.parameters()),
                   'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                   'seconds': time.time() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                   'bounds': windows.bounds, 'channels': windows.channels, 'torch': torch.__version__,
                   'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                   'code_sha256': entry['source_sha256'], 'data_sha256': c['expected_data_sha256'],
                   'validation': cu.evaluate(model, windows, 'val', c['test_batch_size'], False),
                   'training_windows': len(windows.indices['train']), 'validation_windows': len(windows.indices['val']),
                   'initialization': 'from_scratch_in_new_campaign', 'resume_same_recipe': resumed,
                   'criterion': c.get('loss', 'MSE'), 'pretrain_plan_sha256': a.plan_sha256,
                   'checkpoint_sha256': cu.sha(output / 'best.pt'), 'test_values_accessed': False}
        cu.write_json(output / 'metrics.json', metrics, exclusive=True)
        # Only disposable optimizer/resume state in this new run is removed;
        # frozen best/config/history remain intact. Never touch the old cohort.
        if not all((output/n).is_file() for n in ('config.json','best.pt','history.json','metrics.json')) or cu.sha(output / 'best.pt') != metrics['checkpoint_sha256']:
            raise ValueError('completed evidence must close before resume-state cleanup')
        if state_path.exists():
            cleanup = {'status': 'DISPOSABLE_RESUME_STATE_REMOVED_AFTER_COMPLETE',
                       'run_id': rid, 'plan_sha256': a.plan_sha256,
                       'removed_relative_path': 'results/' + rid + '/training_state.pt',
                       'removed_sha256': cu.sha(state_path), 'removed_bytes': state_path.stat().st_size,
                       'preserved_best_checkpoint_sha256': metrics['checkpoint_sha256'],
                       'utc_epoch': time.time()}
            cu.write_json(output / 'DISPOSABLE_RESUME_STATE_CLEANUP.json', cleanup, exclusive=True)
            state_path.unlink()
        print(json.dumps({'status': 'COMPLETE', 'run_id': rid, 'test_values_accessed': False}), flush=True)
    except Exception as exc:
        cu.write_json(output / ('failure_' + str(time.time_ns()) + '.json'),
                      {'error': repr(exc), 'run_id': rid, 'config': config,
                       'seconds': time.time() - started, 'batch_size_unchanged': c['batch_size']}, exclusive=True)
        raise


if __name__=='__main__':main()
