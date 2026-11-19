"""Fresh ADD3DATA comparison training; frozen recipe, shared old window semantics."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import core_utils as cu


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--authorization-sha256', required=True)
    p.add_argument('--config', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    plan = cu.load_plan(root, a.plan_sha256)
    cu.authorize(root, a.plan_sha256, a.authorization_sha256)
    cu.verify_tool(plan, __file__)
    submitted = Path(a.config).resolve()
    if submitted.parent not in [(root / 'running').resolve(), (root / 'queue').resolve()]:
        raise ValueError('training config must belong to this cohort registry')
    config = cu.read_json(submitted)
    rid = config['run_id']
    entry = plan['runs'][rid]
    if cu.sha(submitted) != entry['config_sha256'] or config != entry['config']:
        raise ValueError('submitted config identity mismatch')
    c = cu.validate_config(config, rid, plan)
    output = cu.inside(root, 'results/' + rid)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.json').exists():
        if not cu.train_complete(root, rid, entry):
            raise ValueError('incomplete completed-run collision')
        print(json.dumps({'status': 'ALREADY_COMPLETE', 'run_id': rid}), flush=True)
        return
    for name, h in entry['source_sha256'].items():
        if cu.sha(Path(__file__).parent / name) != h:
            raise ValueError('run runtime source identity mismatch: ' + name)
    snapshot = output / 'code_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in entry['source_sha256']:
        destination = snapshot / name
        source = Path(__file__).parent / name
        if destination.exists() and cu.sha(destination) != cu.sha(source):
            raise ValueError('resume source collision')
        if not destination.exists():
            shutil.copyfile(source, destination)
    cu.write_json(output / 'config.json', config)
    data_path = Path(plan['data_root']) / (c['dataset'] + '.npz')
    if cu.sha(data_path) != c['expected_data_sha256']:
        raise ValueError('prepared data bytes differ from the frozen source')
    torch = cu.initialize_tensor_runtime(root / 'dependencies')
    if not torch.cuda.is_available():
        raise RuntimeError('authorized research training requires the production CUDA runtime')
    cu.seed_all(2026)
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
    shuffle.manual_seed(2026 + 12345)
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
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=c.get('amp', False)):
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
            validation = cu.evaluate(model, windows, 'val', c['test_batch_size'], c.get('amp', False))
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
        cu.verify_inventory(root, plan['source_inventory'])
        if cu.sha(data_path) != c['expected_data_sha256']:
            raise ValueError('data changed during training')
        metrics = {'config': config, 'dataset': c['dataset'], 'model': c['model'],
                   'seed': 2026, 'best_epoch': best_epoch, 'best_val_mse': best_val,
                   'epochs_run': len(history), 'parameters': sum(p.numel() for p in model.parameters()),
                   'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                   'seconds': time.time() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                   'bounds': windows.bounds, 'channels': windows.channels, 'torch': torch.__version__,
                   'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                   'code_sha256': entry['source_sha256'], 'data_sha256': c['expected_data_sha256'],
                   'validation': cu.evaluate(model, windows, 'val', c['test_batch_size'], c.get('amp', False)),
                   'training_windows': len(windows.indices['train']), 'validation_windows': len(windows.indices['val']),
                   'initialization': 'from_scratch_in_new_campaign', 'resume_same_recipe': resumed,
                   'criterion': c.get('loss', 'MSE'), 'pretrain_plan_sha256': a.plan_sha256,
                   'checkpoint_sha256': cu.sha(output / 'best.pt'), 'test_values_accessed': False}
        cu.write_json(output / 'metrics.json', metrics, exclusive=True)
        # Only disposable optimizer/resume state in this new run is removed;
        # frozen best/config/history remain intact. Never touch the old cohort.
        if not cu.train_complete(root, rid, entry) or cu.sha(output / 'best.pt') != metrics['checkpoint_sha256']:
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


if __name__ == '__main__':
    main()
