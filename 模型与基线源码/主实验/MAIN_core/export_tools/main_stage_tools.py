"""H-aware complete-cohort freeze and immutable best-weight test inference.

No model/recipe selection occurs here. Freeze binds all240 predeclared runs;
testing runs in independent processes and emits score-free completion receipts.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write_new(path,value):
    path=Path(path);payload=json.dumps(value,indent=2,ensure_ascii=False).encode()
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('xb') as f:f.write(payload)
    assert not path.exists(), 'immutable output collision'
    temp.rename(path)


def verify_source_inventory(root,master):
    stages=sorted({v['stage'] for v in master['runs'].values() if 'stage' in v})
    result={}
    for stage in stages:
        p=root/(stage+'_FROZEN_SUBMISSION_PLAN.json')
        expected=next(v['stage_plan_sha256'] for v in master['runs'].values() if v.get('stage')==stage)
        assert sha(p)==expected
        plan=json.loads(p.read_text())
        for name,meta in plan['files'].items():
            if name.startswith('configs/'):
                source=root/'done'/Path(name).name
            elif name=='RECIPES_SOURCE.json':
                source=root/('incoming_'+stage)/name
            else:
                source=root/name
            assert source.stat().st_size==meta['bytes'] and sha(source)==meta['sha256'],name
        result[stage]={'stage_plan_sha256':expected,'files_verified':len(plan['files'])}
    dependency=root/'DEPENDENCY_RESOLUTION.json'
    if dependency.exists():result['dependency_resolution_sha256']=sha(dependency)
    installed=root/'dependencies'
    if installed.exists():
        result['installed_dependency_source_sha256']={
            p.relative_to(installed).as_posix():sha(p) for p in sorted(installed.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    return result


def freeze(root):
    root=Path(root).resolve()
    assert str(root)=='/root/autodl-tmp/paper5/main20261009'
    master_path=root/'COHORT_PRETRAIN_PLAN.json';master=json.loads(master_path.read_text())
    expected={(model,ds,h) for model in master['models'] for ds in master['datasets'] for h in master['horizons']}
    actual={(v['config']['display_model'],v['config']['dataset'],v['config']['horizon']) for v in master['runs'].values()}
    assert len(master['runs'])==len(expected)==240 and actual==expected
    boundary=root/'TRAINING_BOUNDARY.json'
    done={p.stem for p in (root/'done').glob('*.json')}
    missing=sorted(set(master['runs'])-done)
    if missing or not boundary.exists():
        print(json.dumps({'status':'PENDING','done':len(done),'expected':240,'missing':missing,'test_scores_read':False}))
        return
    b=json.loads(boundary.read_text())
    assert b['status']=='COMPLETE' and b['failed']==0 and b['done']==240
    assert done==set(master['runs']) and len(done)==240
    assert not list((root/'queue').glob('*.json')) and not list((root/'running').glob('*.json'))
    source_inventory=verify_source_inventory(root,master)
    import numpy as np
    data={}
    for ds in master['datasets']:
        p=Path('/root/autodl-tmp/paper5/data')/(ds+'.npz')
        actual=sha(p)
        with np.load(p,allow_pickle=False) as z:
            bounds=[int(v) for v in z['split_counts']]
        assert len(bounds)==3 and bounds[0]<bounds[1]<bounds[2]
        data[ds]={'sha256':actual,'bounds':bounds}
    runs={}
    for key,planned in sorted(master['runs'].items()):
        output=root/'results'/key
        assert output.resolve().parent==(root/'results').resolve()
        config_file=output/'config.json';cfg=json.loads(config_file.read_text())
        submitted=root/'done'/(key+'.json')
        assert sha(submitted)==planned['config_sha256']
        assert json.loads(submitted.read_text())==cfg==planned['config']
        assert cfg['seed']==2026 and cfg['length']==96 and cfg['horizon'] in master['horizons']
        assert cfg['evaluate_test'] is False and cfg['initialization']=='from_scratch_no_old_checkpoint'
        metrics_file=output/'metrics.json';m=json.loads(metrics_file.read_text())
        assert m['config']==cfg and 'test' not in m
        assert m['data_sha256']==data[cfg['dataset']]['sha256']==cfg['expected_data_sha256']
        assert m['bounds']==data[cfg['dataset']]['bounds']
        assert m['code_sha256']==planned['source_sha256']
        for name,h in m['code_sha256'].items():
            assert sha(output/'code_snapshot'/name)==h
        history=json.loads((output/'history.json').read_text())
        assert len(history)==m['epochs_run'] and history[0]['epoch']==1
        assert 1<=m['best_epoch']<=len(history)<=cfg['epochs']
        assert not any(line.startswith('RESUME') for line in (root/'logs'/(key+'.log')).read_text().splitlines())
        h=cfg['horizon'];c=m['channels'];bounds=m['bounds']
        assert c=={'ETTh2':7,'ETTm2':7,'Electricity':321,'PEMS03':358,'PEMS04':307,'PEMS08':170}[cfg['dataset']]
        training_count=bounds[0]-96-h+1;validation_count=bounds[1]-bounds[0]-h+1
        assert m['validation']['windows']==validation_count and m['validation']['elements']==validation_count*h*c
        if 'training_windows' in m:assert m['training_windows']==training_count
        if 'validation_windows' in m:assert m['validation_windows']==validation_count
        windows=bounds[2]-bounds[1]-h+1
        assert windows>0
        if cfg['display_model']=='OURS':assert m['parameters']==1113058+(h-96)*290
        entry={'run_id':key,'dataset':cfg['dataset'],'model':cfg['model'],'display_model':cfg['display_model'],
               'comparison_family':cfg.get('comparison_family','ours'),'length':96,'horizon':h,'seed':2026,
               'evaluation_seed':2026,'channels':c,'bounds':bounds,'test_windows':windows,'test_elements':windows*h*c,
               'training_windows':bounds[0]-96-h+1,'validation_windows':bounds[1]-bounds[0]-h+1,
               'parameters':m['parameters'],'epochs_run':m['epochs_run'],'best_epoch':m['best_epoch'],
               'config':cfg,'config_sha256':sha(config_file),'submitted_config_sha256':sha(submitted),
               'training_metrics_sha256':sha(metrics_file),'checkpoint_sha256':sha(output/'best.pt'),
               'history_sha256':sha(output/'history.json'),'data_sha256':m['data_sha256'],
               'source_sha256':m['code_sha256'],'runner_relative':cfg['runner_relative'],
               'test_batch_size':cfg['batch_size']*2,'amp':cfg['amp']}
        runs[key]=entry
    result={'status':'FROZEN_BEST_VALIDATION_240','utc_epoch':time.time(),'remote_root':str(root),
            'pretrain_plan_sha256':sha(master_path),'run_count':240,'datasets':master['datasets'],
            'evaluation_tool_sha256':sha(Path(__file__)),
            'evaluation_driver_sha256':sha(Path(__file__).with_name('test_driver.py')),
            'horizons':master['horizons'],'models':master['models'],'length':96,'seed':2026,
            'evaluation_seed':2026,'runs':runs,'source_inventory':source_inventory,
            'configuration_serialization':'submitted_transport_bytes_and_actual_Linux_config_bytes_separately_bound_strict_equal_dictionaries',
            'test_policy':'one_full_test_pass_per_model_dataset_horizon_best_validation_weight',
            'DUET_stochastic_evaluation':'original_Gumbel_channel_mask_retained_seed2026_before_model_construction_ordered_windows',
            'scope':master['selection_scope'],'test_scores_read':False}
    path=root/'COHORT_PRETEST_FREEZE.json'
    write_new(path,result)
    print(json.dumps({'status':'FROZEN','runs':240,'freeze_sha256':sha(path),'test_scores_read':False}))


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_one(root,key,expected_freeze):
    root=Path(root).resolve();freeze_path=root/'COHORT_PRETEST_FREEZE.json'
    freeze_hash=sha(freeze_path);assert freeze_hash==expected_freeze
    f=json.loads(freeze_path.read_text());assert f['status']=='FROZEN_BEST_VALIDATION_240' and f['run_count']==240
    assert sha(Path(__file__))==f['evaluation_tool_sha256']
    entry=f['runs'][key];output=root/'results'/key;cfg=entry['config']
    master_path=root/'COHORT_PRETRAIN_PLAN.json'
    assert sha(master_path)==f['pretrain_plan_sha256']
    master=json.loads(master_path.read_text())
    assert verify_source_inventory(root,master)==f['source_inventory']
    result_path=output/'test_result.json';receipt_path=output/'test_receipt.json'
    assert not result_path.exists() and not receipt_path.exists(), 'already tested; verify receipt rather than repeat inference'
    assert sha(output/'config.json')==entry['config_sha256']
    assert sha(output/'metrics.json')==entry['training_metrics_sha256']
    assert sha(output/'best.pt')==entry['checkpoint_sha256']
    for name,h in entry['source_sha256'].items():assert sha(output/'code_snapshot'/name)==h
    data_path=Path('/root/autodl-tmp/paper5/data')/(entry['dataset']+'.npz')
    assert sha(data_path)==entry['data_sha256']
    sys.path.insert(0,str(root/'dependencies'))
    sys.path.insert(0,str(output/'code_snapshot'))
    runner_name=Path(entry['runner_relative']).name
    runner=load_module('_frozen_main_training_runtime',output/'code_snapshot'/runner_name)
    runner.seed_all(entry['evaluation_seed'])
    import torch
    if entry['display_model']=='OURS':
        from models import create_model
    else:
        from comparison_models import create_model
    model=create_model(cfg,entry['channels']).cuda()
    assert sum(p.numel() for p in model.parameters())==entry['parameters']
    model.load_state_dict(torch.load(output/'best.pt',map_location='cuda',weights_only=True),strict=True)
    windows=runner.Windows(data_path,96,entry['horizon'],'cuda',False)
    assert windows.channels==entry['channels'] and windows.bounds==entry['bounds']
    assert len(windows.indices['test'])==entry['test_windows']
    started=time.time()
    test=runner.evaluate(model,windows,'test',entry['test_batch_size'],entry['amp'])
    assert test['windows']==entry['test_windows'] and test['elements']==entry['test_elements']
    assert torch.isfinite(torch.tensor([test['mse'],test['mae']])).all()
    assert sha(output/'best.pt')==entry['checkpoint_sha256']
    assert sha(output/'metrics.json')==entry['training_metrics_sha256']
    assert sha(data_path)==entry['data_sha256']
    for name,h in entry['source_sha256'].items():assert sha(output/'code_snapshot'/name)==h
    assert verify_source_inventory(root,master)==f['source_inventory']
    header={k:entry[k] for k in ['run_id','dataset','model','display_model','horizon','seed','evaluation_seed',
                                'config_sha256','training_metrics_sha256','checkpoint_sha256','data_sha256',
                                'source_sha256','test_windows','test_elements']}
    header.update(status='COMPLETE',freeze_sha256=freeze_hash)
    body={**header,'test':test,'seconds':time.time()-started,'torch':torch.__version__,
          'gpu':torch.cuda.get_device_name(0),'best_checkpoint_unchanged':True}
    write_new(result_path,body)
    receipt={**header,'test_result_sha256':sha(result_path),'test_result_bytes':result_path.stat().st_size}
    write_new(receipt_path,receipt)
    print(json.dumps({'status':'COMPLETE','run_id':key,'test_scores_read_by_orchestrator':False}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--freeze',action='store_true');p.add_argument('--test-run');p.add_argument('--freeze-sha256')
    a=p.parse_args()
    if a.freeze:freeze(a.root)
    elif a.test_run:test_one(a.root,a.test_run,a.freeze_sha256)
    else:p.error('choose freeze or one frozen test run')


if __name__=='__main__':main()
