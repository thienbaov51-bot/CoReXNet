"""Two explicit scheduler phases under one pinned120 validation-only plan."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import batch_common as bc
import runtime as r
from secure_paths import safe_path,safe_tree,move_registry

def owned_alive(pid,root,rid):
    path=Path('/proc')/str(pid)/'cmdline'
    if not path.exists():return False
    argv=path.read_bytes().decode(errors='replace').split('\0')
    return '--root' in argv and argv[argv.index('--root')+1]==str(root) and '--config' in argv and Path(argv[argv.index('--config')+1]).stem==rid and any(Path(x).name=='runtime.py' for x in argv)

def gpu_state():
    q=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
    rows=q.stdout.strip().splitlines();r.require(q.returncode==0 and len(rows)==1,'one NVIDIA GPU required')
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    r.require(p.returncode==0,'GPU process ownership query failed')
    return float(rows[0].split(',')[0]),{int(x.split(',')[0]) for x in p.stdout.splitlines() if x.strip()},rows[0]

def event(root,value):
    with safe_path(root/'CONTROLLER_EVENTS.jsonl').open('a',encoding='utf-8') as s:s.write(json.dumps(dict(utc_epoch=time.time(),**value))+'\n')

def admission(active_gpu,active_disk,cfg,free_gpu,free_disk,plan):
    return (active_gpu+cfg['reserve_gpu_mib']<=plan.get('memory_budget_mib',22500)
            and free_gpu>=cfg['reserve_gpu_mib']+256
            and active_disk+cfg['reserve_disk_bytes']<=plan.get('disk_budget_bytes',4*(1<<30))
            and free_disk-active_disk>=cfg['reserve_disk_bytes']+plan.get('disk_free_floor_bytes',1536*(1<<20)))

def run(root,plan,auth,a):
    selected=bc.ids(plan,a.phase);expected=30 if a.phase=='h96' else 90
    r.require(len(selected)==expected,'fixed phase scope')
    boundary=root/('H96_TRAINING_BOUNDARY.json' if a.phase=='h96' else 'TRAINING_BOUNDARY.json')
    r.require(not boundary.exists(),'phase already complete/failed; no rerun')
    bc.prestart(root,plan,a.plan_sha256,a.authorization_sha256,a.prestart_manifest_sha256)
    if a.phase=='remaining90':bc.authorize_extension(root,plan,a.plan_sha256,a.extension_authorization_sha256)
    groups=bc.registries(root,plan)
    r.require(not(groups['running']-selected) and not groups['failed'],'no other-phase or failed work')
    if a.phase=='h96':r.require(not(groups['done']-selected),'remaining90 cannot precede whole30 boundary')
    r.require(shutil.disk_usage(root).free>=plan['required_free_before_research_bytes' if a.phase=='h96' else 'remaining90_required_free_before_research_bytes'],'fresh joint best/resume/free-floor disk bound')
    active={}
    for rid in groups['running']:
        path=root/'running'/(rid+'.json');info=r.read(root/'logs'/(rid+'.pid.json'))
        if owned_alive(info['pid'],root,rid):active[rid]={'pid':info['pid'],'process':None,'path':path}
        elif r.completed_clean(root,rid,plan,a.plan_sha256):move_registry(path,root/'done'/path.name)
        elif a.resume_interrupted and not(root/'results'/rid/'FAILED_VALIDATION_TRAINING.json').exists():
            move_registry(path,root/'queue'/path.name);event(root,{'event':'same_recipe_rng_interruption_requeued','run_id':rid})
        else:raise RuntimeError('explicit same-recipe interruption review required; no automatic retry')
    start={'pid':os.getpid(),'phase':a.phase,'plan_sha256':a.plan_sha256,'authorization_sha256':a.authorization_sha256,'prestart_manifest_sha256':a.prestart_manifest_sha256,'extension_authorization_sha256':a.extension_authorization_sha256,'workers':8,'command':sys.argv,'utc_epoch':time.time(),'validation_only':True}
    r.write(root/f'CONTROLLER_START_{a.phase}_{time.time_ns()}.json',start,exclusive=True)
    r.write(root/f'CONTROLLER_{a.phase.upper()}_START_RECEIPT.json',start)
    while True:
        for rid,item in list(active.items()):
            rc=item['process'].poll() if item['process'] is not None else (None if owned_alive(item['pid'],root,rid) else -999)
            if rc is None:continue
            ok=rc in (0,-999) and bool(r.completed_clean(root,rid,plan,a.plan_sha256))
            move_registry(item['path'],root/('done' if ok else 'failed')/item['path'].name)
            event(root,{'event':'done' if ok else 'failed','run_id':rid,'exit_code':rc});del active[rid]
        groups=bc.registries(root,plan);failed=groups['failed']
        pending=sorted((root/'queue'/(rid+'.json') for rid in groups['queue']&selected),key=lambda x:(plan['runs'][x.stem]['config'].get('priority',50),x.name))
        free_gpu,pids,gpu_csv=gpu_state();external=pids-{v['pid'] for v in active.values()};free_disk=shutil.disk_usage(root).free
        reserved_gpu=sum(plan['runs'][rid]['config']['reserve_gpu_mib'] for rid in active);reserved_disk=sum(plan['runs'][rid]['config']['reserve_disk_bytes'] for rid in active)
        if not failed and not external:
            for path in pending:
                if len(active)>=auth['workers']:break
                rid=path.stem;cfg=plan['runs'][rid]['config']
                if not admission(reserved_gpu,reserved_disk,cfg,free_gpu,free_disk,plan):continue
                current=root/'running'/path.name;move_registry(path,current)
                cmd=[sys.executable,'-u',str(root/'runtime.py'),'--root',str(root),'--plan-sha256',a.plan_sha256,'--authorization-sha256',a.authorization_sha256,'--prestart-manifest-sha256',a.prestart_manifest_sha256,'--action','train','--config',str(current)]
                if a.phase=='remaining90':cmd+=['--extension-authorization-sha256',a.extension_authorization_sha256]
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                with safe_path(root/'logs'/f'{rid}.log').open('ab') as log:proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
                r.write(root/'logs'/f'{rid}.pid.json',{'pid':proc.pid,'command':cmd,'utc_epoch':time.time()})
                active[rid]={'pid':proc.pid,'process':proc,'path':current};reserved_gpu+=cfg['reserve_gpu_mib'];reserved_disk+=cfg['reserve_disk_bytes'];free_gpu-=cfg['reserve_gpu_mib']
                event(root,{'event':'spawn','run_id':rid,'pid':proc.pid})
        groups=bc.registries(root,plan);completed=len(groups['done']&selected);finished=completed==expected and not active and not failed
        state={'status':'RUNNING','phase':'h96' if a.phase=='h96' else 'whole120','execution_phase':a.phase,'expected':30 if a.phase=='h96' else 120,'done_in_phase':completed if a.phase=='h96' else len(groups['done']),'done':len(groups['done']),'completed_new_runs_in_execution_phase':completed,'failed':sorted(failed),'active':{rid:v['pid'] for rid,v in active.items()},'queued':len(groups['queue']),'queued_in_execution_phase':len(groups['queue']&selected),'plan_sha256':a.plan_sha256,'authorization_sha256':a.authorization_sha256,'utc_epoch':time.time(),'gpu_csv':gpu_csv,'external_gpu_pids':sorted(external),'disk_free_bytes':free_disk,'validation_scores_decoded':False,'test_values_accessed':False,'automatic_full_or_test':False,'H96_retrained':False}
        r.write(root/'TRAINING_STATUS.json',state)
        if finished or (failed and not active):
            state['status']='COMPLETE_VALIDATION_TRAINING' if finished else 'FAILED';r.write(boundary,state,exclusive=True)
            r.require(not failed,'failed cohort sealed; no score-driven retry');return
        time.sleep(2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--plan-sha256');p.add_argument('--authorization-sha256');p.add_argument('--phase',choices=('h96','remaining90'));p.add_argument('--prestart-manifest-sha256');p.add_argument('--extension-authorization-sha256');p.add_argument('--resume-interrupted',action='store_true');a=p.parse_args()
    if not all((a.root,a.plan_sha256,a.phase)):print('PENDING_EXPLICIT_NEW5_PHASE_PLAN_PRESTART_NO_ACTION');return
    import fcntl
    root=safe_tree(a.root).resolve();plan=r.load_plan(root,a.plan_sha256);auth=r.authorize(root,a.plan_sha256,a.authorization_sha256,plan)
    r.require(r.sha(__file__)==r.sha(root/'controller.py') and 'controller.py' in plan['source_inventory'],'pinned executing controller')
    # Kernel lock prevents concurrent H96/remaining90 schedulers, without touching old roots.
    with safe_path(root/'CONTROLLER_ACTIVE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(root,plan,auth,a)

if __name__=='__main__':main()
