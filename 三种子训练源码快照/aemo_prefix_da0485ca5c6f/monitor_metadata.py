"""Explicit server-side metadata progress only; no result/history decoding."""
import argparse
import json
from pathlib import Path
import time
import batch_common as bc
import controller
import runtime as r
from secure_paths import safe_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--root');p.add_argument('--plan-sha256');p.add_argument('--authorization-sha256');a=p.parse_args()
    if not a.check:print('PENDING_EXPLICIT_NEW5_READONLY_METADATA_CHECK_NO_ACTION');return
    if not all((a.root,a.plan_sha256,a.authorization_sha256)):p.error('root/plan/authorization exact pins required')
    root=safe_path(a.root).resolve();plan=r.load_plan(root,a.plan_sha256);r.authorize(root,a.plan_sha256,a.authorization_sha256,plan);groups=bc.registries(root,plan)
    state=r.read(root/'TRAINING_STATUS.json') if (root/'TRAINING_STATUS.json').exists() else None
    if state:r.require(state['plan_sha256']==a.plan_sha256 and state['validation_scores_decoded'] is False and state['test_values_accessed'] is False,'metadata-only status identity')
    running={}
    for rid in sorted(groups['running']):
        pid_info=r.read(root/'logs'/(rid+'.pid.json'));running[rid]={'pid':pid_info['pid'],'owned_runtime_alive':controller.owned_alive(pid_info['pid'],root,rid),'dataset':plan['runs'][rid]['config']['dataset'],'model':plan['runs'][rid]['config']['model'],'horizon':plan['runs'][rid]['config']['horizon']}
    free_gpu,gpu_pids,gpu_csv=controller.gpu_state()
    report={'status':'READONLY_METADATA_PROGRESS_NOT_A_RESEARCH_RESULT','plan_sha256':a.plan_sha256,'utc_epoch':time.time(),'expected_count':120,'registries':{k:len(v) for k,v in groups.items()},'H96_done':len(groups['done']&bc.ids(plan,'h96')),'remaining90_done':len(groups['done']&bc.ids(plan,'remaining90')),'running':running,'controller_state_metadata':state,'free_gpu_mib':free_gpu,'gpu_csv':gpu_csv,'external_gpu_pids':sorted(gpu_pids-{v['pid'] for v in running.values()}),'disk_free_bytes':controller.shutil.disk_usage(root).free,'validation_or_history_or_test_bodies_read':False,'scientific_success_claimed':False}
    print(json.dumps(report))

if __name__=='__main__':main()
