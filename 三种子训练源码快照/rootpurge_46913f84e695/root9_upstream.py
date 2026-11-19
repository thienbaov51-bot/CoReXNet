"""Read-only isolated native Root10 guard and flattened retained-origin ledger."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
from collect_root18_recovery10 import metadata_bytes, require, identity, safe_path, valid_sha, sha_bytes

UPSTREAM='/root/autodl-tmp/paper5/models2026_pilot_20261011/rootpurge_recovery10'
PLAN_SHA='a42eb42b2a4b67e23d5f732f99b777de1d774885670877c050e675e1cace4eca'
AUTH_SHA='578c6a4f27d5b23802615d3557dd0cb2d21d5baa1dd3f31b96b4757efa61092c'
START_SHA='e7532c37e5f0d45df92876093d5547452d403ddd4c8c5e484ae7fcb66e2fcc2c'
STATE_SHA='94bb72945a9da598be5250417588aca280f778137cae6742d4d9dbaa8b0cbb28'
BOUNDARY_SHA='f7f37e36158dd005358105eaf79153de1310c7a575d14ba4273f676b873a786b'
NATIVE_COMMON_SHA='7c75ea97a4184f99b5444e47ed42e8f02769797b6249eab04c83b9e089f361ca'
SCRIPT=r'''
import base64,hashlib,importlib,json,os,re,sys
from pathlib import Path
r=Path(sys.argv[1]);p=sys.argv[2];a=sys.argv[3];sp=sys.argv[4];st=sys.argv[5];bp=sys.argv[6]
assert os.geteuid()==0 and r.resolve()==r and all(not q.is_symlink() for q in (r,*r.parents))
assert hashlib.sha256((r/'root10_common.py').read_bytes()).hexdigest()==sys.argv[7]
sys.path.insert(0,str(r));n=importlib.import_module('root10_common')
root,plan,manifest,evidence,original,docs,freeze=n.load_frozen(p,a)
assert 'numpy' not in sys.modules and 'torch' not in sys.modules and root==r
start=n.pinned(r,'ROOT10_ACTUAL_START.json',sp);state=n.pinned(r,'ROOT10_STATE.json',st);boundary=n.pinned(r,'ROOT10_TRAINING_BOUNDARY.json',bp)
assert state['status']==boundary['status']=='INCOMPLETE_REQUIRES_ROOT_REVIEW_NO_RETRY'
assert state['done']==boundary['done'] and len(state['done'])==1 and len(state['queue'])==9 and state['queue']==boundary['queue']
assert state['admission_halt']==boundary['admission_halt']=='FRESH_ADMISSION_OR_START_REQUIRES_ROOT_REVIEW: GPU PID membership changed during ownership sampling'
for v in (state,boundary):
 assert v['active']==v['failed']==v['unverified_started_workers']==v['unverified_worker_exits']=={} and v['scores_released'] is v['test_accessed'] is v['automatic_retry'] is False
 assert v['plan_sha256']==p and v['authorization_sha256']==a and v['actual_start_sha256']==sp
assert start['plan_sha256']==p and start['authorization_sha256']==a and start['source_inventory']==plan['files'] and start['test_allowed'] is start['automatic_retry'] is False
assert set(state['done'])|set(state['queue'])==set(plan['authorized_run_ids']) and not set(state['done'])&set(state['queue'])
pre=n.pinned(r,start['prestart_name'],start['prestart_sha256']);claim=n.pinned(r,'ROOT10_START_CLAIM.json',start['claim_sha256']);launch=n.pinned(r,'ROOT10_LAUNCH_RECEIPT.json',start['launch_sha256'])
birth=n.metadata(r/'ROOT10_POPEN_RETURNED.json')
n.validate_prestart(pre,plan,p,a,freeze,start['utc_epoch'],documents=docs)
assert pre['candidate_run_id'] is None and pre['owned']==pre['active_configs']=={} and pre['controller_identity'] is None
for v in (claim,launch):
 assert v['plan_sha256']==p and v['authorization_sha256']==a and v['prestart_name']==start['prestart_name'] and v['prestart_sha256']==start['prestart_sha256'] and v['automatic_retry'] is v['test_allowed'] is False
assert claim['freeze_sha256']==start['freeze_sha256']==freeze and launch['claim_sha256']==start['claim_sha256']
assert claim['status']=='ROOT_EXPLICIT_ONE_ROOT10_CONTROLLER_START_CLAIM' and launch['status']=='ACTUAL_ROOT10_ONE_CONTROLLER_LAUNCHED' and start['status']=='ACTUAL_ROOT10_ONE_CONTROLLER_STARTED_NO_TEST'
assert {k:launch[k] for k in ('pid','start_ticks','argv')}=={k:start[k] for k in ('pid','start_ticks','argv')}
assert birth['status']=='ACTUAL_ROOT10_CONTROLLER_POPEN_RETURNED_UNVERIFIED' and birth['pid']==start['pid'] and birth['expected_argv']==start['argv']
assert birth['plan_sha256']==p and birth['authorization_sha256']==a and birth['prestart_name']==start['prestart_name'] and birth['prestart_sha256']==start['prestart_sha256'] and birth['claim_sha256']==start['claim_sha256']
assert birth['ownership_verified'] is birth['automatic_retry'] is birth['test_allowed'] is False
assert start['argv']==['/root/miniconda3/bin/python','-u',str(r/'root10_controller.py'),'--execute-reviewed-controller','--plan-sha256',p,'--authorization-sha256',a,'--prestart-sha256',start['prestart_sha256'],'--prestart-name',start['prestart_name']]
assert pre['utc_epoch']<=claim['utc_epoch']<=launch['utc_epoch']<=start['utc_epoch']<=pre['utc_epoch']+120
assert claim['utc_epoch']<=birth['utc_epoch']<=launch['utc_epoch']
controller=importlib.import_module('root10_controller');retained={};workers={};caps={}
for path in r.glob('ROOT_ROOT10_RUN_CAPABILITY_*.json'):
 cap=n.metadata(path);assert cap['run_id'] in state['done'] and cap['run_id'] not in caps;caps[cap['run_id']]=(path.name,cap,n.sha(path))
 assert cap['status']=='ROOT_EXPLICIT_ONE_RUN_ROOT10_CAPABILITY_AFTER_FRESH_INDEPENDENT_AUDIT' and cap['expires_after_seconds']==120 and cap['automatic_retry'] is cap['test_allowed'] is False
assert set(caps)==set(state['done'])
for rid,done in state['done'].items():
 assert controller.complete(r,rid,plan,p,a)==done
 cfg=n.metadata(r/'results'/rid/'config.json');assert cfg==plan['runs'][rid]
 files={name:n.identity(n.safe_path(r,'results/'+rid+'/'+name)) for name in (*n.CORE,'COMPLETE_RECEIPT.json')}
 claim=n.metadata(r/('ROOT10_RUN_CLAIM_'+rid+'.json'));name,cap,pin=caps[rid];pre=n.pinned(r,cap['prestart_name'],cap['prestart_sha256'])
 assert claim['capability_name']==name and claim['capability_sha256']==pin and claim['controller_actual_start_sha256']==cap['controller_actual_start_sha256']==sp
 assert cap['freeze_sha256']==freeze and cap['run_config_sha256']==n.config_sha(cfg)
 assert cap['owned']==pre['owned'] and cap['active_configs']==pre['active_configs'] and cap['controller_identity']=={k:start[k] for k in ('pid','start_ticks','argv')}
 c=pre['controller_identity'];assert c['uid']==0 and c['actual_start_sha256']==sp and {k:c[k] for k in ('pid','start_ticks','argv')}==cap['controller_identity']
 n.validate_prestart(pre,plan,p,a,freeze,claim['utc_epoch'],rid,docs)
 worker=claim['worker_identity'];expected=['/root/miniconda3/bin/python','-u',str(r/'root18_trainval_runtime_pending.py'),'--execute-reviewed-run','--run-id',rid,'--plan-sha256',p,'--authorization-sha256',a,'--capability-sha256',pin,'--capability-name',name]
 assert worker['uid']==0 and worker['pid']>0 and worker['start_ticks']>0 and worker['parent_pid']==start['pid'] and worker['argv']==expected
 assert pre['utc_epoch']<=cap['utc_epoch']<=claim['utc_epoch']<=pre['utc_epoch']+120
 workers[rid]=worker
 retained[rid]={'source_root':str(r),'plan_sha256':p,'config':cfg,'files':files,'completion_receipt':{k:v for k,v in done.items() if k!='complete_receipt_sha256'},'numerical_values_decoded':False}
pending=set(state['queue'])
assert {q.name for q in (r/'results').iterdir()}==set(state['done']) and {q.name for q in (r/'logs').iterdir()}=={rid+'.log' for rid in state['done']}
assert {q.name[len('ROOT10_RUN_CLAIM_'):-5] for q in r.glob('ROOT10_RUN_CLAIM_*.json')}==set(state['done'])
for rid in pending:
 assert not (r/'results'/rid).exists() and not (r/'logs'/(rid+'.log')).exists() and not (r/('ROOT10_RUN_CLAIM_'+rid+'.json')).exists()
inventory={};headers={}
for q in r.iterdir():
 assert not q.is_symlink()
 if q.is_file():
  inventory[q.name]=n.identity(q)
  if q.suffix=='.json':n.metadata(q);headers[q.name]=base64.b64encode(q.read_bytes()).decode()
for rid in state['done']:
 for q in (r/'results'/rid).iterdir():
  assert q.is_file() and not q.is_symlink();name='results/'+rid+'/'+q.name;inventory[name]=n.identity(q)
  if q.name in ('config.json','COMPLETE_RECEIPT.json','PROGRESS.json'):n.metadata(q);headers[name]=base64.b64encode(q.read_bytes()).decode()
 inventory['logs/'+rid+'.log']=n.identity(r/'logs'/(rid+'.log'))
for name,pin in plan['files'].items():assert n.sha(n.safe_path(r,name))==pin
print(json.dumps({'plan':plan,'original_plan':original,'old_evidence':evidence,'start':start,'state':state,'boundary':boundary,'upstream_complete1':retained,'upstream_workers':workers,'inventory':inventory,'headers_base64':headers,'framework_imported':False,'numerical_values_decoded':False}))
'''

def read_upstream(runner=None):
    runner=subprocess.run if runner is None else runner
    result=runner([sys.executable,'-I','-c',SCRIPT,UPSTREAM,PLAN_SHA,AUTH_SHA,START_SHA,STATE_SHA,BOUNDARY_SHA,NATIVE_COMMON_SHA],check=True,capture_output=True,text=True,timeout=180)
    value=metadata_bytes(result.stdout.encode())
    require(value['framework_imported'] is value['numerical_values_decoded'] is False,'upstream numerical/framework guard')
    return value

def validate_flat_ledger(value):
    plan=value['original_plan'];old=value['old_evidence'];new=value['upstream_complete1']
    complete={**old['original_complete8'],**new};refs=old['retained_root6_references'];pending=value['state']['queue']
    require(len(old['original_complete8'])==8 and len(new)==1 and len(complete)==9 and len(set(pending))==9
            and not set(complete)&set(pending) and set(complete)|set(pending)==set(plan['runs']),'exact old8 + completed1 + untouched9 partition')
    require(len(refs['runs'])==6 and type(refs['runs']) is list,'retained native Root6 LIST6')
    for rid,row in complete.items():
        require(row['config']==plan['runs'][rid] and row['numerical_values_decoded'] is False,'flat original unchanged config')
    ledger={rid:{'source_root':row['source_root'],'origin_run_id':rid,'origin_plan_sha256':row['plan_sha256'],
        'checkpoint_path':row['source_root']+'/results/'+rid+'/best.pt','config_path':row['source_root']+'/results/'+rid+'/config.json',
        'files':row['files'],'config':row['config']} for rid,row in complete.items()}
    for row in refs['runs']:
        rid=row['run_id']+'_L96_H96_seed2026'
        ledger[rid]={'source_root':row['source_root'],'origin_run_id':row['run_id'],'origin_plan_sha256':row['plan_sha256'],
            'checkpoint_path':row['best_path'],'config_path':row['source_root']+'/results/'+row['run_id']+'/config.json','files':row['physical']['files']}
    require(len(ledger)==15,'flat retained 15 Root checkpoint origins')
    return complete,refs,sorted(pending),ledger

def main():
    print(json.dumps({'status':'PENDING_EXPLICIT_ROOT9_READONLY_UPSTREAM_EVIDENCE_REQUIRED','actual_upstream_read':False,'training_started':False,'new_scores_accessed':False}))
if __name__=='__main__':main()
