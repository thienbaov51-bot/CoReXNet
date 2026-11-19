import argparse,json,subprocess,time,hashlib,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,p)
def main(expected):
    assert sha(ROOT/'MINIMUM12_PLAN.json')==expected
    plan=json.loads((ROOT/'MINIMUM12_PLAN.json').read_text())
    for n,h in plan['files'].items():assert sha(ROOT/n)==h,n
    assert not (ROOT/'MINIMUM12_ACTUAL_START.json').exists(),'no duplicate controller'
    write(ROOT/'MINIMUM12_ACTUAL_START.json',{'pid':os.getpid(),'utc_epoch':time.time(),'plan_sha256':expected,'research_started':True,'test_accessed':False})
    queue=list(plan['runs']);active={};done=[];failed=[]
    (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True)
    while queue or active:
        for rid,(p,log) in list(active.items()):
            if p.poll() is not None:
                log.close();active.pop(rid)
                if p.returncode==0 and (ROOT/'results'/rid/'COMPLETE_RECEIPT.json').exists():done.append(rid)
                else:failed.append(rid)
        reserved=sum(plan['runs'][r]['reserve_gpu_mib'] for r in active)
        for rid in list(queue):
            c=plan['runs'][rid]
            if len(active)>=8 or reserved+c['reserve_gpu_mib']>22500:continue
            if shutil.disk_usage(ROOT).free < (1536<<20)+sum(plan['runs'][r]['reserve_disk_bytes'] for r in active)+c['reserve_disk_bytes']:continue
            log=(ROOT/'logs'/f'{rid}.log').open('x')
            p=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(ROOT/'trainval_runtime.py'),'--run-id',rid,'--plan-sha256',expected],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            active[rid]=(p,log);queue.remove(rid);reserved+=c['reserve_gpu_mib']
        write(ROOT/'MINIMUM12_STATE.json',{'status':'RUNNING','plan_sha256':expected,'done':done,'active':{r:p.pid for r,(p,l) in active.items()},'queue':queue,'failed':failed,'utc_epoch':time.time(),'scores_released':False})
        time.sleep(3)
    write(ROOT/'MINIMUM12_TRAINING_BOUNDARY.json',{'status':'COMPLETE_VALIDATION_TRAINING' if len(done)==12 and not failed else 'FAILED_REQUIRES_REVIEW','done':done,'failed':failed,'plan_sha256':expected,'utc_epoch':time.time(),'scores_released':False,'test_accessed':False})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan-sha256',required=True);a=p.parse_args();main(a.plan_sha256)
