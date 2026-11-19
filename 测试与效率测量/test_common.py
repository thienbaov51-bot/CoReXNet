"""Administrative-only primitives for the once-only426 new-seed pure tests."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),"regular pinned file required")
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def write(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    body=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8")
    if exclusive:
        with path.open("xb") as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
    else:
        temp=path.with_name(path.name+".tmp."+str(os.getpid())+"."+str(time.time_ns()))
        with temp.open("xb") as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)


def owned(root,relative):
    root=Path(root).resolve();raw=Path(relative)
    require(not raw.is_absolute() and ".." not in raw.parts,"relative owned child path")
    target=root/raw;require(target.resolve().is_relative_to(root),"child path escapes owned root")
    for part in [target,*target.parents]:
        if part==root.parent:break
        require(not part.is_symlink(),"owned child path parents must not be symbolic links")
    return target


def ticks(pid):
    try:
        text=(Path("/proc")/str(pid)/"stat").read_text()
        return int(text[text.rfind(")")+2:].split()[19])
    except (FileNotFoundError,ValueError,IndexError):return None


def alive(pid,start):return start is not None and ticks(pid)==start


def verify_pins(job):
    for path,digest in job["pins"].items():require(sha(path)==digest,"exact original source/config/full-data pin "+path)


def plan(root,digest):
    root=Path(root).resolve();path=root/"TEST_PLAN.json"
    require(sha(path)==digest,"explicit child test plan SHA")
    value=read(path)
    require(value["scope"]=="THREE_SEED_ONCE426_TEST_20261012","fixed426 pure-test scope")
    require(value["hostname"]==socket.gethostname() and value["remote_root"]==str(root),"exact test child host/root")
    require(value["global_expected_test_jobs"]==426 and value["new_seeds"]==[2027,2028],"426 new-seed tests only")
    require(1<=value["max_workers"]<=8 and value["gpu_budget_mib"]<=22500 and value["disk_floor_bytes"]>=4*(1<<30),"unchanged bounded-resource limits")
    authority=value["authority"]
    require(sha(authority["training_closure_path"])==authority["training_closure_sha256"],"whole474 closed training archive proof")
    closure=read(authority["training_closure_path"])
    require(closure["status"]=="COMPLETE_ALL474_TRAINING_AND_BEST_LOCAL_ARCHIVES" and closure["completed_training_jobs"]==474 and closure["best_archives_sha_verified"] is True,"all science closed before any test")
    closed_jobs=closure["jobs"]
    require(len(closed_jobs)==474 and len({j["job_id"] for j in closed_jobs})==474,"complete474 distinct sealed training identities")
    closed={j["job_id"]:j for j in closed_jobs}
    require(sha(authority["full_resource_review_path"])==authority["full_resource_review_sha256"],"actual full-resident pure-inference resource review")
    review=read(authority["full_resource_review_path"])
    require(review["actual_cuda_full_inference_resource_admission"] is True,"actual full-data shape/batch resource proof")
    for name,digest in value["runtime_source_sha256"].items():require(sha(owned(root,name))==digest,"test administrative source identity")
    require(value["jobs"] and len({j["job_id"] for j in value["jobs"]})==len(value["jobs"]),"unique fixed child tests")
    for job in value["jobs"]:
        safe_id(job["job_id"]);safe_id(job["training_job_id"])
        require(job["experiment"] in ("main","ablation") and job["seed"] in (2027,2028),"sensitivity remains validation only")
        require(job["cache_relative"]=="cache/"+job["job_id"]+"/best.pt" and job["output_relative"]=="results/"+job["job_id"],"exact owned per-job cache/output")
        require(job["best_sha256"]==job["local_best_archive"]["sha256"] and job["best_bytes"]==job["local_best_archive"]["bytes"],"sealed LOCAL best identity")
        require(job["local_best_archive"]["verified"] is True,"verified LOCAL checkpoint archive required")
        training=closed[job["training_job_id"]]
        require(training["seed"]==job["seed"] and training["experiment"]==job["experiment"] and training["best_sha256"]==job["best_sha256"] and training["best_bytes"]==job["best_bytes"] and training["training_identity"]==job["training_identity"],"test bound to actual completed original training and archived checkpoint")
        argv=job["argv"]
        require(len(argv)>=3 and argv[0]==value["python_executable"] and argv[1]=="-u" and Path(argv[2]).is_absolute() and argv[2] in job["pins"],"literal pinned pure-inference CLI")
        require(all(isinstance(x,str) and "\x00" not in x for x in argv),"literal argv strings without shell evaluation")
        require(job["optimizer_allowed"] is False and job["checkpoint_selection_allowed"] is False,"pure inference only")
        resource=job["resource"]
        require(resource["gpu_reserve_mib"]>0 and resource["disk_peak_bytes"]>=job["best_bytes"] and job["required_artifacts"],"actual inference resource and checkpoint cache included")
        for name in job["required_artifacts"]:owned(owned(root,job["output_relative"]),name)
        owned(root,job["cache_relative"]);owned(root,job["output_relative"])
    return value


def selected(value,job_id):
    jobs=[j for j in value["jobs"] if j["job_id"]==job_id]
    require(len(jobs)==1,"one frozen test job")
    return jobs[0]


def safe_id(value):
    require(isinstance(value,str) and value and len(value)<=200 and all(ch.isalnum() or ch in "_-" for ch in value),"literal bounded job identifier")
    return value


def gpu():
    command=["nvidia-smi","--query-gpu=memory.free,memory.used,utilization.gpu","--format=csv,noheader,nounits"]
    run=subprocess.run(command,capture_output=True,text=True)
    rows=run.stdout.strip().splitlines();require(run.returncode==0 and len(rows)==1,"one actual GPU")
    result=[float(x) for x in rows[0].split(",")]
    run=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True)
    require(run.returncode==0,"GPU ownership query")
    return dict(free_mib=result[0],used_mib=result[1],utilization_percent=result[2],pids={int(x.split(",")[0]) for x in run.stdout.splitlines() if x.strip()})


def descendants(pids):
    owned_pids=set(pids);parents={}
    for file in Path("/proc").glob("[0-9]*/stat"):
        try:
            text=file.read_text();tail=text[text.rfind(")")+2:].split();parents[int(file.parent.name)]=int(tail[1])
        except (FileNotFoundError,ValueError,IndexError):pass
    while True:
        more={pid for pid,parent in parents.items() if parent in owned_pids}
        if more<=owned_pids:return owned_pids
        owned_pids|=more
