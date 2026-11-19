"""Finite serial administrative profiler. Scientific build/step comes from a pinned adapter."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import traceback
import test_common as c

MODELS=["OURS","DLinear","SparseTSF","ModernTCN","MixLinear","PhaseFormer","RootPurgeLinear","ITSMixer"]


def validate(value,root):
    c.require(value["scope"]=="THREE_SEED_EFFICIENCY16_20261012" and value["hostname"]==socket.gethostname() and value["remote_root"]==str(root),"fixed canonical profiling host/root")
    c.require(value["input_seed"]==7442026 and value["batch_size"]==64 and value["length"]==96 and value["horizon"]==96 and value["channels"]==137 and value["dataset"]=="Solar","same original common profiling case")
    c.require(value["warmup_steps"]==10 and value["timed_steps"]==30 and value["wall_scope"]=="THIRTY_CONTIGUOUS_CUDA_EVENT_STEPS_FINAL_SYNC_ELAPSED_DIV30","exact original timing reduction")
    cases=value["cases"];c.require(len(cases)==16 and {(r["model"],r["seed"]) for r in cases}=={(m,s) for m in MODELS for s in [2027,2028]},"fixed sixteen new cases")
    c.require(len({r["case_id"] for r in cases})==16,"unique profile case IDs")
    for row in cases:
        c.safe_id(row["case_id"])
        c.require(row["precision"]==("BF16_WITH_TF32" if row["model"]=="OURS" else "FP32_WITH_TF32"),"original per-model precision")
        c.require(row["checkpoint_loaded"] is False and row["clone_weights_saved"] is False,"disposable synthetic profile only")
        c.require(row["parameters"]>0 and row["adapter_path"] in row["pins"],"exact original constructor and pinned adapter")
    return cases


def validate_authority(value,root):
    cases=validate(value,root)
    authority=value["authority"]
    c.require(c.sha(authority["training_closure_path"])==authority["training_closure_sha256"],"all474 completed training archive identity")
    closed=c.read(authority["training_closure_path"])
    c.require(closed["status"]=="COMPLETE_ALL474_TRAINING_AND_BEST_LOCAL_ARCHIVES" and closed["completed_training_jobs"]==474 and closed["best_archives_sha_verified"] is True,"all training closed and idle before profiles")
    c.require(c.sha(authority["original_profile_review_path"])==authority["original_profile_review_sha256"],"profile scientific protocol review")
    review=c.read(authority["original_profile_review_path"])
    c.require(review["original_batch_precision_and_measurement_scope_verified"] is True and review["synthetic_input_call_order_and_generator_verified"] is True,"same original timing/inputs/source call order")
    for name,digest in value["runtime_source_sha256"].items():c.require(c.sha(c.owned(root,name))==digest,"exact profiling administrative source")
    return cases


def worker(value,root,case):
    destination=c.owned(root,"cases/"+case["case_id"]);destination.mkdir(exist_ok=False,parents=True)
    c.write(destination/"CLAIM.json",dict(case_id=case["case_id"],pid=os.getpid(),ticks=c.ticks(os.getpid()),utc_epoch=time.time()),exclusive=True)
    outcome=dict(case_id=case["case_id"],model=case["model"],seed=case["seed"],input_seed=7442026,batch_size=64,length=96,horizon=96,channels=137,warmup_steps=10,timed_steps=30,original_batch_precision_and_measurement_scope_verified=True,checkpoint_loaded=False,clone_weights_saved=False,scientific_scores_decoded=False)
    try:
        c.verify_pins(case);c.require(not c.gpu()["pids"],"canonical GPU idle before disposable profile constructor")
        c.require(c.read(case["config_path"])==case["config"] and c.sha(case["config_path"])==case["config_sha256"],"disposable profile uses byte-pinned exact trained new-seed configuration")
        for path in reversed(case.get("python_paths",[])):
            if path not in sys.path:sys.path.insert(0,path)
        spec=importlib.util.spec_from_file_location("pinned_profile_adapter",case["adapter_path"]);adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        built=adapter.build_case(case);torch=built["torch"];model=built["model"];step=built["step"]
        c.require(torch.cuda.get_device_name()==value["gpu_name"],"same original canonical GPU model")
        signature=built["input_signature"]
        c.require(signature["input_seed"]==7442026 and signature["shape"]==[64,96,137] and signature["origin_start"]==96 and signature["origin_end_exclusive"]==160,"same original independently seeded synthetic input shape and start call order")
        if value.get("same_input_signature") is not None:c.require(signature==value["same_input_signature"],"exact pinned reference synthetic input bytes")
        parameters=sum(parameter.numel() for parameter in model.parameters());c.require(parameters==case["parameters"],"same original architecture parameter count")
        c.require(torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32,"original TF32 enabled")
        for _ in range(10):loss=step()
        torch.cuda.synchronize();c.require(torch.isfinite(loss).item(),"finite original synthetic warmup objective")
        torch.cuda.reset_peak_memory_stats()
        events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(30)]
        started=time.perf_counter()
        for start,stop in events:
            start.record();loss=step();stop.record()
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        gpu_ms=[start.elapsed_time(stop) for start,stop in events]
        allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        c.require(torch.isfinite(loss).item() and all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),"finite disposable original model after profile")
        c.require(elapsed>0 and all(math.isfinite(x) and x>0 for x in gpu_ms),"finite observed timings")
        c.verify_pins(case)
        external=c.gpu()["pids"]-c.descendants([os.getpid()]);c.require(not external,"exclusive profiling GPU throughout case")
        outcome.update(status="COMPLETE_SEQUENTIAL_SYNTHETIC_TRAIN_STEP_PROFILE",parameters=parameters,precision=case["precision"],input_signature=built["input_signature"],train_wall_ms_per_step=elapsed*1000/30,train_cuda_ms_mean=statistics.mean(gpu_ms),train_cuda_ms_median=statistics.median(gpu_ms),train_cuda_ms_each_step=gpu_ms,train_peak_allocated_gpu_bytes=allocated,train_peak_reserved_gpu_bytes=reserved,loss_scope=built["loss_scope"],optimizer_initialization=built["optimizer_initialization"],gpu_name=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,source_pins=case["pins"],idle_gpu_before_worker=True)
    except Exception as error:
        outcome.update(status="FAILED_PROFILE_NO_AUTOMATIC_RETRY",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    c.write(destination/"PROFILE.json",outcome,exclusive=True)
    return 0 if outcome["status"]=="COMPLETE_SEQUENTIAL_SYNTHETIC_TRAIN_STEP_PROFILE" else 1


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path);p.add_argument("--plan-sha256");p.add_argument("--start",action="store_true");p.add_argument("--case-id");a=p.parse_args()
    if not(a.root and a.plan_sha256 and (a.start or a.case_id)):print("PENDING_EXPLICIT_COMPLETE474_IDLE_SERIAL_PROFILE16_NO_ACTION");return 0
    root=a.root.resolve();path=root/"EFFICIENCY_PLAN.json";c.require(c.sha(path)==a.plan_sha256,"reviewed finite profiling plan SHA")
    value=c.read(path);cases=validate_authority(value,root)
    if a.case_id:
        rows=[r for r in cases if r["case_id"]==a.case_id];c.require(len(rows)==1,"one frozen profile case")
        c.require(c.read(root/"EFFICIENCY_START_CLAIM.json")["plan_sha256"]==a.plan_sha256,"ROOT one-shot serial profile start admitted")
        return worker(value,root,rows[0])
    import fcntl
    lock=(root/"EFFICIENCY_ACTIVE.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c.require(not c.gpu()["pids"],"canonical GPU idle before serial profile scope")
    c.write(root/"EFFICIENCY_START_CLAIM.json",dict(plan_sha256=a.plan_sha256,pid=os.getpid(),ticks=c.ticks(os.getpid()),utc_epoch=time.time()),exclusive=True)
    results=[]
    input_signatures=[]
    for case in cases:
        command=[value["python_executable"],"-u",str(root/"efficiency16.py"),"--root",str(root),"--plan-sha256",a.plan_sha256,"--case-id",case["case_id"]]
        log=c.owned(root,"logs/"+case["case_id"]+".log");log.parent.mkdir(exist_ok=True)
        with log.open("xb") as stream:code=subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,shell=False).returncode
        profile=c.owned(root,"cases/"+case["case_id"]+"/PROFILE.json")
        if code or not profile.is_file():
            c.write(root/"EFFICIENCY_BOUNDARY.json",dict(status="FAILED_PROFILE16_VISIBLE_NO_AUTO_RETRY",failed_case=case["case_id"],return_code=code,completed=results,plan_sha256=a.plan_sha256),exclusive=True);return 1
        results.append(dict(case_id=case["case_id"],path=str(profile),sha256=c.sha(profile),bytes=profile.stat().st_size))
        input_signatures.append(c.read(profile)["input_signature"])
        c.require(input_signatures[-1]==input_signatures[0],"every completed model/seed sees bit-identical synthetic inputs")
        c.write(root/"EFFICIENCY_STATUS.json",dict(status="RUNNING_SERIAL_PROFILE16",completed=len(results),expected=16,plan_sha256=a.plan_sha256))
    c.require(not c.gpu()["pids"],"all disposable profile processes exited with GPU idle")
    c.write(root/"EFFICIENCY_BOUNDARY.json",dict(status="COMPLETE16_ORIGINAL_SCOPE_SEQUENTIAL_PROFILES",expected=16,completed=16,new_seeds=[2027,2028],members=results,same_input_signature=input_signatures[0],all16_inputs_bit_identical=True,plan_sha256=a.plan_sha256),exclusive=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
