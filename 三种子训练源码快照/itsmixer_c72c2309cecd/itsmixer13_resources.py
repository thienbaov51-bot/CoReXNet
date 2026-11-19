"""Score-free, prospective itsmixer18 process ownership and resource admission.

Importing this module and its default CLI perform no resource or process reads.
Only an explicitly authorized operator/controller may call collect_telemetry.
Neither API launches, terminates, retries, cleans up, or changes a batch size.
"""
import copy
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

CHILD = '/root/autodl-tmp/paper5/models2026_pilot_20261011/itsmixer_recovery13_isolated3'
RUNTIME_NAME = 'itsmixer13_runtime.py'
MAX_WORKERS = 8
GPU_BUDGET_MIB = 22500
ACTIVE_DISK_CAP_BYTES = 4 << 30
DISK_FLOOR_BYTES = 1536 << 20
MIN_DISK_RESERVE_BYTES = 64 << 20
DATASETS = frozenset(('Electricity', 'PEMS03', 'PEMS04', 'PEMS08', 'Solar', 'AEMO5'))
MODELS = ('ITSMixer',)
HORIZONS = (192,336,720)
RUN_IDS = frozenset(('ITSMixer_Electricity_H720', 'ITSMixer_PEMS03_H192', 'ITSMixer_PEMS03_H336', 'ITSMixer_PEMS03_H720', 'ITSMixer_PEMS04_H192', 'ITSMixer_PEMS04_H336', 'ITSMixer_PEMS04_H720', 'ITSMixer_PEMS08_H192', 'ITSMixer_PEMS08_H336', 'ITSMixer_PEMS08_H720', 'ITSMixer_Solar_H192', 'ITSMixer_Solar_H336', 'ITSMixer_Solar_H720'))


class PendingAdmission(ValueError):
    """Insufficient current capacity; preserve processes and queue unchanged."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_int(value, label, allow_zero=False):
    require(type(value) is int and value >= (0 if allow_zero else 1), label + ' must be an integer')
    return value


def validate_worker_argv(argv, rid, root=CHILD):
    """Reject wrappers, another runtime, unknown flags, and incomplete pinning."""
    require(root == CHILD and rid in RUN_IDS, 'itsmixer18 process root/run identity differs')
    require(type(argv) is list and all(type(item) is str and item for item in argv), 'worker argv must be an exact nonempty string vector')
    require(len(argv) in (13, 14), 'worker argv length differs from the reviewed runtime CLI')
    require(PurePosixPath(argv[0]).is_absolute()
            and re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?', PurePosixPath(argv[0]).name), 'worker must use an absolute Python interpreter')
    offset = 2 if argv[1] == '-u' else 1
    require(argv[offset] == root + '/' + RUNTIME_NAME, 'worker runtime path differs')
    args = argv[offset + 1:]
    require(args and args[0] == '--execute-reviewed-run' and len(args) == 11, 'worker execution flag or argument count differs')
    parsed = {}
    for index in range(1, len(args), 2):
        key, value = args[index:index + 2]
        require(key not in parsed, 'worker duplicate runtime flag')
        parsed[key] = value
    require(set(parsed) == {'--run-id', '--plan-sha256', '--authorization-sha256', '--capability-sha256', '--capability-name'}
            and parsed['--run-id'] == rid, 'worker runtime flags/run ID differ')
    for key in ('--plan-sha256', '--authorization-sha256', '--capability-sha256'):
        require(re.fullmatch(r'[0-9a-f]{64}', parsed[key]) is not None, 'worker source/admission pin malformed')
    require(re.fullmatch(r'ROOT_ITSMIXER13_RUN_CAPABILITY_[0-9a-f]{32}\.json', parsed['--capability-name']) is not None,
            'worker capability filename differs')


def validate_owned(owned, root=CHILD):
    require(root == CHILD and type(owned) is dict, 'itsmixer18 owned process mapping/root differs')
    seen = set()
    for rid, identity in owned.items():
        require(rid in RUN_IDS and type(identity) is dict, 'unknown owned itsmixer18 run')
        require({'pid', 'start_ticks', 'argv'} <= set(identity)
                and set(identity) <= {'pid', 'start_ticks', 'argv', 'controller_pid'}, 'owned process identity fields differ')
        pid = positive_int(identity['pid'], 'owned PID')
        positive_int(identity['start_ticks'], 'owned process start ticks')
        require(pid not in seen, 'two itsmixer18 run identities share one PID')
        seen.add(pid)
        if 'controller_pid' in identity:
            positive_int(identity['controller_pid'], 'owned controller PID')
        validate_worker_argv(identity['argv'], rid, root)


def config_identity(config):
    require(type(config) is dict and config.get('model')=='ITSMixer' and config.get('dataset') in DATASETS
        and type(config.get('length')) is int and config['length']==96 and type(config.get('horizon')) is int and config['horizon'] in HORIZONS
        and type(config.get('seed')) is int and config['seed']==2026 and type(config.get('batch_size')) is int and config['batch_size']==256
        and type(config.get('epochs')) is int and config['epochs']=={192:15,336:15,720:20}[config['horizon']] and type(config.get('patience')) is int and config['patience']==100,
        'unchanged ITSMixer18 scientific identity/budget required')
    positive_int(config.get('reserved_gpu_mib'),'measured GPU reserve')
    positive_int(config.get('reserved_disk_bytes'),'measured disk reserve')
    require(config['reserved_disk_bytes']>=MIN_DISK_RESERVE_BYTES,'measured disk reserve floor')
    return 'ITSMixer_'+config['dataset']+'_H'+str(config['horizon'])


def validate_resource_admission(telemetry, active_configs, candidate=None):
    """Pure validation of physical telemetry and conservative active reservations.

    A dead owned process may remain active for a finish/reap race; its full
    reservation remains counted. A reused PID or an unowned GPU PID is rejected.
    candidate is an unchanged measured plan config, or None for current-state QA.
    """
    require(type(telemetry) is dict and telemetry.get('root') == CHILD, 'physical telemetry child root differs')
    require(type(telemetry.get('gpu_count')) is int and telemetry['gpu_count'] == 1, 'exactly one physical GPU required')
    owned = telemetry.get('owned')
    validate_owned(owned)
    require(type(active_configs) is dict and set(active_configs) == set(owned), 'active configs and owned process identities differ')
    for rid, config in active_configs.items():
        require(config_identity(config) == rid, 'active config/run identity differs')
    used = positive_int(telemetry.get('gpu_used_mib'), 'physical GPU used MiB', allow_zero=True)
    free = positive_int(telemetry.get('disk_free_bytes'), 'physical free disk bytes', allow_zero=True)
    gpu_pids = telemetry.get('gpu_compute_pids')
    require(type(gpu_pids) is list and all(type(pid) is int and pid > 0 for pid in gpu_pids)
            and len(set(gpu_pids)) == len(gpu_pids), 'GPU compute PID telemetry malformed')
    identities = {identity['pid']: identity for identity in owned.values()}
    require(set(gpu_pids) <= set(identities), 'unowned GPU compute PID blocks itsmixer18 admission')
    processes = telemetry.get('processes')
    require(type(processes) is dict and set(processes) == {str(pid) for pid in set(identities) | set(gpu_pids)}, 'process telemetry membership differs')
    for pid, expected in identities.items():
        process = processes[str(pid)]
        require(type(process) is dict and type(process.get('alive')) is bool, 'process liveness telemetry absent')
        if process['alive'] is False:
            require(pid not in gpu_pids, 'GPU PID cannot be absent or zombie')
            if process.get('observed_zombie') is True:
                require(set(process) == {'alive', 'observed_zombie', 'uid', 'start_ticks', 'parent_pid'}
                        and process['uid'] == 0 and process['start_ticks'] == expected['start_ticks'],
                        'zombie PID reuse/UID identity differs')
                if 'controller_pid' in expected:require(process['parent_pid']==expected['controller_pid'],'zombie owned parent differs')
            else:
                require(set(process) == {'alive'}, 'unknown dead process evidence')
            continue
        require(set(process) == {'alive', 'uid', 'start_ticks', 'argv', 'parent_pid'}, 'physical process identity fields differ')
        require(type(process['uid']) is int and process['uid'] == 0, 'itsmixer18 worker must be ROOT UID 0')
        require(type(process['start_ticks']) is int and process['start_ticks'] == expected['start_ticks'], 'owned PID was reused or its start tick differs')
        require(process['argv'] == expected['argv'], 'physical owned worker argv differs')
        validate_worker_argv(process['argv'], next(rid for rid, value in owned.items() if value['pid'] == pid))
        positive_int(process['parent_pid'], 'physical worker parent PID')
        if 'controller_pid' in expected:
            require(process['parent_pid'] == expected['controller_pid'], 'worker is not a child of its owned controller')
    if candidate is not None:
        candidate_id = config_identity(candidate)
        require(candidate_id not in active_configs, 'candidate itsmixer18 run is already active')
    candidate_gpu = 0 if candidate is None else candidate['reserved_gpu_mib']
    candidate_disk = 0 if candidate is None else candidate['reserved_disk_bytes']
    count = len(active_configs) + int(candidate is not None)
    gpu_reserved = sum(config['reserved_gpu_mib'] for config in active_configs.values()) + candidate_gpu
    disk_reserved = sum(config['reserved_disk_bytes'] for config in active_configs.values()) + candidate_disk
    if count > MAX_WORKERS:
        raise PendingAdmission('itsmixer18 maximum eight concurrent workers reached')
    if gpu_reserved > GPU_BUDGET_MIB:
        raise PendingAdmission('itsmixer18 joint reserved GPU budget exceeds 22500 MiB')
    if used + candidate_gpu > GPU_BUDGET_MIB:
        raise PendingAdmission('physical GPU use plus candidate reserve exceeds 22500 MiB')
    if disk_reserved > ACTIVE_DISK_CAP_BYTES:
        raise PendingAdmission('itsmixer18 joint active disk reserve exceeds 4 GiB')
    if free < DISK_FLOOR_BYTES + disk_reserved:
        raise PendingAdmission('physical disk cannot retain the 1536 MiB floor plus all active/candidate reserves')
    return {'status': 'PASS_ITSMIXER13_RESOURCE_ADMISSION', 'active_workers': len(active_configs),
            'workers_after_candidate': count, 'reserved_gpu_mib': gpu_reserved,
            'physical_gpu_used_mib': used, 'physical_gpu_with_candidate_mib': used + candidate_gpu,
            'reserved_disk_bytes': disk_reserved, 'physical_disk_free_bytes': free,
            'minimum_free_disk_bytes': DISK_FLOOR_BYTES + disk_reserved,
            'automatic_retry': False, 'batch_size_changed': False}


def _stat_identity(stat, pid):
    ending = stat.rfind(')')
    require(ending > 0 and stat[:stat.find('(')].strip() == str(pid), 'malformed /proc worker stat')
    fields = stat[ending + 1:].split()
    require(len(fields) > 19, 'incomplete /proc worker stat')
    require(fields[0] in ('R','S','D','Z','T','t','X','I','P'), 'unknown owned process state')
    return int(fields[19]), int(fields[1]), fields[0]


def _process_identity(pid, gpu_pids=None):
    folder = Path('/proc') / str(pid)
    try:
        stat = (folder / 'stat').read_text(encoding='utf-8')
        status = (folder / 'status').read_text(encoding='utf-8')
        raw_argv = (folder / 'cmdline').read_bytes()
        last_stat = (folder / 'stat').read_text(encoding='utf-8')
    except FileNotFoundError:
        return {'alive': False}
    start_ticks, _, _ = _stat_identity(stat, pid)
    last_ticks, parent_pid, state = _stat_identity(last_stat, pid)
    require(start_ticks == last_ticks, 'PID reused while physical worker identity was being read')
    uid_rows = [line.split()[1:] for line in status.splitlines() if line.startswith('Uid:')]
    require(len(uid_rows) == 1 and len(uid_rows[0]) == 4, 'incomplete /proc worker UID')
    uids = [int(value) for value in uid_rows[0]]
    require(all(value >= 0 for value in uids), 'malformed signed owned process UID')
    if state == 'Z':
        require(gpu_pids is not None and pid not in gpu_pids, 'zombie cannot be declared dead without non-GPU proof')
        require(uids == [0, 0, 0, 0], 'zombie owned process ROOT UID differs')
        return {'alive': False, 'observed_zombie': True, 'uid': 0, 'start_ticks': last_ticks, 'parent_pid': parent_pid}
    require(len(raw_argv) > 0 and raw_argv.endswith(b'\0'), 'empty or malformed live worker argv')
    argv = [value.decode('utf-8') for value in raw_argv[:-1].split(b'\0')]
    # Non-ROOT real/effective/saved/filesystem UID remains visibly nonzero.
    uid = max(uids)
    return {'alive': True, 'uid': uid, 'start_ticks': last_ticks,
            'argv': argv, 'parent_pid': parent_pid}


def _nvidia_query(field, group):
    response = subprocess.run(['nvidia-smi', '--query-' + group + '=' + field,
                               '--format=csv,noheader,nounits'], check=True,
                              capture_output=True, text=True, timeout=10)
    values = []
    for line in response.stdout.splitlines():
        line = line.strip()
        require(re.fullmatch(r'[0-9]+', line) is not None, 'unavailable or malformed physical NVIDIA telemetry')
        values.append(int(line))
    return values


def validate_sampled_ownership(owned, gpu_pids, processes):
    """Pure strict ownership checks before an unstable census can be deferred."""
    validate_owned(owned);expected={v['pid']:v for v in owned.values()}
    require(set(gpu_pids)<=set(expected),'unowned GPU compute PID blocks itsmixer18 admission')
    require(set(processes)=={str(pid) for pid in expected},'sampled process membership differs')
    for pid,identity in expected.items():
        v=processes[str(pid)]
        require(type(v.get('alive')) is bool,'sampled process liveness differs')
        if not v['alive']:
            require(pid not in gpu_pids,'GPU PID cannot be absent or zombie')
            if v.get('observed_zombie'):
                require(v['uid']==0 and v['start_ticks']==identity['start_ticks'],'zombie PID reused or UID differs')
                if 'controller_pid' in identity:require(v['parent_pid']==identity['controller_pid'],'zombie owned parent differs')
            else:require(set(v)=={'alive'},'unknown dead identity')
            continue
        require(v['uid']==0 and v['start_ticks']==identity['start_ticks'] and v['argv']==identity['argv'],'owned UID/PIDtick/argv differs')
        if 'controller_pid' in identity:require(v['parent_pid']==identity['controller_pid'],'owned worker parent differs')


def classify_gpu_samples(first, last, owned, processes, raw_diagnostics=None):
    """Unknown/reused identities hard block; owned-only change returns no PASS."""
    require(type(first) is list and type(last) is list and all(type(v) is int and v>0 for v in first+last)
        and len(first)==len(set(first)) and len(last)==len(set(last)),'GPU PID samples malformed')
    validate_owned(owned);expected={v['pid'] for v in owned.values()}
    require(set(first)<=expected and set(last)<=expected,'unknown GPU PID in either sample')
    validate_sampled_ownership(owned,last,processes)
    if sorted(first)!=sorted(last):
        error=PendingAdmission('owned GPU membership changed; defer fresh admission without Popen')
        error.diagnostic={'status':'PENDING_OWNED_ONLY_UNSTABLE_GPU_CENSUS_NO_ADMISSION','first_gpu_compute_pids':sorted(first),
            'last_gpu_compute_pids':sorted(last),'sampled_owned_processes':processes,'raw_process_samples':raw_diagnostics or {},
            'unknown_gpu_pid_ignored':False,'resource_PASS_issued':False}
        raise error
    return sorted(last)


def raw_process_sample(pid):
    folder=Path('/proc')/str(pid)
    try:return {'first_stat':(folder/'stat').read_text(encoding='utf-8'),'status':(folder/'status').read_text(encoding='utf-8'),
        'argv':(folder/'cmdline').read_bytes(),'last_stat':(folder/'stat').read_text(encoding='utf-8')}
    except (FileNotFoundError,ProcessLookupError):return None


def decode_raw_process_sample(pid,sample,last_gpu_pids):
    if sample is None:return {'alive':False}
    first,_,_=_stat_identity(sample['first_stat'],pid);ticks,parent,state=_stat_identity(sample['last_stat'],pid)
    require(first==ticks,'PID reused during physical sampling')
    rows=[line.split()[1:] for line in sample['status'].splitlines() if line.startswith('Uid:')]
    require(len(rows)==1 and len(rows[0])==4,'UID sample malformed');uids=[int(v) for v in rows[0]]
    require(all(v>=0 for v in uids),'signed UID sample')
    if state=='Z':
        require(pid not in last_gpu_pids and uids==[0,0,0,0],'zombie lacks ROOT/nonGPU evidence')
        return {'alive':False,'observed_zombie':True,'uid':0,'start_ticks':ticks,'parent_pid':parent}
    raw=sample['argv'];require(raw and raw.endswith(b'\0'),'empty or malformed live worker argv')
    argv=[v.decode('utf-8') for v in raw[:-1].split(b'\0')]
    return {'alive':True,'uid':max(uids),'start_ticks':ticks,'argv':argv,'parent_pid':parent}


def _owned_sample_header(pid, sample, expected):
    first,parent0,state0=_stat_identity(sample['first_stat'],pid)
    ticks,parent,state=_stat_identity(sample['last_stat'],pid)
    require(first==ticks==expected['start_ticks'],'owned PID reused during either stat sample')
    rows=[line.split()[1:] for line in sample['status'].splitlines() if line.startswith('Uid:')]
    require(len(rows)==1 and len(rows[0])==4 and [int(v) for v in rows[0]]==[0,0,0,0], 'owned ROOT UID differs')
    if 'controller_pid' in expected:
        require(parent0==parent==expected['controller_pid'],'owned parent changed during stat samples')
    return ticks,parent,state0,state


def _owned_exit_pending(pid, first, second, gpu_pids, expected, reason):
    # An already advertised owned PID may linger in NVIDIA's later sample.
    # This yields only PENDING after all pairs are checked, never dead/PASS.
    # No available ownership sample plus a present GPU PID remains unproved.
    require(pid not in gpu_pids or first is not None or second is not None,
        'GPU PID with no current ownership sample cannot be excused as dead')
    error=PendingAdmission(reason+'; defer before capability/log/Popen')
    def raw(value):
        return None if value is None else {'first_stat':value['first_stat'],'last_stat':value['last_stat'],
            'status':value['status'],'argv_hex':value['argv'].hex()}
    error.diagnostic={'status':'PENDING_CONFIRMED_OWNED_EXIT_OR_ABSENCE_NO_ADMISSION' if pid not in gpu_pids
        else 'PENDING_OWNED_EXIT_WITH_GPU_MEMBERSHIP_NOT_DECLARED_DEAD_NO_ADMISSION','pid':pid,
        'expected_owned_identity':expected,'first_raw_process_sample':raw(first),'second_raw_process_sample':raw(second),
        'last_gpu_compute_pids':sorted(gpu_pids),'unknown_PID_or_GPU_ignored':False,
        'process_declared_dead':False,'resource_PASS_issued':False,'historical_IT18_instantaneous_Z_inferred':False}
    raise error


def decode_owned_process_pair(pid, first, second, gpu_pids, expected):
    """Only explicit current ownership may convert confirmed exit into PENDING.

    Two live empty/malformed argv samples never become dead or PASS. Every
    available sample must preserve UID, PID start ticks and parent. Nonempty
    unexpected argv hard blocks even when a subsequent sample observes exit.
    """
    require(type(expected) is dict and expected['pid']==pid,'exact advertised owned PID required')
    decoded=[]
    for sample in (first,second):
        if sample is None:decoded.append(None);continue
        ticks,parent,state0,state=_owned_sample_header(pid,sample,expected)
        raw=sample['argv']
        if raw:
            require(raw.endswith(b'\0'),'malformed nonempty live owned argv')
            argv=[v.decode('utf-8') for v in raw[:-1].split(b'\0')]
            require(argv==expected['argv'],'owned sampled argv changed')
        else:argv=None
        decoded.append({'alive':state!='Z','uid':0,'start_ticks':ticks,'parent_pid':parent,'argv':argv,'state':state})
    a,b=decoded
    # A first observed zombie cannot legitimately become a live same process.
    require(not (a is not None and a['state']=='Z' and b is not None and b['state']!='Z'), 'zombie-to-live identity cannot be excused')
    if b is None or b['state']=='Z':
        _owned_exit_pending(pid,first,second,gpu_pids,expected,'second sample confirms owned process absent or zombie')
    require(a is not None,'inconsistent absence followed by live PID requires ROOT review')
    require(a['argv'] is not None and b['argv'] is not None,'empty or malformed live worker argv remains unproved after second sample')
    return {key:b[key] for key in ('alive','uid','start_ticks','argv','parent_pid')}


def read_owned_process_identity(pid, expected, gpu_pids):
    first=raw_process_sample(pid);second=raw_process_sample(pid)
    return decode_owned_process_pair(pid,first,second,gpu_pids,expected)


def decode_owned_process_pairs(identities, first, second, gpu_pids):
    """Check every owned identity; hard faults have priority over any deferral."""
    require(set(first)==set(second)==set(identities),'all owned process sample membership differs')
    require(set(gpu_pids)<=set(identities),'unknown GPU PID cannot be hidden by aggregate owned deferral')
    processes={};pending=[]
    for pid,expected in sorted(identities.items()):
        try:processes[str(pid)]=decode_owned_process_pair(pid,first[pid],second[pid],gpu_pids,expected)
        except PendingAdmission as error:pending.append(error)
    if pending:
        error=pending[0]
        error.diagnostic['all_available_owned_pairs_checked_before_deferral']=True
        error.diagnostic['owned_exit_deferrals']=[value.diagnostic for value in pending[1:]]
        raise error
    return processes


def collect_telemetry(root, owned):
    """Two consistent process samples, strict unknown-GPU checks, no retry."""
    require(str(root)==CHILD and sys.platform=='linux' and os.getuid()==0,'exact Linux ROOT itsmixer13 child')
    path=Path(root);require(path.is_dir() and path.resolve(strict=True)==Path(CHILD)
        and all(not q.is_symlink() for q in (path,*path.parents)),'exact child filesystem')
    validate_owned(owned);identities={v['pid']:v for v in owned.values()};expected=set(identities)
    first=sorted(set(_nvidia_query('pid','compute-apps')))
    require(set(first)<=expected,'unknown GPU PID in first sample')
    used=_nvidia_query('memory.used','gpu');require(len(used)==1,'one physical GPU required')
    samples={pid:raw_process_sample(pid) for pid in sorted(expected)}
    confirms={pid:raw_process_sample(pid) for pid in sorted(expected)}
    last=sorted(set(_nvidia_query('pid','compute-apps')))
    require(set(last)<=expected,'unknown GPU PID in last sample')
    processes=decode_owned_process_pairs(identities,samples,confirms,last)
    def raw(sample):return None if sample is None else {'first_stat':sample['first_stat'],'last_stat':sample['last_stat'],'status':sample['status'],'argv_hex':sample['argv'].hex()}
    diagnostics={str(pid):{'first':raw(samples[pid]),'second':raw(confirms[pid])} for pid in sorted(expected)}
    stable=classify_gpu_samples(first,last,owned,processes,diagnostics)
    disk=os.statvfs(path)
    return {'root':CHILD,'owned':copy.deepcopy(owned),'utc_epoch':time.time(),'gpu_used_mib':sum(used),'gpu_count':1,
        'gpu_compute_pids':stable,'disk_free_bytes':disk.f_bavail*disk.f_frsize,'processes':processes,
        'sampling_diagnostics':{'first_gpu_compute_pids':first,'last_gpu_compute_pids':last,'raw_process_samples':diagnostics,'owned_identity_verified':True}}

def main():
    print(json.dumps({'status': 'PENDING_EXPLICIT_ROOT_ITSMIXER13_CONTROLLER_ADMISSION_REQUIRED',
                      'physical_telemetry_read': False, 'training_started': False,
                      'test_accessed': False, 'automatic_retry': False}))


if __name__ == '__main__':
    main()
