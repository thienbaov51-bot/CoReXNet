"""Metadata-only phase closures; scientific result bodies stay opaque here."""
import json
from pathlib import Path
import runtime as r
from secure_paths import safe_path

MODELS=r.MODELS
HORIZONS=r.HORIZONS

def ids(plan,phase):
    r.require(phase in ('h96','remaining90','whole120'),'known fixed phase')
    return {rid for rid,row in plan['runs'].items() if phase=='whole120' or (row['config']['horizon']==96)==(phase=='h96')}

def registries(root,plan):
    root=safe_path(root)
    for label in ('queue','running','done','failed'):safe_path(root/label)
    groups={label:{p.stem for p in (root/label).glob('*.json')} for label in ('queue','running','done','failed')}
    r.require(set().union(*groups.values())==set(plan['runs']) and sum(map(len,groups.values()))==120,'disjoint complete120 registries')
    for label,members in groups.items():
        for rid in members:
            path=root/label/(rid+'.json');entry=plan['runs'][rid]
            r.require(r.sha(path)==entry['config_sha256'] and r.read(path)==entry['config'],'registry config identity')
    return groups

def inventory_row(root,receipt):
    return {**{key:receipt[key] for key in r.HEADER},'origin_root':str(root),
            **{key:receipt[key] for key in ('checkpoint_bytes','validation_result_sha256','validation_result_bytes')},
            'receipt_sha256':r.sha(root/'results'/receipt['run_id']/'validation_receipt.json')}

def collect(root,plan,plan_sha,phase,allow_remaining_active=False):
    root=safe_path(root);selected=ids(plan,phase);expected=30 if phase=='h96' else 120
    r.require(phase in ('h96','whole120') and len(selected)==expected,'phase cohort size')
    groups=registries(root,plan)
    boundary_path=root/('H96_TRAINING_BOUNDARY.json' if phase=='h96' else 'TRAINING_BOUNDARY.json')
    boundary=r.read(boundary_path)
    r.require(boundary['status']=='COMPLETE_VALIDATION_TRAINING' and boundary['plan_sha256']==plan_sha and boundary['phase']==phase and boundary['done_in_phase']==expected and not boundary['active'] and not boundary['failed'],'sealed actual training boundary')
    r.require(selected<=groups['done'] and not selected.intersection(groups['queue']|groups['running']|groups['failed']),'all selected runs completed once')
    if phase=='whole120':r.require(groups['done']==selected and not(groups['queue']|groups['running']|groups['failed']),'complete120 physical registry boundary')
    elif not allow_remaining_active:r.require(groups['done']==selected and groups['queue']==ids(plan,'remaining90') and not(groups['running']|groups['failed']),'first30 release before90 continuation')
    inventory={}
    for rid in sorted(selected):
        receipt=r.complete_metadata(root,rid,plan,plan_sha)
        r.require(receipt is not None and r.completed_clean(root,rid,plan,plan_sha),'best/config/source/data/history/cleanup physically complete: '+rid)
        inventory[rid]=inventory_row(root,receipt)
    return {'status':f'ALL{expected}_PHYSICAL_METADATA_CLOSED_BEFORE_VAL_RELEASE','run_count':expected,'phase':phase,'plan_sha256':plan_sha,
            'datasets':plan['datasets'],'models':list(MODELS),'horizons':[96] if phase=='h96' else list(HORIZONS),'length':96,'seed':2026,
            'physical_inventory':inventory,'test_values_accessed':False,'H96_retrained':False}

def authorize_extension(root,plan,plan_sha,expected_auth_sha,physical_check=True):
    root=safe_path(root);path=root/'ROOT_EXTENSION90_AUTHORIZATION.json'
    r.require(expected_auth_sha and r.sha(path)==expected_auth_sha,'explicit ROOT extension90 authority pin')
    authorization=r.read(path);release_path=root/'ROOT_ALL30_VALIDATION_SCORE_RELEASE.json'
    r.require(authorization['status']=='ROOT_AUTHORIZED_REMAINING90_AFTER_COMPLETE30' and authorization['plan_sha256']==plan_sha,'ROOT phase authority/plan')
    r.require(authorization['h96_release_sha256']==r.sha(release_path) and authorization['new_run_count']==90 and authorization['H96_retraining_allowed'] is False and authorization['test_allowed'] is False,'ROOT no-retrain/no-test phase contract')
    r.require(authorization['h96_audit_sha256']==r.sha(root/'ALL30_VALIDATION_AUDIT.json'),'completed whole30 observation audit pin')
    release=r.read(release_path)
    r.require(release['status']=='ROOT_ALL30_METADATA_RELEASED_BEFORE_VAL_DECODING','ROOT complete30 release identity')
    closed=release['closure'];r.require(closed['status']=='ALL30_PHYSICAL_METADATA_CLOSED_BEFORE_VAL_RELEASE' and closed['run_count']==30 and closed['plan_sha256']==plan_sha and set(closed['physical_inventory'])==ids(plan,'h96') and closed['test_values_accessed'] is False,'original30 complete release grid')
    if physical_check:
        closure=collect(root,plan,plan_sha,'h96',allow_remaining_active=True)
        r.require(closed==closure,'original30 physical identity retained through continuation')
    # Controller checks raw30 once before90 starts, whole120 checks them again.
    # Per-new-run startup checks immutable ROOT SHA/metadata, avoiding90 full
    # rereads of the same30 checkpoint payloads while preserving release gates.
    return authorization

def prestart(root,plan,plan_sha,authorization_sha,expected_manifest_sha):
    path=Path(root)/'PRESTART_MANIFEST.json';r.require(expected_manifest_sha and r.sha(path)==expected_manifest_sha,'explicit actual prestart manifest SHA')
    receipt=r.read(path)
    r.require(receipt['status']=='PASS_NEW5_ACTUAL120_PRESTART' and receipt['plan_sha256']==plan_sha and receipt['authorization_sha256']==authorization_sha,'actual120 independent prestart review')
    r.require(receipt['actual_resource_qa_sha256']==plan['production_resource_qa_sha256'] and receipt['actual_config_count']==120 and receipt['actual_dataset_count']==5,'all real recipe dimensions admitted')
    r.require(receipt['workers']==8 and receipt['memory_budget_mib']==22500 and receipt['disk_budget_bytes']==4*(1<<30) and receipt['disk_free_floor_bytes']==1536*(1<<20),'joint resource contract')
    r.require(receipt['required_free_before_research_bytes']==plan['required_free_before_research_bytes'] and receipt['observed_free_disk_bytes']>=receipt['required_free_before_research_bytes'],'fresh all120 best/live/floor bound')
    r.require(receipt['test_arrays_deployed'] is False and receipt['scientific_scores_read'] is False and receipt['independent_review_status']=='PASS','prefix-only/no-score/review prestart')
    return receipt

def decoded_rows(closure):
    """Called only after explicit ROOT release containing this entire closure."""
    count=closure['run_count'];r.require(count in (30,120) and closure['status']==f'ALL{count}_PHYSICAL_METADATA_CLOSED_BEFORE_VAL_RELEASE','all-group physical closure prerequisite')
    rows={}
    for rid,pins in closure['physical_inventory'].items():
        path=Path(pins['origin_root'])/'results'/rid/'validation_result.json';r.require(r.sha(path)==pins['validation_result_sha256'],'opaque score body byte pin')
        value=r.read(path)
        for key in r.HEADER:r.require(value[key]==pins[key],'released body/header '+key)
        r.require(value['status']=='COMPLETE_VALIDATION_ONLY' and value['test_values_accessed'] is False and value['test_indices_constructed'] is False,'released validation-only identity')
        metric=value['validation'];r.require(metric['windows']==pins['validation_windows'] and metric['elements']==pins['validation_elements'],'all-window/all-element identity')
        r.require(all(r.math.isfinite(metric[k]) and metric[k]>=0 for k in ('mse','mae')),'finite nonnegative pooled metrics')
        channel=metric['channel_mse'];r.require(len(channel)==pins['channels'] and all(r.math.isfinite(v) and v>=0 for v in channel) and r.math.isclose(sum(channel)/pins['channels'],metric['mse'],rel_tol=1e-10,abs_tol=1e-12),'per-channel/pooled MSE identity')
        rows[pins['dataset'],pins['horizon'],pins['model']]=value
    r.require(len(rows)==count,'decoded complete unique grid')
    return rows
