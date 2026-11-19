"""Explicit local author preparation, source/metadata only. Default no IO."""
import argparse,ast,difflib,hashlib,json,re
from pathlib import Path

SIX='MODELS2026_ITSMIXER_MINIMUM6_SCIENCE_SOURCE_PENDING_20261012'
MANIFEST='ITSMIXER6_SOURCE_PACKAGE_MANIFEST_PENDING.json'
MANIFEST_SHA='e8fe24e644b72be8b90c8df70b3f113d7480b76c34e126e1036933d434bfc100'
MODEL_NAMES=('RevIN','Mlp_time','Mixer_Layer','Backbone','Model')

def sh(body):return hashlib.sha256(body).hexdigest()
def enc(x):return (json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
def keep(path,body):
    if path.exists():assert path.read_bytes()==body,'existing immutable source differs '+str(path)
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)

def prepare():
    stage=Path(__file__).resolve().parent;work=stage.parents[1];original=work/'main_experiments'/SIX
    mb=(original/MANIFEST).read_bytes();assert sh(mb)==MANIFEST_SHA
    manifest=json.loads(mb)
    for name,pin in manifest['files'].items():assert sh((original/name).read_bytes())==pin,'original science leaf pin '+name
    retained=('ITSMixer_ECL96_author_exact.py','itsmixer6_protocol.py','itsmixer6_science_kernel.py',
              'ITSMIXER_MINIMUM6_CONFIG_RECIPE_PENDING.json','ITSMIXER_MINIMUM6_SCIENCE_PROTOCOL_PENDING.json',
              'FUTURE_NATIVE_ITSMIXER6_OUTPUT_CONTRACT_PENDING.json','AUTHOR_MODEL_EXACT_AST_PROOF.json',
              'SCIENTIFIC_SOURCE_ADAPTATION_PROOF_PENDING.json')
    for name in retained:keep(stage/name,(original/name).read_bytes())
    keep(stage/'H96_SCIENCE_ORIGINAL_PACKAGE_MANIFEST.json',mb)
    keep(stage/'LICENSE.ITSMixer',(original/'author_source/LICENSE').read_bytes())
    # The historical common15 proposal is disclosed as a pre-code draft; no
    # claim that an executable H72015 experiment or frozen protocol existed.
    keep(stage/'SUPERSEDED_COMMON15_PRE_CODE_SOURCE_PROPOSAL_ONLY.json',enc({
        'status':'SUPERSEDED_PRE_CODE_SOURCE_PROPOSAL_NO_AUTHORITY_NEVER_EXECUTED',
        'initial_proposed_max_epochs_by_horizon':{'192':15,'336':15,'720':15},
        'originally_proposed_by_ROOT_before_per_horizon_log_evidence_was_checked':True,
        'corrected_before_any_remaining18_candidate_values_or_execution':True,
        'reason':'Pinned ECL720 comments demonstrate20 epochs; ROOT now adopts20 prospectively, retaining15 for192/336',
        'executable_15_epoch_h720_kernel_created':False,'actual_15_epoch_h720_run_existed':False,
        'actual_authority_existed':False,'original_H96_15_epoch_recipe_changed':False,
    }))
    import itsmixer18_protocol as p
    parent_path=work/'main_experiments/MODELS2026_PILOT_20261011/source_qa_package/MINIMUM12_PLAN.json'
    pb=parent_path.read_bytes();assert sh(pb)==p.PARENT_PLAN_SHA256;parent=json.loads(pb)
    root_policy=work/'main_experiments/MODELS2026_PILOT_20261011'/p.ROOT_REMAINING18_POLICY_NAME
    rb=root_policy.read_bytes();assert sh(rb)==p.ROOT_REMAINING18_POLICY_SHA256
    rp=json.loads(rb)
    assert rp['maximum_epochs_by_horizon']=={'192':15,'336':15,'720':20}
    assert rp['H96_checkpoint_retraining_allowed'] is False
    assert rp['actual_remaining18_allowed_before_minimum6_strict12_gate_pass'] is False
    keep(stage/p.ROOT_REMAINING18_POLICY_NAME,rb)
    keep(stage/'ROOT_H96_PRE_SCORE_POLICY_ORIGINAL.json',(original/'provenance/ROOT_ITSMIXER_PROSPECTIVE_REPLACEMENT_PRE_SCORE_PROTOCOL_ADOPTION.json').read_bytes())
    models={}
    for h,v in p.VARIANTS.items():
        source=original/f'author_source/Experiments/electricity-{h}/seq-len-512.py';body=source.read_bytes()
        assert sh(body)==v['source_sha256']==rp['corresponding_ECL_horizon_source_sha256'][str(h)]
        blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest();assert blob==v['git_blob_sha1']
        prov=original/f'provenance/Experiments__electricity-{h}__seq-len-512.py.GITHUB_BLOB.json'
        assert json.loads(prov.read_bytes())['sha']==blob
        keep(stage/v['source_file'],body);keep(stage/'provenance'/prov.name,prov.read_bytes())
        tree=ast.parse(body.decode('utf-8'));classes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
        assert tuple(n.name for n in classes)==MODEL_NAMES
        assert [ast.unparse(n.iter) for n in ast.walk(tree) if isinstance(n,ast.For)]==['range('+str(v['first_loop'])+')','range('+str(v['second_loop'])+')']
        comment_epochs=sorted(set(map(int,re.findall(r'Epoch: (\d+)',body.decode()))))
        assert comment_epochs==list(range(1,v['epochs']+1))
        models[str(h)]={**v,'author_original_relative_source':source.relative_to(work).as_posix(),
            'copy_byte_identical':True,'git_blob_sha1_verified':blob,
            'class_order':list(MODEL_NAMES),'class_ast_sha256':{n.name:sh(ast.dump(n,include_attributes=False).encode()) for n in classes},
            'historical_comment_epochs_demonstrated':comment_epochs,'historical_score_values_not_transcribed':True,
            'exact_original_training_CLI_verified':False,'framework_model_instantiated':False}
    before=(original/'itsmixer6_science_kernel.py').read_text(encoding='utf-8')
    replacements=[
      ('Prospective ITSMixer ECL96 science core','Prospective exact-horizon ITSMixer remaining18 science core'),
      ('from itsmixer6_protocol import require, type3_lr_after_epoch, replaces_checkpoint',
       'from itsmixer18_protocol import require, type3_lr_after_epoch, replaces_checkpoint, source_model_for_horizon'),
      ("    args = SimpleNamespace(seq_len=96, pred_len=96, enc_in=config['enc_in'])",
       "    require(config['model_source_file'] == source_model_for_horizon(config['horizon']), 'exact per-horizon author dispatch')\n    args = SimpleNamespace(seq_len=96, pred_len=config['horizon'], enc_in=config['enc_in'])"),
      ('def predict(model, x, channels):','def predict(model, x, channels, horizon):'),
      ("(len(x), 96, channels), 'exact full forecast shape'","(len(x), horizon, channels), 'exact full forecast shape'"),
      ("predict(model, x, config['channels'])","predict(model, x, config['channels'], config['horizon'])"),
      ('(seen_windows * 96)',"(seen_windows * config['horizon'])"),
      ("list(range(96, config['train_end'] - 95))","list(range(96, config['train_end'] - config['horizon'] + 1))"),
      ("torch.arange(config['train_end'], config['validation_end'] - 95, device='cuda')",
       "torch.arange(config['train_end'], config['validation_end'] - config['horizon'] + 1, device='cuda')"),
      ("self.yoff = torch.arange(96, device='cuda')","self.yoff = torch.arange(config['horizon'], device='cuda')"),
      ('for epoch in range(1, 16):',"for epoch in range(1, config['epochs'] + 1):"),
      ('full 15-epoch budget','full corresponding15/15/20 epoch budget'),
      ('unexpected early stop within frozen 15-epoch budget','unexpected early stop within frozen per-horizon epoch budget'),
      ('type3_lr_after_epoch(epoch)',"type3_lr_after_epoch(epoch, config['horizon'])"),
      ("config['expected_optimizer_steps_at_full_15_epoch_budget'], 'full 15-epoch optimizer budget'",
       "config['expected_optimizer_steps_at_full_epoch_budget'], 'full corresponding per-horizon optimizer budget'"),
      ("'epochs': 15, 'steps': total_steps","'epochs': config['epochs'], 'steps': total_steps"),
      ('PENDING_ITSMIXER6_SCIENCE_SOURCE_ONLY_NO_IO_NO_FRAMEWORK_NO_RUN',
       'PENDING_ITSMIXER_REMAINING18_SCIENCE_SOURCE_ONLY_NO_IO_NO_FRAMEWORK_NO_RUN'),
    ]
    after=before;proof=[]
    for old,new in replacements:
        count=after.count(old);assert count>0,'targeted original kernel anchor missing '+old
        after=after.replace(old,new);proof.append({'original':old,'future':new,'replacement_count':count})
    restored=after
    for old,new in reversed(replacements):restored=restored.replace(new,old)
    assert restored==before,'round-trip exact kernel text equivalence outside declared dispatch/geometry/budget'
    ast.parse(after)
    keep(stage/'itsmixer18_science_kernel.py',after.encode('utf-8'))
    diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='unchanged_e8fe/itsmixer6_science_kernel.py',tofile='future_remaining18/itsmixer18_science_kernel.py'))
    keep(stage/'H96_TO_REMAINING18_EXACT_SCIENCE_KERNEL.diff',diff.encode())
    runs={}
    for h in p.HORIZONS:
        for ds in p.DATASETS:
            cfg=p.make_config(ds,h,parent['data_metadata'][ds]);runs[cfg['run_id']]=cfg
    assert len(runs)==18 and not any(x.endswith('_H96') for x in runs)
    protocol={
        'status':'PENDING_PROSPECTIVE_ITSMIXER_REMAINING18_SCIENCE_SOURCE_ONLY_NO_QUALIFICATION_OR_PRODUCTION',
        'root_remaining18_pre_score_policy_name':p.ROOT_REMAINING18_POLICY_NAME,
        'root_remaining18_pre_score_policy_sha256':p.ROOT_REMAINING18_POLICY_SHA256,
        'original_h96_science_package_sha256':MANIFEST_SHA,
        'original_h96_science_snapshot_sha256':manifest['snapshot_sha256'],
        'original_parent_plan_sha256':p.PARENT_PLAN_SHA256,'author_commit':p.AUTHOR_COMMIT,
        'length':96,'horizons':list(p.HORIZONS),'datasets':list(p.DATASETS),
        'epoch_budgets_by_horizon':{str(h):p.VARIANTS[h]['epochs'] for h in p.HORIZONS},
        'model_variants':models,'runs':runs,'native_data_metadata':parent['data_metadata'],
        'opaque_parent_prefix_calendar_file_references':{n:v for n,v in parent['files'].items() if n.startswith('data/')},
        'h96_reuse_requirements':{
            'minimum6_strict_common_STD_MSE_MAE_12_gate_PASS_required':True,
            'independent_full_whole_minimum6_physical_and_numeric_audit_required':True,
            'native6_LIST_exact_references_all5_each_required_before_future_production':True,
            'prior_H96_best_checkpoint_five_file_identity_reused_without_retraining':True,
            'H96_retraining_allowed':False,'actual_H96_qualification_known_in_this_source_package':False,
            'future_ref_placeholder_is_not_authority':True},
        'actual_missing_requirements':[
            'qualified minimum6 strict12 gate PASS with independent exact whole physical/native6 numeric closure',
            'different independent review of this exact remaining18 source/prescore variant budget snapshot',
            'qualified pair ROOT policy with exact H96 native LIST6 all5 immutable checkpoint reference reuse',
            'separate production wrappers/process census/owned admission before any framework/data/RNG',
            'original B256 synthetic fullresident two-step CUDA all18 memory/disk QA and independent measured review',
            'fresh native fullprefix/scaler bitwise proof then ROOT source/resource freeze/prestart/exclusive start',
            'global raw process/GPU idle excluding all original/corrected/Root18/Root10/Root9/VP6/VP18/ITSMixer6/18 producers; no joint GPU'],
        'metric_space':'COMMON_EXTERNAL_TRAIN_STANDARDIZED','inverse_external_scaler_before_metrics':False,
        'TRAIN_shuffle_drop_last_retained':True,'all_available_TRAIN_windows_visited_each_epoch_guaranteed':False,
        'VAL_all_chronological_windows_including_tail':True,'full_H_C_elements_unmasked':True,
        'checkpoint_selector':'0.5*(pooled_STANDARDIZED_MSE+pooled_STANDARDIZED_MAE)',
        'checkpoint_latest_equal_ties':True,'best_state_includes_all_BN_buffers':True,
        'literal_historical_CLI_verified':False,'original_historical_framework_revision_known':False,
        'historical_logs_are_budget_evidence_not_an_optimal_configuration_claim':True,
        'all_negative_FACT_VPNet_and_future_attempts_retained':True,
        'VPNet_qualification_not_a_prerequisite':True,'VPNet18_or_VP_pair_authorized':False,
        'production_wrappers_implemented':False,'actual_resources_measured':False,'actual_source_deployed':False,
        'actual_arrays_opened':False,'framework_imported':False,'actual_scores_read':False,'test_accessed':False,
        'plan_frozen':False,'training_authorized':False,'training_started':False,
        'minimum6_qualified_by_this_package':False,'pair_selected_by_this_package':False,
    }
    keep(stage/'ITSMIXER_REMAINING18_SCIENCE_PROTOCOL_PENDING.json',enc(protocol))
    keep(stage/'ITSMIXER_REMAINING18_CONFIG_RECIPE_PENDING.json',enc({
        'status':'PENDING_FIXED_REMAINING18_PRE_SCORE_CONFIG_RECIPE_NO_PRODUCTION_AUTHORITY',
        'root_pre_score_policy_sha256':p.ROOT_REMAINING18_POLICY_SHA256,'original_h96_science_sha256':MANIFEST_SHA,
        'parent_plan_sha256':p.PARENT_PLAN_SHA256,'runs':runs,'resources_measured':False,
        'production_authority':False,'qualified_pair':False,'test_allowed':False}))
    keep(stage/'AUTHOR_EXACT_VARIANT_AND_KERNEL_ADAPTATION_PROOF.json',enc({
        'status':'AUTHOR_SOURCE_ONLY_PROOF_NOT_INDEPENDENT_REVIEW_OR_PRODUCTION_AUTHORITY',
        'original_e8fe_h96_bytes_retained':{n:sh((stage/n).read_bytes()) for n in retained},
        'original_h96_manifest_sha256':MANIFEST_SHA,'per_horizon_author_model_byte_and_AST_proof':models,
        'kernel_h96_original_sha256':sh((original/'itsmixer6_science_kernel.py').read_bytes()),
        'future_kernel_sha256':sh((stage/'itsmixer18_science_kernel.py').read_bytes()),
        'exact_diff_sha256':sh(diff.encode()),'targeted_kernel_replacements':proof,
        'all_replacements_round_trip_restore_original_exact_text':True,
        'per_horizon_original_author_class_AST_modified':False,
        'kernel_optimizer_loss_checkpoint_rule_changed':False,
        'full_validation_reduction_math_changed_except_horizon_denominator':False,
        'checkpoint_budget_epoch_loop_bound_changed_as_pre_score_policy':True,
        'allowed_future_differences':['explicit corresponding horizon model dispatch','forecast H and legal same-prefix windows/element geometry','native corresponding15/15/20 full epoch budgets','pure protocol import/defaultPENDING labels'],
        'actual_model_framework_arrays_GPU_scores_network_test_accessed':False,
    }))
    print(json.dumps({'status':'AUTHOR_REMAINING18_SOURCE_PREPARED_PENDING_PRIVATE_QA','run_count':len(runs),
                      'kernel_sha256':sh((stage/'itsmixer18_science_kernel.py').read_bytes()),
                      'protocol_sha256':sh((stage/'ITSMIXER_REMAINING18_SCIENCE_PROTOCOL_PENDING.json').read_bytes())}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-source-only',action='store_true');a=p.parse_args()
    if not a.prepare_source_only:print('PENDING_SOURCE_ONLY_NO_IO_NO_FRAMEWORK_NO_NETWORK_NO_ACTUAL_ACTION');return
    prepare()
if __name__=='__main__':main()
