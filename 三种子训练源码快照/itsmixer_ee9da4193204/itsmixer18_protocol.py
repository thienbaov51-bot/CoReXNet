"""Prospective remaining18 metadata/scalar recipe. Import has no IO or authority.

H96 source is retained unchanged. Exact ECL horizon variants and demonstrated
15/15/20 epoch budgets are selected before any candidate remaining18 scores.
Qualification, physical H96 reuse, production and resources remain pending.
"""
from itsmixer6_protocol import (
    AUTHOR_COMMIT, AUTHOR_CLASSES, PARENT_PLAN_SHA256, PARENT_ROOT, DATASETS,
    encoded, sha_bytes, require, typed_equal, compound_scalar,
    replaces_checkpoint, make_config as make_h96_config,
)

HORIZONS = (192, 336, 720)
PROSPECTIVE_CHILD_ROOT = PARENT_ROOT + '/itsmixer_remaining18'
ROOT_REMAINING18_POLICY_NAME = 'ROOT_ITSMIXER_REMAINING18_PRE_SCORE_HORIZON_BUDGET_ADOPTION_20261012.json'
ROOT_REMAINING18_POLICY_SHA256 = '3f42cd092488a9de60c487cc5523c0bd380248c9a3aa5956faf821bba643ede6'
H96_SCIENCE_MANIFEST_SHA256 = 'e8fe24e644b72be8b90c8df70b3f113d7480b76c34e126e1036933d434bfc100'
VARIANTS = {
    192: {'epochs': 15, 'first_loop': 8, 'second_loop': 8, 'passes': 25,
          'author_hidden_width': 512, 'effective_hidden_width': 96, 'unused_layer_list_length': 10,
          'source_file': 'ITSMixer_ECL192_author_exact.py',
          'source_sha256': 'cad8ec381697edc070dd263b764f64be3854263249c127efad245c9ed7d5707d',
          'git_blob_sha1': '5caf8139bf43a5772731f9c93c4cea95ccf0bab7'},
    336: {'epochs': 15, 'first_loop': 9, 'second_loop': 9, 'passes': 28,
          'author_hidden_width': 512, 'effective_hidden_width': 96, 'unused_layer_list_length': 10,
          'source_file': 'ITSMixer_ECL336_author_exact.py',
          'source_sha256': '1c0bc5d9e563db2883a8beb756b1a59c3add943fdde48d863f631eee1778b643',
          'git_blob_sha1': 'd8ab4af1d372480c0974ec3ef38feb73d4e0692f'},
    720: {'epochs': 20, 'first_loop': 8, 'second_loop': 8, 'passes': 25,
          'author_hidden_width': 64, 'effective_hidden_width': 64, 'unused_layer_list_length': 50,
          'source_file': 'ITSMixer_ECL720_author_exact.py',
          'source_sha256': 'c2db425241bb538d56f487b128bf0a674c5f81334b2574b287c81c561aebbac5',
          'git_blob_sha1': '42318155c334c6ec46ddd5bf7108ed1c245407af'},
}

def source_model_for_horizon(horizon):
    require(type(horizon) is int and horizon in HORIZONS, 'only new192/336/720; H96 must be reused')
    return VARIANTS[horizon]['source_file']

def coverage(train_end, val_end, channels, horizon):
    require(all(type(x) is int for x in (train_end,val_end,channels,horizon)), 'typed dimensions')
    require(horizon in HORIZONS and 96+horizon <= train_end < val_end
            and channels > 0 and val_end-train_end >= horizon, 'legal same-prefix horizon geometry')
    ntrain, nval = train_end-96-horizon+1, val_end-train_end-horizon+1
    full_batches = ntrain // 256
    require(full_batches > 0, 'original B256 must have an optimizer step')
    return {
        'train_target_start_inclusive': 96, 'train_target_start_exclusive': train_end-horizon+1,
        'validation_target_start_inclusive': train_end, 'validation_target_start_exclusive': val_end-horizon+1,
        'train_windows_available': ntrain, 'train_windows_used_per_epoch': full_batches*256,
        'train_windows_dropped_per_epoch': ntrain % 256, 'train_optimizer_steps_per_epoch': full_batches,
        'expected_optimizer_steps_at_full_epoch_budget': full_batches*VARIANTS[horizon]['epochs'],
        'validation_windows': nval, 'validation_elements': nval*horizon*channels,
        'validation_full_batches': nval//256, 'validation_tail_windows': nval % 256,
    }

def make_config(dataset, horizon, metadata):
    source_model_for_horizon(horizon)
    # Original H96 metadata verifies unchanged prefix/bounds/scaler/calendar.
    config = make_h96_config(dataset, metadata)
    variant = VARIANTS[horizon]
    for obsolete in ('author_ecl96_git_blob_sha1','epochs_15_basis',
                     'expected_optimizer_steps_at_full_15_epoch_budget'):
        config.pop(obsolete)
    config.update({
        'run_id': 'ITSMixer_'+dataset+'_H'+str(horizon), 'horizon': horizon,
        'author_horizon': horizon, 'epochs': variant['epochs'],
        'author_hidden_width': variant['author_hidden_width'],
        'effective_hidden_width': variant['effective_hidden_width'],
        'author_first_loop_iterations': variant['first_loop'],
        'author_second_loop_iterations': variant['second_loop'],
        'author_temporal_mlp_passes_per_forward': variant['passes'],
        'author_unused_layer_list_length': variant['unused_layer_list_length'],
        'author_h720_bottleneck64_retained': horizon == 720,
        'model_source_file': variant['source_file'], 'model_source_sha256': variant['source_sha256'],
        'author_corresponding_ecl_git_blob_sha1': variant['git_blob_sha1'],
        'scientific_recipe_basis': 'EXACT_CORRESPONDING_AUTHOR_ECL_HORIZON_PRE_SCORE_TRANSFER',
        'epoch_budget_basis': 'ROOT_PRE_SCORE_ADOPTION_OF_ECL'+str(horizon)+'_HISTORICAL_'+str(variant['epochs'])+'_EPOCH_DEMONSTRATION',
        'literal_original_training_cli_verified': False,
        'new_horizon_source_only': True, 'requires_qualified_h96_pair': True,
        'h96_checkpoint_must_be_reused': True, 'h96_retraining_allowed': False,
        'remaining18_production_implemented': False,
        'root_remaining18_pre_score_policy_sha256': ROOT_REMAINING18_POLICY_SHA256,
    })
    config.update(coverage(config['train_end'],config['validation_end'],config['channels'],horizon))
    return config

def validate_config(config, metadata):
    require(type(config) is dict, 'config object')
    require(typed_equal(config,make_config(config.get('dataset'),config.get('horizon'),metadata)),
            'exact per-horizon model/budget/loss/window/source config; no outcome tuning')
    return config

def type3_lr_after_epoch(epoch, horizon, base=0.001):
    source_model_for_horizon(horizon)
    require(type(epoch) is int and 1 <= epoch <= VARIANTS[horizon]['epochs'], 'bounded corresponding epoch budget')
    require(type(base) is float and base == 0.001, 'frozen author base LR')
    return base if epoch < 3 else base*(0.9**(epoch-3))

def main():
    print('PENDING_ITSMIXER_REMAINING18_SCIENCE_ONLY_NO_IO_NO_FRAMEWORK_NO_QUALIFICATION_NO_RUN')

if __name__ == '__main__': main()
