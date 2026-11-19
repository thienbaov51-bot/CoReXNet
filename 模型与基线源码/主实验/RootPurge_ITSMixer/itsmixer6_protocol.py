"""Pure metadata/scalar contract. Importing this file performs no IO.

This is a prospective scientific recipe, not a production plan or permission.
Author classes and external framework provenance are never executed here.
"""
from decimal import Decimal
import hashlib
import json
import math

AUTHOR_COMMIT = '56963974cb2db7bea054dc941be37fe95ddf855f'
AUTHOR_ECL96_BLOB = 'accde6586614316bac2982cbfac68858f99ff3fa'
PARENT_PLAN_SHA256 = '16a9dacee118f30b966489e0e17ad90274cfd56828ceaffb0c612b6fca9144ed'
PARENT_ROOT = '/root/autodl-tmp/paper5/models2026_pilot_20261011'
PROSPECTIVE_CHILD_ROOT = PARENT_ROOT + '/itsmixer_minimum6'
DATASETS = ('Electricity', 'PEMS03', 'PEMS04', 'PEMS08', 'Solar', 'AEMO5')
SOURCE_MODEL_FILE = 'ITSMixer_ECL96_author_exact.py'
AUTHOR_CLASSES = ('RevIN', 'Mlp_time', 'Mixer_Layer', 'Backbone', 'Model')
FIXED = {
    'model': 'ITSMixer', 'length': 96, 'horizon': 96, 'seed': 2026,
    'batch_size': 256, 'epochs': 15, 'patience': 100, 'early_stopping_delta': 0.0,
    'learning_rate': 0.001, 'optimizer': 'AdamW', 'optimizer_weight_decay': 0.01,
    'optimizer_initial_betas': [0.9, 0.999], 'optimizer_eps': 1e-8,
    'optimizer_amsgrad': False, 'lradj': 'type3', 'pct_start': 0.3,
    'onecycle_constructed': True, 'onecycle_scheduler_step_called': False,
    'onecycle_initial_lr': 0.00004, 'onecycle_initial_beta1': 0.95,
    'criterion': '0.5_MSE_plus_0.5_L1',
    'checkpoint_criterion': '0.5_pooled_standardized_VAL_MSE_plus_0.5_pooled_standardized_VAL_MAE',
    'checkpoint_tie_policy': 'LATEST_EQUAL',
    'features': 'M', 'label_len': 0, 'output_attention': False,
    'train_shuffle': True, 'train_drop_last': True,
    'validation_shuffle': False, 'validation_drop_last': False,
    'validation_all_windows': True, 'window_masking': False,
    'data_loader_num_workers': 0, 'torch_threads': 2,
    'full_original_trainval_prefix_resident': True, 'epoch_window_cap': None,
    'all_available_train_windows_visited_every_epoch_guaranteed': False,
    'calendar_used_by_model': False, 'decoder_input_used_by_model': False,
    'calendar_concatenated_to_channels': False,
    'amp': False, 'precision': 'FP32', 'tf32_matmul': True,
    'tf32_cudnn': True, 'cudnn_benchmark': False,
    'metric_space': 'COMMON_EXTERNAL_TRAIN_STANDARDIZED',
    'metric_reduction': 'FLOAT32_ERRORS_FLOAT64_POOLED_SUMS_ALL_HORIZON_CHANNEL_ELEMENTS',
    'inverse_external_scaler_before_metrics': False,
    'automatic_retry': False, 'test_allowed': False,
    'author_test_each_epoch_removed': True, 'profiling_removed': True,
    'author_model_classes_modified': False,
    'reported_checkpoint_selection_metric_is_mse_only': False,
}


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(typed_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def coverage(train_end, val_end, channels):
    require(type(train_end) is int and type(val_end) is int and type(channels) is int, 'typed dimensions')
    require(192 <= train_end < val_end and channels > 0 and val_end - train_end >= 96, 'window dimensions')
    ntrain, nval = train_end - 191, val_end - train_end - 95
    full_batches = ntrain // 256
    require(full_batches > 0, 'author batch must have an optimizer step')
    return {
        'train_target_start_inclusive': 96,
        'train_target_start_exclusive': train_end - 95,
        'validation_target_start_inclusive': train_end,
        'validation_target_start_exclusive': val_end - 95,
        'train_windows_available': ntrain,
        'train_windows_used_per_epoch': full_batches * 256,
        'train_windows_dropped_per_epoch': ntrain % 256,
        'train_optimizer_steps_per_epoch': full_batches,
        'expected_optimizer_steps_at_full_15_epoch_budget': full_batches * 15,
        'validation_windows': nval,
        'validation_elements': nval * 96 * channels,
        'validation_full_batches': nval // 256,
        'validation_tail_windows': nval % 256,
    }


def make_config(dataset, metadata):
    require(dataset in DATASETS and metadata.get('dataset') == dataset, 'known exact dataset')
    bounds = metadata['bounds']
    require(type(bounds) is list and len(bounds) in (2, 3), 'native prefix bounds')
    train_end, val_end = bounds[:2]
    channels = metadata['channels']
    counts = coverage(train_end, val_end, channels)
    require(metadata['prefix_rows'] == val_end, 'full original prefix')
    require(metadata['train_windows'] == counts['train_windows_available'], 'native TRAIN coverage')
    require(metadata['validation_windows'] == counts['validation_windows'], 'native VAL coverage')
    config = dict(FIXED)
    config.update({
        'run_id': 'ITSMixer_' + dataset + '_H96', 'dataset': dataset,
        'channels': channels, 'enc_in': channels, 'prefix_rows': val_end,
        'train_end': train_end, 'validation_end': val_end,
        'native_original_bounds': list(bounds),
        'freq_metadata': 'h' if dataset in ('Electricity', 'AEMO5') else 'n',
        'author_seq_len': 512, 'author_hidden_width': 512,
        'common_seq_len': 96, 'effective_hidden_width': 96,
        'author_horizon': 96, 'author_time_mlp_dropout': 0.1,
        'author_declared_inactive_mixer_dropout': 0.03,
        'author_mixer_dropout_executed': False, 'author_mixer_layers': 1,
        'author_first_loop_iterations': 8, 'author_second_loop_iterations': 8,
        'author_temporal_mlp_passes_per_forward': 25,
        'author_shared_batchnorm_per_mixer': True,
        'author_revin_affine': True, 'author_revin_eps': 1e-5,
        'author_revin_subtract_last': False,
        'model_source_file': SOURCE_MODEL_FILE,
        'author_commit': AUTHOR_COMMIT, 'author_ecl96_git_blob_sha1': AUTHOR_ECL96_BLOB,
        'scientific_recipe_basis': 'EXACT_AUTHOR_ECL_H96_FIXED_PRE_SCORE_TRANSFER',
        'electricity_literal_training_cli_verified': False,
        'epochs_15_basis': 'ROOT_PRE_SCORE_ADOPTION_OF_ECL96_HISTORICAL_15_EPOCH_DEMONSTRATION',
        'production_plan_frozen': False, 'production_authorized': False,
        'source_path': metadata['source_path'], 'source_sha256': metadata['source_sha256'],
        'prefix_path': PARENT_ROOT + '/data/' + dataset + '_trainval.npz',
        'prefix_sha256': metadata['prefix_sha256'],
        'calendar_path': PARENT_ROOT + '/data/' + dataset + '_known_calendar.npy',
        'calendar_sha256': metadata['calendar_sha256'],
        'calendar_channels': metadata['calendar_channels'],
        'scaler_mean_sha256': metadata['scaler_mean_sha256'],
        'scaler_std_sha256': metadata['scaler_std_sha256'],
        'scaler_rule': 'ORIGINAL_COMMON_TRAIN_PREFIX_FLOAT64_MEAN_STD_CAST_FLOAT32_STD_LT_1e-6_SET_1',
    })
    config.update(counts)
    return config


def validate_config(config, metadata):
    require(type(config) is dict, 'config object')
    expected = make_config(config.get('dataset'), metadata)
    require(typed_equal(config, expected), 'exact pre-score scientific config; no budget/model/loss/window tuning')
    return config


def type3_lr_after_epoch(epoch, base=0.001):
    require(type(epoch) is int and 1 <= epoch <= 15, 'one-based bounded epoch')
    require(type(base) is float and base == 0.001, 'frozen author base LR')
    return base if epoch < 3 else base * (0.9 ** (epoch - 3))


def compound_scalar(mse, mae):
    require(isinstance(mse, (int, float, Decimal)) and not isinstance(mse, bool), 'MSE scalar')
    require(isinstance(mae, (int, float, Decimal)) and not isinstance(mae, bool), 'MAE scalar')
    require(math.isfinite(float(mse)) and math.isfinite(float(mae)) and mse >= 0 and mae >= 0, 'finite nonnegative errors')
    return (mse + mae) / 2


def replaces_checkpoint(candidate_compound, best_compound):
    require(math.isfinite(candidate_compound) and candidate_compound >= 0, 'finite criterion')
    require(best_compound == float('inf') or (math.isfinite(best_compound) and best_compound >= 0), 'finite prior criterion')
    return candidate_compound <= best_compound


def main():
    print('PENDING_ITSMIXER6_SOURCE_PROTOCOL_ONLY_NO_IO_NO_FRAMEWORK_NO_AUTHORITY')


if __name__ == '__main__':
    main()
