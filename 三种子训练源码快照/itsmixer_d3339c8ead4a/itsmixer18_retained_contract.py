"""Pure conditional Root18 retention contract; no I/O or model/score imports."""
import json
import re

ORIGINAL = '/root/autodl-tmp/paper5/models2026_pilot_20261011'
ORIGINAL_PLAN_SHA = '16a9dacee118f30b966489e0e17ad90274cfd56828ceaffb0c612b6fca9144ed'
RETENTION_SHA = '605f030f852fbf7b04fdae610c560e937440b8af8b692463b5a144f1735b6e04'
REFERENCES_SHA = '06fa1a3584f45f8cdc11c9c75d140f4fc3f1bf2950d00a7b11576f4c573fa46d'
DATASETS = frozenset(('Electricity', 'PEMS03', 'PEMS04', 'PEMS08', 'Solar', 'AEMO5'))
ROOT6 = {'RootPurgeLinear_' + d for d in DATASETS}
ORIGINAL12 = ROOT6 | {'FACT_' + d for d in DATASETS}
INVALID4 = {'FACT_' + d for d in ('PEMS03', 'PEMS04', 'PEMS08', 'Solar')}
FILES5 = {'best.pt', 'config.json', 'history.json', 'VALIDATION_RESULT.json', 'COMPLETE_RECEIPT.json'}
SCORE_KEYS = {'test', 'validation', 'mse', 'mae', 'channel_mse', 'mse_raw', 'mae_raw', 'scores', 'metrics', 'values'}
STRING = re.compile(r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"')


def require(value, message):
    if not value: raise ValueError(message)


def valid_sha(value): return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def metadata_bytes(body):
    text = body.decode('utf-8-sig')
    for token in STRING.finditer(text):
        if text[token.end():].lstrip().startswith(':'):
            require(json.loads(token.group()) not in SCORE_KEYS, 'ROOT18 metadata cannot decode score keys including escaped keys')
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate score-free ROOT18 key'); result[k] = v
        return result
    def invalid(value): raise ValueError('nonfinite ROOT18 metadata')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def physical5(row):
    require(set(row['files']) == FILES5 and all(valid_sha(v['sha256']) and type(v['bytes']) is int and v['bytes'] > 0
            for v in row['files'].values()), 'all five historical training artifacts require physical SHA/bytes')


def reference_index(references):
    """Validate native LIST6 without changing the immutable ROOT reference bytes."""
    rows = references['runs']
    require(type(rows) is list and len(rows) == 6 and type(references['run_count']) is int
            and references['run_count'] == 6, 'native retained references must be LIST6')
    indexed = {}
    for row in rows:
        require(type(row) is dict and set(row) == {'cohort', 'source_root', 'plan_sha256', 'physical', 'run_id', 'best_path'},
                'exact native retained reference row schema')
        rid = row['run_id']
        require(isinstance(rid, str) and rid in ROOT6 and rid not in indexed, 'duplicate/unknown retained Root H96 reference')
        require(row['cohort'] == 'original_valid8' and row['source_root'] == ORIGINAL
                and row['plan_sha256'] == ORIGINAL_PLAN_SHA
                and row['best_path'] == ORIGINAL + '/results/' + rid + '/best.pt', 'native retained Root origin/path differs')
        physical = row['physical']; physical5(physical)
        require(physical['model'] == 'RootPurgeLinear' and physical['dataset'] == rid.split('_', 1)[1]
                and physical['actual_result_window_elements_not_decoded'] is True,
                'native retained identity must use nested physical metadata')
        for key in ('expected_train_windows', 'expected_validation_windows', 'expected_validation_elements', 'expected_channels'):
            require(type(physical[key]) is int and physical[key] > 0, 'native retained window/channel count differs')
        require(physical['expected_validation_elements'] == physical['expected_validation_windows'] * 96 * physical['expected_channels'],
                'native retained validation elements identity differs')
        indexed[rid] = row
    require(set(indexed) == ROOT6, 'exact complete retained Root6 reference IDs')
    return indexed


def validate_retained_metadata(selection, closure, references):
    """Retention is a prerequisite; it never authorizes eighteen training runs."""
    require(selection['status'] == 'ROOT_PROSPECTIVE_FACT_TO_VPNET_REPLACEMENT_AUTHORIZED_AFTER_COMPLETE_CANONICAL12_NUMERIC_AUDIT'
            and selection['retained_model'] == 'RootPurgeLinear' and selection['retained_h96_count'] == 6
            and selection['retained_h96_strict_ours_wins'] == 12 and selection['replaced_model'] == 'FACT'
            and selection['replacement_model'] == 'VPNet' and selection['original_plan_sha256'] == ORIGINAL_PLAN_SHA
            and selection['two_model_pair_selected'] is False and selection['full_training_authorized_by_this_record'] is False
            and selection['test_authorized_by_this_record'] is False and valid_sha(selection['independent_numeric_audit_sha256']),
            'exact ROOT retained-model decision; replacement record is not full training/test authority')
    require(closure['status'] == 'COMPLETE_CANONICAL12_AND_ORIGINAL_INVALID4_PHYSICAL_CLOSED'
            and closure['original_plan_sha256'] == ORIGINAL_PLAN_SHA and closure['canonical_count'] == 12
            and closure['all_attempts_physical_count'] == 16 and set(closure['canonical']) == ORIGINAL12
            and closure['corrected4_plan_sha256'] == selection['corrected4_plan_sha256']
            and closure['scores_decoded'] is False and closure['invalid_original4_scores_decoded'] is False
            and closure['test_accessed'] is False and set(closure['invalid_original4']) == INVALID4,
            'all original historical16 canonical physical closure required; invalid4 stays opaque')
    for cohort, status, ids, plan_sha in (
            ('original12', 'COMPLETE_ORIGINAL12_PHYSICAL_CLOSURE_NO_SCORES_DECODED', ORIGINAL12, ORIGINAL_PLAN_SHA),
            ('corrected4', 'COMPLETE_CORRECTED4_PHYSICAL_NO_SCORES_DECODED', INVALID4, selection['corrected4_plan_sha256'])):
        value = closure[cohort]
        require(value['status'] == status and value['plan_sha256'] == plan_sha and set(value['runs']) == ids
                and value['controller_and_workers_ended'] is True and value['scores_decoded'] is False
                and value['test_accessed'] is False, 'old16 complete controllers/workers physical metadata differs')
        for rid, row in value['runs'].items():
            physical5(row)
            if cohort == 'original12' and rid in INVALID4:
                require(row == closure['invalid_original4'][rid], 'invalid4 must retain original opaque physical evidence')
    require(references['status'] == 'ROOT_RETAINED_ROOTPURGE6_H96_PHYSICAL_REFERENCES_AFTER_COMPLETE_NUMERIC_AUDIT'
            and references['run_count'] == 6 and references['weights_copied'] is False
            and references['retraining_permitted'] is False and references['replacement_record_sha256'] == RETENTION_SHA
            and references['independent_numeric_audit_sha256'] == selection['independent_numeric_audit_sha256']
            and references['physical_closure_sha256'] == selection['physical_closure_sha256'],
            'exact retained Root6 H96 references required; no pair freeze, copy or retrain')
    for rid, row in reference_index(references).items():
        member = closure['canonical'][rid]
        require(row['run_id'] == rid and row['cohort'] == member['cohort'] == 'original_valid8'
                and row['source_root'] == member['source_root'] == ORIGINAL
                and row['plan_sha256'] == member['plan_sha256'] == ORIGINAL_PLAN_SHA
                and row['best_path'] == ORIGINAL + '/results/' + rid + '/best.pt'
                and row['physical'] == member['physical'] == closure['original12']['runs'][rid], 'retained Root H96 exact all5 original physical origin identity')
        physical5(row['physical'])
