"""Pure per-candidate six-of-eight VAL eligibility; no training/test or IO."""
import math

MODELS=('OURS','DLinear','SparseTSF','ModernTCN','MixLinear','PhaseFormer')
HORIZONS=(96,192,336,720)

def audit(rows,closure):
    count=closure['run_count'];horizons=(96,) if count==30 else HORIZONS
    assert count in (30,120) and closure['status']==f'ALL{count}_PHYSICAL_METADATA_CLOSED_BEFORE_VAL_RELEASE'
    assert len(closure['datasets'])==len(set(closure['datasets']))==5
    assert closure['test_values_accessed'] is False and closure['length']==96 and closure['seed']==2026
    assert set(rows)=={(d,h,m) for d in closure['datasets'] for h in horizons for m in MODELS}
    assert {value['run_id'] for value in rows.values()}==set(closure['physical_inventory'])
    decisions={}
    for dataset in closure['datasets']:
        wins=0;groups={}
        for horizon in horizons:
            values={model:rows[dataset,horizon,model] for model in MODELS}
            assert len({(row['data_sha256'],row['channels'],row['training_windows'],row['validation_windows'],row['validation_elements']) for row in values.values()})==1
            for model,row in values.items():
                assert row['dataset']==dataset and row['model']==model and row['horizon']==horizon and row['length']==96 and row['seed']==2026
                for key in ('checkpoint_sha256','config_sha256','source_sha256','data_sha256'):assert row[key]==closure['physical_inventory'][row['run_id']][key]
            for metric in ('mse','mae'):
                raw={m:values[m]['validation'][metric] for m in MODELS};assert all(math.isfinite(v) and v>=0 for v in raw.values())
                strict=all(raw['OURS']<raw[m] for m in MODELS[1:]);wins+=int(strict)
                groups[f'H{horizon}_{metric}']={'strict_win':strict,'all_six_raw_values':raw}
        decisions[dataset]={'status':'H96_OBSERVATION_ONLY' if count==30 else ('PASS_AT_LEAST_SIX_OF_EIGHT_VALIDATION_METRICS' if wins>=6 else 'FAIL_VALIDATION_GATE'),
                            'strict_metric_wins':wins,'total_metric_groups':2 if count==30 else 8,'metric_groups':groups,
                            'H96_based_early_stop_allowed':False,'test_eligible_after_independent_freeze':count==120 and wins>=6,'test_result_guarantee':False}
    return {'status':'COMPLETE_ALL30_H96_OBSERVATIONS' if count==30 else 'COMPLETE_ALL120_VALIDATION_CANDIDATE_GATES',
            'candidate_order':closure['datasets'],'decisions':decisions,
            'eligible_candidates':[] if count==30 else [d for d in closure['datasets'] if decisions[d]['test_eligible_after_independent_freeze']],
            'automatic_test_started':False,'all_positive_and_negative_results_retained':True}
