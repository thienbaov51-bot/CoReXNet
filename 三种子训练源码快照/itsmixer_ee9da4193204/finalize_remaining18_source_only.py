"""Explicit local source author finalizer; default inert, no scientific authority."""
import argparse,ast,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--finalize-source-only',action='store_true');a=p.parse_args()
    if not a.finalize_source_only:
        print('PENDING_SOURCE_FINALIZER_NO_IO_NO_NETWORK_NO_FRAMEWORK_NO_ACTUAL_ACTION');return
    stage=Path(__file__).resolve().parent;name='ITSMIXER_REMAINING18_SCIENCE_SOURCE_AUTHOR_MANIFEST.json'
    files={f.relative_to(stage).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()
           for f in sorted(stage.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!=name}
    for n in files:
        if n.endswith('.py'):ast.parse((stage/n).read_text(encoding='utf-8'))
    qa=json.loads((stage/'AUTHOR_REMAINING18_PRIVATE_SOURCE_QA.json').read_bytes())
    assert qa['actual_arrays_or_scores_or_GPU_or_SSH_or_framework_accessed'] is False and qa['independent_review'] is False
    assert all(hashlib.sha256((stage/n).read_bytes()).hexdigest()==v for n,v in qa['source_snapshot'].items())
    norm=json.dumps(files,sort_keys=True,separators=(',',':')).encode()
    manifest={'status':'SOURCE_READY_PENDING_DIFFERENT_INDEPENDENT_REMAINING18_SCIENCE_REVIEW_AND_H96_QUALIFICATION',
        'author':'vpnet_independent_source_review; author of this future18 source, not its independent reviewer',
        'files':files,'snapshot_sha256':hashlib.sha256(norm).hexdigest(),'author_private_checks':qa['checks_passed'],
        'original_h96_science_manifest_sha256':'e8fe24e644b72be8b90c8df70b3f113d7480b76c34e126e1036933d434bfc100',
        'original_h96_science_snapshot_sha256':'a178c9760b5d57c0c80f3a02add9cc01074ec22cf99cbcc02d44528e63996c7e',
        'root_remaining18_pre_score_policy_sha256':'3f42cd092488a9de60c487cc5523c0bd380248c9a3aa5956faf821bba643ede6',
        'epoch_budgets_by_horizon':{'192':15,'336':15,'720':20},'h96_retraining_allowed':False,
        'minimum6_qualification_decided':False,'pair_selected':False,'production_implemented':False,
        'resources_measured':False,'actual_deployed_or_frozen_or_started':False,
        'actual_arrays_framework_GPU_SSH_scores_test_accessed':False,'independent_whole_source_PASS_self_signed':False}
    data=(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
    with (stage/name).open('xb') as f:f.write(data)
    print(json.dumps({'manifest_sha256':hashlib.sha256(data).hexdigest(),'snapshot_sha256':manifest['snapshot_sha256'],
        'files':len(files),'author_private_checks':qa['checks_passed']}))
if __name__=='__main__':main()
