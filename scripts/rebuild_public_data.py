"""Download six pinned sources and release already normalized forecasting data.

Usage: python build_processed_data.py --output-dir data
Companion files: dataset_specs.json and original_prepare_data.py.
No training or score evaluation is performed.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests
import original_prepare_data as original

SOURCE_DIR=Path(__file__).resolve().parent
PINS=json.loads((SOURCE_DIR/'dataset_specs.json').read_text(encoding='utf-8'))
SPECS=PINS['datasets']
NAMES=['Electricity','PEMS03','PEMS04','PEMS08','Solar','AEMO5']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def arr_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def raw_filename(name):
    return PINS['raw_sha256'][name]['filename']
def download(name,raw_dir):
    pin=PINS['raw_sha256'][name]
    path=raw_dir/raw_filename(name)
    if not path.is_file():
        part=path.with_name(path.name+'.part')
        with requests.get(pin['url'],stream=True,timeout=(30,180),headers={'User-Agent':'CoReXNet-reproducible-release/1.0'}) as r:
            r.raise_for_status()
            with part.open('wb') as f:
                for b in r.iter_content(1<<20):
                    if b:f.write(b)
        part.replace(path)
    if path.stat().st_size!=pin['bytes'] or sha(path)!=pin['sha256']:
        raise ValueError('Raw source SHA/size mismatch: '+name)
    print(json.dumps({'stage':'RAW_VERIFIED','dataset':name,'sha256':pin['sha256']}),flush=True)
    return path
def force_container_os(path,system):
    """Normalize only ZIP made-by OS fields, retaining exact compressed members."""
    data=bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as z:
        offset=z.start_dir
        for info in z.infolist():
            if data[offset:offset+4]!=b'PK\x01\x02':raise ValueError('Invalid central directory')
            data[offset+5]=system
            name_len=int.from_bytes(data[offset+28:offset+30],'little')
            extra_len=int.from_bytes(data[offset+30:offset+32],'little')
            comment_len=int.from_bytes(data[offset+32:offset+34],'little')
            offset+=46+name_len+extra_len+comment_len
    path.write_bytes(data)
def aemo_panel(path):
    first=dt.datetime(2010,1,1);T=SPECS['AEMO5']['rows'];columns={};metadata=[]
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise ValueError('AEMO source CRC failed')
        members=[n for n in z.namelist() if n.lower().endswith('.tsf')]
        if len(members)!=1:raise ValueError('AEMO member ambiguity')
        attributes=[];in_data=False
        with z.open(members[0]) as f:
            for line in io.TextIOWrapper(f,encoding='cp1252'):
                line=line.strip()
                if not line:continue
                if not in_data:
                    if line.lower()=='@data':in_data=True
                    elif line.lower().startswith('@attribute '):attributes.append(line.split()[1])
                    continue
                meta,tokens=line.rsplit(':',1);attrs=dict(zip(attributes,meta.split(':')))
                sid=attrs['series_name'];start=dt.datetime.strptime(attrs['start_timestamp'],'%Y-%m-%d %H-%M-%S')
                offset=(first-start).total_seconds()/1800
                if not offset.is_integer() or offset<0:raise ValueError('AEMO source alignment mismatch')
                offset=int(offset);all_tokens=tokens.split(',');part=all_tokens[offset:offset+T]
                if len(part)!=T or sid in columns:raise ValueError('AEMO incomplete/duplicate series')
                columns[sid]=np.array([float(v) if v not in ('?','','NA','N/A','null','None') else np.nan for v in part],dtype=np.float64)
                metadata.append({'id':sid,'attributes':attrs,'source_count':len(all_tokens),'selected_start':offset,'selected_stop':offset+T})
    if set(columns)!={'T1','T2','T3','T4','T5'}:raise ValueError('AEMO five-series identity mismatch')
    a=np.column_stack([columns[s] for s in ['T1','T2','T3','T4','T5']]);finite=np.isfinite(a);filled=a.copy()
    train_end=SPECS['AEMO5']['bounds'][0]
    for c in range(5):
        valid=a[:train_end,c][np.isfinite(a[:train_end,c])]
        if len(valid)==0:raise ValueError('No finite TRAIN observations')
        leading_mean=valid.mean(dtype=np.float64)
        idx=np.maximum.accumulate(np.where(finite[:,c],np.arange(T),-1));early=idx<0
        filled[~early,c]=a[idx[~early],c];filled[early,c]=leading_mean
    return filled.astype(np.float32),{'channel_ids':['T1','T2','T3','T4','T5'],'series':metadata,'missing_imputed_count':int((~finite).sum()),'first':'2010-01-01T00:00:00','last':'2014-12-31T23:30:00','clock':'TSF start_timestamp plus regular index-derived30-minute alignment'}
def prepare(name,raw_dir,legacy_dir,out_dir):
    s=SPECS[name];raw=raw_dir/raw_filename(name);legacy=legacy_dir/(name+'.npz')
    if name in ('Electricity','PEMS03','PEMS04','PEMS08'):
        a,extra=original.read_raw(name,raw)
        bounds=s['bounds']
        mean0,std0=original.train_scaler(a,bounds[0])
        metadata=dict(dataset=name,raw_file=raw.name,raw_size_bytes=raw.stat().st_size,raw_sha256=sha(raw),prepared_file=name+'.npz',dtype='float32',original_rows=len(a),used_rows=len(a),channels=a.shape[1],unused_tail_rows=0,frequency_minutes=original.SPECS[name]['minutes'],split_counts=bounds,split_counts_semantics='cumulative exclusive row boundaries: train_end, val_end, used_T',split_row_counts=[bounds[0],bounds[1]-bounds[0],bounds[2]-bounds[1]],seq_len=96,pred_len=96,seed=2026,scaler='runner fits channelwise population mean/std on training rows only; prepared data remain raw',metric='unmasked MSE and MAE averaged over all forecast windows, horizons, and channels in train-standardized units',window_stride=1,history_overlap=True,target_crosses_split=False,imputation='none; finite-value audit required',zero_values='preserved as in original benchmark files',zero_fraction=float(np.mean(a==0)),window_counts={x:len(original.forecast_origins(bounds,x,96,96)) for x in ('train','val','test')},pems_protocol_note='96-to-96 common forecasting experiment; does not reproduce published 12-to-12 short-term PeMS metrics' if name.startswith('PEMS') else None,**original.source_info(name),**extra)
        np.savez_compressed(legacy,data=a,split_counts=np.array(bounds,dtype=np.int64),train_mean=mean0,train_std=std0,metadata_json=json.dumps(metadata,ensure_ascii=False))
        force_container_os(legacy,3)
    elif name=='Solar':
        with gzip.open(raw,'rt') as f:a=pd.read_csv(f,header=None).to_numpy(dtype=np.float32)
        a=np.ascontiguousarray(a,dtype=np.float32);extra={'channel_ids':list(range(137)),'zero_policy':'All genuine zero values retained','source_format':'LSTNet solar_AL.txt.gz comma-separated no-header matrix'}
        np.savez(legacy,data=a,split_counts=np.array(s['bounds'],dtype=np.int64));force_container_os(legacy,0)
    else:
        a,extra=aemo_panel(raw)
        np.savez_compressed(legacy,data=a,split_counts=np.array(s['bounds'],dtype=np.int64));force_container_os(legacy,0)
    if a.dtype!=np.float32 or a.shape!=(s['rows'],s['channels']) or not np.isfinite(a).all():raise ValueError('Full matrix shape/dtype/finite mismatch: '+name)
    if arr_sha(a)!=s['raw_array_sha256']:raise ValueError('Full raw numerical panel SHA mismatch: '+name)
    train_end=s['bounds'][0]
    mean=a[:train_end].mean(axis=0,dtype=np.float64).astype(np.float32)
    std=a[:train_end].std(axis=0,dtype=np.float64).astype(np.float32);std[std<1e-6]=1
    if arr_sha(mean)!=s['scaler_mean_sha256'] or arr_sha(std)!=s['scaler_std_sha256']:raise ValueError('Original TRAIN scaler SHA mismatch: '+name)
    normalized=(a-mean)/std
    payload={'data':a,'normalized_data':normalized,'scaler_mean':mean,'scaler_std':std,'split_counts':np.array(s['bounds'],dtype=np.int64)}
    origin_metadata={}
    for h in [96,192,336,720]:
        origin_metadata[str(h)]={}
        for split,start,end in [('train',96,s['bounds'][0]),('validation',s['bounds'][0],s['bounds'][1]),('test',s['bounds'][1],s['bounds'][2])]:
            ix=np.arange(start,end-h+1,dtype=np.int64);payload[f'{split}_origins_h{h}']=ix
            origin_metadata[str(h)][split]={'count':len(ix),'first':int(ix[0]),'last':int(ix[-1]),'sha256':arr_sha(ix)}
    target=out_dir/(name+'.npz');np.savez_compressed(target,**payload)
    legacy_sha=sha(legacy)
    result={'dataset':name,'shape':list(a.shape),'bounds':s['bounds'],'dtype':'float32','raw_source_sha256':sha(raw),'raw_array_sha256':arr_sha(a),'normalized_array_sha256':arr_sha(normalized),'scaler_mean_sha256':arr_sha(mean),'scaler_std_sha256':arr_sha(std),'legacy_expected_npz_sha256':s['full_data_sha256'],'rebuilt_legacy_npz_sha256':legacy_sha,'legacy_container_identity_verified':legacy_sha==s['full_data_sha256'],'raw_array_pin_basis':s['array_hash_pin_basis'],'file':name+'.npz','sha256':sha(target),'bytes':target.stat().st_size,'array_keys':list(payload),'origin_metadata':origin_metadata,'source_url':PINS['raw_sha256'][name]['url'],'extra':extra}
    dump(out_dir/(name+'.metadata.json'),result)
    print(json.dumps({'stage':'PROCESSED','dataset':name,'bytes':result['bytes'],'legacy_container_identity_verified':result['legacy_container_identity_verified']}),flush=True)
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=Path('build/data'));p.add_argument('--cache-dir',type=Path);p.add_argument('--datasets',nargs='+',choices=NAMES,default=NAMES);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();release_root=args.output_dir.resolve();out=release_root/'processed';cache=(args.cache_dir or release_root/'.source_cache').resolve();raw=cache/'raw';legacy=cache/'legacy_format'
    for d in [out,raw,legacy]:d.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(lambda ds:download(ds,raw),args.datasets))
    results={ds:prepare(ds,raw,legacy,out) for ds in args.datasets}
    manifest={'status':'COMPLETE_PREPROCESSED_DATA_VERIFIED','datasets':results,'normalization':'TRAIN-only float64 population mean/std castfloat32, std<1e-6 maps to1, float32 transform','length':96,'horizons':[96,192,336,720],'window_definition':'x=normalized_data[o-96:o]; y=normalized_data[o:o+H]','target_never_crosses_split':True,'history_context_may_precede_current_split':True,'stride':1,'all_source_and_array_and_scaler_hashes_verified':True,'legacy_container_identity_verified_count':sum(r['legacy_container_identity_verified'] for r in results.values()),'original_legacy_container_identity_mismatch_datasets':[ds for ds,r in results.items() if not r['legacy_container_identity_verified']],'new_training_or_metric_evaluation_performed':False,'specification_sha256':sha(SOURCE_DIR/'dataset_specs.json'),'prepare_source_sha256':sha(SOURCE_DIR/'original_prepare_data.py')}
    dump(out/'data_manifest.json',manifest)
    dump(out/'DATA_MANIFEST.json',manifest)
    readme='''CoReXNet preprocessed forecasting data v1

Each dataset NPZ contains the full original-value panel (data), the already
TRAIN-standardized float32 panel (normalized_data), scaler_mean, scaler_std,
and split_counts with cumulative exclusive TRAIN, validation and TEST ends.
The 12 precomputed integer origin arrays cover train, validation and test at
H=96,192,336,720. For any origin o, x=normalized_data[o-96:o] and
y=normalized_data[o:o+H]. No additional split construction or normalization
is needed. Validation/test histories can precede their target split, while
every target stays fully inside its own split.

The TRAIN scaler uses float64 population mean/std, casts them to float32,
and maps std below1e-6 to1. Both subtraction and division are float32. Raw
source, full numerical panel and scaler SHA256 values are pinned and checked.
Genuine zero values are retained. Evaluation is unmasked and includes all
chronological stride1 windows without dropping the last evaluation batch.

Six datasets: Electricity, PEMS03, PEMS04, PEMS08, Solar and AEMO5. Each
*.metadata.json gives shapes, exact boundaries, sources and hashes. The
DATA_MANIFEST.json distinguishes legacy archive byte identity from the
new processed release archive. See repository dataset/source notices for
attribution and source-specific terms.

The data release is a fresh packaging of the fixed experimental panels.
No new model training, checkpoint selection or metric evaluation is run.
'''
    (out/'README.txt').write_text(readme,encoding='utf-8')
    files=[out/(ds+ext) for ds in args.datasets for ext in ('.npz','.metadata.json')]+[out/'DATA_MANIFEST.json',out/'data_manifest.json',out/'README.txt']
    (out/'SHA256SUMS').write_text(''.join(sha(f)+'  '+f.name+'\n' for f in files),encoding='utf-8')
    files.append(out/'SHA256SUMS')
    bundle=release_root/'CoReXNet_preprocessed_data_v1.zip'
    with zipfile.ZipFile(bundle,'w',allowZip64=True) as archive:
        for f in files:archive.write(f,arcname=f.name,compress_type=zipfile.ZIP_STORED if f.suffix=='.npz' else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:raise ValueError('Final release ZIP CRC mismatch')
        for f in files:
            if hashlib.sha256(archive.read(f.name)).hexdigest()!=sha(f):raise ValueError('Final ZIP member SHA mismatch')
    (release_root/(bundle.name+'.sha256')).write_text(sha(bundle)+'  '+bundle.name+'\n',encoding='utf-8')
    dump(release_root/'DATA_BUILD_RECEIPT.json',{'status':'COMPLETE_PREPROCESSED_RELEASE_ZIP_SHA_CRC_MEMBERS_VERIFIED','dataset_count':len(results),'legacy_container_identity_verified_count':manifest['legacy_container_identity_verified_count'],'zip_filename':bundle.name,'zip_bytes':bundle.stat().st_size,'zip_sha256':sha(bundle),'zip_member_count':len(files),'all_sources_arrays_scalers_checked':True,'all_origins_precomputed':True,'model_training_or_scoring_performed':False})
    print(json.dumps({'status':manifest['status'],'dataset_count':len(results),'legacy_container_identity_verified_count':manifest['legacy_container_identity_verified_count']}),flush=True)
if __name__=='__main__':main()
