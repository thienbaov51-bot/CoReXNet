"""Create six TRAIN/VAL-only copies from archived inputs, no test windows/scores."""
from pathlib import Path
import hashlib,json,time,shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
assert str(ROOT)=='/root/autodl-tmp/paper5/models2026_pilot_20261011'
OUT=ROOT/'data';assert not OUT.exists();OUT.mkdir()
P=Path('/root/autodl-tmp/paper5');rows={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for ds in ['Electricity','PEMS03','PEMS04','PEMS08','Solar','AEMO5']:
    base=P/('new5_pilot_20261011/data' if ds=='AEMO5' else 'add3data_20261010/data' if ds=='Solar' else 'data')
    src=base/(ds+'_trainval.npz' if ds=='AEMO5' else ds+'.npz')
    before=sha(src)
    with np.load(src,allow_pickle=False) as z:
        bounds=list(map(int,z['split_counts']));a=np.array(z['data'][:bounds[1]],dtype=np.float32,copy=True)
    assert a.ndim==2 and len(a)==bounds[1] and np.isfinite(a).all()
    target=OUT/(ds+'_trainval.npz');np.savez_compressed(target,data=a,split_counts=np.array(bounds[:2],dtype=np.int64))
    with np.load(target,allow_pickle=False) as z:assert np.array_equal(z['data'].view(np.uint32),a.view(np.uint32))
    mean=a[:bounds[0]].mean(0,dtype=np.float64).astype(np.float32);std=a[:bounds[0]].std(0,dtype=np.float64).astype(np.float32);std[std<1e-6]=1
    if ds=='Electricity':
        raw=P/'data/raw/Electricity.csv';dates=pd.to_datetime(pd.read_csv(raw,usecols=[0],nrows=bounds[1]).iloc[:,0]);freq='h'
    elif ds=='AEMO5':dates=pd.Series(pd.date_range('2010-01-01',periods=bounds[1],freq='30min'));freq='h'
    else:dates=None;freq='h'
    if dates is None:marks=np.zeros((bounds[1],1),dtype=np.float32)
    else:
        marks=np.stack([dates.dt.hour.to_numpy()/23-0.5,dates.dt.dayofweek.to_numpy()/6-0.5,(dates.dt.day.to_numpy()-1)/30-0.5,(dates.dt.dayofyear.to_numpy()-1)/365-0.5],axis=1).astype(np.float32)
    np.save(OUT/(ds+'_known_calendar.npy'),marks,allow_pickle=False)
    assert sha(src)==before
    rows[ds]={'dataset':ds,'channels':a.shape[1],'bounds':bounds,'prefix_rows':len(a),'source_path':str(src),'source_sha256':before,'prefix_sha256':sha(target),'calendar_sha256':sha(OUT/(ds+'_known_calendar.npy')),'calendar_channels':marks.shape[1],'scaler_mean_sha256':hashlib.sha256(mean.tobytes()).hexdigest(),'scaler_std_sha256':hashlib.sha256(std.tobytes()).hexdigest(),'train_windows':bounds[0]-191,'validation_windows':bounds[1]-bounds[0]-95,'existing_full_archive_NPZ_library_decompressed_but_only_prefix_selected':ds!='AEMO5','new_test_array_or_indices_constructed':False,'test_scores_read':False,'old_source_unchanged':True}
(OUT/'PREFIX_PREPARATION_RECEIPT.json').write_text(json.dumps({'status':'ACTUAL_SIX_PREFIXES_PREPARED_NO_SCORES','utc_epoch':time.time(),'datasets':rows},indent=2))
print(json.dumps({'status':'ACTUAL_SIX_PREFIXES_PREPARED_NO_SCORES','datasets':rows}))
