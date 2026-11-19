"""Exact original completed2026 Root/IT PUREtest scientific definitions."""
import hashlib,math
MODELS=('RootPurgeLinear','ITSMixer')
def require(value,message):
    if not value:raise ValueError(message)
def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def config_view(config):
    model=config['model'];require(model in MODELS,'no FACT/VP/undeclared model admission')
    if model=='RootPurgeLinear':
        return {k:config[k] for k in ('model','dataset','seq_len','pred_len','channels','seed','batch_size')}
    return {'model':model,'dataset':config['dataset'],'seq_len':config['length'],'pred_len':config['horizon'],
        'channels':config['channels'],'seed':config['seed'],'batch_size':config['batch_size']}


class FullTestWindows:
    def __init__(self,np,torch,config,data):
        view=config_view(config);self.C=view['channels'];self.H=view['pred_len'];self.L=96
        with np.load(data['full_data_path'],allow_pickle=False) as archive:
            require(set(archive.files)==({'data','split_counts','train_mean','train_std','metadata_json'} if view['dataset'] in ('Electricity','PEMS03','PEMS04','PEMS08') else {'data','split_counts'}),'original full NPZ schema')
            values=archive['data'];bounds=archive['split_counts'].tolist()
        with np.load(data['prefix_path'],allow_pickle=False) as archive:
            require(set(archive.files)=={'data','split_counts'},'frozen prefix NPZ schema')
            prefix=archive['data'];pbounds=archive['split_counts'].tolist()
        require(values.dtype==prefix.dtype==np.float32 and values.shape==(data['rows'],self.C)
            and prefix.shape==(bounds[1],self.C) and bounds==data['bounds'] and pbounds==bounds[:2]
            and np.isfinite(values).all() and np.array_equal(values[:bounds[1]].view(np.uint32),prefix.view(np.uint32)),
            'actual original full-prefix bit identity/shape/splits before windows')
        mean=values[:bounds[0]].mean(0,dtype=np.float64).astype(np.float32)
        std=values[:bounds[0]].std(0,dtype=np.float64).astype(np.float32);std[std<1e-6]=1
        require(sha_bytes(mean.tobytes())==data['scaler_mean_sha256'] and sha_bytes(std.tobytes())==data['scaler_std_sha256'],
            'bit-identical original TRAIN-only scaler, no TEST statistic fit')
        require(bounds[1]>=96,'legal TEST context')
        self.values=torch.as_tensor((values-mean)/std,device='cuda')
        self.indices=torch.arange(bounds[1],bounds[2]-self.H+1,device='cuda')
        self.xoff=torch.arange(-96,0,device='cuda');self.yoff=torch.arange(self.H,device='cuda')
        require(len(self.indices)==data['rows']-bounds[1]-self.H+1>0,'full chronological uncapped TEST suffix')
    def batch(self,ix):return self.values[ix[:,None]+self.xoff],self.values[ix[:,None]+self.yoff]


def predict(model,kernel,inputs,view):
    if view['model']=='RootPurgeLinear':return model(inputs) # original forward_test, alpha training loss is not applied
    if view['pred_len']==96:return kernel.predict(model,inputs,view['channels'])
    return kernel.predict(model,inputs,view['channels'],view['pred_len'])


def evaluate(torch,model,kernel,windows,config):
    view=config_view(config);model.eval()
    sums=torch.zeros(3,device='cuda',dtype=torch.float64)
    channels=torch.zeros(windows.C,device='cuda',dtype=torch.float64)
    with torch.inference_mode():
        for ix in windows.indices.split(view['batch_size']):
            x,target=windows.batch(ix);forecast=predict(model,kernel,x,view)
            require(forecast.shape==target.shape and bool(torch.isfinite(forecast).all()),'native forecast complete horizon/channel finite shape')
            error=forecast.float()-target
            sums[0]+=error.square().sum(dtype=torch.float64);sums[1]+=error.abs().sum(dtype=torch.float64)
            sums[2]+=error.numel();channels+=error.square().sum((0,1),dtype=torch.float64)
    sse,sae,ne=sums.cpu().tolist();n=len(windows.indices)
    require(int(ne)==n*windows.H*windows.C,'pooled complete elements')
    result={'mse':sse/ne,'mae':sae/ne,'squared_error_sum':sse,'absolute_error_sum':sae,
        'channel_mse':(channels/(n*windows.H)).cpu().tolist(),'windows':n,'elements':int(ne),
        'metric_space':'COMMON_EXTERNAL_TRAIN_STANDARDIZED','masking':False,'test_drop_last':False,'test_shuffle':False}
    require(all(math.isfinite(result[k]) and result[k]>=0 for k in ('mse','mae')),'finite native standardized pooled metrics')
    return result
