"""Original SolarB64 disposable profile kernel, model initialization seed only new."""
import hashlib
from universal_frozen_inference import initialize_family
def require(value,message):
    if not value:raise ValueError(message)

def build_case(case):
    require(case['seed'] in (2027,2028) and case['dataset']=='Solar' and case['horizon']==96,'fixed new-seed Solar profile')
    config=dict(case['config']);require(config['seed']==case['seed'],'exact new model seed')
    science=case['science'];runtime=initialize_family(config,science)
    torch,np,model=runtime['torch'],runtime['np'],runtime['model']
    kernel=runtime['evaluator'] if science['family']=='adapter_core' else runtime['kernel']
    row={'loader_kind':'ADD3DATA' if science['family']=='adapter_core' else 'NATIVE','model':case['model'],'amp':science['amp']}
    if science['family']=='adapter_core':
        predict=lambda inputs,starts:runtime['evaluator'].predict(model,inputs,starts)
    else:
        view=runtime['evaluator'].config_view(config)
        predict=lambda inputs,starts:runtime['evaluator'].predict(model,runtime['kernel'],inputs,view)
    generator = torch.Generator(device='cuda').manual_seed(7442026)
    inputs = torch.randn((64, 96, 137), device='cuda', generator=generator)
    targets = torch.randn((64, 96, 137), device='cuda', generator=generator)
    starts = torch.arange(96, 160, device='cuda', dtype=torch.long)
    if row['loader_kind'] == 'ADD3DATA':
        optimizer = kernel.make_optimizer(model, config)
        scheduler, per_batch = kernel.make_scheduler(optimizer, config, 1)
        require(not per_batch, 'profile expected first-epoch step without per-batch schedule')
        def loss_call():
            forecast = predict(inputs, starts)
            difference = forecast.float() - targets
            loss = difference.abs().mean() if config.get('loss', 'MSE') == 'MAE' else difference.square().mean()
            if hasattr(model, 'auxiliary_loss'):
                require(config.get('additional_loss'), 'original auxiliary loss must be declared')
                loss = loss + model.auxiliary_loss()
            return loss
        loss_scope = config.get('loss', 'MSE')
    elif row['model'] == 'RootPurgeLinear':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        loss_call = lambda: model(inputs, targets)['loss']
        loss_scope = 'Original MSE plus Root Purge regularizer'
    else:
        optimizer, scheduler = kernel.build_optimizer_and_scheduler(torch, model, config)
        loss_call = lambda: kernel.compound_loss(torch.nn.functional, predict(inputs, starts), targets)
        loss_scope = 'Original 0.5 MSE plus 0.5 L1'
    model.train()
    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=row['amp']):
            loss = loss_call()
        loss.backward()
        if config.get('grad_clip') is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()
        return loss
    signature={'input_seed':7442026,'input_sha256':hashlib.sha256(inputs.detach().cpu().numpy().tobytes()).hexdigest(),'target_sha256':hashlib.sha256(targets.detach().cpu().numpy().tobytes()).hexdigest(),'starts_sha256':hashlib.sha256(starts.detach().cpu().numpy().tobytes()).hexdigest(),'shape':[64,96,137],'origin_start':96,'origin_end_exclusive':160}
    return {'torch':torch,'model':model,'step':step,'input_signature':signature,'loss_scope':loss_scope,'optimizer_initialization':{'class':type(optimizer).__name__,'lr':optimizer.param_groups[0]['lr'],'betas':list(optimizer.param_groups[0]['betas']),'weight_decay':optimizer.param_groups[0]['weight_decay']},'parameters':sum(p.numel() for p in model.parameters())}
