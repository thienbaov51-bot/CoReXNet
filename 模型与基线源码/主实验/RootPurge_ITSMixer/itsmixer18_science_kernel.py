"""Prospective exact-horizon ITSMixer remaining18 science core; never imports a framework on import.

The author model is a byte-identical MIT source copy. These small training and
pooled-reduction functions are independently implemented from the documented
mathematical policy. This module has no data loader, disk writer, test function,
research execution CLI, admission authority, or retry path. A future separately
reviewed production wrapper must admit a run before constructing its inputs.
"""
import math
from types import SimpleNamespace
from itsmixer18_protocol import require, type3_lr_after_epoch, replaces_checkpoint, source_model_for_horizon


def configure_framework(torch, np):
    """Future wrapper calls this after metadata/process/resource admission."""
    import random
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False


def build_model(author_module, config):
    """Author class constructors execute in their original order."""
    require(config['model_source_file'] == source_model_for_horizon(config['horizon']), 'exact per-horizon author dispatch')
    args = SimpleNamespace(seq_len=96, pred_len=config['horizon'], enc_in=config['enc_in'])
    return author_module.Model(args).float().to('cuda')


def build_optimizer_and_scheduler(torch, model, config):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'],
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False,
    )
    # Preserve the author's unconditional construction even for type3. In the
    # inherited scheduler semantics, construction initializes LR=0.00004 and
    # AdamW beta1=0.95. No explicit scheduler.step occurs during this recipe.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        steps_per_epoch=config['train_optimizer_steps_per_epoch'],
        pct_start=config['pct_start'], epochs=config['epochs'],
        max_lr=config['learning_rate'],
    )
    require(len(optimizer.param_groups) == 1, 'single original AdamW group')
    group = optimizer.param_groups[0]
    require(math.isclose(group['lr'], 0.00004, rel_tol=1e-12, abs_tol=1e-15), 'OneCycle constructor initial LR')
    require(tuple(group['betas']) == (0.95, 0.999), 'OneCycle constructor initial beta1')
    require(group['weight_decay'] == 0.01 and group['eps'] == 1e-8, 'original AdamW defaults')
    require(scheduler.last_epoch == 0, 'constructor initial step semantics')
    return optimizer, scheduler


def compound_loss(functional, forecast, target):
    # Exact author CustomLoss mathematics; args.loss='mse' is unused metadata.
    return 0.5 * functional.mse_loss(forecast, target) + 0.5 * functional.l1_loss(forecast, target)


def predict(model, x, channels, horizon):
    # The exact author forward ignores all three auxiliary arguments. No
    # synthetic calendar columns, extra channels, or decoder targets are made.
    forecast = model(x, None, None, None)
    require(tuple(forecast.shape) == (len(x), horizon, channels), 'exact full forecast shape')
    return forecast


def pooled_validation(torch, model, resident, config):
    model.eval()
    sums = torch.zeros(3, dtype=torch.float64, device='cuda')
    channel_sse = torch.zeros(config['channels'], dtype=torch.float64, device='cuda')
    seen_windows = 0
    with torch.inference_mode():
        for starts in resident.validation_target_starts.split(config['batch_size']):
            x, y = resident.batch(starts)
            forecast = predict(model, x, config['channels'], config['horizon'])
            require(bool(torch.isfinite(forecast).all()), 'finite forecasts')
            require(tuple(y.shape) == tuple(forecast.shape), 'full target identity')
            # External standardization is retained. RevIN denorm is already
            # part of author Model.forward. There is no physical-unit inverse.
            error = forecast.float() - y
            sums[0] += error.square().sum(dtype=torch.float64)
            sums[1] += error.abs().sum(dtype=torch.float64)
            sums[2] += error.numel()
            channel_sse += error.square().sum((0, 1), dtype=torch.float64)
            seen_windows += len(starts)
    sse, sae, elements_float = sums.cpu().tolist()
    elements = int(elements_float)
    require(seen_windows == config['validation_windows'], 'all chronological VAL windows including tail')
    require(elements == config['validation_elements'], 'all H*C elements without masks')
    mse, mae = sse / elements, sae / elements
    require(math.isfinite(mse) and math.isfinite(mae), 'finite pooled errors')
    return {
        'mse': mse, 'mae': mae, 'windows': seen_windows, 'elements': elements,
        'squared_error_sum': sse, 'absolute_error_sum': sae,
        'channel_mse': (channel_sse / (seen_windows * config['horizon'])).cpu().tolist(),
        'checkpoint_criterion': 0.5 * (mse + mae),
        'metric_space': config['metric_space'],
        'masking': False, 'validation_drop_last': False,
    }


class ResidentWindows:
    """Future guard supplies the already verified standardized full prefix."""
    def __init__(self, torch, standardized_values, config):
        require(tuple(standardized_values.shape) == (config['prefix_rows'], config['channels']), 'exact original prefix shape')
        require(standardized_values.dtype == torch.float32, 'FP32 full prefix')
        require(standardized_values.device.type == 'cuda', 'full prefix resident on admitted GPU')
        self.values = standardized_values
        self.train_target_starts = list(range(96, config['train_end'] - config['horizon'] + 1))
        self.validation_target_starts = torch.arange(config['train_end'], config['validation_end'] - config['horizon'] + 1, device='cuda')
        self.xoff = torch.arange(-96, 0, device='cuda')
        self.yoff = torch.arange(config['horizon'], device='cuda')
        require(len(self.train_target_starts) == config['train_windows_available'], 'complete legal TRAIN universe')
        require(len(self.validation_target_starts) == config['validation_windows'], 'complete legal VAL universe')

    def batch(self, starts):
        starts = starts.to(device='cuda')
        return self.values[starts[:, None] + self.xoff], self.values[starts[:, None] + self.yoff]


def trainval_science_core(torch, model, resident, config):
    """No IO or authority. Future wrapper owns all five native output files.

The initial B256 full-loader optimizer-steps and actual memory must be measured
in future synthetic resource QA. This function is never invoked in source QA.
"""
    optimizer, scheduler = build_optimizer_and_scheduler(torch, model, config)
    loader = torch.utils.data.DataLoader(
        resident.train_target_starts, batch_size=256, shuffle=True,
        drop_last=True, num_workers=0,
    )
    require(len(loader) == config['train_optimizer_steps_per_epoch'], 'full shuffled author B256 loader')
    best_criterion, best_state, best_epoch = float('inf'), None, None
    non_improving_epochs, total_steps = 0, 0
    history = []
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        train_batch_losses, epoch_steps, epoch_windows = [], 0, 0
        lr_used = float(optimizer.param_groups[0]['lr'])
        beta1_used = float(optimizer.param_groups[0]['betas'][0])
        for starts in loader:
            require(len(starts) == 256, 'original full training batch')
            optimizer.zero_grad()
            x, y = resident.batch(starts)
            forecast = predict(model, x, config['channels'], config['horizon'])
            require(bool(torch.isfinite(forecast).all()), 'finite training forecast')
            loss = compound_loss(torch.nn.functional, forecast, y)
            require(bool(torch.isfinite(loss)), 'finite composite training criterion')
            train_batch_losses.append(float(loss.detach().cpu()))
            loss.backward()
            optimizer.step()
            total_steps += 1
            epoch_steps += 1
            epoch_windows += len(starts)
        require(epoch_steps == config['train_optimizer_steps_per_epoch'], 'no optimizer budget truncation')
        require(epoch_windows == config['train_windows_used_per_epoch'], 'exact author drop-tail accounting')
        score = pooled_validation(torch, model, resident, config)
        selected = replaces_checkpoint(score['checkpoint_criterion'], best_criterion)
        if selected:
            best_criterion = score['checkpoint_criterion']
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            non_improving_epochs = 0
        else:
            non_improving_epochs += 1
        history.append({
            'epoch': epoch, 'steps': total_steps, 'optimizer_steps_this_epoch': epoch_steps,
            'train_loss': sum(train_batch_losses) / len(train_batch_losses),
            'train_criterion': config['criterion'],
            'train_windows_available': config['train_windows_available'],
            'train_windows_used': epoch_windows,
            'train_windows_dropped': config['train_windows_dropped_per_epoch'],
            'lr_used': lr_used, 'beta1_used': beta1_used,
            'validation': score, 'checkpoint_updated': selected,
        })
        # Patience 100 cannot shorten a full corresponding15/15/20 epoch budget; keep its original
        # comparison and order instead of inventing a different stopping rule.
        if non_improving_epochs >= config['patience']:
            raise RuntimeError('unexpected early stop within frozen per-horizon epoch budget')
        next_lr = type3_lr_after_epoch(epoch, config['horizon'])
        for group in optimizer.param_groups:
            group['lr'] = next_lr
        require(scheduler.last_epoch == 0, 'type3 must not step OneCycleLR')
        history[-1]['lr_after_epoch_update'] = next_lr
    require(total_steps == config['expected_optimizer_steps_at_full_epoch_budget'], 'full corresponding per-horizon optimizer budget')
    require(best_state is not None and best_epoch is not None, 'finite author-criterion checkpoint')
    model.load_state_dict(best_state)
    final_score = pooled_validation(torch, model, resident, config)
    require(final_score['checkpoint_criterion'] == best_criterion, 'selected best compound criterion replay')
    return {
        'best_state': best_state, 'history': history, 'validation': final_score,
        'best_epoch': best_epoch,
        'best_step': best_epoch * config['train_optimizer_steps_per_epoch'],
        'epochs': config['epochs'], 'steps': total_steps, 'checkpoint_criterion': best_criterion,
        'test_windows_constructed': False, 'test_scores_accessed': False,
    }


def main():
    print('PENDING_ITSMIXER_REMAINING18_SCIENCE_SOURCE_ONLY_NO_IO_NO_FRAMEWORK_NO_RUN')


if __name__ == '__main__':
    main()
