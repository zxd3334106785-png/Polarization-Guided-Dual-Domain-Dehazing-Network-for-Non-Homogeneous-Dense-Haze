import argparse
import json
import math
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpoints import load_weights
from dataset import DehazeDataset
from metrics import compute_S0_3channel, psnr_per_image, ssim_per_image
from model import final_net
from training_losses import TotalLoss


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr_sum = ssim_sum = 0.0
    count = 0
    for hazy, target in tqdm(loader, desc='Validation', leave=False):
        hazy = hazy.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        prediction = model(hazy).clamp(0, 1)
        pred_s0 = compute_S0_3channel(prediction)
        target_s0 = compute_S0_3channel(target)
        psnr = psnr_per_image(pred_s0, target_s0)
        ssim = ssim_per_image(pred_s0, target_s0)
        if torch.isnan(psnr).any() or torch.isneginf(psnr).any() or not torch.isfinite(ssim).all():
            raise FloatingPointError('Invalid validation metrics')
        psnr_sum += psnr.sum().item()
        ssim_sum += ssim.sum().item()
        count += hazy.shape[0]
    if count == 0:
        raise ValueError('Validation dataset is empty')
    return {'psnr': psnr_sum / count, 'ssim': ssim_sum / count}


def write_log(path, row):
    # Preserve infinite PSNR (perfect reconstruction) as valid JSON strings.
    row = {k: str(v) if isinstance(v, float) and not math.isfinite(v) else v
           for k, v in row.items()}
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path(__file__).resolve().parent / 'weights/best_model.pth')
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--val-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('weights/finetune'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--patch-size', type=int, default=384)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--encoder-lr', type=float, default=1e-6)
    parser.add_argument('--min-lr', type=float, default=1e-7)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--val-interval', type=int, default=1)
    parser.add_argument('--freeze-encoder', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--from-scratch', action='store_true', help='Initialize a new model instead of loading full weights')
    parser.add_argument('--encoder-pretrained', type=Path, default=None)
    parser.add_argument('--gt-dirname', choices=('auto', 'wuwu', 'wuwu_gt'), default='auto')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.val_interval) < 1 or args.workers < 0:
        parser.error('epochs, batch-size, val-interval must be positive; workers >= 0')
    if args.patch_size < 32 or args.patch_size % 32:
        parser.error('patch-size must be a positive multiple of 32')
    values = (args.lr, args.encoder_lr, args.min_lr, args.weight_decay, args.grad_clip)
    if not all(math.isfinite(v) for v in values):
        parser.error('Optimizer settings must be finite')
    if min(args.lr, args.encoder_lr, args.grad_clip) <= 0 or min(args.min_lr, args.weight_decay) < 0:
        parser.error('Invalid learning rate, weight decay or gradient clip')
    if args.min_lr > min(args.lr, args.encoder_lr):
        parser.error('min-lr must not exceed either initial learning rate')
    if not args.from_scratch and not args.weights.is_file():
        parser.error(f'Weights not found: {args.weights}')
    if args.encoder_pretrained is not None and (not args.from_scratch or not args.encoder_pretrained.is_file()):
        parser.error('encoder-pretrained requires from-scratch and an existing encoder checkpoint')
    train_root, val_root = args.train_dir.resolve(), args.val_dir.resolve()
    if train_root == val_root or train_root in val_root.parents or val_root in train_root.parents:
        parser.error('Train and validation directories must be separate, non-nested roots')
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('output-dir must be new or empty; existing runs and source weights are never overwritten')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('The supplied DCNv4 model requires a working CUDA device')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Seeds aid repeatability; custom CUDA kernels may still be nondeterministic.
    torch.backends.cudnn.benchmark = False
    train_set = DehazeDataset(str(train_root), patch_size=args.patch_size, is_train=True, gt_dirname=args.gt_dirname)
    val_set = DehazeDataset(str(val_root), is_train=False, gt_dirname=args.gt_dirname)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)

    model = final_net(pretrained_path=args.encoder_pretrained if args.from_scratch else None)
    if not args.from_scratch:
        load_weights(model, args.weights)  # Strict full-model loading.
    model = model.to(device)
    spatial_branch = getattr(model.dehazing_model, 'knowledge_adaptation_branch', None)
    encoder = spatial_branch.encoder if spatial_branch is not None else None
    if args.freeze_encoder and encoder is None:
        parser.error('Model encoder was not found')
    if args.freeze_encoder:
        encoder.requires_grad_(False)
    encoder_params, other_params = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            target = encoder_params if name.startswith('dehazing_model.knowledge_adaptation_branch.encoder.') else other_params
            target.append(parameter)
    groups = [{'params': other_params, 'lr': args.lr, 'name': 'restoration'}]
    if encoder_params:
        groups.append({'params': encoder_params, 'lr': args.encoder_lr, 'name': 'encoder'})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)
    criterion = TotalLoss(device).to(device)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['experiment_protocol'] = 'from_scratch' if args.from_scratch else 'checkpoint_conditioned_finetuning'
    if not args.from_scratch:
        import hashlib
        with args.weights.open('rb') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        config['initial_checkpoint_sha256'] = digest.hexdigest()
    config['metrics'] = 'S0=clamp(I0+I90,0,1), repeated to 3 channels; range=1; Kornia SSIM same padding 11x11'

    # Establish the baseline on THIS validation set instead of reusing saved PSNR.
    baseline = validate(model, val_loader, device)
    best_psnr, best_epoch = baseline['psnr'], 0
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')

    def save_best(epoch, scores):
        torch.save({'epoch': epoch, 'PSNR': scores['psnr'], 'SSIM': scores['ssim'],
                    'state_dict': model.state_dict(), 'config': config}, output / 'best_model.pth')

    save_best(0, baseline)
    write_log(output / 'history.jsonl', {'epoch': 0, 'val_psnr': baseline['psnr'], 'val_ssim': baseline['ssim']})
    print(f'Baseline: PSNR={best_psnr:.4f}, SSIM={baseline["ssim"]:.6f}')
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_encoder:
            encoder.eval()  # Also disable encoder stochastic depth/dropout.
        total_loss, samples = 0.0, 0
        term_sums = {}
        learning_rates = {group['name'] + '_lr': group['lr'] for group in optimizer.param_groups}
        for hazy, target in tqdm(train_loader, desc=f'Fine-tune {epoch}/{args.epochs}'):
            hazy, target = hazy.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(hazy)
            loss, loss_terms = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * hazy.shape[0]
            for name, value in loss_terms.items():
                term_sums[name] = term_sums.get(name, 0.0) + value * hazy.shape[0]
            samples += hazy.shape[0]
        scheduler.step()
        row = {'epoch': epoch, 'train_loss': total_loss / samples, **learning_rates}
        row.update({'train_' + name: value / samples for name, value in term_sums.items()})
        scores = None
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            scores = validate(model, val_loader, device)
            row.update(val_psnr=scores['psnr'], val_ssim=scores['ssim'])
            if scores['psnr'] > best_psnr:
                best_psnr, best_epoch = scores['psnr'], epoch
                save_best(epoch, scores)
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'best_psnr': best_psnr, 'best_epoch': best_epoch,
                    'validation': scores, 'config': config}, output / 'last_model.pth')
        write_log(output / 'history.jsonl', row)
        print(row)
    print(f'Best validation PSNR: {best_psnr:.4f}, epoch={best_epoch}; {output / "best_model.pth"}')


if __name__ == '__main__':
    main()
