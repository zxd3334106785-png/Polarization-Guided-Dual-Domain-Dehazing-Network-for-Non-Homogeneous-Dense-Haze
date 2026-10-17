import torch


def compute_S0(tensor, clip=True):
    s0 = tensor[:, 0:1] + tensor[:, 2:3]
    return s0.clamp(0, 1) if clip else s0


def compute_S0_3channel(tensor, clip=True):
    """Construct [S0, S0, S0], not the original [I0, I45, I90] tensor."""
    return compute_S0(tensor, clip=clip).repeat(1, 3, 1, 1)


def psnr_per_image(pred, target, data_range=1.0):
    mse = (pred - target).square().flatten(1).mean(1)
    return 10 * torch.log10(data_range ** 2 / mse)


def ssim_per_image(pred, target, data_range=1.0):
    """Original measure_model_new/genimg3 Kornia SSIM convention (same padding)."""
    from kornia.metrics import ssim
    if min(pred.shape[-2:]) < 11:
        raise ValueError('SSIM requires images at least 11x11 pixels.')
    return ssim(pred, target, window_size=11, max_val=data_range).flatten(1).mean(1)
