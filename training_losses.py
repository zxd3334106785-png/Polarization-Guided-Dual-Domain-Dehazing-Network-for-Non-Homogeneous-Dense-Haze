import math
import torch
import torch.nn as nn
from loss.losses import L1Loss, SSIM
from metrics import compute_S0_3channel


class PolarizationLoss(nn.Module):
    def __init__(self, phi_d=1.0, phi_a=0.05, phi_s=0.5,
                 mask_dark_thresh=0.02, mask_dolp_thresh=0.05, eps=1e-4):
        super().__init__()
        self.phi_d = phi_d
        self.phi_a = phi_a
        self.phi_s = phi_s
        self.mask_dark_thresh = mask_dark_thresh
        self.mask_dolp_thresh = mask_dolp_thresh
        self.eps = eps

    def charbonnier_loss(self, pred, target, eps=1e-4):
        return torch.sqrt((pred - target) ** 2 + eps ** 2)

    def _compute_stokes(self, I):
        I = torch.clamp(I, min=0.0, max=1.0)
        I0 = I[:, 0:1, :, :]
        I45 = I[:, 1:2, :, :]
        I90 = I[:, 2:3, :, :]
        S0 = I0 + I90
        S1 = I0 - I90
        S2 = 2 * I45 - I0 - I90
        return S0, S1, S2

    def _compute_dolp_aop(self, S0, S1, S2):
        S0_safe = torch.clamp(S0, min=self.eps)
        numerator = torch.sqrt(S1 ** 2 + S2 ** 2 + self.eps ** 2)
        dolp = numerator / S0_safe
        dolp = torch.clamp(dolp, 0.0, 1.0)

        zero_mask = (torch.abs(S1) < self.eps) & (torch.abs(S2) < self.eps)
        S1_safe = torch.where(zero_mask, torch.ones_like(S1), S1)
        S2_safe = torch.where(zero_mask, torch.zeros_like(S2), S2)
        aop = 0.5 * torch.atan2(S2_safe, S1_safe)
        
        return dolp, aop

    def forward(self, out_pol, tg_pol):
        S0_out, S1_out, S2_out = self._compute_stokes(out_pol)
        S0_tg, S1_tg, S2_tg = self._compute_stokes(tg_pol)

        dolp_out, aop_out = self._compute_dolp_aop(S0_out, S1_out, S2_out)
        dolp_tg, aop_tg = self._compute_dolp_aop(S0_tg, S1_tg, S2_tg)

        loss_dolp_raw = self.charbonnier_loss(dolp_out, dolp_tg, self.eps)
        loss_dolp = loss_dolp_raw.mean()

        angle_diff = torch.abs(aop_out - aop_tg)
        circular_diff = torch.min(angle_diff, math.pi - angle_diff)
        loss_aop_raw = self.charbonnier_loss(circular_diff, torch.zeros_like(circular_diff), self.eps)

        valid_mask = (S0_tg > self.mask_dark_thresh) & (dolp_tg > self.mask_dolp_thresh)
        loss_aop_masked = loss_aop_raw * valid_mask.float()

        mask_sum = valid_mask.sum()
        loss_aop = loss_aop_masked.sum() / torch.clamp(mask_sum, min=1.0)

        total_polar_loss = (
                self.phi_d * loss_dolp +
                self.phi_a * loss_aop
        )
        return total_polar_loss

class TotalLoss(nn.Module):
    def __init__(self, device):
        super(TotalLoss, self).__init__()
        self.L1_weight = 1.0
        self.D_weight = 1.0
        self.S0_weight = 1.0
        self.Polar_weight = 0.002 #新增

        self.L1_loss = L1Loss(loss_weight=self.L1_weight, reduction='mean').to(device)
        self.D_loss = SSIM(weight=self.D_weight).to(device)
        self.Polar_loss = PolarizationLoss(phi_d=1.0, phi_a=0.05, phi_s=0.5,
                 mask_dark_thresh=0.00, mask_dolp_thresh=0.00, eps=1e-4).to(device)

    def forward(self, out_pol, gt_pol):
        loss_dict = {}

        loss_pol_l1 = self.L1_loss(out_pol, gt_pol)
        loss_pol_polar = self.Polar_loss(out_pol, gt_pol)

        out_S0 = compute_S0_3channel(out_pol)
        gt_S0 = compute_S0_3channel(gt_pol)
        loss_S0_l1 = nn.functional.l1_loss(out_S0, gt_S0)
        loss_S0_ssim = self.D_loss(out_S0, gt_S0)



        loss = (loss_pol_l1 +
                self.S0_weight * (loss_S0_l1 + 0.1*loss_S0_ssim) +
                self.Polar_weight * loss_pol_polar)


        loss_dict['l1'] = loss_pol_l1.item()
        loss_dict['S0_l1'] = self.S0_weight * loss_S0_l1.item()
        loss_dict['S0_ssim'] = self.S0_weight * 0.1*loss_S0_ssim.item()
        loss_dict['Polar'] = self.Polar_weight * loss_pol_polar.item()
        loss_dict['total'] = loss.item()

        return loss, loss_dict


