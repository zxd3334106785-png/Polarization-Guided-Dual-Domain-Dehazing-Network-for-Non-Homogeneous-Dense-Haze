import torch.nn as nn
import torch.nn.functional as F

from model_flash_final import fusion_net


class final_net(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.dehazing_model = fusion_net(pretrained_path=pretrained_path)

    def forward(self, hazy):
        h, w = hazy.shape[-2:]
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            mode = 'reflect' if ph < h and pw < w else 'replicate'
            hazy = F.pad(hazy, (0, pw, 0, ph), mode=mode)
        return self.dehazing_model(hazy)[:, :, :h, :w]
