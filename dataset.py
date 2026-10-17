import os
import random
import math
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class DehazeDataset(Dataset):
    def __init__(self, root_dir, patch_size=192, img_resize=None, is_train=True, mix_clean_prob=0.0, gt_dirname="auto", brighten_percent=0.0, require_gt=True):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.img_resize = img_resize
        self.is_train = is_train
        self.mix_clean_prob = mix_clean_prob
        self.gt_dirname = gt_dirname
        self.require_gt = require_gt
        self.brighten_percent = float(brighten_percent)
        if not math.isfinite(self.brighten_percent) or self.brighten_percent <= -100:
            raise ValueError('brighten_percent must be finite and greater than -100')
        if is_train and not require_gt:
            raise ValueError('Training requires ground truth')
        if gt_dirname not in ('auto', 'wuwu', 'wuwu_gt'):
            raise ValueError('gt_dirname must be auto, wuwu, or wuwu_gt')
        self.angles = ['0', '45', '90'] 
        self.samples = self._scan_files()
        print(f"Load {len(self.samples)} samples. Patch size: {self.patch_size}, Image resize: {self.img_resize}, Train: {self.is_train}, Mix clean prob: {self.mix_clean_prob}")

    def _scan_files(self):
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f'Dataset directory does not exist: {self.root_dir}')
        samples = []
        for root, dirs, files in os.walk(self.root_dir):
            if 'youwu' not in dirs:
                continue
            candidates = [d for d in ('wuwu', 'wuwu_gt') if d in dirs]
            if self.gt_dirname == 'auto':
                if len(candidates) > 1:
                    raise ValueError(f'Both GT folders exist in {root}; select --gt-dirname explicitly')
                gt_kind = candidates[0] if candidates else None
            else:
                gt_kind = self.gt_dirname if self.gt_dirname in dirs else None
            if self.require_gt and gt_kind is None:
                raise FileNotFoundError(f'Missing GT folder under {root}')
            kinds = ['youwu'] + ([gt_kind] if gt_kind else [])
            image_names = set()
            for kind in kinds:
                for angle in self.angles:
                    folder = os.path.join(root, kind, angle)
                    if not os.path.isdir(folder):
                        raise FileNotFoundError(f'Missing polarization directory: {folder}')
                    image_names.update(f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')))
            for img_name in sorted(image_names):
                for kind in kinds:
                    for angle in self.angles:
                        path = os.path.join(root, kind, angle, img_name)
                        if not os.path.isfile(path):
                            raise FileNotFoundError(f'Incomplete polarization sample: {path}')
                samples.append({'dir_root': root, 'filename': img_name, 'gt_kind': gt_kind})
        if not samples:
            raise ValueError(f'No paired PNG samples found under {self.root_dir}')
        return sorted(samples, key=lambda x: (x['dir_root'], x['filename']))

    def _load_3channel_polar(self, dir_root, sub_type, filename):
        tensors = []
        for angle in self.angles:
            img_path = os.path.join(dir_root, sub_type, angle, filename)
            with Image.open(img_path) as source:
                img = source.convert('L')
            if self.brighten_percent != 0:
                arr = np.array(img, dtype=np.float32)
                arr = np.clip(arr * (1 + self.brighten_percent / 100), 0, 255).round().astype(np.uint8)
                img = Image.fromarray(arr)
            if self.img_resize is not None:
                img = F.resize(img, (self.img_resize, self.img_resize), interpolation=InterpolationMode.BILINEAR)
            t = F.to_tensor(img) 
            tensors.append(t)
        if len({tuple(t.shape) for t in tensors}) != 1:
            raise ValueError(f'Polarization angle image sizes differ: {dir_root}/{sub_type}/{filename}')
        return torch.cat(tensors, dim=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        dir_root = sample_info['dir_root']
        fname = sample_info['filename']
        if sample_info['gt_kind'] is None:
            return self._load_3channel_polar(dir_root, 'youwu', fname), None
        gt_tensor = self._load_3channel_polar(dir_root, sample_info['gt_kind'], fname)
        use_clean_as_input = False
        if self.is_train and self.mix_clean_prob > 0:
            if random.random() < self.mix_clean_prob:
                use_clean_as_input = True
        if use_clean_as_input:
            hazy_tensor = gt_tensor.clone() 
        else:
            hazy_tensor = self._load_3channel_polar(dir_root, 'youwu', fname)
        if hazy_tensor.shape != gt_tensor.shape:
            raise ValueError(f'Hazy/GT sizes differ: {dir_root}/{fname}')
        if self.is_train and min(hazy_tensor.shape[-2:]) < self.patch_size:
            raise ValueError(f'Image smaller than crop {self.patch_size}: {dir_root}/{fname}')
        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(
                hazy_tensor, 
                output_size=(self.patch_size, self.patch_size)
            )
            hazy_tensor = F.crop(hazy_tensor, i, j, h, w)
            gt_tensor = F.crop(gt_tensor, i, j, h, w)
            if random.random() > 0.5:
                hazy_tensor = F.hflip(hazy_tensor)
                gt_tensor = F.hflip(gt_tensor)
            if random.random() > 0.5:
                hazy_tensor = F.vflip(hazy_tensor)
                gt_tensor = F.vflip(gt_tensor)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                hazy_tensor = F.rotate(hazy_tensor, angle)
                gt_tensor = F.rotate(gt_tensor, angle)
        return hazy_tensor, gt_tensor

