import argparse
import csv
import json
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from checkpoints import load_weights
from dataset import DehazeDataset
from metrics import compute_S0, compute_S0_3channel, psnr_per_image, ssim_per_image
from model import final_net


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path(__file__).resolve().parent / 'weights/best_model.pth')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gt-dirname', choices=('auto', 'wuwu', 'wuwu_gt'), default='auto')
    parser.add_argument('--brighten-percent', type=float, default=0.0,
                        help='0: original data; 15: genimg3-style uint8 brightening of input AND GT')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('The supplied DCNv4 requires CUDA')
    if not args.weights.is_file():
        parser.error(f'Missing checkpoint: {args.weights}')
    output, root = args.output_dir.resolve(), args.dataset.resolve()
    if output == root or root in output.parents or output in root.parents:
        parser.error('Output and input roots must be separate and non-nested')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Output directory must be new or empty')
    dataset = DehazeDataset(str(root), is_train=False, require_gt=False,
                            gt_dirname=args.gt_dirname, brighten_percent=args.brighten_percent)
    model = final_net(pretrained_path=None)
    load_weights(model, args.weights)
    model.to(device).eval()
    output.mkdir(parents=True, exist_ok=True)

    def save(tensor, category, relative):
        path = output / category / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(tensor, str(path))

    records = []
    for index in tqdm(range(len(dataset)), desc='Generating'):
        hazy, gt = dataset[index]
        sample = dataset.samples[index]
        # Keep scene paths AND original extensions to prevent output collisions.
        relative = Path(sample['dir_root']).relative_to(root) / (sample['filename'] + '.png')
        hazy = hazy.unsqueeze(0).to(device)
        prediction = model(hazy).clamp(0, 1)
        save(compute_S0(hazy)[0], 'hazy_s0', relative)
        save(compute_S0(prediction)[0], 'dehazed_s0', relative)
        for channel, angle in enumerate(('0', '45', '90')):
            save(prediction[0, channel:channel+1], 'polar/' + angle, relative)
        row = {'sample': str(Path(sample['dir_root']).relative_to(root) / sample['filename']),
               'height': prediction.shape[-2], 'width': prediction.shape[-1], 'psnr': '', 'ssim': ''}
        if gt is not None:
            gt = gt.unsqueeze(0).to(device)
            save(compute_S0(gt)[0], 'gt_s0', relative)
            pred3, gt3 = compute_S0_3channel(prediction), compute_S0_3channel(gt)
            row['psnr'] = psnr_per_image(pred3, gt3).item()
            row['ssim'] = ssim_per_image(pred3, gt3).item()
        records.append(row)
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample', 'height', 'width', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(records)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings['metrics'] = 'clipped S0 repeated to 3 channels, range=1, Kornia SSIM window=11 same padding; before PNG quantization'
    (output / 'settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    scored = [row for row in records if row['psnr'] != '']
    if scored:
        print('Average PSNR:', sum(row['psnr'] for row in scored) / len(scored))
        print('Average SSIM:', sum(row['ssim'] for row in scored) / len(scored))
    print(f'Saved {len(records)} samples to {output}')


if __name__ == '__main__':
    main()
