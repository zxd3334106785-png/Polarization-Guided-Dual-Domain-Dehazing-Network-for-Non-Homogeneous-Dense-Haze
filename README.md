# Polarization-Guided-Dual-Domain-Dehazing-Network-for-Non-Homogeneous-Dense-Haze

## Setup

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation ./DCNv4_op
```
## Data

```text
data/train/<scene>/youwu/{0,45,90}/001.png
data/train/<scene>/wuwu/{0,45,90}/001.png
data/val/<scene>/youwu/{0,45,90}/001.png
data/val/<scene>/wuwu/{0,45,90}/001.png
data/infer/<scene>/youwu/{0,45,90}/001.png
```

`youwu` contains hazy input images; `wuwu` (or `wuwu_gt`) contains haze-free ground-truth images.

Each brace denotes three separate angle folders. Use matching filenames and image sizes, read as 8-bit grayscale. Ground truth is required for training/validation and optional for inference. Keep splits separate by scene.

## Pretrained Weights

The pretrained model weights are available from the following links:

- **Google Drive:** [Download](https://drive.google.com/drive/folders/15_FeB-wIvzzrLTVqY89VdafoNfcxkhNm?usp=sharing)
- **Baidu Netdisk:** [Download](https://pan.baidu.com/s/1xN8VtJ3Apu1GaLDO_mqi2A?pwd=2pi4) (Access code: `2pi4`)

Download the pretrained weights and place them in the `weights/` directory.

## Train from Scratch

```bash
python train.py --from-scratch --train-dir data/train --val-dir data/val --output-dir outputs/train --epochs 2000 --batch-size 8 --lr 1e-4 --encoder-lr 1e-4 --min-lr 1e-6
```

All parameters, including the encoder, are randomly initialized. To use a pretrained encoder instead, add `--encoder-pretrained weights/encoder.pth`.

## Fine-tune

```bash
python train.py --weights weights/best_model.pth --train-dir data/train --val-dir data/val --output-dir outputs/finetune --epochs 100 --batch-size 2
```

## Inference

```bash
python generate.py --weights weights/best_model.pth --dataset data/infer --output-dir outputs/images
```
