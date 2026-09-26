# Conditional DDPM for i-CLEVR

Generate 64 × 64 images from combinations of colored objects with a conditional denoising diffusion probabilistic model (DDPM). The model uses a conditional U-Net, self-attention, classifier-free guidance, and optional exponential moving average (EMA) weights. See the [lab report](LAB6_E94111253_report.pdf) for architecture details and experiments.

## Results

The report records classification accuracy of **0.8056** on `test.json` and **0.8452** on `new_test.json` with guidance scale 2.6. The repository contains 32 generated PNG images per split in [`images/test/`](images/test/) and [`images/new_test/`](images/new_test/). These are results from the original experiment; no generator checkpoint is included for immediate resampling.

## Files

```text
DL_Lab6/
├── README.md
├── requirements.txt
├── LAB6_E94111253_report.pdf
├── images/
│   ├── test/                 # 0.png to 31.png
│   └── new_test/             # 0.png to 31.png
├── file/
│   ├── objects.json         # 24 color-shape classes
│   ├── test.json             # 32 test conditions
│   ├── new_test.json         # 32 new test conditions
│   └── evaluator.py         # Instructor-provided evaluator definition
└── Source code/
    ├── main.py              # Train the DDPM
    ├── sample.py            # Generate images and grids
    ├── denoise_process.py   # Denoising process grid
    ├── dataloader.py        # Images and multi-label condition vectors
    ├── models.py            # Conditional U-Net
    ├── diffusion.py         # Diffusion training and sampling
    └── evaluator_wrapper.py # Accuracy for the two splits
```

The original training dataset and evaluator weights are **not redistributed**. To reproduce training and evaluation, obtain the lab-provided files through the course materials and put `train.json` and `checkpoint.pth` inside `file/`. Put the training images inside `iclevr/` at the repository root, with filenames matching `file/train.json`. The lab instructions say not to upload the dataset. The generator checkpoint `checkpoints/ddpm_final.pt` is also not included; train it locally or provide a compatible checkpoint.

## Setup

Run all commands **from the repository root** (`DL_Lab6/`). Script paths are relative to this directory.

```bash
python -m venv .venv
# Activate .venv in your shell, then:
python -m pip install -r requirements.txt
```

Install compatible PyTorch and TorchVision builds for your CPU or CUDA environment. The supplied evaluator calls `.cuda()` directly, so classifier evaluation requires CUDA. Training and sampling support `--cpu`, although sampling can be slow.

## Train

```bash
python "Source code/main.py" --resume "" --epochs 100 --lr 2e-4 --use_ema
```

The defaults are batch size 64, 1,000 diffusion steps, a linear noise schedule, MSE noise prediction, and 10% condition dropout. `--use_ema` enables EMA checkpoint weights. Continue from a saved checkpoint for the fine-tuning stage described in the report:

```bash
python "Source code/main.py" --resume checkpoints/ddpm_final.pt --epochs 200 --lr 1e-4 --use_ema
```

`--epochs` is the **total** epoch number. The script saves `checkpoints/ddpm_final.pt` and periodic `ddpm_epoch_*.pt` files. Starting with `--resume ""` prevents automatic loading of an existing default checkpoint.

## Generate images

Once the condition JSON files and a trained checkpoint are present:

```bash
python "Source code/sample.py" --checkpoint checkpoints/ddpm_final.pt --use_ema \
  --output_test_dir images/test --output_new_test_dir images/new_test \
  --grid_test_path images/test_grid.png \
  --grid_new_test_path images/new_test_grid.png
```

This overwrites existing numbered results; choose different output directories to retain them. Change classifier-free guidance using `--guidance_scale` (default 2.6). Omit `--use_ema` if your checkpoint has no EMA weights.

Generate a denoising sequence for `red sphere`, `cyan cylinder`, and `cyan cube`:

```bash
python "Source code/denoise_process.py" --checkpoint checkpoints/ddpm_final.pt \
  --output_path images/denoising_process.png
```

This script defaults to EMA weights, so its checkpoint should contain an `ema_state_dict`.

## Evaluate

Put the lab-provided `checkpoint.pth` in `file/`, then run on a CUDA-capable system:

```bash
python "Source code/evaluator_wrapper.py" \
  --test_image_dir images/test --new_test_image_dir images/new_test
```

The wrapper expects exactly `0.png` through `31.png` per split. It normalizes images and compares predictions with the corresponding condition JSON. The instructor's `file/evaluator.py` and weights are unmodified.
