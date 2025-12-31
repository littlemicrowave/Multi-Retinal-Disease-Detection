# Multi-Retinal-Disease-Detection

Code for multi-label retinal disease detection and related experiments (CNN, ViT/Swin, GAN/VAE, Grad-CAM).

## Project structure
- `redd_task*.py`: runnable scripts for each task and model variant.
- `utils/`: shared dataset, model blocks, training and evaluation utilities.
- `images/`: dataset folders (train/val/offsite_test/onsite_test).
- `train.csv`, `val.csv`, `offsite_test.csv`, `onsite_test_submission.csv`: labels and test lists.
- `pretrained_backbone/`: pretrained backbone checkpoints (e.g., ResNet).
- `trained_models/`: saved checkpoints for tasks 1-3.
- `task1/`, `task2/`, `task3/`, `task4/`: outputs, reports, submissions, visualizations and models for task 4.

Swin is in ZIP archive, due to the GitHub limitations, so if you want to run it, unzip to task4, if model file don't have "classifer" or "head" in name, it is fully tuned one. We suggest to use our module wrappers for models for successeful loading. Scripts for all tasks can have commented code, for instance for classifer head tuning, uncomment, if you want to retune.

## Data layout
Expected paths used by the scripts (see `utils/train_eval.py`):
- `images/train/`
- `images/val/`
- `images/offsite_test/`
- `images/onsite_test/`
- `train.csv`
- `val.csv`
- `offsite_test.csv`
- `onsite_test_submission.csv`
- `pretrained_backbone/ckpt_resnet18_ep50.pt`

CSV format is assumed as: `image_name, D, G, A` (multi-label targets in columns 2+).

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install it from the official site first, then run `pip install -r requirements.txt`.

## How to run
Each task script can be executed directly. Examples:
```bash
python redd_task1_resnet.py
python redd_task1_efficientnet.py
python redd_task2_resnet.py
python redd_task2_efficientnet.py
python redd_task3_resnet.py
python redd_task3_efficientnet.py
python redd_task4_vit.py
python redd_task4_swin.py
python redd_task4_vae.py
python redd_task4_dcgan.py
python redd_task4_grad_cam.py
python redd_task4_gradcam_triplet_loss.py
```

Scripts automatically use GPU if available (`cuda`), otherwise fall back to CPU.

## Outputs
Depending on the script, outputs include:
- Metrics and reports saved under `task1/`, `task2/`, `task3/`, `task4/`.
- Submission CSVs saved under task folders.
- Checkpoints saved under `trained_models/`.
