# Multi-Retinal-Disease-Detection

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


## Motivation
In medical imaging, obtaining large-scale, labeled datasets is often challenging due to privacy concerns, high annotation costs, and limited availability of expert knowledge. To effectively learn and boost performance on small-scale datasets, we leverage transfer learning
techniques which consist of models that are trained on large amounts of data.  
## Goal
Improve the performance of multi-label retinal image classification using transfer learning by fine-tuning models. 
## Task Overview
In this project, we address the problem of multi-label retinal disease detection, focusing on three major conditions: Diabetic Retinopathy (DR), Glaucoma (G), and Age-related Macular Degeneration (AMD). To tackle the challenge of limited annotated medical data, we adopt
transfer learning strategies, leveraging models pretrained on large-scale datasets and finetuning them for multi-label retinal image classification. The experiments are conducted on the **ODIR dataset**, which is divided into a training set of 800 images, a validation set of 200
images, an offsite test set of 300 images, and an onsite test set of 250 images, with all images standardized to a resolution of 256×256. The evaluation metrics include precision, recall, F-score of each disease and the average F-score over the three diseases.  
## Project stages
### Stage 1
Perform transfer learning with three different setups using EfficientNet and ResNet18 and evaluate their performances on both off-site test set and on-site test set:
1.  No fine-tuning: Evaluate directly on ODIR test set.
2.  Frozen backbone, fine-tuning classifier only: Backbone weights are fixed, classifier is updated on ODIR training set.
3.  Full fine-tuning: Both backbone and classifier are updated on ODIR training set.
---
### Stage 2
Evaluation of class-balancing techniques such as Focal Loss and Weighted BCE loss. 
1.  Focal Loss: A loss function designed to address class imbalance by downweighting easy examples and focusing training on hard, misclassified ones. 
2.  Class-Balanced Loss: Re-weight the BCE loss according to class frequency.
---
### Stage 3
Incorporation into the backbone and evaluation attention mechanisms:
1.  Squeeze-and-Excitation (SE)
2.  Multi-head Attention (MHA)
---
### Stage 4
Further performance development using:
1.  More powerful backbone such as Swin Transformer and Vision Transformer to improve the disease detection performance.
2.  GradCAM to analyze what features in the image are contributing the most and the least in the model's decision-making process, then use the attention map to guide the learning, thereby potentially improving the performance.
3.  Ensemble learning methods (Stacking, Boosting, Weighted Average, Max Voting, Bagging) and analyze whether the performance increases or not.
4.  VAE to generate new retinal images in order to augment the training set.
---
### Stage 5
Reporting the result.
## Timings
|Stage|Descripition|Deadline|
|-----|------------|--------|
|  1  |Evaluation of ResNet and EfficientNet bare backbones| December 7 |
|  2  |Class balancing| December 7 |
|  3  |Attention| December 14 |
|  4  |Transformer/GradCAM/Ensemble/Augmentation| December 21 |
|  5  |Report| December 31 |
