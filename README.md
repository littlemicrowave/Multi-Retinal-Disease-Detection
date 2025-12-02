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
### Stage 2
Evaluation of class-balancing techniques such as Focal Loss and Weighted BCE loss. 
1.  Focal Loss: A loss function designed to address class imbalance by downweighting easy examples and focusing training on hard, misclassified ones. 
2.  Class-Balanced Loss: Re-weight the BCE loss according to class frequency.
### Stage 3
Incorporation into the backbone and evaluation attention mechanisms:
1.  Squeeze-and-Excitation (SE)
2.  Multi-head Attention (MHA)
### Stage 4
Further performance development using:
1.  More powerful backbone such as Swin Transformer and Vision Transformer to improve the disease detection performance.
2.  GradCAM to analyze what features in the image are contributing the most and the least in the model's decision-making process, then use the attention map to guide the learning, thereby potentially improving the performance.
3.  Ensemble learning methods (Stacking, Boosting, Weighted Average, Max Voting, Bagging) and analyze whether the performance increases or not.
4.  VAE to generate new retinal images in order to augment the training set.
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
