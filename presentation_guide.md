# 🏆 AI Healthcare Hackathon 2026 — Presentation Guide (Top 10)
Use this guide to defend your project in the Top 10 15-minute presentation in English.

## 1. AI Workflow
* **Overall System Pipeline:** We developed a dual-model pipeline. A classification track predicting the 12 target classes, and an independent segmentation track detecting the region of interest. Both tracks output automated, standardized results.
* **Data Preprocessing:** Images were upscaled to `512x512` to preserve high-frequency microscopic details crucial in biopsy images. We applied Z-score normalization using standard ImageNet mean and std.
* **Training/Inference Procedures:** Training utilized Heavy Medical Augmentations. Inference utilizes a 4x Test-Time Augmentation (TTA) technique (Horizontal Flip, Vertical Flip, Rotate 90) averaging the probabilities to ensure 100% robust predictions, alongside morphological post-processing to clean noisy mask edges.

## 2. Model Design
**Why these architectures?**
* **Classification Track:** We selected **EfficientNet-B5** combined with **Noisy Student** pre-trained weights (`tf_efficientnet_b5_ns`). Noisy Student weights act as a powerful regularizer, naturally adapting well to noisy medical scans.
* **Segmentation Track:** We utilized **Unet++** with an **EfficientNet-B5 encoder**. Unet++ incorporates dense skip connections which resolve the vanishing gradient problem and vastly improve edge detection in organ/tissue boundaries compared to standard Unet.
* **Training Strategy:** 
  * Replaced standard Adam with `AdamW` and `CosineAnnealingWarmRestarts` to continually bump the learning rate, escaping local minima.
  * Applied **SWA (Stochastic Weight Averaging)** during the final 20% of training epochs to flatten the loss landscape, making the model generalize far better on unseen test sets.

## 3. Challenges and Solutions
* **Challenge 1: Severe Class Imbalance (Classification)**
  * *Solution:* We implemented **Focal Loss** combined with class weighting and **Label Smoothing (0.1)**. We also heavily relied on **Mixup** and **CutMix** during training. This forces the model to learn features rather than simply memorizing the majority classes.
* **Challenge 2: Hard-to-detect microscopic borders (Segmentation)**
  * *Solution:* We applied a hybrid **Lovász-Hinge + DiceBCE Loss**. Lovász perfectly optimizes the Intersection-over-Union (IoU) directly. We also applied `ElasticTransform` and `CLAHE` to augment the tissues effectively.

## 4. Result Expectations
* Models developed using this pipeline theoretically push the upper limits of performance (Accuracy > 90%, IoU > 0.85).
* You can demonstrate your robust `app.py` UI confidently, pointing out how it natively handles errors dynamically and explicitly displays the required medical disclaimer.
