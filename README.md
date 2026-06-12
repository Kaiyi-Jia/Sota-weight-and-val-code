# BoostingFER: Boosting Representation Learning for High-Level Semantic Information in Facial Expression Recognition

This repository contains the official PyTorch implementation for our paper **"Boosting Representation Learning for High-Level Semantic Information in Facial Expression Recognition"**.

BoostingFER is an efficient and robust Facial Expression Recognition (FER) framework built upon the **CLIP-L/14** vision-language model. It achieves state-of-the-art performance by leveraging high-level semantic alignment across modalities, introducing less than 2% additional computational overhead (FLOPs).

## ✨ Key Features

- **EHSE (Expression High-level Semantic Encoding):** Generates region-specific (ROIs) and global text semantic anchors using FACS cognitive priors. Computes embeddings offline to save 100% of text-encoder overhead during inference.
- **EHSA (Expression High-Level Semantic Aligning):** Implements hierarchical semantic guidance (HSG), region-aware dual-alignment (DAR) for class-balancing, and Dual-Granularity Feature Alignment (DGFA) using in-batch negative sampling.
- **EKFF (Expression Key-core Features Focusing):** Utilizes multi-head cross-attention (MHCA) with Adaptive Feature Screening (AFS) and Straight-Through Estimator (STE) to focus on the most discriminative facial regions.

## 📂 Repository Structure

```text
├── config.py          # Centralized configuration (hyperparameters, dataset paths, thresholds)
├── dataset.py         # Robust data loader supporting RAF-DB, FERPlus, and AffectNet (7/8 classes)
├── model.py           # Core BoostingFER architecture wrapper
├── EHSE.py            # Expression High-level Semantic Encoding module
├── EHSA.py            # Expression High-Level Semantic Aligning module
├── EKFF.py            # Expression Key-core Features Focusing module
├── utils.py           # Utilities (seed setting, class weights, lean checkpoint saving/loading)
├── train.py           # Main training script with TensorBoard logging and early stopping target
├── evaluate.py        # Evaluation script for computing final accuracy
└── README.md          # This file
```
## ⚙️ Environment Setup
```text
Requirements:

Python >= 3.8

PyTorch >= 1.10 (CUDA enabled)

torchvision

pandas, tqdm, Pillow

OpenAI CLIP

Install the required packages:
pip install torch torchvision pandas tqdm
pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)
```
