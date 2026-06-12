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
## 📊 Data Preparation
The dataset.py script is highly robust and automatically detects your dataset structures. Place your datasets in the data_root directory defined in config.py.

Supported formats include the official _label.txt format or standard folder-per-class format:
```text
data_root/
├── RAF-DB/
│   ├── train/ (0/ 1/ 2/ ...)
│   └── val/   (0/ 1/ 2/ ...)
├── FERPlus/
│   ├── train/ (0/ 1/ 2/ ...)
│   └── val/   (0/ 1/ 2/ ...)
└── AffectNet8/
    ├── train/ (0/ 1/ 2/ ...)
    └── val/   (0/ 1/ 2/ ...)
```
## 🚀 Training
To train the BoostingFER model from scratch, adjust the hyperparameters in config.py (such as batch_size, accumulation_steps depending on your VRAM) and run:
```text
python train.py
```
Training Highlights:

- **Semantic anchors are automatically pre-computed and cached before training starts.**

- **Model checkpoints (Lean versions) are saved in the configured save_dir.**

- **TensorBoard logs are recorded in the log_dir.**
## 📦 Pre-trained Models

The pre-trained model weights have been uploaded to the GitHub Releases page. You can download them directly from the link below:

- [📥 Download Best Model Weights (best_model.pth)](https://github.com/Kaiyi-Jia/BoostingFER/archive/refs/tags/weight.zip)

*After downloading, please place the `.pth` file in your specified `save_dir` (e.g., `./checkpoints/`) for evaluation.*
## 🧪 Evaluation
To evaluate a trained model, open test.py, set the WEIGHT_PATH to your saved .pth file, and run:
```text
python evaluate.py
```
Note: During inference, the EHSA alignment constraints and the CLIP text encoder are completely discarded, ensuring ultra-fast execution.
## 📝 Citation

Our paper is currently under review. If you find this repository or our code useful for your research, please consider giving this repository a star ⭐! 

The formal citation details will be updated upon acceptance. For now, you can cite our project as follows:

```bibtex
@misc{BoostingFER2026,
  title={Boosting Representation Learning for High-Level Semantic Information in Facial Expression Recognition},
  author={Min},
  note={Under Review},
  year={2026}
}
