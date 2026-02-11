
* Overleaf document: <https://www.overleaf.com/5873327455zvrhvjcsjhkj#c0eac7>


# 🧬 AN2DL – Challenge 2: Molecular Subtype Classification (WSI)

Deep learning pipeline for **molecular subtype classification** of histopathology Whole Slide Images (WSIs) at low magnification. Target classes: **Luminal A**, **Luminal B**, **HER2(+)**, **Triple Negative**. 

## 👥 Team
**ANeuronInFour**  


## 🎯 Goal
Build a robust model to classify WSIs into one of the four molecular subtypes, addressing stain variability, noise, limited data, and class imbalance through ROI-aware patch extraction and transfer learning. 

## 📦 Dataset
- **1168 Image/Mask pairs** for training (labels in `train_labels.csv`), plus a **test set without labels**. 
- **Binary masks** highlight regions of interest (tumor/abnormal tissue) to guide ROI extraction.

### 🧹 Data cleaning
A total of **110 images** were removed from the training set:
- **50** affected by *green stain* and duplicates in the provided dataset
- **60** considered unsuitable as training examples

## ⚠️ Key challenges
- Limited data and noisy samples
- High-dimensional variability typical of WSIs
- Need to focus on tumor tissue (ROI)
- Strong stain inconsistency across samples
- Significant class imbalance

## 🧠 Method
### 🧩 1) ROI-aware grid tiling
WSIs are split into **224×224** patches with **30% overlap**.
- Keep patches intersecting the mask by at least **2%**
- Non-tissue/empty areas are filled with **white** to standardize background

### 🎨 2) Two-step normalization
1. **Macenko stain normalization** using reference HE vectors and maxC statistics estimated from 100 sampled WSIs (mask applied to exclude non-tumor/background during estimation). 
2. **Statistical normalization**: channel-wise mean/std recomputed from a random 20% subset of Macenko-normalized data. 

### 🏋️ 3) Training strategy
- **Transfer learning + fine-tuning** with **EfficientNet** backbone. 
- **6-fold stratified cross-validation** to preserve class proportions. 
- Class imbalance handled via **Focal Loss** + **dynamic per-fold class weights**. 
- Optimizer: **Lion** (better performance than AdamW in our tests). 
- Scheduler: **ReduceLROnPlateau** (more consistent than CosineAnnealing in our experiments). 

### 🧪 4) Data augmentation
- Spatial invariance transforms (random resize/crop, H/V flips, rotations)
- **Elastic transformations** to emulate realistic tissue deformations
- Photometric transforms (ColorJitter on brightness/contrast/saturation/hue, mild Gaussian blur)
- **MixUp** during fine-tuning (α = 0.2) 

## 🧰 Additional experiments (beyond the final baseline)
- **One-vs-Rest stacking** (4 binary models) + **XGBoost** ensemble: not better than direct K-fold ensembling. 
- **Second-stage XGBoost** trained on K-fold model probabilities (soft-voting features): no significant improvement. 

## 📊 Results (summary)
Among the evaluated alternatives (EfficientNet, ResNet, PhiCoN, UniMahmood, MobileNet), **EfficientNet** achieved the best inference-time F1 in the final configuration used for the ensemble submission.

## 🔍 Interpretability
We used **Grad-CAM** to verify that the model attends to biologically relevant regions within extracted patches, supporting the ROI tiling approach.

## 🚀 How to run (template)
This section is a standard layout—adapt script names/paths to your repository.

1) Install dependencies  
- `pip install -r requirements.txt`

2) Preprocess (cleaning + tiling + normalization)  
- Example: `python scripts/preprocess.py --data <DATASET_PATH> --out <OUTPUT_PATH>`

3) Train (6-fold CV)  
- Example: `python scripts/train_kfold.py --config configs/effnet.yaml`

4) Inference + submission  
- Example: `python scripts/infer.py --weights <WEIGHTS_DIR> --test <TEST_PATH> --out submission.csv`

## 🗂️ Suggested repository layout
- `configs/` training/inference configuration (folds, LR, augmentations, etc.)
- `scripts/` entry points (preprocess, train, infer)
- `src/`
  - `data/` dataset loading, tiling, transforms
  - `models/` EfficientNet + head, Focal Loss, optimizer/scheduler wrappers
  - `utils/` seeding, logging, metrics, checkpointing
- `notebooks/` analysis, debugging, Grad-CAM
- `reports/` report and figures
- `requirements.txt`

## 📝 Notes and limitations
Main observed limitation was a **validation-loss plateau** despite trying multiple architectures/optimizers. Potential future work includes broader hyperparameter search (Lion + schedulers), richer classification heads, and stronger augmentation policies (e.g., RandAugment/AugMix).

## 📚 References
See the report’s references section for sources on EfficientNet/torchvision, Lion, Macenko normalization, XGBoost, and related literature.
