# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Replaced 2-layer CNN backbone with pretrained InceptionResnetV1 (FaceNet, VGGFace2) (#7)
- Froze early layers; fine-tuning last 2 blocks (repeat_3, block8) + new FC head (512→128) (#7)
- Optimizer now targets only trainable parameters (#7)
- Input resolution increased from 56×56 to 112×112 (#8)
- Training epochs reduced from 50 to 10 (pretrained backbone converges faster)
- Input normalization updated to [-1, 1] range (mean=0.5, std=0.5 per channel) (#8)

### Added
- `facenet-pytorch` dependency in `run.sh` (#7)
- Training data augmentation: RandomHorizontalFlip, RandomRotation(±10°), ColorJitter (#8)
- Separate `train_transform` and `eval_transform` pipelines (#8)
- `ImagePairDataset` now accepts a `transform` parameter for split-specific pipelines (#8)

## [0.5.0] - 2026-03-09

### Changed
- Use all training pairs (~180K+ balanced) instead of 2K subsample (#4)
- Family-aware 70/15/15 split — no family appears in multiple splits, eliminating data leakage (#4)
- Negative pairs now sampled only from families within the same split (#4)
- Batch size increased from 32 to 64 (#4)
- Training loop now reports both train and validation loss every 10 epochs (#4)

### Added
- Validation set with dedicated DataLoader (`val_loader`) (#4)
- `validate()` function for per-epoch validation loss (#4)

### Removed
- `num_training_samples` / `num_testing_samples` caps (#4)
- "Only 2K of ~257K pairs used for training" from known issues (#4)

## [0.4.0] - 2026-03-09

### Added
- Evaluation metrics cell: AUC-ROC, accuracy, precision, recall at optimal threshold (#6)
- ROC curve plot with optimal operating point (Youden's J) (#6)
- Distance distribution histograms for related vs unrelated pairs (#6)
- `test()` function now returns per-sample distances and labels for downstream analysis (#6)

### Removed
- "No evaluation metrics beyond raw loss" from known issues (#6)

## [0.3.0] - 2026-03-09

### Fixed
- `missing_relations_list.count` bug — was a method reference (always truthy), replaced with `if missing_relations_list:` (#3)
- Windows-only path separator `split('\\')` replaced with `os.path.basename()` for cross-platform compatibility (#3)
- `set` variable shadowing Python builtin, renamed to `image_files` (#3)
- Misleading comment "Load the (trained) model to a file" corrected to describe actual behavior (#3)
- Prediction rounding from `.1f` (11 distinct values) to `.6f` (full float precision) for AUC-ROC ranking (#5)

### Added
- Random seeds for reproducibility: `random.seed(42)`, `numpy.random.seed(42)`, `torch.manual_seed(42)` (#3)

## [0.2.0] - 2026-03-09

### Fixed
- Contrastive loss label convention was inverted — `(1-label)` was used for the similar term, but `label=1` means related. Swapped to `(label)` for the distance-minimizing term and `(1-label)` for the margin term (#2)

### Added
- Post-training sanity check cell that compares mean euclidean distance for related vs unrelated pairs

### Removed
- "Contrastive loss label convention may be inverted" from known issues (confirmed and fixed)

## [0.1.0] - 2024-12-01

### Added
- Siamese CNN for kinship verification (contrastive loss, 2-layer backbone)
- Kaggle data download and extraction pipeline
- Training data loading, cleaning, and pair generation (positive + negative)
- `ImagePairDataset` and DataLoader for batched training
- Training loop (50 epochs, Adam, lr=0.0001)
- Test evaluation (contrastive loss on held-out pairs)
- Inference function and submission CSV generation for ~11.8M test pairs
- Model save/load via `state_dict`
- Image normalization (56x56 resize, tensor transform)
- `.gitignore` for data directories (`_*/`)

### Known Issues
- Only 2K of ~257K pairs used for training (fixed in 0.5.0)
