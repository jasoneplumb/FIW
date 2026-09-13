# Recognizing Faces in the Wild (FIW)

![CI](https://github.com/jasoneplumb/FIW/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)
![AUC](https://img.shields.io/badge/AUC--ROC-0.784-yellow)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Kinship verification from facial images using an ensemble of frozen pretrained face-embedding models and a lightweight learned scoring head, on the [Families in the Wild](https://www.kaggle.com/c/recognizing-faces-in-the-wild) dataset from Northeastern University's SMILE Lab.

Given a pair of face images, the pipeline scores how likely the two individuals are to be related.

## Motivation

This project explores a core challenge in computer vision: can a neural network learn to recognize familial resemblance from unconstrained face images? Beyond the Kaggle competition, it serves as a case study in deliberate, iterative model development — evolving from a naive 2-layer CNN (v0.1) through a fine-tuned FaceNet backbone (v0.6) and a simpler, stronger frozen-embedding pipeline (v0.7) to a frozen-encoder ensemble (v0.8) through a series of principled improvements:

- **v0.1** — Baseline Siamese CNN with a 2K-pair subsample
- **v0.2–v0.3** — Fixed inverted contrastive loss, added reproducibility seeds, cross-platform support
- **v0.4** — Added proper evaluation metrics (AUC-ROC, Youden's J threshold)
- **v0.5** — Scaled to full dataset with family-aware splits to eliminate data leakage
- **v0.6** — Pretrained FaceNet backbone with selective fine-tuning and data augmentation
- **v0.7** — Replaced fine-tuning with frozen embeddings + cosine/logreg rank-blend after measuring that the unmodified pretrained encoder outperformed the fine-tuned network zero-shot (0.743 vs 0.674 on the v0.6 evaluation protocol)
- **v0.8** — Ensembled three frozen encoders (FaceNet/VGGFace2, FaceNet/CASIA-WebFace, ArcFace): individually weaker encoders contribute complementary signal, lifting test AUC 0.764 → 0.784

Each version addressed a specific weakness exposed by the previous iteration. The full history is in the [CHANGELOG](CHANGELOG.md).

## Results

![AUC Visualization](auc_visualization.png)

Measured on the held-out test split (23,776 pairs: 11,888 related + 11,888 unrelated, from families unseen during training):

| Metric | Value |
|--------|------|
| **AUC-ROC (rank blend)** | 0.784 |
| AUC-ROC (VGGFace2 cosine) | 0.749 |
| AUC-ROC (CASIA cosine) | 0.718 |
| AUC-ROC (ArcFace cosine) | 0.691 |
| AUC-ROC (concat logreg head) | 0.768 |
| Accuracy (stale, see note) | 0.712 |
| Precision (stale, see note) | 0.709 |
| Recall (stale, see note) | 0.719 |
| **Threshold Selection** | Youden's J (argmax TPR−FPR), selected on the validation split |
| **Head Training Pairs** | ~216K balanced (108K related + 108K unrelated) |
| **Runtime** | one-time embedding per encoder (~40 min total CPU, minutes on GPU), then head training ~15 min; embeddings are cached so re-runs skip the expensive steps |

> **Note on the three threshold-dependent metrics.** Until 2026-09-13,
> `visualize_auc.py` selected the operating point by maximizing Youden's J on
> the *test* split and then reported accuracy, precision, and recall at that
> threshold on the same split — a best case rather than a held-out result. The
> threshold is now selected on validation, matching how the blend weights were
> always chosen. The three values above predate that fix and will change when
> the pipeline is re-run; they are left visible rather than deleted so the
> correction is legible. **AUC-ROC is threshold-free and is unaffected** — 0.784
> stands, as does every per-encoder AUC.

Progression: fine-tuned Siamese (v0.6) 0.674 → single frozen encoder blend (v0.7) 0.764 → three-encoder ensemble (v0.8) 0.784, all without training any component larger than a single linear layer. Individually, ArcFace scores only 0.691 on these unaligned crops — its value is complementary signal, not standalone strength. The project's original success criterion of AUC ≥ 0.80 has not been met.

## Evidence

| | |
| --- | --- |
| **Contribution** | Sole author of the pipeline, pair-sampling policy, family-aware splits, evaluation, and the measurement sequence that replaced fine-tuning with frozen encoders. AI-assisted implementation. The encoders themselves are third-party pretrained models (facenet-pytorch, insightface); the dataset is Families in the Wild from Northeastern's SMILE Lab. |
| **Status** | Personal research project. Reported figures are the repository's own measurements, not independently reproduced. |
| **Evidence** | AUC-ROC 0.784 (rank blend) on the held-out test split of 23,776 pairs from families unseen in training; per-encoder cosines 0.749 / 0.718 / 0.691 and concat-logreg head 0.768 on the same split. Progression 0.674 (v0.6 fine-tuned) → 0.764 (v0.7) → 0.784 (v0.8). |
| **Reproduction** | Requires the Kaggle dataset and the pretrained encoder weights. README setup, then `./run.sh --headless`. Embeddings are cached per encoder, so the expensive step runs once. |
| **Limitations** | The 0.80 target was not met. The v0.6 → v0.7 comparison spans a protocol change and is not a clean controlled A/B. ArcFace runs on unaligned crops and scores 0.691 alone; its contribution is complementary signal, not standalone strength. Youden's J selects the operating point on the validation split as of 2026-09-13; the accuracy, precision, and recall currently shown predate that fix, were computed at a test-selected threshold, and will change on the next run. AUC is threshold-free and unaffected. Kinship inference from face images has obvious misuse potential and is published here as an evaluation exercise, not a deployable classifier. |

## Architecture

The pipeline (`kinship.py`) has no fine-tuning. Three frozen pretrained encoders embed each face image exactly once (cached per encoder): `InceptionResnetV1` with VGGFace2 and CASIA-WebFace weights at their native 160×160, and the ArcFace w600k_r50 ONNX recognizer (insightface's buffalo_l) at its native 112×112 — all preprocessing is `(x−127.5)/127.5` RGB, and all embeddings are L2-normalized 512-d vectors. Pairs are scored by rank-blending four signals: the three per-encoder cosine similarities and the logit of a logistic-regression head (`Linear(3072, 1)`) over the concatenated symmetric pair features `[|e1−e2|, e1⊙e2]`. The blend weights are selected on the validation split over a 0.1-step simplex.

- **Inputs**: 160×160 RGB (FaceNet variants), 112×112 RGB (ArcFace) — each encoder's native size
- **Encoders**: InceptionResnetV1 (VGGFace2), InceptionResnetV1 (CASIA-WebFace), ArcFace w600k_r50 (ONNX) — all frozen, eval mode
- **Trained component**: a single linear layer (6,145 parameters), Adam (lr=0.01, cosine decay), early-stopped on validation AUC
- **Final score**: rank-blend with val-selected weights (cos_vggface2 0.3, cos_casia 0.1, cos_arcface 0.3, head 0.3)
- **Pairs**: ~216K balanced training pairs, family-aware 70/15/15 split

### Key Engineering Decisions

- **Family-aware splits** prevent data leakage — no family appears in more than one split, ensuring evaluation on unseen families rather than memorized individuals
- **Frozen encoder over fine-tuning** — measurement showed v0.6's fine-tuning *hurt*: it dropped the encoder's final L2 normalization, scrambled the pretrained embedding through a randomly initialized projection, let BatchNorm statistics drift on frozen layers, and used 112×112 inputs instead of the native 160×160. On the v0.6 protocol the unmodified encoder scored 0.743 zero-shot vs 0.674 fine-tuned
- **Policy-checked negative sampling** (`pair_sampling.py`) — negatives are never the same member, a known relation, or two members of the same family; positive/negative sets are asserted disjoint
- **Embed once, score cheaply** — each image is embedded exactly once per encoder and cached (`_cache/`), so iterating on the scoring head or blend takes seconds, and submission scoring needs one forward pass per image per encoder instead of one per pair
- **Ensemble of frozen encoders** — encoders trained on different data (VGGFace2, CASIA-WebFace, WebFace600K) make different mistakes; rank-blending their cosine signals with a concat-feature head adds +0.020 AUC over the best single encoder pipeline

## Quick Start

### Prerequisites

- Python 3.11. On macOS, install it with `brew install python@3.11`.
- A [Kaggle API key](https://www.kaggle.com/docs/api) at `~/.kaggle/kaggle.json`

### Run

```bash
# Interactive Jupyter session
./run.sh

# Headless execution
./run.sh --headless
```

The script creates a virtual environment, installs the pinned dependencies from `requirements.txt` (`torch`, `torchvision`, `pandas`, `scikit-learn`, `matplotlib`, `kaggle`, `jupyter`, `facenet-pytorch`, `onnxruntime`), and launches the notebook.

### Run on Kaggle

For cloud execution with free GPU, see the [Kaggle Setup Guide](KAGGLE_SETUP.md) or open `kaggle.ipynb` directly in a Kaggle Notebook.

## Pipeline

1. **Download** — Fetches competition data via the Kaggle API
2. **Clean** — Loads `train_relationships.csv`, removes pairs with missing image data (1,108 of 3,598 removed)
3. **Split** — Family-aware 70/15/15 split (no family leaks across splits), balanced positive/negative pairs under the `pair_sampling.py` policy
4. **Embed + Train** — Three frozen encoders (FaceNet/VGGFace2, FaceNet/CASIA-WebFace, ArcFace) embed each face once (cached per encoder); a logistic-regression head is trained on concatenated pair features with early stopping on validation AUC
5. **Evaluate** — AUC-ROC per encoder cosine, head, and rank-blend scores; accuracy, precision, recall at optimal threshold (Youden's J); ROC curve and similarity distribution plots
6. **Submit** — Scores exactly the 5,310 image pairs requested by `sample_submission.csv` (from the `test.zip` image set) and writes `submission.csv` with the keys preserved verbatim

## Project Structure

```
main.ipynb              # End-to-end pipeline (download -> embed -> train head -> evaluate -> submit)
kaggle.ipynb            # Cloud-optimized variant for Kaggle Notebooks
run.sh                  # Setup and launch script
kinship.py              # Scoring pipeline: embeddings, logreg head, rank-blend
pair_sampling.py        # Negative-sampling policy and pair-set helpers
test_invariants.py      # Fast invariant checks (run by CI on every push/PR)
visualize_auc.py        # Generates AUC visualization dashboard
quick_verify.py         # Component verification (encoder, head, blend, metrics)
requirements.txt        # Python dependencies (pinned to tested versions)
requirements-dev.txt    # Dev/CI extras (pytest) on top of requirements.txt
.github/workflows/ci.yml  # CI: env install + sampling/evaluation invariants
```

## Citations

Per the [competition rules](https://www.kaggle.com/competitions/recognizing-faces-in-the-wild/rules#7-competition-data), usage of FIW data should cite:

> Robinson, J.P., Shao, M., Liu, H., Wu, Y., Gillis, T., & Fu, Y. "Visual Kinship Recognition of Families In the Wild." *IEEE TPAMI Special Edition: Computational Face* (2018).
>
> Robinson, J.P., Shao, M., Zhao, H., Wu, Y., Gillis, T., & Fu, Y. "Recognizing Families In the Wild (RFIW): Data Challenge Workshop in conjunction with ACM MM 2017." *ACM Multimedia Conference: Workshop on RFIW* (2017).
>
> Wang, S., Robinson, J.P., & Fu, Y. "Kinship Verification on Families in the Wild with Marginalized Denoising Metric Learning." *IEEE Automatic Face and Gesture Recognition* (2017).
>
> Robinson, J.P., Shao, M., Wu, Y., & Fu, Y. "Families In the Wild (FIW): Large-Scale Kinship Image Database and Benchmarks." *ACM on Multimedia Conference* (2016).

## License

[MIT](LICENSE)
