# Recognizing Faces in the Wild (FIW)

![CI](https://github.com/jasoneplumb/FIW/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)
![AUC](https://img.shields.io/badge/AUC--ROC-0.764-yellow)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Kinship verification from facial images using frozen pretrained FaceNet embeddings and a lightweight learned scoring head, on the [Families in the Wild](https://www.kaggle.com/c/recognizing-faces-in-the-wild) dataset from Northeastern University's SMILE Lab.

Given a pair of face images, the pipeline scores how likely the two individuals are to be related.

## Motivation

This project explores a core challenge in computer vision: can a neural network learn to recognize familial resemblance from unconstrained face images? Beyond the Kaggle competition, it serves as a case study in deliberate, iterative model development — evolving from a naive 2-layer CNN (v0.1) through a fine-tuned FaceNet backbone (v0.6) to a simpler, stronger frozen-embedding pipeline (v0.7) through a series of principled improvements:

- **v0.1** — Baseline Siamese CNN with a 2K-pair subsample
- **v0.2–v0.3** — Fixed inverted contrastive loss, added reproducibility seeds, cross-platform support
- **v0.4** — Added proper evaluation metrics (AUC-ROC, Youden's J threshold)
- **v0.5** — Scaled to full dataset with family-aware splits to eliminate data leakage
- **v0.6** — Pretrained FaceNet backbone with selective fine-tuning and data augmentation
- **v0.7** — Replaced fine-tuning with frozen embeddings + cosine/logreg rank-blend after measuring that the unmodified pretrained encoder outperformed the fine-tuned network zero-shot (0.743 vs 0.674 on the v0.6 evaluation protocol)

Each version addressed a specific weakness exposed by the previous iteration. The full history is in the [CHANGELOG](CHANGELOG.md).

## Results

![AUC Visualization](auc_visualization.png)

Measured on the held-out test split (23,776 pairs: 11,888 related + 11,888 unrelated, from families unseen during training):

| Metric | Value |
|--------|------|
| **AUC-ROC (rank blend)** | 0.764 |
| AUC-ROC (cosine only) | 0.749 |
| AUC-ROC (logreg head only) | 0.754 |
| **Accuracy** | 0.697 |
| **Precision** | 0.695 |
| **Recall** | 0.701 |
| **Threshold Selection** | Youden's J statistic (0.489) |
| **Head Training Pairs** | ~216K balanced (108K related + 108K unrelated) |
| **Runtime** | one-time embedding ~40 min (CPU) or minutes (GPU), then head training ~5 min; embeddings are cached so re-runs skip the expensive step |

For comparison, the previous fine-tuned Siamese network (v0.6) reported 0.674 under its own evaluation protocol, which permitted same-family negative pairs; v0.7 numbers use the stricter `pair_sampling.py` policy. Related and unrelated pairs separate in the expected direction — related pairs have higher mean cosine similarity — but the distributions overlap substantially, which the 0.764 AUC reflects. The operating point is selected via Youden's J on the ROC curve. The project's original success criterion of AUC ≥ 0.80 has not been met.

## Architecture

The pipeline (`kinship.py`) has no fine-tuning. A frozen pretrained InceptionResnetV1 (FaceNet, VGGFace2) encoder embeds each face image exactly once at its native 160×160 input size; its forward pass L2-normalizes the 512-d embeddings. Pairs are scored by rank-blending two signals: raw cosine similarity between the embeddings, and the logit of a logistic-regression head (`Linear(1024, 1)`) over the symmetric pair features `[|e1−e2|, e1⊙e2]`. The blend weight is selected on the validation split.

- **Input**: 160x160 RGB face images (the encoder's native size)
- **Encoder**: InceptionResnetV1 (pretrained on VGGFace2), frozen, eval mode
- **Trained component**: a single linear layer (2,049 parameters), Adam (lr=0.01, cosine decay), early-stopped on validation AUC
- **Final score**: rank-blend of cosine similarity (weight 0.7, chosen on val) and head logit
- **Pairs**: ~216K balanced training pairs, family-aware 70/15/15 split

### Key Engineering Decisions

- **Family-aware splits** prevent data leakage — no family appears in more than one split, ensuring evaluation on unseen families rather than memorized individuals
- **Frozen encoder over fine-tuning** — measurement showed v0.6's fine-tuning *hurt*: it dropped the encoder's final L2 normalization, scrambled the pretrained embedding through a randomly initialized projection, let BatchNorm statistics drift on frozen layers, and used 112×112 inputs instead of the native 160×160. On the v0.6 protocol the unmodified encoder scored 0.743 zero-shot vs 0.674 fine-tuned
- **Policy-checked negative sampling** (`pair_sampling.py`) — negatives are never the same member, a known relation, or two members of the same family; positive/negative sets are asserted disjoint
- **Embed once, score cheaply** — each image is embedded exactly once and cached (`_cache/`), so iterating on the scoring head or blend takes seconds, and submission scoring needs one forward pass per image instead of one per pair

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

The script creates a virtual environment, installs the pinned dependencies from `requirements.txt` (`torch`, `torchvision`, `pandas`, `scikit-learn`, `matplotlib`, `kaggle`, `jupyter`, `facenet-pytorch`), and launches the notebook.

### Run on Kaggle

For cloud execution with free GPU, see the [Kaggle Setup Guide](KAGGLE_SETUP.md) or open `kaggle.ipynb` directly in a Kaggle Notebook.

## Pipeline

1. **Download** — Fetches competition data via the Kaggle API
2. **Clean** — Loads `train_relationships.csv`, removes pairs with missing image data (1,108 of 3,598 removed)
3. **Split** — Family-aware 70/15/15 split (no family leaks across splits), balanced positive/negative pairs under the `pair_sampling.py` policy
4. **Embed + Train** — Frozen FaceNet embeds each face once (cached); a logistic-regression head is trained on pair features with early stopping on validation AUC
5. **Evaluate** — AUC-ROC for cosine, head, and rank-blend scores; accuracy, precision, recall at optimal threshold (Youden's J); ROC curve and similarity distribution plots
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
