# Recognizing Faces in the Wild (FIW)

Kinship verification from facial images using a Siamese convolutional neural network trained on the [Families in the Wild](https://www.kaggle.com/c/recognizing-faces-in-the-wild) dataset from Northeastern University's SMILE Lab.

Given a pair of face images, the model predicts the probability that the two individuals are related.

## Architecture

A Siamese CNN with shared weights processes each image independently through two convolutional layers (3→32→64 channels, with ReLU and max-pooling), followed by fully connected layers (9216→512→256→128) to produce a 128-dimensional embedding. Pairs are compared via Euclidean distance and trained with contrastive loss.

- **Input**: 56×56 RGB face images
- **Training**: ~216K balanced pairs (108K related + 108K unrelated), family-aware 70/15/15 split
- **Optimizer**: Adam (lr=0.0001), 50 epochs, batch size 64

## Quick Start

### Prerequisites

- Python 3.8+
- A [Kaggle API key](https://www.kaggle.com/docs/api) at `~/.kaggle/kaggle.json`

### Run

```bash
# Interactive Jupyter session
./run.sh

# Headless execution
./run.sh --headless
```

The script creates a virtual environment, installs dependencies (`torch`, `torchvision`, `pandas`, `scikit-learn`, `matplotlib`, `kaggle`, `jupyter`), and launches the notebook.

## Project Structure

```
main.ipynb              # End-to-end pipeline (download → train → evaluate → submit)
run.sh                  # Setup and launch script
_provided-data/         # Raw Kaggle competition archive
_recognizing-faces-in-the-wild/  # Extracted competition files
_train-faces/           # 786 family directories with member face images
_test-faces/            # 4,866 unlabeled test images
_model-state-dict.pth   # Saved model weights (generated after training)
submission.csv          # Kaggle submission file (generated after inference)
```

## Pipeline

1. **Download** — Fetches competition data via the Kaggle API
2. **Clean** — Loads `train_relationships.csv`, removes pairs with missing image data (1,108 of 3,598 removed)
3. **Split** — Family-aware 70/15/15 split (no family leaks across splits), balanced positive/negative pairs
4. **Train** — Siamese network with contrastive loss for 50 epochs
5. **Evaluate** — AUC-ROC, accuracy, precision, recall at optimal threshold (Youden's J); ROC curve and distance distribution plots
6. **Submit** — Generates `submission.csv` with pairwise relatedness probabilities for all ~11.8M test pairs

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
