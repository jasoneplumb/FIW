# Running on Kaggle Notebooks

This guide shows how to run the FIW kinship verification model on Kaggle's cloud infrastructure.

## Quick Start (5 minutes)

### 1. Create a Kaggle Notebook
- Go to https://www.kaggle.com/code
- Click **"New Notebook"**
- Select **Python** language

### 2. Add the Competition Dataset
- Click **"+ Input"** on the right sidebar
- Search for: **"Recognizing Faces in the Wild"**
- Add it to your notebook

### 3. Copy & Paste the Code
- Open [kaggle.ipynb](./kaggle.ipynb) in your browser
- Copy all cells into your Kaggle notebook
- Or: Import directly from GitHub repository

### 4. Run the Notebook
- Click **"Run All"** (or run cells sequentially)
- Expected runtime: **~15 minutes** with GPU enabled (mostly one-time embedding)
- Monitor progress in the output cells

## Features of the Kaggle Notebook

✅ **Auto-configured for Kaggle environment:**
- Uses `/kaggle/input/` dataset path automatically
- Outputs saved to `/kaggle/working/`
- Pre-installed libraries (PyTorch, scikit-learn, pandas, etc.)
- One-click GPU acceleration

✅ **Optimized pipeline:**
- Skips Kaggle API authentication (dataset auto-mounted)
- Efficient data extraction from zips
- All paths use cross-platform `os.path.join()`
- Memory-efficient batch processing

✅ **Real-time evaluation:**
- AUC-ROC, accuracy, precision, recall
- ROC curve visualization
- Similarity distribution plots
- Success metric check (AUC ≥ 0.80)

## Enable GPU (Recommended)

For fastest embedding (a few minutes instead of ~40 on CPU):

1. In your Kaggle notebook, click **⚙️ Settings** (top right)
2. Under **Accelerator**, select **GPU**
3. Choose **P100** or **T4**
4. Run the notebook

Estimated times:
- **GPU (T4)**: 12-15 min ✓
- **GPU (P100)**: 8-10 min ✓✓
- **CPU only**: 30-45 min (still works!)

## Outputs

After execution, check `/kaggle/working/`:
- **logreg-head.pth** — Trained scoring-head weights
- **evaluation.png** — ROC curve + similarity plots

Download these for local use.

## Troubleshooting

**"facenet_pytorch not found"**
- The notebook installs it automatically (cell 1)
- Wait for the pip install to complete

**"Memory limit exceeded"**
- Reduce the embedding `batch_size` from 128 to 64 in the embedding cell

**"CUDA out of memory" (GPU only)**
- Use a smaller embedding batch size: `batch_size = 64`
- Switch to P100 or CPU if T4 fails

**Notebook timeouts (>10 hours)**
- GPU is not enabled (see above)
- Or reduce dataset size for testing

## Alternative: Fork & Run Directly

1. Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/jasoneplumb/fiw)
2. Click **"New Notebook"** button
3. Our notebook is pre-linked and ready to run!

## Results Expected

On first run, you should see:

```
=== TEST RESULTS ===
AUC — cosine only:      ~0.75
AUC — logreg head only: ~0.75
AUC — rank blend:       ~0.76
Optimal Threshold:      ~0.49
Accuracy:               ~0.70
Precision:              ~0.70
Recall:                 ~0.70
```

The pipeline is deterministic up to hardware differences (seeded splits, frozen encoder), so results should closely match the values above.

## Next Steps

- Use the cached embeddings and trained head for inference on new images
- Tune the scoring head (features, regularization) or the blend-weight grid
- Ensemble additional pretrained face encoders (e.g., ArcFace)
- Add additional training data

## Links

- **Notebook**: [kaggle.ipynb](./kaggle.ipynb)
- **Main (local) notebook**: [main.ipynb](./main.ipynb)
- **Original dataset**: https://www.kaggle.com/c/recognizing-faces-in-the-wild
- **FIW Paper**: https://www.northeastern.edu/smile/
