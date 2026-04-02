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
- Expected runtime: **~15 minutes** with GPU enabled
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
- Distance distribution plots
- Success metric check (AUC ≥ 0.80)

## Enable GPU (Recommended)

For fastest training (8-12 min instead of 30+):

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
- **model.pth** — Trained model weights
- **evaluation.png** — ROC curve + distance plots

Download these for local use.

## Troubleshooting

**"facenet_pytorch not found"**
- The notebook installs it automatically (cell 1)
- Wait for the pip install to complete

**"Memory limit exceeded"**
- Reduce `batch_size` from 64 to 32 in the DataLoader cells
- Or reduce training epochs from 10 to 5

**"CUDA out of memory" (GPU only)**
- Use smaller batch size: `batch_size = 32`
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
AUC-ROC:           0.7X to 0.8X (depends on random initialization)
Optimal Threshold: 0.XX
Accuracy:          0.XX
Precision:         0.XX
Recall:            0.XX

✓ SUCCESS: AUC 0.80+ meets epic target
```

## Next Steps

- Use the trained model for inference on new images
- Fine-tune hyperparameters (margin, learning rate, epochs)
- Add additional training data
- Experiment with different backbone architectures

## Links

- **Notebook**: [kaggle.ipynb](./kaggle.ipynb)
- **Main (local) notebook**: [main.ipynb](./main.ipynb)
- **Original dataset**: https://www.kaggle.com/c/recognizing-faces-in-the-wild
- **FIW Paper**: https://www.northeastern.edu/smile/
