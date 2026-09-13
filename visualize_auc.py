#!/usr/bin/env python3
"""Generate the AUC visualization dashboard for the kinship pipeline.

Rebuilds the notebook's family split, scores the held-out test pairs with
the multi-encoder rank-blend, and renders auc_visualization.png. Requires
a completed main.ipynb run: the per-encoder embedding caches and trained
head under _cache/ are reused (missing embeddings are computed on the fly).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

import kinship

plt.style.use('seaborn-v0_8-darkgrid')

TARGET_AUC = 0.80
HEAD_PATH = '_cache/logreg-head.pth'

# name -> (encoder factory, transform, cache path)
ENCODER_SPECS = {
    'vggface2': (lambda: kinship.load_encoder(), kinship.eval_transform,
                 '_cache/embeddings-160.npz'),
    'casia': (lambda: kinship.load_encoder(pretrained='casia-webface'),
              kinship.eval_transform, '_cache/embeddings-casia-160.npz'),
    'arcface': (lambda: kinship.ArcFaceEncoder(kinship.download_arcface()),
                kinship.arcface_transform, '_cache/embeddings-arcface-112.npz'),
}

if not os.path.exists(HEAD_PATH):
    print(f'[ERROR] Trained head not found: {HEAD_PATH}')
    print('        Run main.ipynb (Step 5) first.')
    sys.exit(1)

print('Rebuilding the family split...')
relations_df = pd.read_csv('_recognizing-faces-in-the-wild/train_relationships.csv')
relations_df = kinship.clean_relations(relations_df)
splits = kinship.build_pairs(relations_df, seed=42)

print('Loading embeddings and trained head...')
unique_images = sorted({img for data in splits.values()
                        for pair in data for img in pair[:2]})
tables = {}
for name, (encoder_factory, transform, cache) in ENCODER_SPECS.items():
    encoder = None if os.path.exists(cache) else encoder_factory()
    tables[name], _ = kinship.embed_images(encoder, unique_images,
                                           '_train-faces', transform=transform,
                                           cache_path=cache)

head = kinship.make_head(embedding_dim=512 * len(tables))
head.load_state_dict(torch.load(HEAD_PATH, weights_only=True))

print('Scoring validation and test pairs...')
e1_val, e2_val, y_val, _ = kinship.concat_embeddings(list(tables.values()),
                                                     splits['val'])
e1_test, e2_test, y_test, _ = kinship.concat_embeddings(list(tables.values()),
                                                        splits['test'])

signal_names = [f'cos_{name}' for name in tables] + ['head']
signals_val = kinship.segment_cosines(e1_val, e2_val) + \
    [kinship.logit_scores(head, e1_val, e2_val)]
signals_test = kinship.segment_cosines(e1_test, e2_test) + \
    [kinship.logit_scores(head, e1_test, e2_test)]

blend_weights, blend_val_auc = kinship.select_blend_weights(signals_val, y_val)
scores = kinship.rank_blend(signals_test, blend_weights)
print('Blend weights (selected on val, val AUC '
      f'{blend_val_auc:.4f}): '
      + ', '.join(f'{n}={w:.1f}' for n, w in zip(signal_names, blend_weights)))

auc = roc_auc_score(y_test, scores)
fpr, tpr, thresholds = roc_curve(y_test, scores)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
predictions = (scores >= optimal_threshold).astype(float)
acc = accuracy_score(y_test, predictions)
prec = precision_score(y_test, predictions)
rec = recall_score(y_test, predictions)

# --- Dashboard ---------------------------------------------------------------

fig = plt.figure(figsize=(16, 12))

# 1. ROC Curve
ax1 = plt.subplot(2, 3, 1)
ax1.plot(fpr, tpr, linewidth=2.5, label=f'ROC Curve (AUC = {auc:.4f})', color='#2E86AB')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier', alpha=0.7)
ax1.scatter(fpr[optimal_idx], tpr[optimal_idx], s=200, color='red', zorder=5,
            label=f'Optimal Threshold ({optimal_threshold:.3f})', marker='*')
ax1.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax1.set_title('ROC Curve - AUC Visualization', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])

# 2. Blend components vs target
ax2 = plt.subplot(2, 3, 2)
categories = [n.replace('cos_', 'cos\n') for n in signal_names] + ['Rank\nBlend', 'Target']
values = [roc_auc_score(y_test, sig) for sig in signals_test] + [auc, TARGET_AUC]
colors = ['#06A77D' if v >= TARGET_AUC else '#D62828' for v in values]
bars = ax2.bar(categories, values, color=colors, alpha=0.8,
               edgecolor='black', linewidth=2)
ax2.set_ylabel('AUC Score', fontsize=11, fontweight='bold')
ax2.set_title('AUC by Scoring Component', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1.0])
ax2.axhline(y=TARGET_AUC, color='red', linestyle='--', linewidth=2, alpha=0.7,
            label=f'Target ({TARGET_AUC:.2f})')
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax2.tick_params(axis='x', labelsize=8)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Similarity distributions (VGGFace2 cosine)
ax3 = plt.subplot(2, 3, 3)
cosine_test = signals_test[0]
pos_cosine = cosine_test[y_test == 1.0]
neg_cosine = cosine_test[y_test == 0.0]
ax3.hist(pos_cosine, bins=40, alpha=0.65, label=f'Related (n={len(pos_cosine)})',
         color='#06A77D', edgecolor='black', linewidth=0.5)
ax3.hist(neg_cosine, bins=40, alpha=0.65, label=f'Unrelated (n={len(neg_cosine)})',
         color='#D62828', edgecolor='black', linewidth=0.5)
ax3.set_xlabel('Cosine Similarity (VGGFace2)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Embedding Similarity Distribution', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Metrics comparison
ax4 = plt.subplot(2, 3, 4)
metrics = ['AUC', 'Accuracy', 'Precision', 'Recall']
values = [auc, acc, prec, rec]
colors_metrics = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bars = ax4.bar(metrics, values, color=colors_metrics, alpha=0.8,
               edgecolor='black', linewidth=2)
ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
ax4.set_title('Classification Metrics', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 1.0])
ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax4.axhline(y=TARGET_AUC, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
            label='Target AUC')
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# 5. Confusion matrix
ax5 = plt.subplot(2, 3, 5)
cm = confusion_matrix(y_test, predictions)
im = ax5.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax5.figure.colorbar(im, ax=ax5)
ax5.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
        yticklabels=['Unrelated', 'Related'], xticklabels=['Unrelated', 'Related'])
ax5.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax5.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax5.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax5.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black',
                 fontsize=12, fontweight='bold')

# 6. Performance summary table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_data = [
    ['Metric', 'Value'],
    ['AUC-ROC (rank blend)', f'{auc:.4f}'],
    ['Blend weights', ' / '.join(f'{w:.1f}' for w in blend_weights)],
    ['Accuracy', f'{acc:.4f}'],
    ['Precision', f'{prec:.4f}'],
    ['Recall', f'{rec:.4f}'],
    ['Optimal Threshold', f'{optimal_threshold:.4f}'],
    ['True Positives', f'{cm[1, 1]}'],
    ['True Negatives', f'{cm[0, 0]}'],
    ['False Positives', f'{cm[0, 1]}'],
    ['False Negatives', f'{cm[1, 0]}'],
]
table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                  colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
for i in range(2):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')
for i in range(1, len(summary_data)):
    for j in range(2):
        table[(i, j)].set_facecolor('#F0F0F0' if i % 2 == 0 else '#FFFFFF')
ax6.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('auc_visualization.png', dpi=300, bbox_inches='tight')
print('Chart saved: auc_visualization.png')

print('\nSummary:')
print(f'  AUC-ROC:     {auc:.4f}')
print(f'  Target:      {TARGET_AUC:.4f}')
print(f'  Difference:  {auc - TARGET_AUC:.4f}')
