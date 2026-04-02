#!/usr/bin/env python3
"""Extract and calculate final AUC metrics from the trained model"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

print("=" * 70)
print("FIW Notebook - Final Metrics Extraction")
print("=" * 70)

# Check for model and data
model_path = '_model-state-dict.pth'
if not os.path.exists(model_path):
    print(f"\n[ERROR] Model not found: {model_path}")
    print("        Training may not have completed yet")
    sys.exit(1)

print(f"\n[OK] Model file found: {model_path} ({os.path.getsize(model_path) / 1e9:.2f} GB)")

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"[OK] Device: CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"[OK] Device: MPS")
else:
    device = torch.device("cpu")
    print(f"[OK] Device: CPU")

# Rebuild model
print("\n[Loading model architecture...]")

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        backbone = InceptionResnetV1(pretrained='vggface2')
        for param in backbone.parameters():
            param.requires_grad = False
        for name, param in backbone.named_parameters():
            if name.startswith('repeat_3.') or name.startswith('block8.'):
                param.requires_grad = True
        self.backbone = nn.Sequential(
            backbone.conv2d_1a, backbone.conv2d_2a, backbone.conv2d_2b,
            backbone.maxpool_3a, backbone.conv2d_3b, backbone.conv2d_4a,
            backbone.conv2d_4b, backbone.repeat_1, backbone.mixed_6a,
            backbone.repeat_2, backbone.mixed_7a, backbone.repeat_3,
            backbone.block8, backbone.avgpool_1a, nn.Flatten(),
            backbone.dropout, backbone.last_linear, backbone.last_bn,
        )
        self.fc1 = nn.Linear(512, 128)

    def forward_once(self, x):
        output = self.backbone(x)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
print("[OK] Model architecture rebuilt and weights loaded")

# Load test data
print("\n[Loading test data...]")

eval_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load test relationships
dataset_path = '_recognizing-faces-in-the-wild/'
relations_csv = dataset_path + 'train_relationships.csv'
relations_df = pd.read_csv(relations_csv)

# Get test split (last 15% of families)
import random
import glob
from collections import defaultdict
import itertools

random.seed(42)
np.random.seed(42)

# Load and prepare test data
training_image_path = '_train-faces/'
all_families = sorted(set(row.split('/')[0] for row in relations_df['p1'].values))
random.shuffle(all_families)
n = len(all_families)
val_cutoff = int(0.85 * n)
test_families = set(all_families[val_cutoff:])

# Build member images dict
members = sorted(set(m for row in relations_df.values for m in row))
member_images = {}
for member in members:
    if os.path.exists(training_image_path + '/' + member):
        image_files = os.listdir(training_image_path + '/' + member)
        if len(image_files) > 0:
            member_images[member] = [member + '/' + img for img in image_files]

# Extract test relations
test_relations = relations_df[relations_df['p1'].apply(lambda x: x.split('/')[0] in test_families)]
print(f"[OK] Test set: {len(test_relations)} related pairs from {len(test_families)} families")

# Generate test pairs
positives = []
for _, row in test_relations.iterrows():
    p1_images = member_images.get(row.p1, [])
    p2_images = member_images.get(row.p2, [])
    for img1, img2 in itertools.product(p1_images, p2_images):
        positives.append([img1, img2, 1.0])

# Generate negative pairs
split_members = [m for m in members if m.split('/')[0] in test_families and m in member_images]
negatives = []
while len(negatives) < len(positives):
    p1 = random.choice(split_members)
    p2 = random.choice(split_members)
    if p1 == p2:
        continue
    p1_images = member_images[p1]
    p2_images = member_images[p2]
    for img1, img2 in itertools.product(p1_images, p2_images):
        negatives.append([img1, img2, 0.0])
        if len(negatives) >= len(positives):
            break

test_data = positives + negatives[:len(positives)]
random.shuffle(test_data)
print(f"[OK] Generated {len(test_data)} test pairs ({len(positives)} pos + {len(negatives[:len(positives)])} neg)")

# Dataset
class ImagePairDataset(Dataset):
    def __init__(self, data, transform):
        self.image_pairs = [sublist[:-1] for sublist in data]
        self.labels = torch.tensor([sublist[-1] for sublist in data], dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            img0 = Image.open('_train-faces/' + self.image_pairs[idx][0])
            img1 = Image.open('_train-faces/' + self.image_pairs[idx][1])
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            return img0, img1, self.labels[idx]
        except Exception as e:
            print(f"[Warning] Failed to load image pair {idx}: {e}")
            # Return dummy tensors
            return torch.zeros(3, 112, 112), torch.zeros(3, 112, 112), self.labels[idx]

test_dataset = ImagePairDataset(test_data, eval_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(f"[OK] Test loader: {len(test_dataset)} samples, {len(test_loader)} batches")

# Evaluate
print("\n[Evaluating model...]")

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

criterion = ContrastiveLoss()
test_distances = []
test_labels = []
running_loss = 0.0

with torch.no_grad():
    for i, (data1, data2, label) in enumerate(test_loader):
        data1, data2, label = data1.to(device), data2.to(device), label.to(device)
        output1, output2 = model(data1, data2)
        loss = criterion(output1, output2, label)
        running_loss += loss.item() * data1.size(0)
        distances = F.pairwise_distance(output1, output2)
        test_distances.extend(distances.cpu().numpy())
        test_labels.extend(label.cpu().numpy())
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(test_loader)}")

epoch_loss = running_loss / len(test_dataset)
print(f"\n[OK] Test Loss: {epoch_loss:.4f}")

# Metrics
distances = np.array(test_distances)
labels = np.array(test_labels)
scores = 1 - (distances / distances.max())

auc = roc_auc_score(labels, scores)
fpr, tpr, thresholds = roc_curve(labels, scores)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

predictions = (scores >= optimal_threshold).astype(float)
acc = accuracy_score(labels, predictions)
prec = precision_score(labels, predictions)
rec = recall_score(labels, predictions)

print("\n" + "=" * 70)
print("FINAL EVALUATION METRICS")
print("=" * 70)
print(f"AUC-ROC Score:             {auc:.4f}")
print(f"Optimal Threshold:         {optimal_threshold:.4f}")
print(f"Accuracy:                  {acc:.4f}")
print(f"Precision:                 {prec:.4f}")
print(f"Recall:                    {rec:.4f}")
print("=" * 70)

# Distance analysis
pos_distances = distances[labels == 1.0]
neg_distances = distances[labels == 0.0]
mean_pos = np.mean(pos_distances)
mean_neg = np.mean(neg_distances)

print("\nDistance Distribution:")
print(f"  Related pairs (mean):     {mean_pos:.4f} +/- {np.std(pos_distances):.4f}")
print(f"  Unrelated pairs (mean):   {mean_neg:.4f} +/- {np.std(neg_distances):.4f}")
print(f"  Separation ratio:         {mean_neg / mean_pos:.2f}x")

# Success criterion
print("\n" + "=" * 70)
print("SUCCESS CRITERIA (from Issue #1)")
print("=" * 70)
if auc >= 0.80:
    print(f"[PASS] AUC >= 0.80: {auc:.4f}")
else:
    print(f"[FAIL] AUC < 0.80: {auc:.4f}")

if mean_pos < mean_neg:
    print(f"[PASS] Model learns meaningful embeddings")
else:
    print(f"[FAIL] Model does not separate related/unrelated")

print("=" * 70)
