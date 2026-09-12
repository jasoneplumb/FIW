"""Kinship scoring from frozen face embeddings (issue #14).

The pipeline replaces fine-tuning with three fixed stages:

1. A frozen pretrained ``InceptionResnetV1`` (FaceNet, VGGFace2) encoder
   embeds each face image once at its native 160x160 input size. The
   encoder's ``forward()`` L2-normalizes embeddings, so cosine similarity
   and Euclidean distance rank pairs identically.
2. A logistic-regression head (``nn.Linear(1024, 1)``) scores pair features
   ``[|e1 - e2|, e1 * e2]``, trained with early stopping on validation AUC.
3. The final score rank-blends cosine similarity with the head logit; the
   blend weight is selected on the validation split.

Image-load failures are tracked per image: failed images are excluded from
the embedding table and every pair touching one is dropped from evaluation
(``stack_embeddings``), so zero-filled tensors never enter a metric.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

from pair_sampling import (assert_pair_sets_disjoint, build_excluded_pairs,
                           generate_negative_pairs)

EMBED_INPUT_SIZE = 160

eval_transform = transforms.Compose([
    transforms.Resize((EMBED_INPUT_SIZE, EMBED_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_encoder(device='cpu'):
    """Load the frozen pretrained FaceNet encoder in eval mode."""
    from facenet_pytorch import InceptionResnetV1
    encoder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def embed_images(encoder, image_paths, image_root, device='cpu',
                 batch_size=64, cache_path=None, transform=eval_transform,
                 progress_every=1024):
    """Embed each image once; return ({path: embedding}, {failed paths}).

    ``encoder`` is any callable mapping a (B, 3, H, W) tensor to (B, D)
    embeddings. When ``cache_path`` is given, previously computed
    embeddings are loaded from it and only missing images are embedded;
    the merged table is written back. Images that fail to load are
    returned in the failed set and never produce an embedding.
    """
    embeddings = {}
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        embeddings = {key: cached[key] for key in cached.files}

    todo = [p for p in image_paths if p not in embeddings]
    failed = set()
    batch, names = [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            out = encoder(torch.stack(batch).to(device))
        for name, emb in zip(names, out):
            embeddings[name] = emb.cpu().numpy().astype(np.float32)
        batch.clear()
        names.clear()

    for i, path in enumerate(todo):
        try:
            image = Image.open(os.path.join(image_root, path))
            batch.append(transform(image))
            names.append(path)
        except Exception as error:
            print(f'[Warning] Failed to load {path}: {error}')
            failed.add(path)
            continue
        if len(batch) >= batch_size:
            flush()
        if progress_every and i % progress_every == 0:
            print(f'  embedded {i}/{len(todo)} new images')
    flush()

    if cache_path and todo:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.savez(cache_path, **embeddings)
    return embeddings, failed


def build_pairs(relations_df, image_root='_train-faces', seed=42):
    """Family-aware 70/15/15 split with balanced, policy-compliant pairs.

    Positive pairs are all image combinations of each labeled relationship.
    Negative pairs follow the exclusion policy in ``pair_sampling`` (never
    the same member, a known relation, or the same family). Returns a dict
    with 'train', 'val', and 'test' lists of [img1, img2, label] entries.
    """
    rng = random.Random(seed)
    all_families = sorted({p.split('/')[0] for p in relations_df['p1']})
    rng.shuffle(all_families)
    n = len(all_families)
    split_families = {
        'train': set(all_families[:int(0.70 * n)]),
        'val': set(all_families[int(0.70 * n):int(0.85 * n)]),
        'test': set(all_families[int(0.85 * n):]),
    }

    members = sorted({m for row in relations_df.values for m in row})
    member_images = {}
    for member in members:
        member_dir = os.path.join(image_root, member)
        if os.path.isdir(member_dir):
            files = os.listdir(member_dir)
            if files:
                member_images[member] = [member + '/' + f for f in files]

    excluded_pairs = build_excluded_pairs(relations_df, members)

    splits = {}
    for name, family_set in split_families.items():
        subset = relations_df[relations_df['p1'].apply(
            lambda p: p.split('/')[0] in family_set)]
        positives = []
        for _, row in subset.iterrows():
            for img1 in member_images.get(row.p1, []):
                for img2 in member_images.get(row.p2, []):
                    positives.append([img1, img2, 1.0])
        candidates = [m for m in members
                      if m.split('/')[0] in family_set and m in member_images]
        negatives = generate_negative_pairs(positives, candidates,
                                            member_images, excluded_pairs, rng)
        assert_pair_sets_disjoint(positives, negatives)
        data = positives + negatives[:len(positives)]
        rng.shuffle(data)
        splits[name] = data
    return splits


def stack_embeddings(embeddings, pairs):
    """Return (e1, e2, labels, kept_pairs) for pairs whose images embedded.

    Pairs touching an image absent from ``embeddings`` (e.g. a load
    failure) are dropped, mirroring the evaluation policy of issue #10.
    """
    kept = [p for p in pairs if p[0] in embeddings and p[1] in embeddings]
    dropped = len(pairs) - len(kept)
    if dropped:
        print(f'[Warning] Dropped {dropped} pairs with missing embeddings')
    e1 = torch.from_numpy(np.stack([embeddings[p[0]] for p in kept]))
    e2 = torch.from_numpy(np.stack([embeddings[p[1]] for p in kept]))
    labels = np.array([p[2] for p in kept], dtype=np.float32)
    return e1, e2, labels, kept


def pair_features(e1, e2):
    """Symmetric pair features for the head: [|e1 - e2|, e1 * e2]."""
    return torch.cat([(e1 - e2).abs(), e1 * e2], dim=1)


def cosine_scores(e1, e2):
    return torch.nn.functional.cosine_similarity(e1, e2).numpy()


def make_head(embedding_dim=512, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(2 * embedding_dim, 1)


def logit_scores(head, e1, e2, batch_size=8192):
    head.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(e1), batch_size):
            features = pair_features(e1[start:start + batch_size],
                                     e2[start:start + batch_size])
            out.append(head(features).squeeze(1))
    return torch.cat(out).numpy()


def train_logreg_head(e1_train, e2_train, y_train, e1_val, e2_val, y_val,
                      epochs=200, lr=1e-2, weight_decay=1e-4, batch_size=1024,
                      patience=25, seed=0, verbose_every=10):
    """Train the logistic-regression head, early-stopped on validation AUC.

    Returns (head restored to its best-validation state, best val AUC).
    """
    head = make_head(e1_train.shape[1], seed=seed)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    targets = torch.from_numpy(y_train)
    best_val, best_state, best_epoch = 0.0, None, -1
    n = len(e1_train)
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = head(pair_features(e1_train[idx], e2_train[idx]))
            loss = loss_fn(logits.squeeze(1), targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_auc = roc_auc_score(y_val, logit_scores(head, e1_val, e2_val))
        if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0):
            print(f'  epoch {epoch + 1:3d}  val AUC {val_auc:.4f}')
        if val_auc > best_val:
            best_val, best_epoch = val_auc, epoch
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f'  early stop at epoch {epoch + 1} '
                  f'(best val AUC {best_val:.4f} @ epoch {best_epoch + 1})')
            break
    head.load_state_dict(best_state)
    return head, best_val


def _normalized_ranks(scores):
    scores = np.asarray(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[np.argsort(scores, kind='stable')] = np.arange(1, len(scores) + 1)
    return ranks / len(scores)


def rank_blend(cos, logits, weight):
    """Blend two score lists by their normalized ranks; result in (0, 1]."""
    return weight * _normalized_ranks(cos) + (1 - weight) * _normalized_ranks(logits)


def select_blend_weight(cos_val, logit_val, y_val,
                        grid=np.arange(0.0, 1.01, 0.1)):
    """Pick the blend weight maximizing validation AUC. Returns (w, auc)."""
    best_weight, best_auc = 0.5, 0.0
    for weight in grid:
        auc = roc_auc_score(y_val, rank_blend(cos_val, logit_val, weight))
        if auc > best_auc:
            best_auc, best_weight = auc, float(weight)
    return best_weight, best_auc
