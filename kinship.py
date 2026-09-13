"""Kinship scoring from frozen face embeddings (issues #14, #16).

The pipeline has no fine-tuning:

1. Frozen pretrained encoders embed each face image exactly once, cached
   per encoder: two ``InceptionResnetV1`` variants (VGGFace2 and
   CASIA-WebFace weights, native 160x160) and an ArcFace ONNX recognizer
   (w600k_r50, 112x112). All embeddings are L2-normalized, so cosine
   similarity and Euclidean distance rank pairs identically.
2. A logistic-regression head scores the concatenated symmetric pair
   features ``[|e1 - e2|, e1 * e2]`` across encoders, trained with early
   stopping on validation AUC.
3. The final score rank-blends the per-encoder cosine similarities with
   the head logit; the blend weights are selected on the validation split
   over a simplex grid.

Image-load failures are tracked per image: failed images are excluded from
the embedding tables and every pair touching one is dropped from evaluation
(``stack_embeddings``/``concat_embeddings``), so zero-filled tensors never
enter a metric.
"""

import itertools
import os
import random
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

from pair_sampling import (assert_pair_sets_disjoint, build_excluded_pairs,
                           generate_negative_pairs)

EMBED_INPUT_SIZE = 160
ARCFACE_INPUT_SIZE = 112
ARCFACE_URL = ('https://huggingface.co/immich-app/buffalo_l/resolve/main/'
               'recognition/model.onnx')
ARCFACE_MODEL_PATH = '_cache/arcface-w600k-r50.onnx'

eval_transform = transforms.Compose([
    transforms.Resize((EMBED_INPUT_SIZE, EMBED_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ArcFace expects (x - 127.5) / 127.5 RGB at 112x112 — the same math as
# ToTensor + Normalize(0.5, 0.5), only at the smaller input size.
arcface_transform = transforms.Compose([
    transforms.Resize((ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_encoder(device='cpu', pretrained='vggface2'):
    """Load a frozen pretrained FaceNet encoder ('vggface2' or 'casia-webface')."""
    from facenet_pytorch import InceptionResnetV1
    encoder = InceptionResnetV1(pretrained=pretrained).eval().to(device)
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def download_arcface(dest=ARCFACE_MODEL_PATH, url=ARCFACE_URL):
    """Download the ArcFace w600k_r50 ONNX recognizer (~174MB) once."""
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    print(f'Downloading ArcFace model to {dest} ...')
    urllib.request.urlretrieve(url, dest)
    return dest


class ArcFaceEncoder:
    """ArcFace ONNX recognizer wrapped as an ``embed_images`` encoder.

    Takes a (B, 3, 112, 112) tensor preprocessed by ``arcface_transform``
    and returns L2-normalized (B, 512) embeddings.
    """

    def __init__(self, model_path=ARCFACE_MODEL_PATH, num_threads=None):
        import onnxruntime as ort
        options = ort.SessionOptions()
        if num_threads:
            options.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(model_path, sess_options=options,
                                            providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, batch):
        out = self.session.run(None, {self.input_name: batch.cpu().numpy()})[0]
        return torch.nn.functional.normalize(torch.from_numpy(out), p=2, dim=1)


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


def clean_relations(relations_df, image_root='_train-faces'):
    """Drop relations whose members have no image directory or no images.

    Mirrors the notebook's Step 2 cleaning so scripts reproduce the same
    family split: a row survives only if both members' directories exist
    and contain at least one file.
    """
    def has_images(member):
        member_dir = os.path.join(image_root, member)
        return os.path.isdir(member_dir) and len(os.listdir(member_dir)) > 0

    keep = relations_df.apply(
        lambda row: has_images(row.p1) and has_images(row.p2), axis=1)
    return relations_df[keep]


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


def concat_embeddings(tables, pairs):
    """Stack pairs across multiple embedding tables, concatenating vectors.

    ``tables`` is an ordered sequence of {path: embedding} dicts. A pair is
    kept only when both of its images embedded in every table, so all
    per-encoder signals stay aligned on the same pair list. Returns
    (e1, e2, labels, kept_pairs).
    """
    kept = [p for p in pairs
            if all(p[0] in t and p[1] in t for t in tables)]
    dropped = len(pairs) - len(kept)
    if dropped:
        print(f'[Warning] Dropped {dropped} pairs with missing embeddings')
    e1 = torch.cat([torch.from_numpy(np.stack([t[p[0]] for p in kept]))
                    for t in tables], dim=1)
    e2 = torch.cat([torch.from_numpy(np.stack([t[p[1]] for p in kept]))
                    for t in tables], dim=1)
    labels = np.array([p[2] for p in kept], dtype=np.float32)
    return e1, e2, labels, kept


def segment_cosines(e1, e2, dim=512):
    """Per-encoder cosine similarities from concatenated embeddings.

    Splits the concatenation into ``dim``-sized segments (one per encoder,
    each unit-norm) and returns a list of cosine-score arrays.
    """
    return [cosine_scores(e1[:, start:start + dim], e2[:, start:start + dim])
            for start in range(0, e1.shape[1], dim)]


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


def rank_blend(scores, weights):
    """Blend score lists by their normalized ranks; result in (0, 1].

    ``scores`` and ``weights`` are same-length sequences; weights should
    sum to 1 for a score in (0, 1].
    """
    return sum(w * _normalized_ranks(s) for s, w in zip(scores, weights))


def _simplex_grid(k, step=0.1):
    """All k-tuples of non-negative multiples of ``step`` summing to 1."""
    n = round(1 / step)
    for combo in itertools.product(range(n + 1), repeat=k - 1):
        if sum(combo) <= n:
            yield tuple(c / n for c in combo) + ((n - sum(combo)) / n,)


def select_blend_weights(scores_val, y_val, step=0.1):
    """Grid-search simplex weights maximizing validation AUC.

    ``scores_val`` is a sequence of per-signal score arrays on the
    validation split. Returns (weights tuple, best val AUC).
    """
    best_weights, best_auc = None, 0.0
    for weights in _simplex_grid(len(scores_val), step):
        auc = roc_auc_score(y_val, rank_blend(scores_val, weights))
        if auc > best_auc:
            best_auc, best_weights = auc, weights
    return best_weights, best_auc
