"""Fast invariant checks for the sampling/evaluation logic (issues #10, #11, #14).

Covers, on synthetic fixtures (no dataset or pretrained weights required):
1. Negative-sampling policy — known relatives and same-family pairs are
   excluded, cross-family unrelated pairs are not.
2. Positive/negative pair-set disjointness.
3. Load-failure handling — images that fail to load are flagged, never
   embedded, and every pair touching one is dropped before evaluation.
4. Scoring invariants — embedding cache round-trip, pair-feature symmetry,
   rank-blend ordering, blend-weight selection, and head training on
   separable data.
"""

import itertools
import random

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

from kinship import (build_pairs, embed_images, pair_features, rank_blend,
                     select_blend_weight, stack_embeddings, train_logreg_head)
from pair_sampling import (assert_pair_sets_disjoint, build_excluded_pairs,
                           generate_negative_pairs)

# --- Fixtures ---------------------------------------------------------------

RELATIONS = [
    ('F0001/MID1', 'F0001/MID2'),
    ('F0002/MID1', 'F0002/MID2'),
    ('F0002/MID2', 'F0002/MID3'),
]

MEMBERS = [
    'F0001/MID1', 'F0001/MID2', 'F0001/MID3',
    'F0002/MID1', 'F0002/MID2', 'F0002/MID3',
    'F0003/MID1',
]


@pytest.fixture
def relations_df():
    return pd.DataFrame(RELATIONS, columns=['p1', 'p2'])


@pytest.fixture
def member_images():
    return {m: [f'{m}/img{i}.jpg' for i in range(2)] for m in MEMBERS}


@pytest.fixture
def excluded_pairs(relations_df):
    return build_excluded_pairs(relations_df, MEMBERS)


def make_positives(relations_df, member_images):
    positives = []
    for _, row in relations_df.iterrows():
        for img1, img2 in itertools.product(member_images[row.p1],
                                            member_images[row.p2]):
            positives.append([img1, img2, 1.0])
    return positives


# --- Negative-sampling policy ------------------------------------------------

def test_excluded_pairs_contain_known_relations_both_directions(excluded_pairs):
    for p1, p2 in RELATIONS:
        assert (p1, p2) in excluded_pairs
        assert (p2, p1) in excluded_pairs


def test_excluded_pairs_contain_same_family_pairs(excluded_pairs):
    # F0001/MID3 has no explicit relation, but shares a family with MID1/MID2
    assert ('F0001/MID1', 'F0001/MID3') in excluded_pairs
    assert ('F0001/MID3', 'F0001/MID1') in excluded_pairs


def test_excluded_pairs_omit_cross_family_unrelated(excluded_pairs):
    assert ('F0001/MID1', 'F0002/MID1') not in excluded_pairs
    assert ('F0003/MID1', 'F0001/MID2') not in excluded_pairs


def test_generated_negatives_respect_policy(relations_df, member_images,
                                            excluded_pairs):
    positives = make_positives(relations_df, member_images)
    negatives = generate_negative_pairs(positives, MEMBERS, member_images,
                                        excluded_pairs, random.Random(0))
    assert len(negatives) == len(positives)
    for img1, img2, label in negatives:
        assert label == 0.0
        m1 = '/'.join(img1.split('/')[:2])
        m2 = '/'.join(img2.split('/')[:2])
        # never the same member, never same-family, never a known relation
        assert m1 != m2
        assert m1.split('/')[0] != m2.split('/')[0]
        assert (m1, m2) not in excluded_pairs


# --- Disjointness -------------------------------------------------------------

def test_positive_and_negative_pair_sets_are_disjoint(relations_df,
                                                      member_images,
                                                      excluded_pairs):
    positives = make_positives(relations_df, member_images)
    negatives = generate_negative_pairs(positives, MEMBERS, member_images,
                                        excluded_pairs, random.Random(0))
    assert_pair_sets_disjoint(positives, negatives)


def test_disjointness_assertion_raises_on_overlap():
    positives = [['a.jpg', 'b.jpg', 1.0]]
    negatives = [['a.jpg', 'b.jpg', 0.0]]
    with pytest.raises(AssertionError):
        assert_pair_sets_disjoint(positives, negatives)


# --- Embedding and load-failure handling ---------------------------------------

class CountingStubEncoder(nn.Module):
    """Deterministic parameter-free encoder that records forward calls."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 4).flatten(1)
        return torch.nn.functional.normalize(pooled, dim=1)


SMALL_TRANSFORM = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor(),
])


@pytest.fixture
def image_root(tmp_path):
    member_dir = tmp_path / 'F0001' / 'MID1'
    member_dir.mkdir(parents=True)
    Image.new('RGB', (16, 16), color=(120, 90, 60)).save(member_dir / 'a.jpg')
    Image.new('RGB', (16, 16), color=(20, 200, 160)).save(member_dir / 'b.jpg')
    return str(tmp_path)


def test_embed_images_flags_failures_and_never_embeds_them(image_root):
    paths = ['F0001/MID1/a.jpg', 'F0001/MID1/b.jpg', 'F0001/MID1/missing.jpg']
    embeddings, failed = embed_images(CountingStubEncoder(), paths, image_root,
                                      transform=SMALL_TRANSFORM,
                                      progress_every=0)
    assert failed == {'F0001/MID1/missing.jpg'}
    assert set(embeddings) == {'F0001/MID1/a.jpg', 'F0001/MID1/b.jpg'}
    for emb in embeddings.values():
        assert emb.dtype == np.float32
        assert np.isfinite(emb).all()


def test_embed_images_cache_roundtrip(image_root, tmp_path):
    paths = ['F0001/MID1/a.jpg', 'F0001/MID1/b.jpg']
    cache = str(tmp_path / 'cache.npz')
    first, _ = embed_images(CountingStubEncoder(), paths, image_root,
                            transform=SMALL_TRANSFORM, cache_path=cache,
                            progress_every=0)
    second_encoder = CountingStubEncoder()
    second, _ = embed_images(second_encoder, paths, image_root,
                             transform=SMALL_TRANSFORM, cache_path=cache,
                             progress_every=0)
    assert second_encoder.calls == 0  # everything served from cache
    for path in paths:
        assert np.allclose(first[path], second[path])


def test_stack_embeddings_drops_pairs_with_missing_images(image_root):
    paths = ['F0001/MID1/a.jpg', 'F0001/MID1/b.jpg']
    embeddings, _ = embed_images(CountingStubEncoder(), paths, image_root,
                                 transform=SMALL_TRANSFORM, progress_every=0)
    pairs = [
        ['F0001/MID1/a.jpg', 'F0001/MID1/b.jpg', 1.0],
        ['F0001/MID1/a.jpg', 'F0001/MID1/missing.jpg', 0.0],
    ]
    e1, e2, labels, kept = stack_embeddings(embeddings, pairs)
    assert len(kept) == 1
    assert kept[0][:2] == ['F0001/MID1/a.jpg', 'F0001/MID1/b.jpg']
    assert labels.tolist() == [1.0]
    assert e1.shape == e2.shape


# --- Scoring invariants ---------------------------------------------------------

def test_pair_features_are_symmetric():
    torch.manual_seed(0)
    a, b = torch.randn(5, 8), torch.randn(5, 8)
    assert torch.allclose(pair_features(a, b), pair_features(b, a))
    assert pair_features(a, b).shape == (5, 16)


def test_rank_blend_extremes_preserve_component_ordering():
    rng = np.random.default_rng(0)
    cos, logits = rng.normal(size=50), rng.normal(size=50)
    assert (np.argsort(rank_blend(cos, logits, 1.0)) == np.argsort(cos)).all()
    assert (np.argsort(rank_blend(cos, logits, 0.0)) == np.argsort(logits)).all()
    blended = rank_blend(cos, logits, 0.5)
    assert (blended > 0).all() and (blended <= 1).all()


def test_select_blend_weight_prefers_the_informative_score():
    rng = np.random.default_rng(0)
    labels = np.array([0.0, 1.0] * 50)
    perfect = labels + rng.normal(scale=0.01, size=100)  # near-perfect signal
    noise = rng.normal(size=100)
    weight, auc = select_blend_weight(perfect, noise, labels)
    assert weight == 1.0
    assert auc > 0.99
    weight, _ = select_blend_weight(noise, perfect, labels)
    assert weight == 0.0


def test_train_logreg_head_learns_separable_data():
    rng = np.random.default_rng(0)

    def make_split(n):
        base = rng.normal(size=(n, 8)).astype(np.float32)
        e1 = torch.from_numpy(base)
        related = torch.from_numpy(
            (base + rng.normal(scale=0.05, size=(n, 8))).astype(np.float32))
        unrelated = torch.from_numpy(
            rng.normal(size=(n, 8)).astype(np.float32))
        return torch.cat([e1, e1]), torch.cat([related, unrelated]), \
            np.concatenate([np.ones(n, dtype=np.float32),
                            np.zeros(n, dtype=np.float32)])

    e1_tr, e2_tr, y_tr = make_split(200)
    e1_va, e2_va, y_va = make_split(80)
    head, best_val = train_logreg_head(e1_tr, e2_tr, y_tr, e1_va, e2_va, y_va,
                                       epochs=30, patience=10,
                                       verbose_every=0)
    assert best_val > 0.9


# --- Split construction -----------------------------------------------------------

def test_build_pairs_is_balanced_disjoint_and_policy_compliant(tmp_path,
                                                               relations_df):
    for member in MEMBERS:
        member_dir = tmp_path / member
        member_dir.mkdir(parents=True)
        for i in range(2):
            Image.new('RGB', (8, 8), color=(i * 40, 80, 120)).save(
                member_dir / f'img{i}.jpg')

    splits = build_pairs(relations_df, image_root=str(tmp_path), seed=42)
    assert set(splits) == {'train', 'val', 'test'}
    families_seen = {}
    for name, data in splits.items():
        positives = [p for p in data if p[2] == 1.0]
        negatives = [p for p in data if p[2] == 0.0]
        assert len(negatives) <= len(positives)
        for img1, img2, _ in data:
            for img in (img1, img2):
                family = img.split('/')[0]
                assert families_seen.setdefault(family, name) == name
