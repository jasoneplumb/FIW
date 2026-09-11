"""Fast invariant checks for the sampling/evaluation logic (issues #10, #11).

Covers, on synthetic fixtures (no dataset or trained model required):
1. Negative-sampling policy — known relatives and same-family pairs are
   excluded, cross-family unrelated pairs are not.
2. Positive/negative pair-set disjointness.
3. Load-failure handling — failed image pairs get the NaN label sentinel
   and are excluded by valid_label_mask, so zero-filled images never enter
   evaluation.
"""

import itertools
import math
import random

import pandas as pd
import pytest
import torch
import torchvision.transforms as transforms
from PIL import Image

from image_pair_dataset import ImagePairDataset, valid_label_mask
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


# --- Load-failure handling ------------------------------------------------------

@pytest.fixture
def image_root(tmp_path):
    member_dir = tmp_path / 'F0001' / 'MID1'
    member_dir.mkdir(parents=True)
    Image.new('RGB', (16, 16), color=(120, 90, 60)).save(member_dir / 'ok.jpg')
    return str(tmp_path) + '/'


@pytest.fixture
def eval_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])


def test_load_failure_returns_nan_sentinel(image_root, eval_transform):
    data = [['F0001/MID1/missing.jpg', 'F0001/MID1/ok.jpg', 1.0]]
    dataset = ImagePairDataset(data, eval_transform, image_root=image_root)
    img0, img1, label = dataset[0]
    assert math.isnan(label.item())
    assert dataset.load_failures == 1
    assert torch.all(img0 == 0) and torch.all(img1 == 0)


def test_valid_pair_keeps_label(image_root, eval_transform):
    data = [['F0001/MID1/ok.jpg', 'F0001/MID1/ok.jpg', 1.0]]
    dataset = ImagePairDataset(data, eval_transform, image_root=image_root)
    img0, img1, label = dataset[0]
    assert label.item() == 1.0
    assert dataset.load_failures == 0
    assert img0.shape == (3, 112, 112)


def test_valid_label_mask_excludes_sentinels(image_root, eval_transform):
    data = [
        ['F0001/MID1/ok.jpg', 'F0001/MID1/ok.jpg', 1.0],
        ['F0001/MID1/missing.jpg', 'F0001/MID1/ok.jpg', 1.0],
        ['F0001/MID1/ok.jpg', 'F0001/MID1/ok.jpg', 0.0],
    ]
    dataset = ImagePairDataset(data, eval_transform, image_root=image_root)
    batch = [dataset[i] for i in range(len(dataset))]
    labels = torch.stack([sample[2] for sample in batch])
    mask = valid_label_mask(labels)
    assert mask.tolist() == [True, False, True]
    # zero-filled images from the failed pair never enter evaluation
    images = torch.stack([sample[0] for sample in batch])[mask]
    assert not torch.any(torch.all(images.flatten(1) == 0, dim=1))
