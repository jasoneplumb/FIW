"""Pair-sampling policy and helpers for FIW evaluation.

Negative-sampling policy (issue #10):
A candidate member pair (p1, p2) may be sampled as a negative (label 0.0)
only if ALL of the following hold:
  1. p1 != p2
  2. (p1, p2) is not a known positive relationship, in either direction
  3. p1 and p2 are not members of the same family
     (family membership implies potential kinship)

Members are identified as 'FAMILY/MEMBER' path fragments, matching the
layout of train_relationships.csv and the _train-faces/ image directory.
"""

import itertools
from collections import defaultdict


def build_excluded_pairs(relations_df, members):
    """Return the set of ordered member pairs excluded from negative sampling.

    The set is the union of:
    - all known positive relationships from relations_df, in both directions
    - all ordered pairs of members that share a family
    """
    positive_member_pairs = set()
    for _, row in relations_df.iterrows():
        positive_member_pairs.add((row.p1, row.p2))
        positive_member_pairs.add((row.p2, row.p1))

    family_members = defaultdict(set)
    for m in members:
        fam = m.split('/')[0]
        family_members[fam].add(m)

    same_family_pairs = set()
    for fam, fam_members in family_members.items():
        for m1, m2 in itertools.permutations(fam_members, 2):
            same_family_pairs.add((m1, m2))

    return positive_member_pairs | same_family_pairs


def generate_negative_pairs(positives, candidate_members, member_images,
                            excluded_pairs, rng, max_attempts_multiplier=100):
    """Sample negative image pairs under the exclusion policy.

    Draws member pairs from candidate_members with rng.choice, skipping any
    pair that violates the policy encoded in excluded_pairs, until the
    negative set matches len(positives) or the attempt budget is exhausted.
    Returns a list of [img1, img2, 0.0] entries.
    """
    negatives = []
    attempts = 0
    max_attempts = len(positives) * max_attempts_multiplier
    while len(negatives) < len(positives) and attempts < max_attempts:
        attempts += 1
        p1 = rng.choice(candidate_members)
        p2 = rng.choice(candidate_members)
        if p1 == p2 or (p1, p2) in excluded_pairs:
            continue
        p1_images = member_images[p1]
        p2_images = member_images[p2]
        for img1, img2 in itertools.product(p1_images, p2_images):
            negatives.append([img1, img2, 0.0])
            if len(negatives) >= len(positives):
                break
    return negatives


def assert_pair_sets_disjoint(positives, negatives):
    """Assert that the positive and negative image-pair sets never overlap."""
    pos_img_pairs = set((p[0], p[1]) for p in positives)
    neg_img_pairs = set((n[0], n[1]) for n in negatives)
    overlap = pos_img_pairs & neg_img_pairs
    assert len(overlap) == 0, f"Positive/negative overlap: {len(overlap)} pairs"
