#!/usr/bin/env python3
"""Quick component verification for the kinship pipeline (no dataset needed)."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import kinship

print('=' * 60)
print('FIW - Quick Component Verification')
print('=' * 60)

# 1. Frozen encoder
print('\n[1] Loading frozen FaceNet encoder...')
encoder = kinship.load_encoder()
assert not any(p.requires_grad for p in encoder.parameters())
print('    [OK] InceptionResnetV1 (VGGFace2) loaded, frozen, eval mode')

# 2. Embedding forward pass at the native input size
print('\n[2] Testing embedding forward pass...')
with torch.no_grad():
    dummy = torch.randn(2, 3, kinship.EMBED_INPUT_SIZE, kinship.EMBED_INPUT_SIZE)
    emb = encoder(dummy)
assert emb.shape == (2, 512), f'Expected (2, 512), got {emb.shape}'
norms = emb.norm(dim=1)
assert torch.allclose(norms, torch.ones(2), atol=1e-4), 'embeddings not L2-normalized'
print(f'    [OK] Output shape {tuple(emb.shape)}, L2-normalized (norms {norms.tolist()})')

# 3. Pair features and head
print('\n[3] Testing pair features and scoring head...')
head = kinship.make_head()
features = kinship.pair_features(emb[:1], emb[1:])
assert features.shape == (1, 1024)
logits = kinship.logit_scores(head, emb[:1], emb[1:])
assert logits.shape == (1,)
print('    [OK] pair_features -> (1, 1024), head logit computed')

# 4. Multi-signal rank blend and AUC on synthetic scores
print('\n[4] Testing multi-signal rank blend and AUC calculation...')
rng = np.random.default_rng(42)
labels = np.array([1.0] * 100 + [0.0] * 100)
signals = [
    np.concatenate([rng.normal(0.6, 0.2, 100), rng.normal(0.2, 0.2, 100)]),
    np.concatenate([rng.normal(0.5, 0.3, 100), rng.normal(0.1, 0.3, 100)]),
    np.concatenate([rng.normal(1.0, 0.5, 100), rng.normal(-1.0, 0.5, 100)]),
]
weights, val_auc = kinship.select_blend_weights(signals, labels)
blended = kinship.rank_blend(signals, weights)
auc = roc_auc_score(labels, blended)
assert abs(sum(weights) - 1.0) < 1e-9
assert 0.5 < auc <= 1.0
print(f'    [OK] blend weights {tuple(round(w, 1) for w in weights)}, synthetic AUC {auc:.4f}')

# 5. Segment cosines recover per-encoder scores from a concatenation
print('\n[5] Testing segment cosines on concatenated embeddings...')
a, b = torch.randn(4, 512), torch.randn(4, 512)
c, d = torch.randn(4, 512), torch.randn(4, 512)
seg = kinship.segment_cosines(torch.cat([a, c], dim=1), torch.cat([b, d], dim=1))
assert len(seg) == 2
assert np.allclose(seg[0], torch.nn.functional.cosine_similarity(a, b).numpy(), atol=1e-5)
print('    [OK] per-encoder cosines recovered from concatenation')

# 6. ArcFace wrapper (only when the model file is already downloaded)
import os
if os.path.exists(kinship.ARCFACE_MODEL_PATH):
    print('\n[6] Testing ArcFace ONNX encoder...')
    arcface = kinship.ArcFaceEncoder(kinship.ARCFACE_MODEL_PATH)
    with torch.no_grad():
        emb_arc = arcface(torch.randn(2, 3, 112, 112))
    assert emb_arc.shape == (2, 512)
    assert torch.allclose(emb_arc.norm(dim=1), torch.ones(2), atol=1e-4)
    print('    [OK] ArcFace output (2, 512), L2-normalized')
else:
    print('\n[6] ArcFace model not downloaded — skipping (run main.ipynb Step 5 first)')

print('\n' + '=' * 60)
print('VERIFICATION COMPLETE — all components functional')
print('=' * 60)
