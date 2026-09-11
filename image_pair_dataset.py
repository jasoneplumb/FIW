"""Image-pair dataset with explicit load-failure handling (issue #10).

A pair whose images fail to load is returned with a NaN label sentinel and
zero-filled tensors. Evaluation code must drop these samples with
valid_label_mask before computing any metric — zero-filled images must
never enter evaluation.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image


class ImagePairDataset(Dataset):
    LOAD_FAILURE_SENTINEL = float('nan')

    def __init__(self, data, transform, image_root='_train-faces/'):
        self.image_pairs = [sublist[:-1] for sublist in data]
        self.labels = torch.tensor([sublist[-1] for sublist in data], dtype=torch.float32)
        self.transform = transform
        self.image_root = image_root
        self.load_failures = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            img0 = Image.open(self.image_root + self.image_pairs[idx][0])
            img1 = Image.open(self.image_root + self.image_pairs[idx][1])
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            return img0, img1, self.labels[idx]
        except Exception as e:
            print(f"[Warning] Failed to load image pair {idx}: {e}")
            self.load_failures += 1
            return (torch.zeros(3, 112, 112), torch.zeros(3, 112, 112),
                    torch.tensor(self.LOAD_FAILURE_SENTINEL))


def valid_label_mask(labels):
    """Boolean mask selecting samples whose labels are not the NaN sentinel."""
    return ~torch.isnan(labels)
