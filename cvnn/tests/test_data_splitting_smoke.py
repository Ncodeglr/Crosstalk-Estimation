import numpy as np
import pytest
from torch.utils.data import Dataset

from cvnn.data import get_label_based_split_indices


class DummySegDataset(Dataset):
    def __init__(self, masks):
        self.masks = masks

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        # return (input, mask)
        return None, self.masks[idx]


@pytest.mark.smoke
def test_label_split_reproducible():
    masks = [np.array([[0, 1], [1, 0]]) for _ in range(30)]
    ds = DummySegDataset(masks)
    cfg = {"data": {"valid_ratio": 0.2, "test_ratio": 0.1}}

    a = get_label_based_split_indices(ds, "segmentation", cfg, random_state=123)
    b = get_label_based_split_indices(ds, "segmentation", cfg, random_state=123)
    assert a[0] == b[0] and a[1] == b[1]
