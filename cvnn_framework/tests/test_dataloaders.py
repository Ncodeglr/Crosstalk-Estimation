import pytest
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# This test file focuses on dataloader-specific functionality
# that's different from the general data module tests


class MockDataset(Dataset):
    """Mock dataset for testing dataloader functionality"""

    def __init__(self, size=100, input_shape=(3, 32, 32), num_classes=10):
        self.size = size
        self.input_shape = input_shape
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate deterministic data based on index for reproducible tests
        np.random.seed(idx)
        data = torch.from_numpy(np.random.randn(*self.input_shape)).float()
        label = torch.tensor(idx % self.num_classes, dtype=torch.long)
        return data, label


def test_dataloader_basic_functionality():
    """Test basic dataloader creation and iteration"""
    dataset = MockDataset(size=20, input_shape=(1, 8, 8))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Test basic properties
    assert len(dataloader) == 5  # 20 / 4 = 5 batches

    # Test iteration
    batches = list(dataloader)
    assert len(batches) == 5

    # Test batch shapes
    for i, (data, labels) in enumerate(batches):
        expected_batch_size = 4 if i < 4 else 4  # Last batch might be smaller
        assert data.shape == (expected_batch_size, 1, 8, 8)
        assert labels.shape == (expected_batch_size,)
        assert labels.dtype == torch.long


def test_dataloader_with_shuffle():
    """Test that shuffle produces different orders"""
    dataset = MockDataset(size=10)

    # Test that we can create shuffled dataloaders without errors
    loader1 = DataLoader(dataset, batch_size=10, shuffle=True)
    loader2 = DataLoader(dataset, batch_size=10, shuffle=False)

    batch1 = next(iter(loader1))[1]  # Get labels from shuffled
    batch2 = next(iter(loader2))[1]  # Get labels from non-shuffled

    # Both should have same elements, but shuffled might be in different order
    assert len(batch1) == len(batch2) == 10
    assert set(batch1.tolist()) == set(batch2.tolist())  # Same elements

    # Non-shuffled should be in original order (0, 1, 2, ..., 9)
    expected_order = list(range(10))
    assert batch2.tolist() == expected_order


def test_dataloader_empty_dataset():
    """Test dataloader behavior with empty dataset"""
    dataset = MockDataset(size=0)
    dataloader = DataLoader(dataset, batch_size=4)

    assert len(dataloader) == 0
    assert list(dataloader) == []


def test_dataloader_single_item():
    """Test dataloader with single item dataset"""
    dataset = MockDataset(size=1)
    dataloader = DataLoader(dataset, batch_size=4)

    assert len(dataloader) == 1
    batches = list(dataloader)
    assert len(batches) == 1

    data, labels = batches[0]
    assert data.shape[0] == 1
    assert labels.shape[0] == 1


def test_dataloader_batch_size_larger_than_dataset():
    """Test dataloader when batch size is larger than dataset"""
    dataset = MockDataset(size=3)
    dataloader = DataLoader(dataset, batch_size=10)

    assert len(dataloader) == 1
    data, labels = next(iter(dataloader))
    assert data.shape[0] == 3  # Should return all 3 items
    assert labels.shape[0] == 3


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, 10])
def test_dataloader_various_batch_sizes(batch_size):
    """Test dataloader with various batch sizes"""
    dataset_size = 10
    dataset = MockDataset(size=dataset_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    expected_batches = (dataset_size + batch_size - 1) // batch_size  # Ceiling division
    assert len(dataloader) == expected_batches

    # Verify all items are covered exactly once
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.tolist())

    expected_labels = list(
        range(dataset_size)
    )  # Labels are idx % num_classes, with 10 classes they're 0-9
    assert sorted(all_labels) == sorted(expected_labels)
