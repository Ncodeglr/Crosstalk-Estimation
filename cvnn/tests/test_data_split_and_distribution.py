import numpy as np
import importlib

cvnn_data = importlib.import_module("cvnn.data")


def test_empty_dataset_distribution():
    class EmptyDataset:
        def __len__(self):
            return 0

        def __getitem__(self, i):
            raise IndexError()

    dist = cvnn_data.calculate_class_distribution(EmptyDataset(), "segmentation")
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (0, 0)


def test_classification_stratified_split_reproducible():
    class TinyDataset:
        def __init__(self, labels):
            import numpy as np
            import importlib

            cvnn_data = importlib.import_module("cvnn.data")

            def test_empty_dataset_distribution():
                class EmptyDataset:
                    def __len__(self):
                        return 0

                    def __getitem__(self, i):
                        raise IndexError()

                dist = cvnn_data.calculate_class_distribution(
                    EmptyDataset(), "segmentation"
                )
                assert isinstance(dist, np.ndarray)
                assert dist.shape == (0, 0)

            def test_classification_stratified_split_reproducible():
                class TinyDataset:
                    def __init__(self, labels):
                        self._labels = labels

                    def __len__(self):
                        return len(self._labels)

                    def __getitem__(self, i):
                        return None, self._labels[i]

                labels = [0, 0, 1, 1, 2, 2]
                ds = TinyDataset(labels)
                cfg = {"data": {"valid_ratio": 0.2, "test_ratio": 0.0}}

                a = cvnn_data.get_label_based_split_indices(
                    ds, "classification", cfg, random_state=1
                )
                b = cvnn_data.get_label_based_split_indices(
                    ds, "classification", cfg, random_state=1
                )
                assert a == b
                train, valid, test = a
                assert len(train) + len(valid) == len(labels)
                # validate at least one sample in validation for this ratio and size
                assert len(valid) >= 1

            def test_segmentation_histogram_small():
                class TinySeg:
                    def __init__(self, masks):
                        self._masks = masks

                    def __len__(self):
                        return len(self._masks)

                    def __getitem__(self, i):
                        return None, self._masks[i]

                masks = [
                    [[0, 0], [0, 0]],
                    [[0, 1], [0, 1]],
                    [[1, 1], [1, 1]],
                ]
                ds = TinySeg(masks)
                dist = cvnn_data.calculate_class_distribution(ds, "segmentation")
                # Should have discovered classes {0,1}
                assert dist.shape[1] == 2
                # first mask is all zeros -> histogram should be [1.0, 0.0]
                assert np.allclose(dist[0], [1.0, 0.0])
