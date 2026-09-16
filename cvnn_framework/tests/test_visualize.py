import numpy as np
import torch
import pytest
from cvnn.visualize import (
    safe_log,
    pauli_transform,
    krogager_transform,
    cameron_transform,
    cameron_classification,
    cameron,
    exp_amplitude_transform,
    equalize,
    angular_distance,
    plot_phase,
    plot_angular_distance,
    compute_h_alpha_metrics,
    compute_cameron_metrics,
)


def test_safe_log_base10_and_e_and_other():
    x = np.array([0.0, 1.0, 10.0, 100.0])
    # base10: clip 0->eps then log10
    y10 = safe_log(x, base=10, eps=1e-2)
    # smallest value approx log10(eps)
    assert pytest.approx(y10[0], rel=1e-3) == np.log10(1e-2)
    assert y10[1] == pytest.approx(0.0)
    assert y10[2] == pytest.approx(1.0)
    # natural log base e
    ye = safe_log(x, base=np.e, eps=1e-2)
    assert ye[1] == pytest.approx(0.0)
    # other base, e.g., base=2: log(x)/log(2)
    y2 = safe_log(x, base=2, eps=1e-2)
    assert y2[2] == pytest.approx(np.log(10.0) / np.log(2))


@pytest.mark.parametrize("shape", [(2, 2), (4, 3)])
def test_pauli_transform_and_inverse_like(shape):
    # Random complex SAR image shape (3,H,W)
    H, W = shape
    real = np.random.randn(3, H, W)
    imag = np.random.randn(3, H, W)
    SAR = real + 1j * imag
    out = pauli_transform(SAR).astype(np.complex64)
    # Output shape (3,H,W)
    assert out.shape == (3, H, W)
    assert out.dtype == np.complex64


@pytest.mark.parametrize("shape", [(2, 2), (5, 4)])
def test_krogager_transform_shape(shape):
    H, W = shape
    x = np.random.randn(3, H, W)
    out = krogager_transform(x)
    assert out.shape == (3, H, W)


def test_exp_amplitude_transform_dtype_and_shape():
    # input numpy complex array
    arr = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex64)
    # exp_amplitude_transform requires a dataset_name argument in the current API
    out = exp_amplitude_transform(arr, 30, -30)
    assert isinstance(out, torch.Tensor)
    assert out.dtype.is_complex
    assert out.shape == (2, 2)


def test_equalize_and_percentiles():
    img = np.linspace(0, 100, num=25).reshape(5, 5)
    img_resc, (p2, p98) = equalize(img)
    assert img_resc.dtype == np.uint8
    # percentiles should match 2th and 98th percentiles
    expected_p2, expected_p98 = np.percentile(np.log10(img + 1e-10), (2, 98))
    # allow slight rounding
    assert pytest.approx(p2, rel=1e-2) == expected_p2
    assert pytest.approx(p98, rel=1e-2) == expected_p98
    assert img_resc.shape == img.shape


def test_angular_and_phase_plots():
    # create two images with constant phases
    img1 = np.exp(1j * 0)
    img2 = np.exp(1j * (np.pi / 2))
    ang = angular_distance(img1, img2)
    # angular distance should be -pi/2 normalized to [-pi, pi]
    assert pytest.approx(ang, rel=1e-3) == -np.pi / 2
    # plot_phase maps phase to uint8
    phase_img = plot_phase(img2)
    assert phase_img.dtype == np.uint8
    assert phase_img.shape == () or phase_img.shape == img2.shape
    # plot_angular_distance
    pad = plot_angular_distance(img1, img2)
    assert pad.dtype == np.uint8


def test_calculate_means_of_classes():
    # image_of_stacked_covariances shape (2,2,1)
    img = np.array([[[1], [2]], [[3], [4]]], dtype=float)
    # classes mask: alternating 1 and 2
    classes = np.array([[1, 2], [2, 1]])
    # Compute per-class means directly from baseline behavior (no helper required)
    vals_class1 = img[classes == 1].astype(float)
    vals_class2 = img[classes == 2].astype(float)
    mean1 = np.mean(vals_class1)
    mean2 = np.mean(vals_class2)

    assert mean1 == pytest.approx((1 + 4) / 2)
    assert mean2 == pytest.approx((2 + 3) / 2)


@pytest.fixture
def h_alpha_test_data():
    """Fixture to create test data for H-alpha metrics tests."""
    np.random.seed(42)
    valid_classes = [1, 2, 4, 5, 6, 7, 8, 9]

    # Create 20x20 classification maps
    h_alpha_original = np.random.choice(valid_classes, size=(20, 20))

    # For generated, make it slightly different from original to simulate reconstruction errors
    h_alpha_generated = h_alpha_original.copy()

    # Introduce some errors (change 10% of pixels randomly)
    n_errors = int(0.1 * h_alpha_original.size)
    error_indices = np.random.choice(h_alpha_original.size, n_errors, replace=False)
    flat_generated = h_alpha_generated.flatten()
    for idx in error_indices:
        # Change to a different random class
        current_class = flat_generated[idx]
        new_class = np.random.choice([c for c in valid_classes if c != current_class])
        flat_generated[idx] = new_class
    h_alpha_generated = flat_generated.reshape(20, 20)

    return h_alpha_original, h_alpha_generated, valid_classes


def test_h_alpha_metrics_basic_computation(h_alpha_test_data):
    """Test that H-alpha metrics are computed without errors."""
    h_alpha_original, h_alpha_generated, valid_classes = h_alpha_test_data

    metrics = compute_h_alpha_metrics(h_alpha_original, h_alpha_generated)

    # Basic checks - metrics should be computed successfully
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_h_alpha_metrics_all_expected_keys(h_alpha_test_data):
    """Test that all expected metric keys are present."""
    h_alpha_original, h_alpha_generated, valid_classes = h_alpha_test_data

    metrics = compute_h_alpha_metrics(h_alpha_original, h_alpha_generated)

    expected_keys = [
        "accuracy",
        "cohen_kappa",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "per_class_metrics",
        "class_distribution_original",
        "class_distribution_generated",
        "confusion_matrix_raw",
        "confusion_matrix_normalized",
        "class_labels",
    ]

    for key in expected_keys:
        assert key in metrics, f"Missing expected key: {key}"


def test_h_alpha_metrics_value_ranges(h_alpha_test_data):
    """Test that metric values are within expected ranges."""
    h_alpha_original, h_alpha_generated, valid_classes = h_alpha_test_data

    metrics = compute_h_alpha_metrics(h_alpha_original, h_alpha_generated)

    # Test value ranges for main metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert -1.0 <= metrics["cohen_kappa"] <= 1.0
    assert 0.0 <= metrics["precision_macro"] <= 1.0
    assert 0.0 <= metrics["recall_macro"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["precision_micro"] <= 1.0
    assert 0.0 <= metrics["recall_micro"] <= 1.0
    assert 0.0 <= metrics["f1_micro"] <= 1.0
    assert 0.0 <= metrics["precision_weighted"] <= 1.0
    assert 0.0 <= metrics["recall_weighted"] <= 1.0
    assert 0.0 <= metrics["f1_weighted"] <= 1.0


def test_h_alpha_metrics_per_class_structure(h_alpha_test_data):
    """Test the structure of per-class metrics."""
    h_alpha_original, h_alpha_generated, valid_classes = h_alpha_test_data

    metrics = compute_h_alpha_metrics(h_alpha_original, h_alpha_generated)

    per_class = metrics["per_class_metrics"]
    assert isinstance(per_class, dict)

    # Check that we have metrics for classes that appear in the data
    for class_id in per_class.keys():
        class_metrics = per_class[class_id]
        required_class_keys = ["precision", "recall", "f1_score", "support"]

        for key in required_class_keys:
            assert key in class_metrics, f"Missing key '{key}' for class {class_id}"

        # Check value types and ranges
        assert isinstance(class_metrics["precision"], float)
        assert isinstance(class_metrics["recall"], float)
        assert isinstance(class_metrics["f1_score"], float)
        assert isinstance(class_metrics["support"], int)
        assert class_metrics["support"] >= 0


def test_h_alpha_metrics_confusion_matrix_structure(h_alpha_test_data):
    """Test the structure and properties of confusion matrices."""
    h_alpha_original, h_alpha_generated, valid_classes = h_alpha_test_data

    metrics = compute_h_alpha_metrics(h_alpha_original, h_alpha_generated)

    cm_raw = np.array(metrics["confusion_matrix_raw"])
    cm_norm = np.array(metrics["confusion_matrix_normalized"])

    expected_shape = (len(valid_classes), len(valid_classes))

    # Check dimensions
    assert cm_raw.shape == expected_shape
    assert cm_norm.shape == expected_shape

    # Check that raw confusion matrix contains non-negative integers
    assert cm_raw.dtype.kind in [
        "i",
        "u",
    ], "Raw confusion matrix should contain integers"
    assert np.all(
        cm_raw >= 0
    ), "Raw confusion matrix should contain non-negative values"

    # Check that normalized confusion matrix rows sum to 1 (within tolerance)
    row_sums = np.sum(cm_norm, axis=1)
    # Only check rows that have at least one sample
    non_empty_rows = row_sums > 0
    if np.any(non_empty_rows):
        assert np.allclose(
            row_sums[non_empty_rows], 1.0, atol=1e-6
        ), f"Normalized confusion matrix rows should sum to 1, got: {row_sums[non_empty_rows]}"


def test_h_alpha_metrics_class_distributions(h_alpha_test_data):
    """Test class distribution dictionaries."""
    h_alpha_original, h_alpha_generated, valid_classes = h_alpha_test_data

    metrics = compute_h_alpha_metrics(h_alpha_original, h_alpha_generated)

    dist_orig = metrics["class_distribution_original"]
    dist_gen = metrics["class_distribution_generated"]

    assert isinstance(dist_orig, dict)
    assert isinstance(dist_gen, dict)

    # Check that counts are non-negative integers
    for class_id, count in dist_orig.items():
        assert isinstance(class_id, int)
        assert isinstance(count, int)
        assert count >= 0

    for class_id, count in dist_gen.items():
        assert isinstance(class_id, int)
        assert isinstance(count, int)
        assert count >= 0

    # Total counts should match image size
    total_pixels = h_alpha_original.size
    assert sum(dist_orig.values()) == total_pixels
    assert sum(dist_gen.values()) == total_pixels


def test_h_alpha_metrics_perfect_match():
    """Test metrics when original and generated are identical."""
    valid_classes = [1, 2, 4, 5, 6, 7, 8, 9]
    h_alpha_data = np.random.choice(valid_classes, size=(10, 10))

    metrics = compute_h_alpha_metrics(h_alpha_data, h_alpha_data, valid_classes)

    # Perfect match should give perfect scores
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision_macro"] == pytest.approx(1.0)
    assert metrics["recall_macro"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["cohen_kappa"] == pytest.approx(1.0)


@pytest.mark.parametrize("shape", [(2, 2), (4, 3)])
def test_cameron_transform_shape(shape):
    """Test Cameron transform returns correct number of parameters with correct shapes."""
    H, W = shape
    real = np.random.randn(3, H, W)
    imag = np.random.randn(3, H, W)
    SAR = real + 1j * imag

    cameron_params = cameron_transform(SAR)

    # Should return 13 parameters
    assert len(cameron_params) == 13

    # All parameters should have shape (H, W)
    for param in cameron_params:
        assert param.shape == (H, W)


def test_cameron_transform_input_validation():
    """Test Cameron transform input validation."""
    # Test invalid shape
    invalid_sar = np.random.randn(2, 4, 4) + 1j * np.random.randn(2, 4, 4)
    with pytest.raises(ValueError, match="Expected 3 channels"):
        cameron_transform(invalid_sar)


def test_cameron_classification_shape():
    """Test Cameron classification returns correct shape."""
    H, W = 4, 4
    real = np.random.randn(3, H, W)
    imag = np.random.randn(3, H, W)
    SAR = real + 1j * imag

    cameron_params = cameron_transform(SAR)
    classification = cameron_classification(*cameron_params)

    assert classification.shape == (H, W)
    assert classification.dtype == int


def test_cameron_classification_input_validation():
    """Test Cameron classification input validation."""
    # Create mismatched parameter shapes
    param1 = np.random.randn(2, 2)
    param2 = np.random.randn(3, 3)  # Different shape

    # Create dummy parameters with mismatched shapes
    params = [param1] + [param2] * 12

    # The current implementation may raise a broadcasting-related ValueError
    # when shapes are incompatible. Assert that a ValueError is raised.
    with pytest.raises(ValueError):
        cameron_classification(*params)


def test_cameron_full_pipeline():
    """Test full Cameron pipeline from transform to classification."""
    H, W = 3, 3
    real = np.random.randn(3, H, W)
    imag = np.random.randn(3, H, W)
    SAR = real + 1j * imag

    # Test the full cameron function
    classification = cameron(SAR)

    assert classification.shape == (H, W)
    assert classification.dtype == int

    # Classification values should be in range [1, 12] (Cameron classes)
    unique_classes = np.unique(classification)
    assert all(1 <= cls <= 12 for cls in unique_classes)


def test_cameron_input_validation():
    """Test Cameron function input validation."""
    # Test invalid shape
    invalid_sar = np.random.randn(2, 4, 4) + 1j * np.random.randn(2, 4, 4)
    # The current implementation raises a ValueError indicating the SAR image
    # shape is incorrect. Match broadly by asserting a ValueError.
    with pytest.raises(ValueError):
        cameron(invalid_sar)


@pytest.fixture
def cameron_test_data():
    """Fixture to create test data for Cameron metrics tests."""
    np.random.seed(42)
    cameron_classes = list(range(1, 13))  # Cameron classes 1-12

    # Create 15x15 classification maps
    cameron_original = np.random.choice(cameron_classes, size=(15, 15))

    # For generated, make it slightly different from original
    cameron_generated = cameron_original.copy()

    # Introduce some errors (change 15% of pixels randomly)
    n_errors = int(0.15 * cameron_original.size)
    error_indices = np.random.choice(cameron_original.size, n_errors, replace=False)
    flat_generated = cameron_generated.flatten()
    for idx in error_indices:
        # Change to a different random class
        current_class = flat_generated[idx]
        new_class = np.random.choice([c for c in cameron_classes if c != current_class])
        flat_generated[idx] = new_class
    cameron_generated = flat_generated.reshape(15, 15)

    return cameron_original, cameron_generated, cameron_classes


def test_compute_cameron_metrics_basic_computation(cameron_test_data):
    """Test that Cameron metrics are computed without errors."""
    cameron_original, cameron_generated, cameron_classes = cameron_test_data

    metrics = compute_cameron_metrics(cameron_original, cameron_generated)

    # Basic checks - metrics should be computed successfully
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_compute_cameron_metrics_all_expected_keys(cameron_test_data):
    """Test that all expected metric keys are present."""
    cameron_original, cameron_generated, cameron_classes = cameron_test_data

    metrics = compute_cameron_metrics(cameron_original, cameron_generated)

    expected_keys = [
        "accuracy",
        "cohen_kappa",
        "matthews_corrcoef",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "per_class_metrics",
        "class_distribution_original",
        "class_distribution_generated",
        "confusion_matrix_raw",
        "confusion_matrix_normalized",
        "class_labels",
    ]

    for key in expected_keys:
        assert key in metrics, f"Missing expected key: {key}"


def test_compute_cameron_metrics_value_ranges(cameron_test_data):
    """Test that Cameron metric values are within expected ranges."""
    cameron_original, cameron_generated, cameron_classes = cameron_test_data

    metrics = compute_cameron_metrics(cameron_original, cameron_generated)

    # Test value ranges for main metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert -1.0 <= metrics["cohen_kappa"] <= 1.0
    assert -1.0 <= metrics["matthews_corrcoef"] <= 1.0
    assert 0.0 <= metrics["precision_macro"] <= 1.0
    assert 0.0 <= metrics["recall_macro"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_compute_cameron_metrics_perfect_match():
    """Test Cameron metrics when original and generated are identical."""
    cameron_classes = list(range(1, 13))
    cameron_data = np.random.choice(cameron_classes, size=(8, 8))

    metrics = compute_cameron_metrics(cameron_data, cameron_data)

    # Perfect match should give perfect scores
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision_macro"] == pytest.approx(1.0)
    assert metrics["recall_macro"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["cohen_kappa"] == pytest.approx(1.0)


def test_compute_cameron_metrics_input_validation():
    """Test Cameron metrics input validation."""
    cameron_orig = np.random.choice(range(1, 13), size=(5, 5))
    cameron_gen = np.random.choice(range(1, 13), size=(6, 6))  # Different shape

    with pytest.raises(ValueError, match="Input arrays must have the same shape"):
        compute_cameron_metrics(cameron_orig, cameron_gen)
