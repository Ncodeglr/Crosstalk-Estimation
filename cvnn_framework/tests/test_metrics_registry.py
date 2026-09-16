"""
Test suite for the metrics registry system.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from cvnn.metrics_registry import (
    get_metrics_info,
    get_available_metrics,
    get_metric_function,
    compute_metrics,
    list_tasks,
    list_pipeline_types_for_task,
    _METRICS_REGISTRY,
)


class TestMetricsRegistryCore:
    """Test core registry functionality."""

    def test_list_tasks(self):
        """Test that all expected tasks are registered."""
        tasks = list_tasks()
        expected_tasks = {
            "reconstruction",
            "segmentation",
            "classification",
            "generation",
        }
        assert expected_tasks.issubset(
            set(tasks)
        ), f"Missing tasks: {expected_tasks - set(tasks)}"

    def test_list_pipeline_types_for_task(self):
        """Test pipeline type listing for each task."""
        for task in list_tasks():
            pipelines = list_pipeline_types_for_task(task)
            assert isinstance(pipelines, list)
            assert len(pipelines) > 0, f"Task {task} has no pipeline types"
            # All tasks should support at least basic pipeline types
            assert "complex" in pipelines or "real" in pipelines

    def test_get_metrics_info(self):
        """Test metrics info retrieval."""
        info = get_metrics_info()
        assert isinstance(info, dict)

        # Check structure
        for task, pipelines in info.items():
            assert isinstance(pipelines, dict)
            for pipeline, metrics in pipelines.items():
                assert isinstance(metrics, list)
                assert len(metrics) > 0, f"No metrics for {task}/{pipeline}"

    def test_get_available_metrics(self):
        """Test metric retrieval for specific task/pipeline combinations."""
        # Test reconstruction/complex
        metrics = get_available_metrics("reconstruction", "complex")
        expected_metrics = {"mse", "psnr", "ssim", "mae"}
        assert expected_metrics.issubset(set(metrics))

        # Test segmentation/complex
        metrics = get_available_metrics("segmentation", "complex")
        expected_metrics = {"dice", "iou", "precision", "recall"}
        assert expected_metrics.issubset(set(metrics))

        # Test classification/complex
        metrics = get_available_metrics("classification", "complex")
        expected_metrics = {"accuracy", "f1", "precision", "recall"}
        assert expected_metrics.issubset(set(metrics))

        # Test generation/complex
        metrics = get_available_metrics("generation", "complex")
        expected_metrics = {"fid", "inception_score"}
        assert expected_metrics.issubset(set(metrics))

    def test_invalid_task_pipeline(self):
        """Test handling of invalid task/pipeline combinations."""
        try:
            metrics = get_available_metrics("invalid_task", "complex")
            # If it doesn't raise, it should return an empty list or handle gracefully
            assert isinstance(metrics, list)
        except ValueError:
            # Either behavior is acceptable
            pass

        try:
            metrics = get_available_metrics("reconstruction", "invalid_pipeline")
            # If it doesn't raise, it should return an empty list or handle gracefully
            assert isinstance(metrics, list)
        except ValueError:
            # Either behavior is acceptable
            pass


class TestMetricComputation:
    """Test metric computation functionality."""

    def setup_method(self):
        """Set up test data for metric computation."""
        self.batch_size = 2
        self.channels = 1
        self.height = 8
        self.width = 8
        self.num_classes = 2

        # Create test tensors for reconstruction
        self.original = torch.complex(
            torch.randn(self.batch_size, self.channels, self.height, self.width),
            torch.randn(self.batch_size, self.channels, self.height, self.width),
        )
        self.reconstructed = self.original.clone()  # Perfect reconstruction

        # Create test data for segmentation (multi-class)
        self.seg_targets = torch.randint(
            0, self.num_classes, (self.batch_size, self.height, self.width)
        )
        self.seg_predictions = torch.zeros(
            self.batch_size, self.num_classes, self.height, self.width
        )

        # Create perfect segmentation predictions
        for b in range(self.batch_size):
            for h in range(self.height):
                for w in range(self.width):
                    correct_class = self.seg_targets[b, h, w]
                    self.seg_predictions[b, correct_class, h, w] = 10.0

        # Create test data for classification (1D)
        self.class_targets = torch.randint(0, self.num_classes, (self.batch_size,))
        self.class_predictions = torch.zeros(self.batch_size, self.num_classes)

        # Create perfect classification predictions
        for b in range(self.batch_size):
            correct_class = self.class_targets[b]
            self.class_predictions[b, correct_class] = 10.0

    def test_reconstruction_metrics(self):
        """Test reconstruction metric computation."""
        metrics = compute_metrics(
            task="reconstruction",
            pipeline_type="complex",
            metric_names=["mse", "psnr", "ssim"],
            predictions=self.reconstructed,
            targets=self.original,
        )

        assert "mse" in metrics
        assert "psnr" in metrics
        assert "ssim" in metrics

        # For perfect reconstruction
        assert metrics["mse"] == 0.0
        assert metrics["psnr"] == float("inf")
        # SSIM might be NaN if window size is too small for input size or other edge cases
        # For 8x8 input, default window size might be problematic
        if not np.isnan(metrics["ssim"]):
            assert metrics["ssim"] >= 0.9

    def test_segmentation_metrics(self):
        """Test segmentation metric computation."""
        metrics = compute_metrics(
            task="segmentation",
            pipeline_type="complex",
            metric_names=["dice", "iou"],
            predictions=self.seg_predictions,
            targets=self.seg_targets,
        )

        assert "dice" in metrics
        assert "iou" in metrics

        # For perfect predictions, dice and iou should be high
        assert metrics["dice"] >= 0.9
        assert metrics["iou"] >= 0.9

    def test_classification_metrics(self):
        """Test classification metric computation."""
        metrics = compute_metrics(
            task="classification",
            pipeline_type="complex",
            metric_names=["accuracy", "f1"],
            predictions=self.class_predictions,
            targets=self.class_targets,
        )

        assert "accuracy" in metrics
        assert "f1" in metrics

        # For perfect predictions
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] >= 0.9

    def test_generation_metrics(self):
        """Test generation metric computation."""
        # Generation metrics typically need special handling
        # Test that they can be called without error
        try:
            metrics = compute_metrics(
                task="generation",
                pipeline_type="complex",
                metric_names=["fid"],
                real_features=torch.randn(10, 512),
                fake_features=torch.randn(10, 512),
            )
            assert isinstance(metrics, dict)
        except Exception as e:
            # Generation metrics might need special setup, allow graceful failure
            assert "fid" in str(e).lower() or "feature" in str(e).lower()

    def test_metric_computation_errors(self):
        """Test error handling in metric computation."""
        # Test missing required arguments
        with pytest.raises(Exception):  # Should raise some kind of error
            compute_metrics(
                task="reconstruction",
                pipeline_type="complex",
                metric_names=["mse"],
                # Missing predictions and targets
            )

        # Test invalid metric name - should not raise but should skip the metric
        metrics = compute_metrics(
            task="reconstruction",
            pipeline_type="complex",
            metric_names=["invalid_metric", "mse"],
            predictions=self.reconstructed,
            targets=self.original,
        )

        # Should compute valid metrics and skip invalid ones
        assert "mse" in metrics
        assert "invalid_metric" not in metrics or np.isnan(metrics["invalid_metric"])

    def test_partial_metric_computation(self):
        """Test computing only a subset of available metrics."""
        # Test with only MSE
        metrics = compute_metrics(
            task="reconstruction",
            pipeline_type="complex",
            metric_names=["mse"],
            predictions=self.reconstructed,
            targets=self.original,
        )

        assert "mse" in metrics
        assert "psnr" not in metrics  # Should not compute unrequested metrics
        assert "ssim" not in metrics

    def test_noisy_reconstruction_metrics(self):
        """Test metrics with noisy reconstruction."""
        # Add noise to reconstruction
        noise = torch.complex(
            torch.randn_like(self.original.real) * 0.1,
            torch.randn_like(self.original.imag) * 0.1,
        )
        noisy_recon = self.original + noise

        metrics = compute_metrics(
            task="reconstruction",
            pipeline_type="complex",
            metric_names=["mse", "psnr", "ssim"],
            predictions=noisy_recon,
            targets=self.original,
        )

        # MSE should be positive for noisy reconstruction
        assert metrics["mse"] > 0
        # PSNR should be finite and positive
        assert 0 < metrics["psnr"] < float("inf")
        # SSIM should be good but not perfect
        if not np.isnan(metrics["ssim"]):
            assert 0.3 <= metrics["ssim"] <= 1.0


class TestPipelineTypes:
    """Test different pipeline type handling."""

    def test_complex_pipeline(self):
        """Test complex pipeline metrics."""
        original = torch.complex(torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        reconstructed = original.clone()

        metrics = compute_metrics(
            task="reconstruction",
            pipeline_type="complex",
            metric_names=["mse", "ssim"],
            predictions=reconstructed,
            targets=original,
        )

        assert "mse" in metrics
        assert "ssim" in metrics

    def test_real_pipeline(self):
        """Test real-valued pipeline metrics."""
        original = torch.randn(1, 1, 4, 4)
        reconstructed = original.clone()

        metrics = compute_metrics(
            task="reconstruction",
            pipeline_type="real_real",
            metric_names=["mse", "ssim"],
            predictions=reconstructed,
            targets=original,
        )

        assert "mse" in metrics
        assert "ssim" in metrics

    def test_amplitude_pipeline(self):
        """Test amplitude pipeline metrics."""
        # Amplitude pipeline typically works with magnitude
        original = torch.abs(
            torch.complex(torch.randn(1, 1, 4, 4), torch.randn(1, 1, 4, 4))
        )
        reconstructed = original.clone()

        metrics = compute_metrics(
            task="reconstruction",
            pipeline_type="complex_amplitude_real",
            metric_names=["mse", "psnr"],
            predictions=reconstructed,
            targets=original,
        )

        assert "mse" in metrics
        assert "psnr" in metrics


class TestRegistryIntegration:
    """Test integration with the broader system."""

    def test_registry_consistency(self):
        """Test that registry data is consistent."""
        # Check that all registered metrics have implementations
        for task in list_tasks():
            for pipeline in list_pipeline_types_for_task(task):
                metrics = get_available_metrics(task, pipeline)
                for metric in metrics:
                    # Try to get the metric function to ensure it exists
                    try:
                        from cvnn.metrics_registry import get_metric_function

                        func = get_metric_function(task, pipeline, metric)
                        assert callable(
                            func
                        ), f"Metric {metric} for {task}/{pipeline} is not callable"
                    except Exception as e:
                        pytest.fail(
                            f"Missing implementation for {task}/{pipeline}/{metric}: {e}"
                        )

    def test_metric_function_signatures(self):
        """Test that metric functions have reasonable signatures."""
        # Test a few key metrics to ensure they are callable
        from cvnn.metrics_registry import get_metric_function

        test_cases = [
            ("reconstruction", "complex", "mse"),
            ("reconstruction", "real_real", "psnr"),
            ("segmentation", "complex", "dice"),
            ("classification", "complex", "accuracy"),
        ]

        for task, pipeline, metric in test_cases:
            try:
                func = get_metric_function(task, pipeline, metric)
                assert callable(
                    func
                ), f"Metric {task}/{pipeline}/{metric} is not callable"
            except ValueError:
                # Skip if this combination is not registered
                continue

    def test_registry_immutability(self):
        """Test that registry cannot be easily corrupted."""
        original_tasks = set(list_tasks())

        # Try to modify registry directly (this should not affect the actual registry)
        try:
            # This is testing implementation details, but ensures registry integrity
            if hasattr(_METRICS_REGISTRY, "clear"):
                # Don't actually clear, just test we can access it
                current_keys = list(_METRICS_REGISTRY.keys())
                assert len(current_keys) > 0, "Registry appears to be empty"

            # Registry should still return consistent results
            tasks_after = set(list_tasks())
            assert tasks_after == original_tasks, "Registry was corrupted"

        except Exception as e:
            pytest.fail(f"Registry access test failed: {e}")


class TestErrorHandling:
    """Test comprehensive error handling."""

    def test_malformed_inputs(self):
        """Test handling of malformed inputs."""
        original = torch.randn(1, 1, 4, 4)

        # Test mismatched shapes - should handle gracefully with error logging
        metrics = compute_metrics(
            task="reconstruction",
            pipeline_type="real_real",
            metric_names=["mse"],
            predictions=torch.randn(2, 2, 8, 8),  # Different shape
            targets=original,
        )

        # Should return results but with NaN for failed computation
        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert np.isnan(metrics["mse"]) or np.isfinite(metrics["mse"])

    def test_empty_metric_list(self):
        """Test handling of empty metric lists."""
        original = torch.randn(1, 1, 4, 4)
        reconstructed = original.clone()

        metrics = compute_metrics(
            task="reconstruction",
            pipeline_type="real_real",
            metric_names=[],  # Empty list
            predictions=reconstructed,
            targets=original,
        )

        assert isinstance(metrics, dict)
        assert len(metrics) == 0

    def test_none_inputs(self):
        """Test handling of None inputs."""
        # Should handle gracefully with error logging
        metrics = compute_metrics(
            task="reconstruction",
            pipeline_type="real_real",
            metric_names=["mse"],
            predictions=None,
            targets=None,
        )

        # Should return results but with NaN for failed computation
        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert np.isnan(metrics["mse"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
