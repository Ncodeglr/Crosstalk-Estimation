import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock, patch

# Test the polyphase downsampling layers
from cvnn.models.learn_poly_sampling.layers.polydown import (
    fixed_component,
    max_p_norm,
    LPS,
    PolyphaseInvariantDown2D,
    Decimation,
    set_pool,
)

from cvnn.models.learn_poly_sampling.layers.polyup import (
    max_p_norm_u,
    LPS_u,
    PolyphaseInvariantUp2D,
    set_unpool,
)

from cvnn.models.learn_poly_sampling.layers.lps_utils import (
    lps_downsample,
    lps_downsampleV2,
    lps_upsample,
    lps_upsampleV2,
)

from cvnn.models.softmax import Softmax, GumbelSoftmax
from cvnn.models.projection import PolyCtoR


class TestPolyphaseDownsampling:
    """Test suite for polyphase downsampling components."""

    def test_fixed_component(self):
        """Test fixed component selection."""
        # Create mock input with 4 polyphase components
        batch_size, channels, height, width = 2, 3, 8, 8
        x = torch.randn(4, batch_size, channels, height, width, dtype=torch.complex64)
        prob = torch.tensor([0, 1])  # Select components for each batch (batch_size=2)

        result, idx = fixed_component(x, prob)

        assert result.shape == (batch_size, channels, height, width)
        assert torch.equal(idx, prob)
        # Verify correct component selection
        for i in range(batch_size):
            assert torch.allclose(result[i], x[prob[i], i])

    def test_max_p_norm_basic(self):
        """Test max p-norm component selection."""
        batch_size, channels, height, width = 2, 3, 4, 4

        # Create input where component 1 has higher norm than others
        x = torch.randn(4, batch_size, channels, height, width, dtype=torch.complex64)
        x[1] = x[1] * 3  # Make component 1 have higher norm

        result, idx = max_p_norm(x, p=2)

        assert result.shape == (batch_size, channels, height, width)
        assert idx.shape == (batch_size,)
        # Component 1 should be selected for all batches
        assert torch.all(idx == 1)

    def test_max_p_norm_with_precomputed_prob(self):
        """Test max p-norm with precomputed probabilities."""
        batch_size, channels, height, width = 2, 3, 4, 4
        x = torch.randn(4, batch_size, channels, height, width, dtype=torch.complex64)
        prob = torch.tensor([2, 0])  # Precomputed selection

        result, idx = max_p_norm(x, p=2, prob=prob)

        assert result.shape == (batch_size, channels, height, width)
        assert torch.equal(idx, prob)

    def test_max_p_norm_with_unfiltered(self):
        """Test max p-norm using unfiltered components for index calculation."""
        batch_size, channels, height, width = 2, 3, 4, 4
        x_filtered = torch.randn(
            4, batch_size, channels, height, width, dtype=torch.complex64
        )
        x_nofilt = torch.randn(
            4, batch_size, channels, height, width, dtype=torch.complex64
        )
        x_nofilt[2] = (
            x_nofilt[2] * 5
        )  # Make component 2 have highest norm in unfiltered

        result, idx = max_p_norm(x_filtered, p=2, x_nofilt=x_nofilt)

        assert result.shape == (batch_size, channels, height, width)
        assert torch.all(idx == 2)  # Should select based on unfiltered norms

    def test_lps_layer_initialization(self):
        """Test LPS layer initialization."""
        stride = 2
        in_channels = 16

        # Create mock components needed for LPS
        conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        softmax = Softmax()
        gumbel_softmax = GumbelSoftmax()
        projection = PolyCtoR()

        # Initialize LPS layer
        layer = LPS(
            stride=stride,
            conv=conv,
            gumbel_softmax=gumbel_softmax,
            softmax=softmax,
            projection=projection,
        )

        assert layer.stride == stride
        assert layer.gumbel_softmax == gumbel_softmax
        assert layer.softmax == softmax

    def test_decimation_layer(self):
        """Test Decimation layer functionality."""
        in_channels = 8
        stride = 2

        # Create a partial function for antialias layer (as expected by Decimation)
        from functools import partial
        from cvnn.models.learn_poly_sampling.layers.lowpass_filter import LowPassFilter

        antialias_layer = partial(
            LowPassFilter,
            filter_size=3,
            padding="same",
            padding_mode="circular",
            layer_mode="complex",
        )

        decimation = Decimation(
            antialias_layer=antialias_layer,
            stride=stride,
            in_channels=in_channels,
            no_antialias=False,
        )

        x = torch.randn(2, in_channels, 8, 8, dtype=torch.complex64)
        result = decimation(x)

        # Should downsample with antialiasing
        assert result.shape == (2, in_channels, 4, 4)


class TestPolyphaseUpsampling:
    """Test suite for polyphase upsampling components."""

    def test_max_p_norm_u_basic(self):
        """Test max p-norm upsampling."""
        batch_size, channels, height, width = 2, 4, 4, 4
        x = torch.randn(batch_size, channels, height, width, dtype=torch.complex64)
        indices = torch.randint(0, 4, (batch_size,))

        result, prob_out = max_p_norm_u(x, indices, stride=2)

        expected_height = height * 2
        expected_width = width * 2
        assert result.shape == (batch_size, channels, expected_height, expected_width)
        assert torch.equal(prob_out, indices)

    def test_lps_u_layer_initialization(self):
        """Test LPS upsampling layer initialization."""
        stride = 2
        in_channels = 16
        hid_channels = 32
        softmax = Softmax()
        gumbel_softmax = GumbelSoftmax()

        lps_u = LPS_u(
            in_channels=in_channels,
            hid_channels=hid_channels,
            stride=stride,
            softmax=softmax,
            gumbel_softmax=gumbel_softmax,
        )

        assert lps_u.stride == stride
        assert lps_u.softmax == softmax


class TestLPSUtils:
    """Test suite for LPS utility functions."""

    def test_lps_downsample_basic(self):
        """Test basic LPS downsampling."""
        batch_size, channels, height, width = 2, 4, 8, 8
        x = torch.randn(batch_size, channels, height, width, dtype=torch.complex64)
        stride = 2
        polyphase_logits = torch.randn(batch_size, stride * stride)

        result = lps_downsample(x, stride, polyphase_logits, mode="test")

        # Should return downsampled tensor
        expected_height = height // stride
        expected_width = width // stride
        assert result.shape == (batch_size, channels, expected_height, expected_width)

    def test_lps_downsampleV2_basic(self):
        """Test LPS downsampling V2."""
        batch_size, channels, height, width = 2, 4, 8, 8
        stride = 2

        # Create polyphase components as a tensor (not list)
        polyphase_components = torch.randn(
            stride * stride,
            batch_size,
            channels,
            height // stride,
            width // stride,
            dtype=torch.complex64,
        )

        polyphase_logits = torch.randn(batch_size, stride * stride)
        gumbel_softmax = GumbelSoftmax()
        softmax = Softmax()

        result, _ = lps_downsampleV2(
            polyphase_components,
            stride,
            polyphase_logits,
            gumbel_softmax,
            softmax,
            mode="test",
        )

        # Should return downsampled tensor
        expected_height = height // stride
        expected_width = width // stride
        assert result.shape == (batch_size, channels, expected_height, expected_width)

    def test_lps_upsample_basic(self):
        """Test basic LPS upsampling."""
        batch_size, channels, height, width = 2, 4, 4, 4
        stride = 2
        x = torch.randn(batch_size, channels, height, width, dtype=torch.complex64)
        polyphase_logits = torch.randn(batch_size, stride * stride)
        prob = torch.softmax(polyphase_logits, dim=-1)

        result = lps_upsample(
            x, stride, polyphase_logits=polyphase_logits, mode="test", prob=prob
        )

        expected_height = height * stride
        expected_width = width * stride
        assert result.shape == (batch_size, channels, expected_height, expected_width)

    def test_lps_upsampleV2_basic(self):
        """Test LPS upsampling V2."""
        batch_size, channels, height, width = 2, 4, 4, 4
        stride = 2
        x = torch.randn(batch_size, channels, height, width, dtype=torch.complex64)

        # Create probability tuple (prob, logits) for train mode
        polyphase_logits = torch.randn(batch_size, stride * stride)
        prob_dist = torch.softmax(polyphase_logits, dim=-1)
        prob = (prob_dist, polyphase_logits)
        softmax = Softmax()
        gumbel_softmax = GumbelSoftmax()

        result, _ = lps_upsampleV2(
            x, prob, stride, softmax, gumbel_softmax, mode="train"
        )

        expected_height = height * stride
        expected_width = width * stride
        assert result.shape == (batch_size, channels, expected_height, expected_width)

    def test_stride_consistency(self):
        """Test that downsample followed by upsample preserves dimensions."""
        batch_size, channels, height, width = 1, 2, 8, 8
        stride = 2

        # Create polyphase components for downsampling
        polyphase_components = torch.randn(
            stride * stride,
            batch_size,
            channels,
            height // stride,
            width // stride,
            dtype=torch.complex64,
        )

        polyphase_logits = torch.randn(batch_size, stride * stride)
        gumbel_softmax = GumbelSoftmax()
        softmax = Softmax()

        # Downsample
        downsampled, prob_out = lps_downsampleV2(
            polyphase_components,
            stride,
            polyphase_logits,
            gumbel_softmax,
            softmax,
            mode="train",
        )

        # Upsample
        upsampled, _ = lps_upsampleV2(
            downsampled, prob_out, stride, softmax, gumbel_softmax, mode="train"
        )

        # Check final dimensions match original
        expected_height = height
        expected_width = width
        assert upsampled.shape == (
            batch_size,
            channels,
            expected_height,
            expected_width,
        )


class TestIntegration:
    """Test integration between different polyphase components."""

    def test_polyphase_invariant_down2d_initialization(self):
        """Test PolyphaseInvariantDown2D initialization."""
        in_channels = 16
        stride = 2

        # Create real components needed for initialization
        conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        gumbel_softmax = GumbelSoftmax()
        softmax = Softmax()
        projection = PolyCtoR()
        mock_component_selection = MagicMock()
        mock_antialias = MagicMock()

        layer = PolyphaseInvariantDown2D(
            conv=conv,
            gumbel_softmax=gumbel_softmax,
            softmax=softmax,
            projection=projection,
            component_selection=mock_component_selection,
            antialias_layer=mock_antialias,
            stride=stride,
            in_channels=in_channels,
            selection_noantialias=False,
        )

        assert layer.stride == stride

    def test_polyphase_invariant_up2d_initialization(self):
        """Test PolyphaseInvariantUp2D initialization."""
        in_channels = 16
        stride = 2
        softmax = Softmax()

        # Mock component selection
        mock_component_selection = MagicMock()

        layer = PolyphaseInvariantUp2D(
            component_selection=mock_component_selection,
            stride=stride,
            in_channels=in_channels,
            softmax=softmax,
        )

        # PolyphaseInvariantUp2D doesn't store stride as an attribute directly
        assert layer.softmax == softmax

    @pytest.mark.parametrize(
        "stride", [2]
    )  # Only test stride=2 as implementation doesn't support stride=4
    @pytest.mark.parametrize("channels", [8, 16, 32])
    def test_different_configurations(self, stride, channels):
        """Test different stride and channel configurations."""
        batch_size, height, width = 2, 16, 16

        # Create polyphase components for testing
        polyphase_components = torch.randn(
            stride * stride,
            batch_size,
            channels,
            height // stride,
            width // stride,
            dtype=torch.complex64,
        )

        polyphase_logits = torch.randn(batch_size, stride * stride)
        gumbel_softmax = GumbelSoftmax()
        softmax = Softmax()

        # Test downsampling preserves batch and channel dimensions
        downsampled, prob_out = lps_downsampleV2(
            polyphase_components,
            stride,
            polyphase_logits,
            gumbel_softmax,
            softmax,
            mode="test",
        )
        assert downsampled.shape[0] == batch_size  # Batch dimension
        assert downsampled.shape[1] == channels  # Channel dimension
        assert downsampled.shape[2] == height // stride  # Height
        assert downsampled.shape[3] == width // stride  # Width

        # Test upsampling restores original dimensions
        upsampled, _ = lps_upsampleV2(
            downsampled, prob_out, stride, softmax, gumbel_softmax, mode="test"
        )
        assert upsampled.shape == (batch_size, channels, height, width)

    def test_complex_dtype_preservation(self):
        """Test that complex dtypes are preserved throughout operations."""
        dtypes = [torch.complex64, torch.complex128]

        for dtype in dtypes:
            batch_size, channels, height, width = 2, 4, 8, 8
            stride = 2

            # Create polyphase components
            polyphase_components = torch.randn(
                stride * stride,
                batch_size,
                channels,
                height // stride,
                width // stride,
                dtype=dtype,
            )

            polyphase_logits = torch.randn(batch_size, stride * stride)
            gumbel_softmax = GumbelSoftmax()
            softmax = Softmax()

            # Test downsampling preserves dtype
            downsampled, prob_out = lps_downsampleV2(
                polyphase_components,
                stride,
                polyphase_logits,
                gumbel_softmax,
                softmax,
                mode="test",
            )
            assert downsampled.dtype == dtype

            # Test upsampling preserves dtype
            upsampled, _ = lps_upsampleV2(
                downsampled, prob_out, stride, softmax, gumbel_softmax, mode="test"
            )
            assert upsampled.dtype == dtype
