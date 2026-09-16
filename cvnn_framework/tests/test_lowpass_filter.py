import pytest
import torch
import torch.nn as nn

from cvnn.models.learn_poly_sampling.layers.lowpass_filter import (
    CircularPad2d,
    LowPassFilter,
    DDAC,
    get_pad_layer,
)


class TestCircularPad2d:
    def test_circular_pad2d_initialization(self):
        padding = (2, 3, 1, 4)
        pad_layer = CircularPad2d(padding)
        assert pad_layer.pad_v == 4 * [padding]


class TestLowPassFilter:
    def test_lowpass_filter_initialization_default(self):
        in_channels = 16
        filter_size = 5
        padding = "same"
        padding_mode = "circular"

        lp_filter = LowPassFilter(
            in_channels, filter_size, padding, padding_mode, layer_mode="real"
        )

        assert lp_filter.channels == in_channels
        assert lp_filter.filter_size == filter_size
        assert lp_filter.padding == padding
        assert lp_filter.padding_mode == padding_mode

    def test_lowpass_filter_forward_real(self):
        in_channels = 8
        batch_size, height, width = 2, 16, 16

        lp_filter = LowPassFilter(
            in_channels,
            filter_size=5,
            padding="same",
            padding_mode="circular",
            layer_mode="real",
        )

        x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float64)
        result = lp_filter(x)

        assert result.shape == x.shape
        assert result.dtype == torch.float64


class TestDDAC:
    def test_ddac_initialization_default(self):
        in_channels = 16
        kernel_size = 3
        ddac = DDAC(in_channels, kernel_size)

        assert ddac.kernel_size == kernel_size
        assert ddac.group == 2
        assert ddac.kernel_scale == 1


class TestGetPadLayer:
    def test_get_pad_layer_reflect(self):
        PadLayer = get_pad_layer("reflect")
        assert PadLayer == nn.ReflectionPad2d

    def test_get_pad_layer_replicate(self):
        PadLayer = get_pad_layer("replicate")
        assert PadLayer == nn.ReplicationPad2d

    def test_get_pad_layer_zero(self):
        PadLayer = get_pad_layer("zero")
        assert PadLayer == nn.ZeroPad2d

    def test_get_pad_layer_circular(self):
        PadLayer = get_pad_layer("circular")
        assert PadLayer == CircularPad2d


class TestAntiAliasingIntegration:
    def test_filter_with_different_input_sizes(self):
        in_channels = 4
        input_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]

        for height, width in input_sizes:
            x = torch.randn(1, in_channels, height, width, dtype=torch.float64)

            lowpass = LowPassFilter(
                in_channels,
                filter_size=3,
                padding="same",
                padding_mode="circular",
                layer_mode="real",
            )

            x_float32 = x.float()
            ddac = DDAC(in_channels, kernel_size=3)

            lowpass_result = lowpass(x)
            ddac_result = ddac(x_float32)

            assert lowpass_result.shape == x.shape
            assert ddac_result.shape == x_float32.shape

    @pytest.mark.parametrize("in_channels", [4, 8, 16])
    def test_filter_parameter_combinations(self, in_channels):
        x = torch.randn(1, in_channels, 16, 16, dtype=torch.float64)

        lowpass = LowPassFilter(
            in_channels,
            filter_size=3,
            padding="same",
            padding_mode="circular",
            layer_mode="real",
        )

        x_float32 = x.float()
        ddac = DDAC(in_channels, kernel_size=3)

        lowpass_result = lowpass(x)
        ddac_result = ddac(x_float32)

        assert lowpass_result.shape == x.shape
        assert ddac_result.shape == x_float32.shape
