import torch
import torch.nn as nn
from cvnn.utils import count_model_parameters, measure_inference_time

def test_count_model_parameters():
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    stats = count_model_parameters(model)
    assert stats["total_params"] > 0
    assert stats["trainable_params"] > 0
    assert "trainable_params_fmt" in stats
    assert "total_params_fmt" in stats

def test_measure_inference_time_cpu():
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    device = torch.device("cpu")
    sample_input = torch.randn(1, 10)
    timing = measure_inference_time(model, device, sample_input, batch_size=1, timed_iters=2)
    assert timing["mean_ms"] > 0
    assert timing["p50_ms"] > 0
    assert timing["p95_ms"] > 0
    assert timing["device"] == "cpu"
