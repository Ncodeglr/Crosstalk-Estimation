# Standard library imports
import logging
import random
from typing import Optional, Union
import time

# Third-party imports
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
import math

def set_seed(seed: int) -> None:
    """
    Set seed for reproducible results across random, numpy, and torch.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    def format_params(n):
        if n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.1f}K"
        return str(n)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_params_fmt": format_params(trainable_params),
        "total_params_fmt": format_params(total_params),
    }

def measure_inference_time(model, device, sample_input, warmup_iters=10, timed_iters=50, batch_size=1):# -> dict[str, Any]:
    # Extract tensor if sample_input is a tuple/list (e.g., [input, label])
    if isinstance(sample_input, (tuple, list)):
        sample_input = sample_input[0]
    
    dtype = sample_input.dtype
    x = torch.randn_like(sample_input, dtype=dtype, device=device)
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(timed_iters):
            start = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    stats = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
    }
    return stats

def estimate_model_flops(model, sample_input, device):
    dtype = sample_input.dtype
    dummy_input = torch.randn_like(sample_input, dtype=dtype, device=device)
    flops = FlopCountAnalysis(model, dummy_input).total()
    def format_flops(n):
        if n >= 1e9:
            return f"{n/1e9:.2f}G"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.1f}K"
        return str(n)
    return {"flops": int(flops), "flops_fmt": format_flops(flops)}

def setup_logging(
    name: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Return a logger with StreamHandler and formatter if not already configured.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger

def safe_log(
    x: Union[float, np.ndarray], base: float = 10, eps: float = 1e-10
) -> Union[float, np.ndarray]:
    """
    Compute logarithm safely by avoiding log(0) and negative values.

    Args:
        x: Input value(s) to compute logarithm of
        base: Logarithm base (10 for log10, math.e for natural log)
        eps: Minimum value to clamp inputs to avoid numerical issues

    Returns:
        Logarithm of the input with the specified base

    Raises:
        ValueError: If base is invalid
    """
    if base <= 0 or base == 1:
        raise ValueError(f"Invalid logarithm base: {base}")

    x_safe = np.clip(x, eps, None)
    if base == 10:
        return np.log10(x_safe)
    elif base == math.e:
        return np.log(x_safe)
    else:
        # Change of base formula: log_b(x) = ln(x) / ln(b)
        return np.log(x_safe) / np.log(base)

@torch.no_grad()
def kl_sanity_checks(device="cuda" if torch.cuda.is_available() else "cpu", D=64):
    """
    Verifies:
      (A) Prior case: mu=0, std=1, delta=0 -> KL=0
      (B) δ=0 case matches closed form: (σ² + |μ|² - 1) - log(σ²)
      (C) Monotonicity near boundary as |δ|↑
    """
    def kl_impl(mu, std, delta, eps=1e-12):
        sigma2 = std**2
        t = (sigma2**2 - delta.abs()**2).clamp_min(eps)
        per_dim = (sigma2 + mu.abs()**2 - 1.0) - 0.5 * torch.log(t)
        return per_dim.view(mu.size(0), -1).sum(1)  # per-sample

    B = 4
    # (A) prior
    mu = torch.zeros(B, D, dtype=torch.complex64, device=device)
    std = torch.ones(B, D, dtype=torch.float32, device=device)
    delta = torch.zeros(B, D, dtype=torch.complex64, device=device)
    KL = kl_impl(mu, std, delta)
    assert KL.abs().max().item() < 1e-6, f"Prior KL not zero: {KL}"

    # (B) δ=0 closed-form
    torch.manual_seed(0)
    mu = (torch.randn(B, D, device=device) + 1j*torch.randn(B, D, device=device)) * 0.1
    std = torch.exp(torch.randn(B, D, device=device) * 0.1)  # positive
    delta = torch.zeros(B, D, dtype=torch.complex64, device=device)
    KL1 = kl_impl(mu, std, delta)
    KL2 = ((std**2 + mu.abs()**2 - 1.0) - torch.log(std**2)).sum(1)
    assert torch.allclose(KL1, KL2, rtol=1e-5, atol=1e-6), f"δ=0 mismatch: {(KL1-KL2).abs().max()}"

    # (C) boundary monotonicity: increase |δ| -> KL increases
    phi = torch.randn(B, D, device=device)
    phase = torch.exp(1j * phi)
    sigma2 = std**2
    deltas = [0.0, 0.5, 0.9, 0.99]
    KLs = []
    for alpha in deltas:
        delta = (alpha := alpha) * sigma2 * phase * 0.999  # keep strictly inside feasible set
        KLs.append(kl_impl(mu, std, delta).mean().item())
    assert KLs == sorted(KLs), f"KL not monotone with |δ|: {KLs}"
    print("[KL sanity] passed:", {"prior": KL.mean().item(), "delta0_eq": (KL1-KL2).abs().max().item(), "monotone": KLs})

class KLAnnealing:
    def __init__(self, kind, warmup_steps):
        self.kind = kind
        self.N = warmup_steps
        self.t = 0

    def progress(self) -> float:
        """
        Returns a scalar in [0,1] indicating how far along the schedule we are.
        """
        # --- NOUVEAU MODE ICI ---
        if self.kind == "constant" or self.kind is None:
            return 1.0
        
        if self.N <= 0:
            return 1.0

        x = min(1.0, self.t / self.N)
        
        if self.kind == "linear":
            return x
        if self.kind == "cosine":
            return 0.5 * (1 - math.cos(math.pi * x))
        if self.kind == "sigmoid":
            s = 10.0
            return 1 / (1 + math.exp(-s * (x - 0.5)))
            
        raise ValueError(f"Unknown schedule kind: {self.kind}")

    def step(self):
        """Advance the schedule by one step (usually one minibatch)."""
        self.t += 1


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution. Works with complex distances.
    """

    # Compute P-row and corresponding perplexity using magnitude of complex numbers
    # Use absolute value for complex distances
    D_abs = np.abs(D.copy())
    P = np.exp(-D_abs * beta)
    sumP = sum(P)
    
    # Avoid division by zero
    if sumP <= 0 or not np.isfinite(sumP):
        sumP = 1e-10
    
    H = np.log(sumP) + beta * np.sum(D_abs * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity. Works with complex tensors.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    # For complex tensors, use magnitude squared: |x|^2 = x * conj(x)
    sum_X = np.sum(np.real(X * np.conj(X)), 1)
    # Compute distance matrix: |x_i - x_j|^2
    D = np.add(np.add(-2 * np.real(np.dot(X, np.conj(X.T))), sum_X).T, sum_X)
    # Take absolute value to ensure real non-negative distances
    D = np.abs(D)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions. Works with complex tensors using SVD.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    # Use SVD for complex tensors instead of eigendecomposition
    # This is more numerically stable for complex matrices
    U, s, Vh = np.linalg.svd(np.dot(np.conj(X.T), X), full_matrices=False)
    # Select the first no_dims principal components
    Y = np.dot(X, U[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, max_iter=None):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims)
    (n, d) = X.shape
    
    # Adaptive max_iter based on dataset size
    if max_iter is None:
        if n > 1000:
            max_iter = 300
        elif n > 500:
            max_iter = 500
        else:
            max_iter = 1000
    
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    # Initialize Y in real space (output of t-SNE is typically real)
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities in output space (real-valued)
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y
