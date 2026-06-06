"""
gnavar_core.py — shared infrastructure for G-NAVAR identifiability experiments.

This module contains the data-generating process, model, training loop, and
recovery metrics used across Experiments 1-4. It is the verbatim extraction
of the code that produced Experiment 1's results, with three additions:
  - simulate_dgp_coupled: DGP variant with rho-controlled coupling between
    x3 and x5 (used in Experiment 2 for the support-collapse phase transition).
  - save_model / load_model: persist trained GNAVAR models to/from disk.
  - effective_rank: support-richness diagnostic (used in Experiments 3-4).

Place this file in /content/drive/MyDrive/GNAVAR/code/ and import via:
    sys.path.insert(0, '/content/drive/MyDrive/GNAVAR/code')
    from gnavar_core import *
"""
import math
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = (DEVICE.type == 'cuda')
print(f'Device: {DEVICE} | Mixed precision: {USE_AMP}')
@dataclass
class Config:
    # DGP
    n_vars: int = 5            # x1 (target) + x2..x5 (sources)
    K: int = 2                 # lag order
    sigma_eps: float = 0.1     # innovation std for target
    burn_in: int = 200
    ar_coefs: tuple = (0.6, 0.5, 0.7, 0.65)

    # Sweep
    sample_sizes: tuple = (1000, 5000, 25000, 100000)
    n_seeds: int = 5
    base_seed: int = 42

    # Model
    hidden_dim: int = 32

    # Training
    n_epochs: int = 300
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # L1 sparsity penalty on gate deviation from 1.
    # Encourages irrelevant gates to collapse to identically-1 (minimality).
    # Tune on a single (T, seed) trial before the full sweep; see lambda-tuning cell below.
    l1_lambda: float = 0.02

    # Minimality threshold: declare g_{ijk} trivial if E[(g - 1)^2] < threshold.
    # Calibrated for L1-trained gates; should be roughly midway between the
    # noise floor of true-trivial gates and the magnitude of true modulators.
    triviality_threshold: float = 0.001

def true_f12(x2_lag1):
    return 0.7 * x2_lag1

def true_g123(x3_lag1, x3_lag2):
    """Change-sensitive saturating gate."""
    change_part = np.exp(-0.5 * (x3_lag1 - x3_lag2) ** 2)
    sat_part = 2.0 / (1.0 + np.exp(-2.0 * x3_lag1))
    return change_part * sat_part

def true_f14(x4_lag1):
    return 0.5 * x4_lag1

def true_g145(x5_lag1, x5_lag2):
    """2D Gaussian inhibitory gate, weighted toward the recent lag."""
    return np.exp(-0.5 * (x5_lag1 ** 2 + 0.5 * x5_lag2 ** 2))

def true_f12_lag2(x2_lag2):
    return 0.3 * x2_lag2
def simulate_dgp(T: int, cfg: Config, seed: int) -> np.ndarray:
    """
    Simulate trajectory of length T (after burn-in).
    Returns shape (T, n_vars). Column 0 is target; columns 1..n_vars-1 are sources.
    """
    rng = np.random.default_rng(seed)
    T_total = T + cfg.burn_in
    X = np.zeros((T_total, cfg.n_vars))
    X[:cfg.K] = 0.1 * rng.standard_normal((cfg.K, cfg.n_vars))

    eps_sources = rng.standard_normal((T_total, cfg.n_vars - 1))
    eps_target = cfg.sigma_eps * rng.standard_normal(T_total)

    for t in range(cfg.K, T_total):
        # Sources: independent AR(1)
        for j in range(cfg.n_vars - 1):
            phi = cfg.ar_coefs[j]
            X[t, j + 1] = phi * X[t - 1, j + 1] + math.sqrt(1 - phi ** 2) * eps_sources[t, j]

        # Target via DGP
        x2_l1, x2_l2 = X[t - 1, 1], X[t - 2, 1]
        x3_l1, x3_l2 = X[t - 1, 2], X[t - 2, 2]
        x4_l1       = X[t - 1, 3]
        x5_l1, x5_l2 = X[t - 1, 4], X[t - 2, 4]

        contrib_23   = true_f12(x2_l1) * true_g123(x3_l1, x3_l2)
        contrib_45   = true_f14(x4_l1) * true_g145(x5_l1, x5_l2)
        contrib_lag2 = true_f12_lag2(x2_l2)

        X[t, 0] = contrib_23 + contrib_45 + contrib_lag2 + eps_target[t]

    return X[cfg.burn_in:]
def make_lag_tensor(X: np.ndarray, K: int):
    """
    Build supervised pairs (X_lag, y) from a trajectory.

    X_lag: (T - K, n_sources, K). X_lag[t, j, ell] is value of source j+1
           at lag (ell+1) relative to target time (t + K).
    y    : (T - K,) target values at time t + K.

    Vectorized implementation: ~100x faster than the original triple-nested
    Python loop for large T.
    """
    T, n_vars = X.shape
    n_sources = n_vars - 1
    n_samples = T - K
    # Targets at times K..T-1
    y = X[K:T, 0].astype(np.float32, copy=False)
    # For each lag ell in 0..K-1, source values from time K-(ell+1) .. T-1-(ell+1)
    # Shape (n_samples, n_sources, K). Order: X_lag[t, j, ell] = X[K + t - (ell+1), j+1].
    X_lag = np.empty((n_samples, n_sources, K), dtype=np.float32)
    sources = X[:, 1:].astype(np.float32, copy=False)  # (T, n_sources)
    for ell in range(K):
        # Time indices: K - (ell+1) .. T - 1 - (ell+1) = K-ell-1 .. T-ell-2
        # which gives n_samples values starting at K-ell-1.
        start = K - (ell + 1)
        X_lag[:, :, ell] = sources[start:start + n_samples, :]
    return X_lag, y
class BatchedMLP(nn.Module):
    """
    A batch of `n_groups` independent small MLPs, each mapping R^K -> R
    via a single hidden layer with Tanh activation.

    Implemented with grouped 1x1 Conv1d so all groups run in parallel on GPU
    without Python loops. Equivalent to having `n_groups` separate Linear layers.

    Input shape : (batch, n_groups, K)
    Output shape: (batch, n_groups)
    """
    def __init__(self, n_groups: int, K: int, hidden_dim: int):
        super().__init__()
        self.n_groups = n_groups
        self.K = K
        self.hidden_dim = hidden_dim
        # First layer: maps n_groups groups, each with K input channels,
        # to n_groups groups, each with hidden_dim output channels.
        self.conv1 = nn.Conv1d(
            in_channels  = n_groups * K,
            out_channels = n_groups * hidden_dim,
            kernel_size  = 1,
            groups       = n_groups,
        )
        # Second layer: maps each group's hidden_dim back to 1.
        self.conv2 = nn.Conv1d(
            in_channels  = n_groups * hidden_dim,
            out_channels = n_groups,
            kernel_size  = 1,
            groups       = n_groups,
        )

    def forward(self, x):
        # x: (batch, n_groups, K)
        batch = x.shape[0]
        # Reshape to (batch, n_groups * K, 1) for grouped conv
        x = x.reshape(batch, self.n_groups * self.K, 1)
        h = torch.tanh(self.conv1(x))           # (batch, n_groups * hidden_dim, 1)
        out = self.conv2(h)                     # (batch, n_groups, 1)
        return out.squeeze(-1)                  # (batch, n_groups)
class GNAVAR(nn.Module):
    """
    Vectorized G-NAVAR for a single target.

    For n_sources sources and lag order K:
      - n_sources base functions f_j: R^K -> R
      - n_sources * (n_sources - 1) gate functions g_{j,k}: R^K -> R, indexed by
        (source j, modulator k != j).
    Bases and gates are each computed via a single BatchedMLP forward pass.
    """
    def __init__(self, n_sources: int, K: int, hidden_dim: int):
        super().__init__()
        self.n_sources = n_sources
        self.K = K
        self.bias  = nn.Parameter(torch.zeros(1))
        self.bases = BatchedMLP(n_groups=n_sources, K=K, hidden_dim=hidden_dim)
        # Gates: n_sources * (n_sources - 1) per-(j, k) functions.
        self.n_gates = n_sources * (n_sources - 1)
        self.gates   = BatchedMLP(n_groups=self.n_gates, K=K, hidden_dim=hidden_dim)

        # Precompute (j, k) index lookups
        # gate_to_jk[g] = (j, k) for the g-th gate
        # k_index_of_gate[g] = k (used to look up x_k from X_lag)
        # j_index_of_gate[g] = j
        # gate_indices_for_source[j] = list of gate indices whose source is j
        gate_to_jk = []
        for j in range(n_sources):
            for k in range(n_sources):
                if k != j:
                    gate_to_jk.append((j, k))
        self.register_buffer(
            'gate_k_idx',
            torch.tensor([k for (_, k) in gate_to_jk], dtype=torch.long),
        )
        self.register_buffer(
            'gate_j_idx',
            torch.tensor([j for (j, _) in gate_to_jk], dtype=torch.long),
        )

    def forward(self, X_lag):
        """
        X_lag: (batch, n_sources, K)
        Returns: (batch,) prediction.
        """
        batch = X_lag.shape[0]

        # Bases: f_j(x_j) for each source j. Pass X_lag directly; group j sees x_j.
        bases_out = self.bases(X_lag)                      # (batch, n_sources)

        # Gates: gather lag blocks corresponding to each gate's modulator k.
        # gate_k_idx: (n_gates,) — for each gate g, which source k to read.
        gate_inputs = X_lag[:, self.gate_k_idx, :]         # (batch, n_gates, K)
        gate_eta    = self.gates(gate_inputs)              # (batch, n_gates)
        gate_vals   = torch.exp(gate_eta)                  # strictly positive

        # For each source j, multiply its gates together.
        # gate_j_idx tells us which source each gate belongs to.
        # We use scatter-prod via log-sum-exp-style trick to keep it differentiable.
        # log(prod_g gate_vals[g]) = sum_g log(gate_vals[g]) = sum_g eta_g.
        # Sum eta_g over each source j using index_add.
        log_gate_product = torch.zeros(batch, self.n_sources, device=X_lag.device, dtype=gate_eta.dtype)
        log_gate_product = log_gate_product.index_add(
            dim=1,
            index=self.gate_j_idx,
            source=gate_eta,
        )
        gate_product = torch.exp(log_gate_product)         # (batch, n_sources)

        # Per-source contribution and sum
        contributions = bases_out * gate_product           # (batch, n_sources)
        return self.bias + contributions.sum(dim=1)

    @torch.no_grad()
    def evaluate_gate(self, j: int, k: int, x_block):
        """
        Evaluate gate g_{j,k} on input x_block of shape (batch, K).

        Implementation: build a fake input where the gate's modulator slot has
        x_block, others are zero (the gate ignores them by construction).
        Then take the corresponding output channel.
        """
        batch = x_block.shape[0]
        # Find the gate index for (j, k)
        gate_idx = None
        cnt = 0
        for jj in range(self.n_sources):
            for kk in range(self.n_sources):
                if kk != jj:
                    if jj == j and kk == k:
                        gate_idx = cnt
                    cnt += 1
        assert gate_idx is not None, f'Gate ({j},{k}) not found'

        gate_inputs = torch.zeros(batch, self.n_gates, self.K, device=x_block.device, dtype=x_block.dtype)
        gate_inputs[:, gate_idx, :] = x_block
        eta = self.gates(gate_inputs)[:, gate_idx]
        return torch.exp(eta)

    @torch.no_grad()
    def evaluate_base(self, j: int, x_block):
        batch = x_block.shape[0]
        base_inputs = torch.zeros(batch, self.n_sources, self.K, device=x_block.device, dtype=x_block.dtype)
        base_inputs[:, j, :] = x_block
        return self.bases(base_inputs)[:, j]
def compute_gate_l1_penalty(model, X_lag):
    """
    Compute mean_{batch, gates} |g_{ijk}(x_k) - 1|.

    This is the L1 sparsity penalty: it pushes gates that do not help prediction
    toward identically 1, which corresponds to the minimality condition of the
    theorem. Gates that genuinely matter for prediction will have |g - 1| > 0
    on average (because they actively modulate); irrelevant gates have no reason
    to deviate from 1 once the penalty is in place.
    """
    batch = X_lag.shape[0]
    # Gather lag blocks for each gate's modulator k (same logic as model.forward)
    gate_inputs = X_lag[:, model.gate_k_idx, :]      # (batch, n_gates, K)
    eta = model.gates(gate_inputs)                   # (batch, n_gates)
    g = torch.exp(eta)                               # (batch, n_gates), strictly positive
    return (g - 1.0).abs().mean()                    # scalar


def fit_gnavar(X: np.ndarray, cfg: Config, seed: int, verbose: bool = False) -> GNAVAR:
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    n_sources = cfg.n_vars - 1
    X_lag_np, y_np = make_lag_tensor(X, cfg.K)

    X_lag = torch.from_numpy(X_lag_np).to(DEVICE)
    y     = torch.from_numpy(y_np).to(DEVICE)

    model = GNAVAR(n_sources=n_sources, K=cfg.K, hidden_dim=cfg.hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    n_samples = X_lag.shape[0]
    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(n_samples, device=DEVICE)
        # Accumulate epoch loss on the GPU to avoid per-batch CPU sync.
        # We only convert to a Python float when verbose logging is needed.
        epoch_mse_sum = torch.zeros((), device=DEVICE)
        epoch_l1_sum  = torch.zeros((), device=DEVICE)
        n_batches = 0
        for i in range(0, n_samples, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = X_lag[idx], y[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                pred = model(xb)
                mse  = F.mse_loss(pred, yb)
                l1   = compute_gate_l1_penalty(model, xb)
                loss = mse + cfg.l1_lambda * l1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # Detach so the accumulator does not retain the graph.
            epoch_mse_sum = epoch_mse_sum + mse.detach()
            epoch_l1_sum  = epoch_l1_sum  + l1.detach()
            n_batches += 1
        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            mse_avg = float(epoch_mse_sum.item()) / n_batches
            l1_avg  = float(epoch_l1_sum.item()) / n_batches
            print(f'  epoch {epoch+1:3d}/{cfg.n_epochs}  '
                  f'mse={mse_avg:.5f}  '
                  f'l1={l1_avg:.5f}', flush=True)
    return model


def fit_gnavar_with_restarts(X, cfg, seed, n_restarts=3, verbose=False):
    """
    Restart-and-keep-best wrapper around fit_gnavar.

    Runs `n_restarts` independent training runs with different initialization
    seeds and returns the model with the lowest final training loss. This
    addresses the bimodality of training outcomes observed at finite sample
    sizes with L1 sparsity penalties: roughly half of single runs converge to
    a clean optimum and roughly half get stuck in a suboptimal regime where
    some gates fail to collapse properly. Best-of-n keeps the successful run.

    Returns the best model. Does not return all candidates (to keep the API
    swappable with fit_gnavar).
    """
    import torch.nn.functional as F
    n_sources = cfg.n_vars - 1
    X_lag_np, y_np = make_lag_tensor(X, cfg.K)
    X_lag = torch.from_numpy(X_lag_np).to(DEVICE)
    y     = torch.from_numpy(y_np).to(DEVICE)

    best_model = None
    best_loss = float('inf')
    losses = []
    for r in range(n_restarts):
        # Use a deterministic but distinct seed per restart, derived from `seed`
        restart_seed = seed + 10_000_000 * r
        model = fit_gnavar(X, cfg, seed=restart_seed, verbose=False)
        model.eval()
        # Evaluate final MSE on the full (in-sample) training data.
        # This is a deterministic comparison across candidates.
        with torch.no_grad():
            pred = model(X_lag)
            final_loss = float(F.mse_loss(pred, y).item())
        losses.append(final_loss)
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model
        if verbose:
            print(f'  restart {r+1}/{n_restarts}: final_loss={final_loss:.6f} '
                  f'(best so far: {best_loss:.6f})', flush=True)
    if verbose:
        print(f'  selected restart with loss {best_loss:.6f} '
              f'(out of {[f"{l:.6f}" for l in losses]})', flush=True)
    return best_model


# ----------------------------------------------------------------------
# Pairwise (additive) NAVAR baseline
# ----------------------------------------------------------------------
# This is the additive-only ablation of G-NAVAR: same architecture for the
# per-source base functions f_j, but NO multiplicative gates. The prediction
# is purely additive: predicted x = bias + sum_j f_j(x_j).
#
# The contrast with G-NAVAR isolates the contribution of the gating structure:
# both models have identical expressive power per source, the only difference
# is whether cross-source interactions are modeled multiplicatively. This is
# the right baseline for testing the paper's central claim — that
# multiplicative interactions must be modeled explicitly to be recovered.


class PairwiseNAVAR(nn.Module):
    """
    Additive (no-gate) baseline for G-NAVAR.

    Architecture mirrors GNAVAR's base functions exactly: one BatchedMLP
    producing n_sources outputs, each f_j: R^K -> R. The forward pass is
    predicted y = bias + sum_j f_j(x_j). No gates.

    Parameter count and per-source capacity match GNAVAR's bases exactly.
    The only difference is the absence of gating.
    """
    def __init__(self, n_sources: int, K: int, hidden_dim: int):
        super().__init__()
        self.n_sources = n_sources
        self.K = K
        self.bias  = nn.Parameter(torch.zeros(1))
        self.bases = BatchedMLP(n_groups=n_sources, K=K, hidden_dim=hidden_dim)

    def forward(self, X_lag):
        """
        X_lag: (batch, n_sources, K)
        Returns: (batch,) prediction.
        """
        bases_out = self.bases(X_lag)                      # (batch, n_sources)
        return self.bias + bases_out.sum(dim=1)

    @torch.no_grad()
    def evaluate_base(self, j: int, x_block):
        """
        Evaluate f_j on input x_block of shape (batch, K).
        Builds a tensor where group j has x_block and others are zero;
        returns the j-th output.
        """
        batch = x_block.shape[0]
        base_inputs = torch.zeros(
            batch, self.n_sources, self.K,
            device=x_block.device, dtype=x_block.dtype,
        )
        base_inputs[:, j, :] = x_block
        return self.bases(base_inputs)[:, j]


def fit_pairwise(X: np.ndarray, cfg: Config, seed: int,
                 verbose: bool = False) -> PairwiseNAVAR:
    """
    Train PairwiseNAVAR on data X with config cfg.

    Uses the same Adam + AMP setup as fit_gnavar, and the same n_epochs.
    NOT given an L1 penalty: the L1 penalty exists in G-NAVAR specifically
    to enforce gate minimality, which doesn't apply here. Including it
    would be a strawman.
    """
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    n_sources = cfg.n_vars - 1
    X_lag_np, y_np = make_lag_tensor(X, cfg.K)
    X_lag = torch.from_numpy(X_lag_np).to(DEVICE)
    y     = torch.from_numpy(y_np).to(DEVICE)

    model = PairwiseNAVAR(n_sources=n_sources, K=cfg.K, hidden_dim=cfg.hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    n_samples = X_lag.shape[0]
    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(n_samples, device=DEVICE)
        # Accumulate on GPU to avoid per-batch CPU sync (see fit_gnavar).
        epoch_loss_sum = torch.zeros((), device=DEVICE)
        n_batches = 0
        for i in range(0, n_samples, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = X_lag[idx], y[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss_sum = epoch_loss_sum + loss.detach()
            n_batches += 1
        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            loss_avg = float(epoch_loss_sum.item()) / n_batches
            print(f'  pairwise epoch {epoch+1:3d}/{cfg.n_epochs}  '
                  f'mse={loss_avg:.5f}', flush=True)
    return model


def fit_pairwise_with_restarts(X, cfg, seed, n_restarts=3, verbose=False):
    """
    Restart-and-keep-best wrapper for PairwiseNAVAR.
    Same convention as fit_gnavar_with_restarts: select the run with lowest
    final training MSE on the full in-sample data.
    """
    n_sources = cfg.n_vars - 1
    X_lag_np, y_np = make_lag_tensor(X, cfg.K)
    X_lag = torch.from_numpy(X_lag_np).to(DEVICE)
    y     = torch.from_numpy(y_np).to(DEVICE)

    best_model = None
    best_loss = float('inf')
    losses = []
    for r in range(n_restarts):
        restart_seed = seed + 10_000_000 * r
        model = fit_pairwise(X, cfg, seed=restart_seed, verbose=False)
        model.eval()
        with torch.no_grad():
            pred = model(X_lag)
            final_loss = float(F.mse_loss(pred, y).item())
        losses.append(final_loss)
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model
        if verbose:
            print(f'  pairwise restart {r+1}/{n_restarts}: final_loss={final_loss:.6f} '
                  f'(best: {best_loss:.6f})', flush=True)
    return best_model


@torch.no_grad()
def edge_product_l2_error_pairwise(model: PairwiseNAVAR, X_lag_t: torch.Tensor,
                                    X_lag_np: np.ndarray, j: int,
                                    true_edge_fn) -> float:
    """
    L2 error between the pairwise model's contribution from source j and the
    TRUE edge contribution (which includes gating in the DGP).

    The pairwise model's contribution from source j is just f_j(x_j) — no
    gates. So this metric directly measures: how well does an additive
    approximation capture the true (multiplicatively-modulated) contribution?

    The pairwise model has no way to model the gating, so this error reflects
    the fundamental limitation of additive models on multiplicatively-structured
    data. We expect it to be substantially larger than G-NAVAR's edge error.
    """
    n_samples = X_lag_t.shape[0]
    f_j = model.evaluate_base(j=j, x_block=X_lag_t[:, j, :]).cpu().numpy()
    fitted = f_j  # No gates -> no product, just the base function
    true_vals = np.array([true_edge_fn(X_lag_np[t]) for t in range(n_samples)])
    return float(np.sqrt(np.mean((fitted - true_vals) ** 2)))


def held_out_mse_pairwise(model: PairwiseNAVAR, X_test: np.ndarray, cfg: Config) -> float:
    """Forecast MSE on held-out data."""
    X_lag_np, y_np = make_lag_tensor(X_test, cfg.K)
    X_lag = torch.from_numpy(X_lag_np).to(DEVICE)
    y     = torch.from_numpy(y_np).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(X_lag)
        return float(F.mse_loss(pred, y).item())


def held_out_mse_gnavar(model: GNAVAR, X_test: np.ndarray, cfg: Config) -> float:
    """Forecast MSE on held-out data."""
    X_lag_np, y_np = make_lag_tensor(X_test, cfg.K)
    X_lag = torch.from_numpy(X_lag_np).to(DEVICE)
    y     = torch.from_numpy(y_np).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(X_lag)
        return float(F.mse_loss(pred, y).item())



@torch.no_grad()
def gate_triviality_score(model: GNAVAR, X_lag_t: torch.Tensor, j: int, k: int) -> float:
    """E[(g_{j,k}(x_k) - 1)^2] over the empirical lag-block samples of x_k."""
    n_sources = model.n_sources
    K = model.K
    x_block = X_lag_t[:, k, :]                       # (n_samples, K)
    g = model.evaluate_gate(j=j, k=k, x_block=x_block)
    return float(((g - 1.0) ** 2).mean().item())

@torch.no_grad()
def detect_modulator_set(model: GNAVAR, X_lag_t: torch.Tensor, j: int, threshold: float):
    """Return the set of modulator indices k for which the gate is non-trivial."""
    active = set()
    for k in range(model.n_sources):
        if k == j:
            continue
        if gate_triviality_score(model, X_lag_t, j, k) >= threshold:
            active.add(k)
    return active

@torch.no_grad()
def gauge_normalize(g_vals: np.ndarray):
    """Rescale so empirical mean is 1."""
    m = float(np.mean(g_vals))
    return (g_vals / m, m) if m > 1e-12 else (g_vals, 1.0)

@torch.no_grad()
def gate_l2_error(model: GNAVAR, X_lag_t: torch.Tensor, j: int, k: int,
                  true_gate_fn) -> float:
    """
    L2 error between fitted gate and true gate, evaluated on the empirical
    lag-block samples of x_k. Both are gauge-normalized to E[g] = 1 first.
    true_gate_fn: callable taking (lag1, lag2) numpy arrays.
    """
    x_block = X_lag_t[:, k, :]
    g_fitted = model.evaluate_gate(j=j, k=k, x_block=x_block).cpu().numpy()
    x_block_np = x_block.cpu().numpy()
    g_true = true_gate_fn(x_block_np[:, 0], x_block_np[:, 1])
    g_fitted_n, _ = gauge_normalize(g_fitted)
    g_true_n,   _ = gauge_normalize(g_true)
    return float(np.sqrt(np.mean((g_fitted_n - g_true_n) ** 2)))

@torch.no_grad()
def edge_product_l2_error(model: GNAVAR, X_lag_t: torch.Tensor, X_lag_np: np.ndarray,
                          j: int, true_edge_fn) -> float:
    """
    L2 error between fitted edge product (source j) and true edge product
    over empirical joint support.
    The fitted contribution from source j is f_j(x_j) * prod_k g_{j,k}(x_k).
    """
    n_sources = model.n_sources
    K = model.K
    n_samples = X_lag_t.shape[0]

    # Fitted base for source j
    f_j = model.evaluate_base(j=j, x_block=X_lag_t[:, j, :]).cpu().numpy()
    # Fitted gate product for source j (over k != j)
    log_prod = np.zeros(n_samples)
    for k in range(n_sources):
        if k == j:
            continue
        g_k = model.evaluate_gate(j=j, k=k, x_block=X_lag_t[:, k, :]).cpu().numpy()
        log_prod = log_prod + np.log(np.clip(g_k, 1e-12, None))
    g_prod = np.exp(log_prod)
    fitted = f_j * g_prod

    true_vals = np.array([true_edge_fn(X_lag_np[t]) for t in range(n_samples)])
    return float(np.sqrt(np.mean((fitted - true_vals) ** 2)))

# True edge contribution functions (matching the DGP)
# Layout reminder: X_lag_np[t, j, ell], j in 0..3 (sources x2..x5), ell in 0..1 (lag 1, lag 2)
def true_edge_source0(row):  # source j=0 = x2 ; truly contributes only at lag 1 (modulated)
    x2_l1, x2_l2 = row[0, 0], row[0, 1]
    x3_l1, x3_l2 = row[1, 0], row[1, 1]
    return true_f12(x2_l1) * true_g123(x3_l1, x3_l2)

def true_edge_source2(row):  # source j=2 = x4 ; modulated by x5
    x4_l1 = row[2, 0]
    x5_l1, x5_l2 = row[3, 0], row[3, 1]
    return true_f14(x4_l1) * true_g145(x5_l1, x5_l2)

# Note: the lag-2 unmodulated x2 contribution is captured by source j=0's base function
# operating on the (x2_lag1, x2_lag2) input. In the vector-input fitter, f_j is a single
# function of the full lag block, so the lag-1 modulated and lag-2 unmodulated contributions
# combine into one source-0 contribution. The edge-product metric for source 0 therefore
# evaluates against the *combined* truth: the modulated lag-1 plus the additive lag-2.
def true_edge_source0_combined(row):
    x2_l1, x2_l2 = row[0, 0], row[0, 1]
    x3_l1, x3_l2 = row[1, 0], row[1, 1]
    return true_f12(x2_l1) * true_g123(x3_l1, x3_l2) + true_f12_lag2(x2_l2)

def simulate_dgp_coupled(T: int, cfg: 'Config', seed: int, rho: float) -> np.ndarray:
    """
    Variant of simulate_dgp with controlled coupling between modulators x3 and x5.

    The coupling is:
        x_{5,t} = rho * x_{3,t} + sqrt(1 - rho^2) * eta_{5,t}
    where eta_{5,t} is independent noise. As rho -> 1, the support of (x_3, x_5)
    contracts from 2D toward a 1D line.

    All other variables follow the original DGP. This is used by Experiment 2
    to test Corollary 8.2: as support collapses, gate identifiability fails.
    """
    # Allow tiny floating-point slop above 1.0; rho gets clamped inside sqrt.
    assert -1e-9 <= rho <= 1.0 + 1e-9, f"rho must be in [0, 1], got {rho}"
    rho = float(min(1.0, max(0.0, rho)))
    rng = np.random.default_rng(seed)
    T_total = T + cfg.burn_in
    X = np.zeros((T_total, cfg.n_vars))
    X[:cfg.K] = 0.1 * rng.standard_normal((cfg.K, cfg.n_vars))

    eps_sources = rng.standard_normal((T_total, cfg.n_vars - 1))
    eps_target = cfg.sigma_eps * rng.standard_normal(T_total)

    # x5's eta noise -- independent of all other sources' innovations
    eta5 = rng.standard_normal(T_total)

    for t in range(cfg.K, T_total):
        # x2: AR(1)
        phi = cfg.ar_coefs[0]
        X[t, 1] = phi * X[t - 1, 1] + math.sqrt(1 - phi ** 2) * eps_sources[t, 0]
        # x3: AR(1)
        phi = cfg.ar_coefs[1]
        X[t, 2] = phi * X[t - 1, 2] + math.sqrt(1 - phi ** 2) * eps_sources[t, 1]
        # x4: AR(1)
        phi = cfg.ar_coefs[2]
        X[t, 3] = phi * X[t - 1, 3] + math.sqrt(1 - phi ** 2) * eps_sources[t, 2]
        # x5: coupled to x3, plus independent noise scaled by sqrt(1 - rho^2)
        # Then apply AR(1) on top so it remains stationary.
        # Construction: define x5_innov_t = rho * x3_innov_t + sqrt(1-rho^2) * eta5_t,
        # then x5_t = phi * x5_{t-1} + sqrt(1-phi^2) * x5_innov_t.
        # This preserves marginal AR(1) structure while inducing controlled
        # correlation between (x3_t, x5_t) via shared innovation.
        x5_innov = rho * eps_sources[t, 1] + math.sqrt(max(0.0, 1.0 - rho ** 2)) * eta5[t]
        phi = cfg.ar_coefs[3]
        X[t, 4] = phi * X[t - 1, 4] + math.sqrt(1 - phi ** 2) * x5_innov

        # Target x1: same DGP as before
        x2_l1, x2_l2 = X[t - 1, 1], X[t - 2, 1]
        x3_l1, x3_l2 = X[t - 1, 2], X[t - 2, 2]
        x4_l1 = X[t - 1, 3]
        x5_l1, x5_l2 = X[t - 1, 4], X[t - 2, 4]
        contrib_23   = true_f12(x2_l1) * true_g123(x3_l1, x3_l2)
        contrib_45   = true_f14(x4_l1) * true_g145(x5_l1, x5_l2)
        contrib_lag2 = true_f12_lag2(x2_l2)
        X[t, 0] = contrib_23 + contrib_45 + contrib_lag2 + eps_target[t]

    return X[cfg.burn_in:]


def save_model(model: 'GNAVAR', path) -> None:
    """Save a trained GNAVAR to disk, including architectural metadata."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'n_sources': model.n_sources,
        'K': model.K,
        'hidden_dim': model.bases.hidden_dim,
    }, path)


def load_model(path, device=None) -> 'GNAVAR':
    """Load a previously saved GNAVAR from disk."""
    if device is None:
        device = DEVICE
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GNAVAR(
        n_sources=ckpt['n_sources'],
        K=ckpt['K'],
        hidden_dim=ckpt['hidden_dim'],
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def effective_rank(X: np.ndarray) -> float:
    """
    Compute the participation-ratio effective rank of a data matrix:
        r_eff = (tr Sigma)^2 / tr(Sigma^2)
    where Sigma is the empirical covariance of X.

    X: (n_samples, n_features) data matrix.
    Returns: a scalar in (0, n_features].
    """
    X = X - X.mean(axis=0, keepdims=True)
    sigma = X.T @ X / max(1, X.shape[0] - 1)
    trace = float(np.trace(sigma))
    trace_sq = float(np.trace(sigma @ sigma))
    if trace_sq <= 1e-20:
        return 0.0
    return trace ** 2 / trace_sq


def effective_rank_subset(X_lag: np.ndarray, subset_indices: list) -> float:
    """
    Effective rank computed on the lag-windowed subset of variables.

    X_lag: (n_samples, n_sources, K) lag tensor.
    subset_indices: list of source indices to include (each 0-indexed in source space).
    Returns: effective rank of the joint lag-block covariance over the subset.
    """
    blocks = [X_lag[:, j, :] for j in subset_indices]  # each (n_samples, K)
    stacked = np.concatenate(blocks, axis=1)            # (n_samples, |subset| * K)
    return effective_rank(stacked)


# ----------------------------------------------------------------------
# Real-data helpers (Beijing experiment and similar applications)
# ----------------------------------------------------------------------

def make_lag_tensor_runs(X: np.ndarray, runs: np.ndarray, K: int,
                         target_col: int = 0):
    """
    Build supervised pairs (X_lag, y) from a trajectory with run boundaries.

    Real-world panel data is often a sequence of contiguous "runs" of clean
    observations separated by gaps where data was missing. We must compute lag
    pairs within each run only -- never spanning gap boundaries -- so that
    the autoregressive structure is correct.

    Parameters:
        X: (T, n_vars) full trajectory (with the target as column `target_col`).
        runs: (n_runs, 2) array of [start, end) indices defining each contiguous run.
        K: lag order.
        target_col: column index of the prediction target (default 0).

    Returns:
        X_lag: (n_samples, n_sources, K) -- only the non-target columns are sources.
        y:     (n_samples,) -- target values aligned to X_lag.
    """
    T, n_vars = X.shape
    n_sources = n_vars - 1
    # source_cols = all columns except target_col, preserving order
    source_cols = [c for c in range(n_vars) if c != target_col]
    sources = X[:, source_cols].astype(np.float32, copy=False)  # (T, n_sources)
    target  = X[:, target_col].astype(np.float32, copy=False)   # (T,)

    pieces_X = []
    pieces_y = []
    for start, end in runs:
        # Need run length >= K + 1 to produce even one usable sample.
        run_len = end - start
        if run_len <= K:
            continue
        # n_samples_run = run_len - K
        # For each lag ell in 0..K-1:
        #   X_lag[i, j, ell] = sources[start + K + i - (ell + 1), j]
        n = run_len - K
        run_X_lag = np.empty((n, n_sources, K), dtype=np.float32)
        for ell in range(K):
            s = start + K - (ell + 1)
            run_X_lag[:, :, ell] = sources[s : s + n, :]
        run_y = target[start + K : start + K + n]
        pieces_X.append(run_X_lag)
        pieces_y.append(run_y)

    if not pieces_X:
        # All runs too short
        return (np.empty((0, n_sources, K), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    X_lag = np.concatenate(pieces_X, axis=0)
    y     = np.concatenate(pieces_y, axis=0)
    return X_lag, y


def effective_rank_full(X_lag_np: np.ndarray) -> float:
    """
    Effective rank of the joint lag-block covariance of ALL sources.

    Used as a pre-fit diagnostic on real data, where there is no a-priori
    "modulator subset" to evaluate. Returns r_eff over the full
    (n_sources * K)-dimensional joint distribution.
    """
    n_samples, n_sources, K = X_lag_np.shape
    if n_samples < 2:
        return 0.0
    flat = X_lag_np.reshape(n_samples, n_sources * K)
    return effective_rank(flat)


@torch.no_grad()
def detect_all_modulators(model: 'GNAVAR', X_lag_t: torch.Tensor,
                          threshold: float) -> dict:
    """
    Return the set of active modulators for EACH source.

    Unlike detect_modulator_set (which targets a specific edge), this returns
    a dict {j: set_of_modulators} for every source j. Used in real-data
    applications where we want to inspect the recovered structure across all
    edges, not just two known-modulated ones.
    """
    out = {}
    for j in range(model.n_sources):
        active = set()
        for k in range(model.n_sources):
            if k == j:
                continue
            score = gate_triviality_score(model, X_lag_t, j, k)
            if score >= threshold:
                active.add(k)
        out[j] = active
    return out


def fit_gnavar_from_lag(X_lag_np: np.ndarray, y_np: np.ndarray, cfg: Config,
                         seed: int, verbose: bool = False) -> 'GNAVAR':
    """
    Train G-NAVAR on a precomputed (X_lag, y) pair instead of building the
    lag tensor internally. Use this when the lag tensor must respect run
    boundaries (real-data applications).

    The cfg must have K and n_vars set correctly. The n_sources is inferred
    from X_lag.shape[1].
    """
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    n_samples, n_sources, K = X_lag_np.shape
    assert K == cfg.K, f"cfg.K={cfg.K} does not match X_lag K={K}"

    X_lag = torch.from_numpy(X_lag_np).to(DEVICE)
    y     = torch.from_numpy(y_np).to(DEVICE)

    model = GNAVAR(n_sources=n_sources, K=K, hidden_dim=cfg.hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(n_samples, device=DEVICE)
        epoch_mse_sum = torch.zeros((), device=DEVICE)
        epoch_l1_sum  = torch.zeros((), device=DEVICE)
        n_batches = 0
        for i in range(0, n_samples, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = X_lag[idx], y[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                pred = model(xb)
                mse  = F.mse_loss(pred, yb)
                l1   = compute_gate_l1_penalty(model, xb)
                loss = mse + cfg.l1_lambda * l1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_mse_sum = epoch_mse_sum + mse.detach()
            epoch_l1_sum  = epoch_l1_sum + l1.detach()
            n_batches += 1
        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            mse_avg = float(epoch_mse_sum.item()) / n_batches
            l1_avg  = float(epoch_l1_sum.item()) / n_batches
            print(f'  epoch {epoch+1:3d}/{cfg.n_epochs}  '
                  f'mse={mse_avg:.5f}  l1={l1_avg:.5f}', flush=True)
    return model


def fit_gnavar_from_lag_with_restarts(X_lag_np, y_np, cfg, seed,
                                       n_restarts=3, verbose=False):
    """Restart-and-keep-best wrapper around fit_gnavar_from_lag."""
    X_lag = torch.from_numpy(X_lag_np).to(DEVICE)
    y     = torch.from_numpy(y_np).to(DEVICE)
    best_model = None
    best_loss = float('inf')
    losses = []
    for r in range(n_restarts):
        restart_seed = seed + 10_000_000 * r
        model = fit_gnavar_from_lag(X_lag_np, y_np, cfg, seed=restart_seed, verbose=False)
        model.eval()
        with torch.no_grad():
            pred = model(X_lag)
            final_loss = float(F.mse_loss(pred, y).item())
        losses.append(final_loss)
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model
        if verbose:
            print(f'  restart {r+1}/{n_restarts}: final_loss={final_loss:.6f} '
                  f'(best so far: {best_loss:.6f})', flush=True)
    if verbose:
        print(f'  selected restart with loss {best_loss:.6f} '
              f'(out of {[f"{l:.6f}" for l in losses]})', flush=True)
    return best_model


def fit_pairwise_from_lag(X_lag_np: np.ndarray, y_np: np.ndarray, cfg: Config,
                          seed: int, verbose: bool = False) -> 'PairwiseNAVAR':
    """Train PairwiseNAVAR on a precomputed (X_lag, y) pair. See fit_gnavar_from_lag."""
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    n_samples, n_sources, K = X_lag_np.shape
    assert K == cfg.K

    X_lag = torch.from_numpy(X_lag_np).to(DEVICE)
    y     = torch.from_numpy(y_np).to(DEVICE)

    model = PairwiseNAVAR(n_sources=n_sources, K=K, hidden_dim=cfg.hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(n_samples, device=DEVICE)
        epoch_loss_sum = torch.zeros((), device=DEVICE)
        n_batches = 0
        for i in range(0, n_samples, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = X_lag[idx], y[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss_sum = epoch_loss_sum + loss.detach()
            n_batches += 1
        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            loss_avg = float(epoch_loss_sum.item()) / n_batches
            print(f'  pairwise epoch {epoch+1:3d}/{cfg.n_epochs}  '
                  f'mse={loss_avg:.5f}', flush=True)
    return model


def fit_pairwise_from_lag_with_restarts(X_lag_np, y_np, cfg, seed,
                                         n_restarts=3, verbose=False):
    """Restart-and-keep-best wrapper around fit_pairwise_from_lag."""
    X_lag = torch.from_numpy(X_lag_np).to(DEVICE)
    y     = torch.from_numpy(y_np).to(DEVICE)
    best_model = None
    best_loss = float('inf')
    losses = []
    for r in range(n_restarts):
        restart_seed = seed + 10_000_000 * r
        model = fit_pairwise_from_lag(X_lag_np, y_np, cfg, seed=restart_seed, verbose=False)
        model.eval()
        with torch.no_grad():
            pred = model(X_lag)
            final_loss = float(F.mse_loss(pred, y).item())
        losses.append(final_loss)
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model
        if verbose:
            print(f'  pairwise restart {r+1}/{n_restarts}: final_loss={final_loss:.6f} '
                  f'(best so far: {best_loss:.6f})', flush=True)
    return best_model


@torch.no_grad()
def held_out_mse_gnavar_from_lag(model: 'GNAVAR', X_lag_np: np.ndarray,
                                   y_np: np.ndarray) -> float:
    """Forecast MSE on a precomputed held-out (X_lag, y) pair."""
    X_lag = torch.from_numpy(X_lag_np).to(DEVICE)
    y     = torch.from_numpy(y_np).to(DEVICE)
    model.eval()
    pred = model(X_lag)
    return float(F.mse_loss(pred, y).item())


@torch.no_grad()
def held_out_mse_pairwise_from_lag(model: 'PairwiseNAVAR', X_lag_np: np.ndarray,
                                     y_np: np.ndarray) -> float:
    """Forecast MSE on a precomputed held-out (X_lag, y) pair."""
    X_lag = torch.from_numpy(X_lag_np).to(DEVICE)
    y     = torch.from_numpy(y_np).to(DEVICE)
    model.eval()
    pred = model(X_lag)
    return float(F.mse_loss(pred, y).item())
