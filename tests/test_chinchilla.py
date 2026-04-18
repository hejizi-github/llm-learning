"""Tests for Chinchilla scaling law concepts (node 13)."""

import numpy as np
import pytest


# ── Core math functions (mirroring the notebook) ─────────────────────────────

A_PARAM = 406.4
B_PARAM = 410.7
ALPHA_C = 0.34
BETA_C  = 0.28
L_INF   = 1.69


def chinchilla_loss(N, D):
    """L(N,D) = A/N^alpha + B/D^beta + L_inf"""
    return A_PARAM / (N ** ALPHA_C) + B_PARAM / (D ** BETA_C) + L_INF


def power_law_loss(N, A=7.0, alpha=0.076):
    """Single-factor power law: L(N) = A / N^alpha"""
    return A / (N ** alpha)


def compute_cost(N, D):
    """Approximate FLOPs: C ≈ 6 * N * D"""
    return 6 * N * D


def optimal_tokens(N_params, ratio=20):
    """Chinchilla rule: ~20 tokens per parameter"""
    return ratio * N_params


def find_optimal_allocation(C_flops, n_candidates=100):
    """Grid search: given compute budget C, find (N*, D*) minimizing L(N,D)."""
    best_loss = float('inf')
    best_N = None
    best_D = None
    upper = C_flops / (6 * 1e6)  # D must be at least 1M tokens
    N_candidates = np.logspace(8, np.log10(max(upper, 1e9)), n_candidates)
    for N in N_candidates:
        D = C_flops / (6 * N)
        if D < 1e6:
            continue
        L = chinchilla_loss(N, D)
        if L < best_loss:
            best_loss = L
            best_N = N
            best_D = D
    return best_N, best_D, best_loss


# ── Tests: power law ──────────────────────────────────────────────────────────

class TestPowerLaw:
    def test_loss_decreases_with_more_params(self):
        """Larger N → lower loss (monotone decreasing)."""
        L1 = power_law_loss(1e8)
        L2 = power_law_loss(1e9)
        L3 = power_law_loss(1e10)
        assert L1 > L2 > L3

    def test_loss_always_positive(self):
        for N in [1e7, 1e9, 1e11]:
            assert power_law_loss(N) > 0

    def test_power_law_ratio(self):
        """10x params → loss ratio = 10^(-alpha)."""
        alpha = 0.076
        A = 7.0
        L1 = power_law_loss(1e9, A, alpha)
        L2 = power_law_loss(10e9, A, alpha)
        expected_ratio = 10 ** alpha  # L1/L2 should equal 10^alpha
        assert abs(L1 / L2 - expected_ratio) < 0.001

    def test_loss_finite_for_large_N(self):
        """Loss must be finite even for 1T parameters."""
        assert np.isfinite(power_law_loss(1e12))


# ── Tests: Chinchilla two-factor loss ─────────────────────────────────────────

class TestChinchillaLoss:
    def test_loss_decreases_with_more_params(self):
        """More params → lower loss (holding D constant)."""
        D = 1e11
        assert chinchilla_loss(1e9, D) > chinchilla_loss(1e10, D)

    def test_loss_decreases_with_more_data(self):
        """More data → lower loss (holding N constant)."""
        N = 70e9
        assert chinchilla_loss(N, 1e11) > chinchilla_loss(N, 1e12)

    def test_loss_lower_bound_is_l_inf(self):
        """Loss should always be >= L_inf."""
        for N in [1e10, 1e11, 1e12]:
            for D in [1e11, 1e12, 1e13]:
                assert chinchilla_loss(N, D) >= L_INF

    def test_chinchilla_beats_gopher(self):
        """Chinchilla (70B, 1.4T) should have lower predicted loss than Gopher (280B, 300B)."""
        L_chinchilla = chinchilla_loss(70e9, 1400e9)
        L_gopher     = chinchilla_loss(280e9, 300e9)
        assert L_chinchilla < L_gopher, (
            f"Chinchilla loss {L_chinchilla:.4f} should be < Gopher loss {L_gopher:.4f}"
        )

    def test_chinchilla_beats_gpt3(self):
        """Chinchilla (70B, 1.4T) should beat GPT-3 (175B, 300B)."""
        L_chinchilla = chinchilla_loss(70e9, 1400e9)
        L_gpt3       = chinchilla_loss(175e9, 300e9)
        assert L_chinchilla < L_gpt3

    def test_symmetric_large_values(self):
        """Very large N and D should converge near L_inf."""
        L = chinchilla_loss(1e14, 1e16)
        assert L < L_INF + 0.1

    def test_loss_is_symmetric_in_spirit(self):
        """Bottleneck side matters: tiny N or tiny D both hurt loss."""
        N_ok  = 70e9
        D_ok  = 1400e9
        L_balanced   = chinchilla_loss(N_ok, D_ok)
        L_tiny_data  = chinchilla_loss(N_ok, 1e9)   # data-bottlenecked
        L_tiny_model = chinchilla_loss(1e8, D_ok)   # model-bottlenecked
        assert L_tiny_data  > L_balanced
        assert L_tiny_model > L_balanced


# ── Tests: compute cost ───────────────────────────────────────────────────────

class TestComputeCost:
    def test_compute_scales_linearly_with_N(self):
        D = 1e11
        assert compute_cost(2e9, D) == pytest.approx(2 * compute_cost(1e9, D))

    def test_compute_scales_linearly_with_D(self):
        N = 70e9
        assert compute_cost(N, 2e11) == pytest.approx(2 * compute_cost(N, 1e11))

    def test_compute_formula_matches_chinchilla(self):
        """C = 6*N*D is the standard approximation."""
        N, D = 70e9, 1400e9
        C = compute_cost(N, D)
        assert C == pytest.approx(6 * 70e9 * 1400e9)


# ── Tests: Chinchilla 20-token rule ───────────────────────────────────────────

class TestTwentyTokenRule:
    def test_seven_billion_model_needs_140b_tokens(self):
        assert optimal_tokens(7e9) == pytest.approx(140e9)

    def test_seventy_billion_model_needs_1_4t_tokens(self):
        assert optimal_tokens(70e9) == pytest.approx(1400e9)

    def test_gpt3_is_undertrained(self):
        """GPT-3 (175B) needs ~3.5T tokens; it was trained on only 300B."""
        gpt3_optimal = optimal_tokens(175e9)
        gpt3_actual  = 300e9
        assert gpt3_actual < gpt3_optimal
        assert gpt3_actual / gpt3_optimal < 0.1  # less than 10% of optimal

    def test_chinchilla_is_near_optimal(self):
        """Chinchilla (70B, 1.4T) token ratio is ~20, close to optimal."""
        ratio = 1400e9 / 70e9
        assert 15 <= ratio <= 25


# ── Tests: optimal allocation ─────────────────────────────────────────────────

class TestOptimalAllocation:
    def test_optimal_loss_is_lower_than_gopher(self):
        """At Gopher-level compute, the optimal allocation beats Gopher's actual."""
        C_gopher = compute_cost(280e9, 300e9)
        _, _, L_opt = find_optimal_allocation(C_gopher)
        L_gopher = chinchilla_loss(280e9, 300e9)
        assert L_opt < L_gopher

    def test_token_param_ratio_in_reasonable_range(self):
        """Optimal token/param ratio from parametric model should be >> 1 (data-heavy)."""
        C = 1e22
        N_opt, D_opt, _ = find_optimal_allocation(C)
        ratio = D_opt / N_opt
        # The parametric model with alpha=0.34, beta=0.28 gives ratios in 50-100 range;
        # the empirical "20 tokens/param" rule is a practical approximation from Table 2
        # of Hoffmann et al. Both confirm: data should outnumber parameters by many-fold.
        assert 10 <= ratio <= 200, f"Unexpected ratio {ratio:.1f}"

    def test_larger_budget_gives_more_optimal_params(self):
        """More compute → larger optimal N."""
        _, _, _ = find_optimal_allocation(1e21)  # warm up
        N1, _, _ = find_optimal_allocation(1e21)
        N2, _, _ = find_optimal_allocation(1e22)
        assert N2 > N1

    def test_larger_budget_gives_more_optimal_data(self):
        """More compute → larger optimal D."""
        _, D1, _ = find_optimal_allocation(1e21)
        _, D2, _ = find_optimal_allocation(1e22)
        assert D2 > D1
