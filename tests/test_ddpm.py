"""
Tests for DDPM (Denoising Diffusion Probabilistic Models) — Node 16
"""
import numpy as np
import pytest


# ── 复用 notebook 核心实现 ────────────────────────────────────────────────────

def make_linear_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_bar = np.cumprod(alphas)
    return betas, alphas, alphas_bar


def q_sample(x0, t, alphas_bar, noise=None):
    """正向过程：x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε"""
    if noise is None:
        noise = np.random.randn(*x0.shape)
    xt = np.sqrt(alphas_bar[t]) * x0 + np.sqrt(1.0 - alphas_bar[t]) * noise
    return xt, noise


def cosine_schedule(T=1000, s=0.008):
    t = np.arange(T + 1)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    betas = 1 - ab[1:] / ab[:-1]
    betas = np.clip(betas, 0, 0.999)
    return betas, 1 - betas, ab[1:]


# ── TestForwardProcess ────────────────────────────────────────────────────────

class TestForwardProcess:
    """验证正向加噪过程的数学性质"""

    def setup_method(self):
        self.betas, self.alphas, self.alphas_bar = make_linear_schedule(T=1000)

    def test_xt_mean_at_t0(self):
        """t=0 时 ᾱ_0 ≈ 1，x_t ≈ x_0"""
        x0 = np.array([5.0])
        xt, _ = q_sample(x0, 0, self.alphas_bar)
        # xt = sqrt(ᾱ_0)*x0 + sqrt(1-ᾱ_0)*ε，ᾱ_0 ≈ 0.9999
        assert abs(xt[0] - x0[0]) < 1.0, "t=0 时 x_t 应接近 x_0"

    def test_xt_distribution_mean(self):
        """q(x_t|x_0) 的均值应为 sqrt(ᾱ_t)*x_0"""
        x0_val = 3.0
        t = 500
        n = 50000
        x0 = np.full((n, 1), x0_val)
        xt, _ = q_sample(x0, t, self.alphas_bar)
        expected_mean = np.sqrt(self.alphas_bar[t]) * x0_val
        assert abs(xt.mean() - expected_mean) < 0.02

    def test_xt_distribution_std(self):
        """q(x_t|x_0) 的标准差应为 sqrt(1-ᾱ_t)"""
        x0_val = 3.0
        t = 500
        n = 50000
        x0 = np.full((n, 1), x0_val)
        xt, _ = q_sample(x0, t, self.alphas_bar)
        expected_std = np.sqrt(1.0 - self.alphas_bar[t])
        assert abs(xt.std() - expected_std) < 0.02

    def test_xt_approaches_noise_at_T(self):
        """t=T-1 时 ᾱ_T ≈ 0，x_T 应接近标准正态（均值≈0，std≈1）"""
        n = 50000
        t = 999
        x0 = np.full((n, 1), 100.0)  # 极大信号
        xt, _ = q_sample(x0, t, self.alphas_bar)
        assert abs(xt.mean()) < 1.0, "t=T 时均值应接近 0"
        assert abs(xt.std() - 1.0) < 0.05, "t=T 时标准差应接近 1"

    def test_noise_injection_is_independent(self):
        """给定相同 x0 和 t，不同噪声应得到不同 x_t"""
        x0 = np.array([2.0])
        t = 300
        xt1, _ = q_sample(x0, t, self.alphas_bar, noise=np.array([1.0]))
        xt2, _ = q_sample(x0, t, self.alphas_bar, noise=np.array([-1.0]))
        assert xt1[0] != xt2[0], "不同噪声应产生不同 x_t"

    def test_signal_coefficient_monotone_decrease(self):
        """sqrt(ᾱ_t) 随 t 单调递减（信号权重越来越小）"""
        signal_coefs = np.sqrt(self.alphas_bar)
        assert np.all(np.diff(signal_coefs) <= 0), "信号系数应单调递减"


# ── TestReparamTrick ──────────────────────────────────────────────────────────

class TestReparamTrick:
    """验证重参数化采样的等价性"""

    def setup_method(self):
        self.betas, self.alphas, self.alphas_bar = make_linear_schedule(T=1000)

    def test_direct_vs_stepwise_distribution_match(self):
        """直接采样 x_t 与逐步两步采样的分布应一致"""
        np.random.seed(0)
        T_small = 2
        beta1, beta2 = 0.1, 0.2
        a1, a2 = 1 - beta1, 1 - beta2
        ab2 = a1 * a2

        n = 50000
        x0 = np.ones(n)

        # 方法1：直接用 ᾱ_2
        eps = np.random.randn(n)
        xt_direct = np.sqrt(ab2) * x0 + np.sqrt(1 - ab2) * eps

        # 方法2：逐步两步
        eps1 = np.random.randn(n)
        x1 = np.sqrt(a1) * x0 + np.sqrt(1 - a1) * eps1
        eps2 = np.random.randn(n)
        x2_step = np.sqrt(a2) * x1 + np.sqrt(1 - a2) * eps2

        # 两种方法的分布均值/方差应一致
        assert abs(xt_direct.mean() - x2_step.mean()) < 0.05
        assert abs(xt_direct.std() - x2_step.std()) < 0.05

    def test_reparameterization_preserves_signal(self):
        """重参数化公式中，信号部分 sqrt(ᾱ_t)*x_0 权重正确"""
        x0_val = 10.0
        t = 200
        n = 10000
        x0 = np.full((n,), x0_val)
        ab = self.alphas_bar[t]
        xt, _ = q_sample(x0.reshape(-1, 1), t, self.alphas_bar)
        # 均值应为 sqrt(ᾱ_t) * x0_val
        assert abs(xt.mean() - np.sqrt(ab) * x0_val) < 0.1


# ── TestNoiseSchedule ─────────────────────────────────────────────────────────

class TestNoiseSchedule:
    """验证噪声调度的性质"""

    def test_linear_betas_monotone_increase(self):
        betas, _, _ = make_linear_schedule()
        assert np.all(np.diff(betas) >= 0)

    def test_linear_betas_range(self):
        betas, _, _ = make_linear_schedule()
        assert betas[0] >= 1e-5
        assert betas[-1] <= 0.05
        assert np.all(betas > 0) and np.all(betas < 1)

    def test_alphas_bar_decreasing(self):
        _, _, ab = make_linear_schedule()
        assert np.all(np.diff(ab) <= 0)

    def test_alphas_bar_starts_near_1(self):
        _, _, ab = make_linear_schedule()
        assert ab[0] > 0.99

    def test_alphas_bar_ends_near_0(self):
        _, _, ab = make_linear_schedule()
        assert ab[-1] < 1e-3

    def test_cosine_betas_in_valid_range(self):
        betas, _, _ = cosine_schedule()
        assert np.all(betas >= 0) and np.all(betas <= 0.999)

    def test_cosine_alphas_bar_decreasing(self):
        _, _, ab = cosine_schedule()
        assert np.all(np.diff(ab) <= 0)

    def test_cosine_starts_near_1(self):
        _, _, ab = cosine_schedule()
        assert ab[0] > 0.99

    def test_cosine_ends_near_0(self):
        _, _, ab = cosine_schedule()
        assert ab[-1] < 0.01


# ── TestTrainingObjective ─────────────────────────────────────────────────────

class TestTrainingObjective:
    """验证训练目标（噪声预测 MSE）的计算"""

    def test_perfect_predictor_zero_loss(self):
        """完美预测噪声时 MSE 应为 0"""
        eps_true = np.array([1.0, -0.5, 0.3])
        eps_pred = eps_true.copy()
        loss = np.mean((eps_pred - eps_true) ** 2)
        assert loss == pytest.approx(0.0)

    def test_zero_predictor_loss_equals_variance(self):
        """全零预测时 MSE = E[ε²] = 1（标准正态）"""
        np.random.seed(7)
        eps_true = np.random.randn(100000)
        eps_pred = np.zeros_like(eps_true)
        loss = np.mean((eps_pred - eps_true) ** 2)
        assert abs(loss - 1.0) < 0.02

    def test_loss_is_nonnegative(self):
        eps_true = np.random.randn(100)
        eps_pred = np.random.randn(100)
        loss = np.mean((eps_pred - eps_true) ** 2)
        assert loss >= 0

    def test_symmetry(self):
        """MSE(a,b) == MSE(b,a)"""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert np.mean((a - b) ** 2) == pytest.approx(np.mean((b - a) ** 2))


# ── TestSamplingLoop ──────────────────────────────────────────────────────────

class TestSamplingLoop:
    """验证 DDPM 采样循环"""

    def setup_method(self):
        self.betas, self.alphas, self.ab = make_linear_schedule(T=200)

    def _reverse_step(self, xt, t_idx, eps_pred):
        """单步反向去噪"""
        coef = self.betas[t_idx] / np.sqrt(1 - self.ab[t_idx])
        mean = (1.0 / np.sqrt(self.alphas[t_idx])) * (xt - coef * eps_pred)
        if t_idx > 0:
            z = np.random.randn(*xt.shape)
            return mean + np.sqrt(self.betas[t_idx]) * z
        return mean

    def test_single_reverse_step_reduces_noise(self):
        """单步去噪后，x_{t-1} 的值不完全等于 x_t（有实质变化）"""
        xt = np.array([2.0, -1.0, 0.5])
        eps_pred = np.array([0.5, -0.3, 0.1])
        t_idx = 100
        xt_prev = self._reverse_step(xt, t_idx, eps_pred)
        assert not np.allclose(xt_prev, xt), "反向步骤应改变 x_t"

    def test_last_step_no_noise(self):
        """t=0 时不添加额外噪声（确定性步骤）"""
        np.random.seed(0)
        xt = np.array([1.0, 0.5])
        eps_pred = np.array([0.2, 0.1])
        r1 = self._reverse_step(xt, 0, eps_pred)
        r2 = self._reverse_step(xt, 0, eps_pred)
        np.testing.assert_array_equal(r1, r2, err_msg="t=0 时应确定性（无噪声）")

    def test_output_shape_preserved(self):
        """采样输出维度与输入一致"""
        x = np.random.randn(50)
        eps_pred = np.random.randn(50)
        x_prev = self._reverse_step(x, 50, eps_pred)
        assert x_prev.shape == x.shape

    def test_sampling_from_noise_changes_distribution(self):
        """从大方差噪声出发，经过多步去噪后，分布应发生改变"""
        np.random.seed(42)
        x = np.random.randn(1000) * 5.0  # 大方差初始噪声
        x_initial = x.copy()
        # 运行 50 步去噪（零预测网络，仅验证机制）
        for t_idx in range(49, -1, -1):
            eps_pred = np.zeros_like(x)  # 极简：预测为 0
            x = self._reverse_step(x, t_idx, eps_pred)
        # 经过去噪后，分布特征应改变
        assert x.std() != x_initial.std() or x.mean() != x_initial.mean()
