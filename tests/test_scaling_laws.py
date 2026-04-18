"""
tests/test_scaling_laws.py — 节点25 Scaling Laws 测试

覆盖：
- 幂律函数基本性质（单调递减、正值）
- 参数量/数据量/计算量幂律指数的合理范围
- 最优模型大小计算（Kaplan 和 Chinchilla 两个版本）
- 对数坐标下的线性关系
- 数值稳定性（极端输入不产生 NaN）
- 幂律拟合精度
"""
import numpy as np
import pytest


# ─── Core functions (mirror notebook) ─────────────────────────────────────

def power_law(x, a, alpha):
    """幂律函数: L = a * x^(-alpha)"""
    return a * x ** (-alpha)


def kaplan_optimal_N(C, scale=1e8):
    """Kaplan 2020: N_opt ∝ C^0.73"""
    return scale * C ** 0.73


def chinchilla_optimal_N(C):
    """Chinchilla 2022: N = sqrt(C/120)"""
    return np.sqrt(C / 120)


def fit_power_law(x_data, y_data):
    """在对数空间拟合幂律，返回 (a, alpha)"""
    log_x = np.log10(x_data)
    log_y = np.log10(y_data)
    coeffs = np.polyfit(log_x, log_y, 1)
    alpha = -coeffs[0]
    a = 10 ** coeffs[1]
    return a, alpha


# ─── Tests ────────────────────────────────────────────────────────────────

class TestPowerLawBasics:
    """幂律函数基本性质"""

    def test_power_law_positive(self):
        """幂律函数对正输入返回正值"""
        x = np.logspace(6, 11, 20)
        y = power_law(x, a=1.0, alpha=0.076)
        assert np.all(y > 0), "幂律值应全为正"

    def test_power_law_monotone_decreasing(self):
        """幂律函数单调递减（随x增大，Loss减小）"""
        x = np.logspace(6, 11, 50)
        y = power_law(x, a=1.0, alpha=0.076)
        diffs = np.diff(y)
        assert np.all(diffs < 0), "幂律函数应单调递减"

    def test_power_law_no_nan(self):
        """幂律函数不产生 NaN 或 inf"""
        x = np.logspace(1, 15, 100)
        y = power_law(x, a=1.0, alpha=0.076)
        assert np.all(np.isfinite(y)), "幂律输出不应含 NaN/inf"

    def test_power_law_alpha_zero_is_constant(self):
        """alpha=0 时幂律为常数"""
        x = np.logspace(6, 11, 10)
        y = power_law(x, a=3.5, alpha=0.0)
        assert np.allclose(y, 3.5), "alpha=0 时函数值应等于常数 a"

    def test_power_law_log_log_linear(self):
        """在对数-对数坐标下，幂律是直线（残差很小）"""
        x = np.logspace(6, 11, 20)
        y = power_law(x, a=1.0, alpha=0.076)
        log_x = np.log10(x)
        log_y = np.log10(y)
        coeffs = np.polyfit(log_x, log_y, 1)
        residuals = log_y - np.polyval(coeffs, log_x)
        assert np.max(np.abs(residuals)) < 1e-10, "对数空间残差应极小"


class TestScalingExponents:
    """Kaplan 2020 报告的幂律指数合理性检验"""

    @pytest.mark.parametrize("alpha, name", [
        (0.076, "参数量指数"),
        (0.095, "数据量指数"),
        (0.050, "计算量指数"),
    ])
    def test_alpha_positive(self, alpha, name):
        """幂律指数应为正值（Loss 随规模增大而减小）"""
        assert alpha > 0, f"{name} 应为正值"

    @pytest.mark.parametrize("alpha, name", [
        (0.076, "参数量指数"),
        (0.095, "数据量指数"),
        (0.050, "计算量指数"),
    ])
    def test_alpha_less_than_one(self, alpha, name):
        """幂律指数应 < 1（收益递减，不是线性）"""
        assert alpha < 1.0, f"{name} 应 < 1（否则意味着线性甚至超线性提升）"

    def test_data_alpha_greater_than_params_alpha(self):
        """数据量指数 > 参数量指数（相同倍数下，数据更有效）"""
        alpha_N = 0.076
        alpha_D = 0.095
        assert alpha_D > alpha_N, "数据量幂律指数应大于参数量指数"

    def test_10x_params_reduction(self):
        """参数量翻10倍，Loss 降低约 16%"""
        alpha_N = 0.076
        reduction = 1 - 10 ** (-alpha_N)
        assert 0.14 < reduction < 0.18, f"10倍参数量的降幅应在14-18%之间，实际{reduction:.3f}"

    def test_10x_data_reduction(self):
        """数据量翻10倍，Loss 降低约 20%"""
        alpha_D = 0.095
        reduction = 1 - 10 ** (-alpha_D)
        assert 0.18 < reduction < 0.22, f"10倍数据量的降幅应在18-22%之间，实际{reduction:.3f}"


class TestOptimalAllocation:
    """最优计算分配（Kaplan vs Chinchilla）"""

    def test_kaplan_optimal_n_increases_with_compute(self):
        """Kaplan 最优模型大小随计算量单调递增"""
        C_values = np.logspace(18, 24, 10)
        N_values = kaplan_optimal_N(C_values)
        assert np.all(np.diff(N_values) > 0), "最优模型大小应随计算量增大"

    def test_kaplan_exponent_073(self):
        """Kaplan 最优 N ∝ C^0.73：10倍计算 → 约5.4倍参数"""
        C1, C2 = 1e20, 1e21  # 10倍计算
        N1 = kaplan_optimal_N(C1)
        N2 = kaplan_optimal_N(C2)
        ratio = N2 / N1
        # C^0.73 → 10^0.73 ≈ 5.37
        assert 4.8 < ratio < 6.0, f"10倍计算下最优N增幅应在4.8-6.0倍，实际{ratio:.2f}"

    def test_chinchilla_optimal_n_increases_with_compute(self):
        """Chinchilla 最优模型大小随计算量单调递增"""
        C_values = np.logspace(18, 24, 10)
        N_values = chinchilla_optimal_N(C_values)
        assert np.all(np.diff(N_values) > 0)

    def test_chinchilla_vs_kaplan_at_large_compute(self):
        """Chinchilla 在大计算量下推荐比 Kaplan 更小的模型"""
        C = 3e23  # GPT-3 级别的计算量
        N_kaplan = kaplan_optimal_N(C)
        N_chin = chinchilla_optimal_N(C)
        assert N_chin < N_kaplan, "在大计算量下 Chinchilla 推荐更小的模型（更强调数据）"

    def test_chinchilla_tokens_to_params_ratio(self):
        """Chinchilla 每参数约20个token的关系"""
        C = 1e22  # 固定计算量
        N_opt = chinchilla_optimal_N(C)
        # D_opt = C / (6 * N_opt)
        D_opt = C / (6 * N_opt)
        ratio = D_opt / N_opt
        assert 18 < ratio < 22, f"最优 token/param 比率应在18-22之间，实际{ratio:.1f}"


class TestPowerLawFitting:
    """从数据拟合幂律指数"""

    def test_fit_recovers_true_alpha(self):
        """无噪声数据下，拟合应精确恢复真实 alpha"""
        true_alpha = 0.076
        x = np.logspace(6, 11, 20)
        y = power_law(x, a=1.0, alpha=true_alpha)
        _, fitted_alpha = fit_power_law(x, y)
        assert abs(fitted_alpha - true_alpha) < 1e-6, f"拟合 alpha={fitted_alpha} 与真实值差异过大"

    def test_fit_robust_to_small_noise(self):
        """小噪声下，拟合误差应在5%以内"""
        rng = np.random.RandomState(0)
        true_alpha = 0.076
        x = np.logspace(6, 11, 20)
        y = power_law(x, a=1.0, alpha=true_alpha) * (1 + rng.randn(20) * 0.02)
        _, fitted_alpha = fit_power_law(x, y)
        error = abs(fitted_alpha - true_alpha) / true_alpha
        assert error < 0.05, f"噪声下拟合误差{error:.3f}超过5%"
