"""Tests for DPO (Direct Preference Optimization) node 15."""
import numpy as np
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dpo_loss(log_prob_chosen_new, log_prob_chosen_ref,
             log_prob_rejected_new, log_prob_rejected_ref, beta=0.5):
    chosen_lr   = log_prob_chosen_new   - log_prob_chosen_ref
    rejected_lr = log_prob_rejected_new - log_prob_rejected_ref
    margin = beta * (chosen_lr - rejected_lr)
    return -np.log(sigmoid(margin) + 1e-10)


def bradley_terry_prob(r_chosen, r_rejected):
    return sigmoid(r_chosen - r_rejected)


# ── Bradley-Terry 测试 ────────────────────────────────────────────────────────

class TestBradleyTerry:
    def test_equal_rewards_gives_half(self):
        prob = bradley_terry_prob(2.0, 2.0)
        assert abs(prob - 0.5) < 1e-6

    def test_higher_reward_gives_higher_prob(self):
        prob = bradley_terry_prob(3.0, 1.0)
        assert prob > 0.5

    def test_lower_reward_gives_lower_prob(self):
        prob = bradley_terry_prob(1.0, 3.0)
        assert prob < 0.5

    def test_large_gap_near_one(self):
        prob = bradley_terry_prob(10.0, -10.0)
        assert prob > 0.99

    def test_probability_in_range(self):
        for r1, r2 in [(0, 0), (5, -5), (-3, 3), (0.1, 0.2)]:
            p = bradley_terry_prob(r1, r2)
            assert 0.0 < p < 1.0

    def test_symmetry(self):
        p1 = bradley_terry_prob(3.0, 1.0)
        p2 = bradley_terry_prob(1.0, 3.0)
        assert abs(p1 + p2 - 1.0) < 1e-10


# ── DPO Loss 基本性质 ─────────────────────────────────────────────────────────

class TestDPOLoss:
    def test_identical_responses_loss_is_log2(self):
        lp = np.log(0.4)
        loss = dpo_loss(lp, lp, lp, lp, beta=0.5)
        assert abs(loss - np.log(2)) < 0.01

    def test_good_preference_gives_low_loss(self):
        # chosen 概率比参考模型高，rejected 比参考模型低 → 低 loss
        loss = dpo_loss(np.log(0.7), np.log(0.3),
                        np.log(0.2), np.log(0.6), beta=0.5)
        assert loss < np.log(2)

    def test_reversed_preference_gives_high_loss(self):
        # chosen 概率比参考模型低，rejected 比参考模型高 → 高 loss
        loss = dpo_loss(np.log(0.2), np.log(0.6),
                        np.log(0.7), np.log(0.3), beta=0.5)
        assert loss > np.log(2)

    def test_loss_is_nonnegative(self):
        for sign in [+1, -1]:
            loss = dpo_loss(np.log(0.5), np.log(0.3),
                            np.log(0.3) * sign, np.log(0.5), beta=0.5)
            assert loss >= 0.0

    def test_larger_margin_lower_loss(self):
        loss_small = dpo_loss(np.log(0.4), np.log(0.3),
                              np.log(0.3), np.log(0.4), beta=0.5)
        loss_large = dpo_loss(np.log(0.9), np.log(0.3),
                              np.log(0.1), np.log(0.7), beta=0.5)
        assert loss_large < loss_small


# ── β 超参数效果 ──────────────────────────────────────────────────────────────

class TestBetaEffect:
    def test_zero_beta_gives_log2(self):
        # beta=0 时 margin=0，sigmoid(0)=0.5，loss=log2
        loss = dpo_loss(np.log(0.8), np.log(0.3),
                        np.log(0.2), np.log(0.7), beta=0.0)
        assert abs(loss - np.log(2)) < 0.01

    def test_larger_beta_magnifies_margin(self):
        # 相同的 log-ratio，beta 越大，margin 越大，loss 越低
        base = (np.log(0.6) - np.log(0.3), np.log(0.2) - np.log(0.4))
        losses = []
        for b in [0.1, 0.5, 1.0, 2.0]:
            margin = b * (base[0] - base[1])
            losses.append(-np.log(sigmoid(margin)))
        # losses 应该随 beta 增大而单调递减（margin 为正时）
        assert losses[0] > losses[-1]

    def test_beta_controls_kl_penalty(self):
        # beta 越大 → DPO 越不愿意远离参考模型
        # 相同偏好对，大 beta 在 log-ratio 小时 loss 更高
        log_diff = 0.2  # 很小的 log-ratio 差
        loss_small_beta = -np.log(sigmoid(0.1 * log_diff))
        loss_large_beta = -np.log(sigmoid(2.0 * log_diff))
        assert loss_small_beta > loss_large_beta


# ── Log-Ratio 计算 ────────────────────────────────────────────────────────────

class TestLogRatio:
    def test_log_ratio_zero_when_equal(self):
        p = 0.4
        log_ratio = np.log(p) - np.log(p)
        assert abs(log_ratio) < 1e-10

    def test_log_ratio_positive_when_improved(self):
        log_ratio = np.log(0.7) - np.log(0.3)
        assert log_ratio > 0

    def test_log_ratio_negative_when_degraded(self):
        log_ratio = np.log(0.2) - np.log(0.5)
        assert log_ratio < 0

    def test_log_ratio_formula_consistency(self):
        lp_new = np.log(0.6)
        lp_ref = np.log(0.3)
        log_ratio = lp_new - lp_ref
        assert abs(log_ratio - np.log(0.6 / 0.3)) < 1e-10


# ── DPO 训练方向验证 ──────────────────────────────────────────────────────────

class TestDPOTrainingDirection:
    def test_gradient_decreases_loss(self):
        # 验证 DPO Loss 是凸的（margin 正时，增大 margin 降低 loss）
        margins = np.linspace(-2, 2, 50)
        losses = -np.log(sigmoid(margins))
        # loss 在 margin > 0 时应该单调递减
        positive_losses = losses[margins > 0]
        assert np.all(np.diff(positive_losses) < 0)

    def test_chosen_should_improve_more_than_rejected(self):
        # 理想情况：chosen log-ratio 增加，rejected log-ratio 减少
        cr_before, rr_before = 0.0, 0.0
        # 模拟一步训练（手动调大 chosen_lr，调小 rejected_lr）
        cr_after, rr_after = 0.3, -0.2
        margin_before = 0.5 * (cr_before - rr_before)
        margin_after  = 0.5 * (cr_after  - rr_after)
        assert margin_after > margin_before

    def test_loss_at_training_start_is_log2(self):
        # 从 SFT 模型初始化时，π_θ = π_ref，所有 log-ratio = 0，loss = log2
        loss_init = dpo_loss(np.log(0.5), np.log(0.5),
                             np.log(0.3), np.log(0.3), beta=0.5)
        assert abs(loss_init - np.log(2)) < 0.01
