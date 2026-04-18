"""
tests/test_chain_of_thought.py — 节点23 Chain-of-Thought Prompting 测试

覆盖：
- CoT 比直接回答准确率更高
- CoT 误差分布更集中（标准差更小）
- Zero-shot CoT（加咒语）比不加咒语准确率更高
- 步骤数越多时，CoT 优势越显著
"""
import numpy as np
import pytest


# ─── Helpers (mirror notebook implementations) ────────────────────────────

def standard_answer(a, b, c, rng):
    """直接回答：对整个 a+b-c 一步猜，误差范围与 a 相关"""
    correct = a + b - c
    err_range = max(2, a // 3)
    err = rng.randint(-err_range, err_range + 1)
    return correct + err


def cot_answer(a, b, c, rng):
    """CoT 回答：分两步，每步允许±1误差"""
    step1 = (a + b) + rng.randint(-1, 2)
    return step1 - c + rng.randint(-1, 2)


def is_correct(pred, true_val, tolerance=1):
    return abs(pred - true_val) <= tolerance


def run_standard(n, rng):
    results = []
    for _ in range(n):
        a = rng.randint(5, 20)
        b = rng.randint(1, 10)
        c = rng.randint(1, 8)
        true_ans = a + b - c
        pred = standard_answer(a, b, c, rng)
        results.append(is_correct(pred, true_ans))
    return np.array(results)


def run_cot(n, rng):
    results = []
    for _ in range(n):
        a = rng.randint(5, 20)
        b = rng.randint(1, 10)
        c = rng.randint(1, 8)
        true_ans = a + b - c
        pred = cot_answer(a, b, c, rng)
        results.append(is_correct(pred, true_ans))
    return np.array(results)


# ─── Tests ────────────────────────────────────────────────────────────────

class TestChainOfThoughtAccuracy:
    def test_cot_higher_accuracy_than_standard(self):
        """CoT 准确率应显著高于直接回答"""
        rng = np.random.RandomState(42)
        n = 500
        standard_acc = run_standard(n, rng).mean()
        cot_acc = run_cot(n, rng).mean()
        assert cot_acc > standard_acc, (
            f"CoT ({cot_acc:.1%}) should beat Standard ({standard_acc:.1%})"
        )

    def test_cot_accuracy_above_threshold(self):
        """CoT 准确率应超过 70%（两步简单运算）"""
        rng = np.random.RandomState(0)
        n = 500
        cot_acc = run_cot(n, rng).mean()
        assert cot_acc > 0.70, f"CoT accuracy {cot_acc:.1%} should be > 70%"

    def test_standard_accuracy_lower(self):
        """直接回答准确率应低于 CoT（验证模拟设计合理）"""
        rng = np.random.RandomState(7)
        n = 500
        std_acc = run_standard(n, rng).mean()
        cot_acc = run_cot(n, rng).mean()
        assert std_acc < cot_acc

    def test_cot_error_smaller_variance(self):
        """CoT 误差方差应小于直接回答误差方差"""
        rng = np.random.RandomState(1)
        n = 500
        std_errors = []
        cot_errors = []
        for _ in range(n):
            a = rng.randint(5, 20)
            b = rng.randint(1, 10)
            c = rng.randint(1, 8)
            true_ans = a + b - c
            std_errors.append(standard_answer(a, b, c, rng) - true_ans)
            cot_errors.append(cot_answer(a, b, c, rng) - true_ans)

        assert np.std(cot_errors) < np.std(std_errors), (
            f"CoT error std {np.std(cot_errors):.2f} should be < "
            f"Standard error std {np.std(std_errors):.2f}"
        )


class TestZeroShotCoT:
    def setup_method(self):
        self.rng = np.random.RandomState(42)
        self.n = 1000

    def _solve_with_strategy(self, use_cot_prob):
        """按给定概率使用 CoT 策略"""
        correct_count = 0
        rng = np.random.RandomState(99)
        for _ in range(self.n):
            a = rng.randint(5, 20)
            b = rng.randint(1, 10)
            c = rng.randint(1, 8)
            true_ans = a + b - c
            if rng.random() < use_cot_prob:
                pred = cot_answer(a, b, c, rng)
            else:
                pred = standard_answer(a, b, c, rng)
            if is_correct(pred, true_ans):
                correct_count += 1
        return correct_count / self.n

    def test_zero_shot_cue_improves_accuracy(self):
        """加了 CoT 提示后准确率应提升"""
        no_cue_acc = self._solve_with_strategy(use_cot_prob=0.5)
        with_cue_acc = self._solve_with_strategy(use_cot_prob=0.9)
        assert with_cue_acc > no_cue_acc, (
            f"With cue ({with_cue_acc:.1%}) should beat no cue ({no_cue_acc:.1%})"
        )


class TestCoTProperties:
    def test_step_decomposition_reduces_error_propagation(self):
        """CoT 总误差上界（±2）比直接回答平均误差更小（a∈[10,20]时直接误差可达±6）"""
        rng = np.random.RandomState(5)
        cot_errors = []
        direct_errors = []
        for _ in range(500):
            a = rng.randint(10, 20)  # 保证 a//3 >= 3，直接回答误差 > CoT 上界
            b = rng.randint(1, 10)
            c = rng.randint(1, 8)
            true_ans = a + b - c
            # CoT：两步各±1，总误差范围 [-2, 2]
            cot_pred = cot_answer(a, b, c, rng)
            cot_errors.append(abs(cot_pred - true_ans))
            # 直接回答：误差范围 ±(a//3)
            std_pred = standard_answer(a, b, c, rng)
            direct_errors.append(abs(std_pred - true_ans))

        # CoT 平均误差 < 直接回答平均误差（如果改大 CoT 步骤噪声这个断言会失败）
        assert np.mean(cot_errors) < np.mean(direct_errors), (
            f"CoT avg error {np.mean(cot_errors):.2f} should < direct avg error "
            f"{np.mean(direct_errors):.2f}"
        )
        # CoT 最大误差上界为 2（两步各±1）
        assert max(cot_errors) <= 2, (
            f"CoT max error {max(cot_errors)} exceeds expected bound of 2"
        )

    def test_more_steps_makes_direct_harder(self):
        """三步运算时，直接回答比两步误差更大"""
        rng = np.random.RandomState(10)
        n = 300

        two_step_errors = []
        three_step_errors = []

        for _ in range(n):
            a, b, c, d_val = (rng.randint(5, 15) for _ in range(4))
            true2 = a + b - c
            true3 = a + b - c + d_val

            # 两步：误差范围 max(2, a//3)
            err2 = rng.randint(-max(2, a//3), max(2, a//3)+1)
            # 三步：误差范围 max(3, a//2)（更多步骤，更难）
            err3 = rng.randint(-max(3, a//2), max(3, a//2)+1)

            two_step_errors.append(abs(err2))
            three_step_errors.append(abs(err3))

        # 三步直接回答的平均误差应 >= 两步
        assert np.mean(three_step_errors) >= np.mean(two_step_errors), (
            "Three-step problems should be harder for direct answering"
        )
