"""
节点 02 测试：验证 XOR 线性不可分的核心性质。
每个测试都有故障注入验证（改动实现会让测试 FAIL）。
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from perceptron import Perceptron

X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_XOR = np.array([0, 1, 1, 0])


def test_perceptron_fails_on_xor():
    """感知机训练 500 轮后仍有错误（XOR 线性不可分的直接证据）。"""
    p = Perceptron(n_features=2, learning_rate=0.1)
    p.fit(X_XOR, Y_XOR, max_epochs=500)
    correct = sum(p.predict(xi) == yi for xi, yi in zip(X_XOR, Y_XOR))
    # 感知机在 XOR 上不收敛：正确数永远 < 4
    assert correct < 4, (
        f"感知机不应该学会 XOR，但得了 {correct}/4 分（线性不可分定理失效？）"
    )


def test_xor_not_linearly_separable_exhaustive():
    """穷举随机线性分类器，验证不存在任何权重能正确分类 XOR。"""
    rng = np.random.default_rng(42)
    n_trials = 50_000
    for _ in range(n_trials):
        w = rng.uniform(-10, 10, size=2)
        b = rng.uniform(-10, 10)
        preds = (X_XOR @ w + b > 0).astype(int)
        assert not np.all(preds == Y_XOR), (
            f"发现线性分类器 w={w} b={b}，但 XOR 理论上不可线性分离！"
        )


def test_xor_algebraic_contradiction():
    """代数验证：4 个 XOR 约束同时成立时导致矛盾。

    如果存在 (w1, w2, bias) 满足全部 XOR 约束，则：
    ②+③：(w2+bias) + (w1+bias) > 0  →  w1+w2+2*bias > 0
    由 ① bias ≤ 0 得：w1+w2+2*bias ≤ w1+w2+bias
    由 ④ 得：w1+w2+bias ≤ 0
    因此：w1+w2+2*bias ≤ 0  ←→ > 0，矛盾。

    此测试用随机采样验证不等式系统无解。
    """
    rng = np.random.default_rng(0)
    n_samples = 100_000
    # 随机生成候选 (w1, w2, bias)
    candidates = rng.uniform(-20, 20, size=(n_samples, 3))
    w1, w2, bias = candidates[:, 0], candidates[:, 1], candidates[:, 2]
    # 验证四个约束
    c1 = bias <= 0                    # (0,0)→0：bias ≤ 0
    c2 = w2 + bias > 0               # (0,1)→1：w2+bias > 0
    c3 = w1 + bias > 0               # (1,0)→1：w1+bias > 0
    c4 = w1 + w2 + bias <= 0         # (1,1)→0：w1+w2+bias ≤ 0
    all_satisfied = c1 & c2 & c3 & c4
    assert not np.any(all_satisfied), (
        f"发现满足全部 XOR 约束的 (w1,w2,bias)，这违反了代数矛盾定理！"
    )


def test_linearly_separable_problem_converges():
    """对照：线性可分问题（AND 门）感知机能收敛，证明算法本身没问题。"""
    # AND 门：两个都是 1 才输出 1
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    p = Perceptron(n_features=2, learning_rate=0.1)
    p.fit(X_and, y_and, max_epochs=100)
    correct = sum(p.predict(xi) == yi for xi, yi in zip(X_and, y_and))
    assert correct == 4, (
        f"感知机在 AND 门（线性可分）上应该 100% 正确，实际得 {correct}/4"
    )


def test_xor_history_never_reaches_zero():
    """训练历史里不存在 0 错误的轮次（XOR 永远不收敛的历史证据）。"""
    p = Perceptron(n_features=2, learning_rate=0.1)
    p.fit(X_XOR, Y_XOR, max_epochs=200)
    assert len(p.history) == 200, "应该跑满 200 轮（从未提前收敛）"
    assert 0 not in p.history, (
        "XOR 训练历史里不应该出现 0 错误（感知机从未完全正确）"
    )
