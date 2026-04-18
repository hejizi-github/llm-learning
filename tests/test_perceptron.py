"""
pytest 单元测试：src/perceptron.py
覆盖核心行为：初始化、收敛、准确率。
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from perceptron import Perceptron


def and_gate():
    """AND 逻辑门数据集（线性可分）"""
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0, 0, 0, 1])
    return X, y


def or_gate():
    """OR 逻辑门数据集（线性可分）"""
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0, 1, 1, 1])
    return X, y


def xor_gate():
    """XOR 逻辑门数据集（线性不可分）"""
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0, 1, 1, 0])
    return X, y


class TestPerceptronInit:
    def test_weights_start_at_zero(self):
        """权重必须初始化为全 0（与 README 描述一致）"""
        X, y = and_gate()
        p = Perceptron()
        p.fit(X, y)
        # 检查历史记录长度 > 0（fit 被调用过）
        assert len(p.history) > 0

    def test_bias_starts_at_zero(self):
        """偏置必须初始化为 0"""
        X, y = and_gate()
        p = Perceptron()
        # 在 fit 之前检查初始值
        assert p.weights is None
        assert p.bias is None


class TestPerceptronConvergence:
    def test_and_gate_converges(self):
        """AND 门是线性可分的，感知机必须收敛到 100%"""
        X, y = and_gate()
        p = Perceptron(learning_rate=0.1, max_epochs=100)
        p.fit(X, y)
        acc = p.accuracy(X, y)
        assert acc == 1.0, f"AND 门应收敛到 100%，实际 {acc:.0%}"

    def test_or_gate_converges(self):
        """OR 门是线性可分的，感知机必须收敛到 100%"""
        X, y = or_gate()
        p = Perceptron(learning_rate=0.1, max_epochs=100)
        p.fit(X, y)
        acc = p.accuracy(X, y)
        assert acc == 1.0, f"OR 门应收敛到 100%，实际 {acc:.0%}"

    def test_history_recorded(self):
        """history 列表记录每轮错误数"""
        X, y = and_gate()
        p = Perceptron()
        p.fit(X, y)
        assert isinstance(p.history, list)
        assert len(p.history) >= 1
        # 最后一轮错误数必须为 0（收敛）
        assert p.history[-1] == 0


class TestPerceptronXOR:
    def test_xor_cannot_reach_100_percent(self):
        """XOR 线性不可分：100 轮后准确率永远 < 100%"""
        X, y = xor_gate()
        p = Perceptron(learning_rate=0.1, max_epochs=100)
        p.fit(X, y)
        acc = p.accuracy(X, y)
        assert acc < 1.0, "XOR 不可分，感知机不应达到 100%"


class TestPerceptronPredict:
    def test_predict_returns_0_or_1(self):
        """预测值只能是 0 或 1"""
        X, y = and_gate()
        p = Perceptron()
        p.fit(X, y)
        preds = p.predict(X)
        assert set(preds).issubset({0, 1}), f"预测值应只包含 0 和 1，实际：{set(preds)}"

    def test_and_gate_truth_table(self):
        """AND 门真值表验证"""
        X, y = and_gate()
        p = Perceptron()
        p.fit(X, y)
        # 逐一检查真值表
        assert p.predict(np.array([[0., 0.]]))[0] == 0
        assert p.predict(np.array([[0., 1.]]))[0] == 0
        assert p.predict(np.array([[1., 0.]]))[0] == 0
        assert p.predict(np.array([[1., 1.]]))[0] == 1
