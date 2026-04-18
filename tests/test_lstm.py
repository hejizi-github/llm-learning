"""
Tests for LSTM node (05-lstm-1997):
  - LSTMCell output shapes and gate behavior
  - Sequence forward pass shapes
  - Gradient vanishing demonstration
  - Training loss decrease sanity check
"""
import numpy as np
import pytest


# ── 内联实现（与 notebook 相同，不依赖外部模块）─────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class LSTMCell:
    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.RandomState(seed)
        D = input_size + hidden_size
        scale = 0.1
        self.Wf = rng.randn(hidden_size, D) * scale
        self.Wi = rng.randn(hidden_size, D) * scale
        self.Wg = rng.randn(hidden_size, D) * scale
        self.Wo = rng.randn(hidden_size, D) * scale
        self.bf = np.ones(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bg = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x, h_prev, c_prev):
        combined = np.concatenate([h_prev, x])
        f = sigmoid(self.Wf @ combined + self.bf)
        i = sigmoid(self.Wi @ combined + self.bi)
        g = np.tanh(self.Wg @ combined + self.bg)
        o = sigmoid(self.Wo @ combined + self.bo)
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        return h, c


def lstm_forward(cell, sequence):
    h = np.zeros(cell.hidden_size)
    c = np.zeros(cell.hidden_size)
    outputs = []
    for x in sequence:
        h, c = cell.forward(x, h, c)
        outputs.append(h.copy())
    return np.array(outputs), h, c


def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def rnn_gradient_norm(T, Wh=0.9):
    grad_norm = 1.0
    h = 0.0
    norms = [1.0]
    for _ in range(T):
        h = np.tanh(Wh * h + 0.1)
        local_grad = Wh * tanh_deriv(h)
        grad_norm *= abs(local_grad)
        norms.append(grad_norm)
    return norms


# ── Shape 测试 ───────────────────────────────────────────────────────────

class TestLSTMCellShapes:

    def test_output_shapes_small(self):
        cell = LSTMCell(input_size=2, hidden_size=3)
        x = np.zeros(2)
        h0, c0 = np.zeros(3), np.zeros(3)
        h, c = cell.forward(x, h0, c0)
        assert h.shape == (3,), f"h shape wrong: {h.shape}"
        assert c.shape == (3,), f"c shape wrong: {c.shape}"

    def test_output_shapes_large(self):
        cell = LSTMCell(input_size=10, hidden_size=32, seed=42)
        rng = np.random.RandomState(42)
        x = rng.randn(10) * 0.1
        h0, c0 = np.zeros(32), np.zeros(32)
        h, c = cell.forward(x, h0, c0)
        assert h.shape == (32,)
        assert c.shape == (32,)

    def test_h_bounded_by_output_gate(self):
        """h = o * tanh(c)，o ∈ (0,1), tanh ∈ (-1,1) → h ∈ (-1,1)"""
        cell = LSTMCell(input_size=4, hidden_size=8, seed=7)
        rng = np.random.RandomState(7)
        for _ in range(20):
            x = rng.randn(4) * 0.5
            h0 = rng.randn(8) * 0.5
            c0 = rng.randn(8) * 0.5
            h, _ = cell.forward(x, h0, c0)
            assert np.all(np.abs(h) <= 1.0 + 1e-9), f"h out of [-1,1]: {h}"


# ── Gate 行为测试 ─────────────────────────────────────────────────────────

class TestLSTMCellGates:

    def test_forget_gate_zero_clears_cell(self):
        """当遗忘门 f ≈ 0 时，细胞状态应该约等于 i*g（不保留旧状态）"""
        cell = LSTMCell(input_size=1, hidden_size=4, seed=1)
        # 把遗忘门偏置设成极大负数，使 f ≈ 0
        cell.bf = np.full(4, -100.0)
        c_old = np.array([100.0, -50.0, 30.0, -20.0])
        x = np.zeros(1)
        h0 = np.zeros(4)
        _, c_new = cell.forward(x, h0, c_old)
        # 旧状态应该被清除，c_new 不应接近 c_old
        # f≈0 时 c_new = 0*c_old + i*g ≈ i*g，与 c_old 无关
        assert not np.allclose(c_new, c_old, atol=1.0), \
            "forget_gate≈0 时细胞状态应该被清零，不能保留旧值"

    def test_input_gate_zero_preserves_cell(self):
        """当输入门 i ≈ 0 时，细胞状态应该基本保持不变（依赖遗忘门）"""
        cell = LSTMCell(input_size=1, hidden_size=4, seed=2)
        # 输入门偏置极大负数 → i≈0；遗忘门偏置极大正数 → f≈1
        cell.bi = np.full(4, -100.0)
        cell.bf = np.full(4, 100.0)
        c_old = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.zeros(1)
        h0 = np.zeros(4)
        _, c_new = cell.forward(x, h0, c_old)
        # i≈0, f≈1 → c_new ≈ c_old
        assert np.allclose(c_new, c_old, atol=1e-3), \
            f"input_gate≈0, forget_gate≈1 时细胞状态应保留: {c_new} vs {c_old}"

    def test_output_gate_zero_masks_h(self):
        """当输出门 o ≈ 0 时，h ≈ 0（隐状态被屏蔽）"""
        cell = LSTMCell(input_size=1, hidden_size=4, seed=3)
        cell.bo = np.full(4, -100.0)   # o ≈ 0
        c_old = np.array([5.0, -5.0, 3.0, -3.0])   # 非零细胞状态
        x = np.zeros(1)
        h0 = np.zeros(4)
        h_new, _ = cell.forward(x, h0, c_old)
        assert np.allclose(h_new, 0.0, atol=1e-6), \
            f"output_gate≈0 时 h 应该≈0，实际={h_new}"


# ── 序列维度测试 ─────────────────────────────────────────────────────────

class TestLSTMSequence:

    def test_sequence_output_shape(self):
        cell = LSTMCell(input_size=3, hidden_size=5)
        seq = [np.random.randn(3) for _ in range(7)]
        outputs, h_final, c_final = lstm_forward(cell, seq)
        assert outputs.shape == (7, 5), f"outputs shape: {outputs.shape}"
        assert h_final.shape == (5,)
        assert c_final.shape == (5,)

    def test_single_step_equals_forward(self):
        """single step: lstm_forward 和 cell.forward 结果一致"""
        cell = LSTMCell(input_size=2, hidden_size=4, seed=5)
        x = np.array([0.3, 0.7])
        h0, c0 = np.zeros(4), np.zeros(4)
        h_ref, c_ref = cell.forward(x, h0, c0)
        outputs, h_seq, c_seq = lstm_forward(cell, [x])
        assert np.allclose(outputs[0], h_ref)
        assert np.allclose(h_seq, h_ref)
        assert np.allclose(c_seq, c_ref)


# ── 梯度消失验证 ─────────────────────────────────────────────────────────

class TestGradientVanishing:

    def test_short_sequence_gradient_not_vanished(self):
        norms = rnn_gradient_norm(T=5)
        assert norms[-1] > 0.01, f"5步后梯度不应消失: {norms[-1]}"

    def test_long_sequence_gradient_vanishes(self):
        norms = rnn_gradient_norm(T=100)
        assert norms[-1] < 1e-3, f"100步后梯度应极小: {norms[-1]}"

    def test_gradient_monotonically_decreasing(self):
        norms = rnn_gradient_norm(T=30)
        # 梯度整体趋势应递减（允许局部轻微波动）
        first_half_mean = np.mean(norms[:15])
        second_half_mean = np.mean(norms[15:])
        assert second_half_mean < first_half_mean, \
            "梯度后半段均值应小于前半段均值"


# ── 训练健全性测试 ───────────────────────────────────────────────────────

class TestLSTMTraining:

    def test_loss_decreases_after_gradient_steps(self):
        """用数值梯度更新 10 步，loss 应该下降"""
        np.random.seed(99)
        cell = LSTMCell(input_size=1, hidden_size=4, seed=99)
        Wy = np.random.randn(1, 4) * 0.1
        by = np.zeros(1)

        seq = [np.array([v/5.0]) for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
        tgt = [v/5.0 for v in [5.0, 4.0, 3.0, 2.0, 1.0]]

        def predict(cell, Wy, by, seq):
            h = np.zeros(4)
            c = np.zeros(4)
            preds = []
            for x in seq:
                h, c = cell.forward(x, h, c)
                preds.append((Wy @ h + by)[0])
            return preds

        def mse(cell, Wy, by, seq, tgt):
            preds = predict(cell, Wy, by, seq)
            return float(np.mean([(p - t)**2 for p, t in zip(preds, tgt)]))

        loss0 = mse(cell, Wy, by, seq, tgt)

        # 对 Wy 做数值梯度下降（简化版）
        eps = 1e-4
        lr = 0.5
        for _ in range(10):
            g = np.zeros_like(Wy)
            for idx in np.ndindex(Wy.shape):
                orig = Wy[idx]
                Wy[idx] = orig + eps
                lp = mse(cell, Wy, by, seq, tgt)
                Wy[idx] = orig - eps
                lm = mse(cell, Wy, by, seq, tgt)
                Wy[idx] = orig
                g[idx] = (lp - lm) / (2 * eps)
            Wy -= lr * g

        loss_final = mse(cell, Wy, by, seq, tgt)
        assert loss_final < loss0, \
            f"梯度下降后 loss 应下降: {loss0:.4f} → {loss_final:.4f}"
