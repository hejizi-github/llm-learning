"""
Tests for node 03 backpropagation math logic.

Inline implementation mirrors gen_nb_03.py / notebooks/03-backprop-1986.ipynb
so tests stay independent of notebook execution environment.
"""
import numpy as np
import pytest


# ── Core functions (mirrored from notebooks/03-backprop-1986.ipynb) ─────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)


def mse_loss(y_hat, y):
    return np.mean((y_hat - y) ** 2)


class TwoLayerNet:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.h  = sigmoid(self.z1)
        self.z2 = self.h @ self.W2 + self.b2
        self.y_hat = sigmoid(self.z2)
        return self.y_hat

    def backward(self, X, y, lr=0.1):
        N = X.shape[0]
        dL_dyhat = self.y_hat - y
        delta2 = dL_dyhat * sigmoid_deriv(self.z2)
        dW2 = self.h.T @ delta2 / N
        db2 = delta2.mean(axis=0, keepdims=True)
        delta1 = (delta2 @ self.W2.T) * sigmoid_deriv(self.z1)
        dW1 = X.T @ delta1 / N
        db1 = delta1.mean(axis=0, keepdims=True)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


# ── XOR dataset ─────────────────────────────────────────────────────────────

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([[0], [1], [1], [0]], dtype=float)


# ── Tests: sigmoid ───────────────────────────────────────────────────────────

def test_sigmoid_range():
    """sigmoid 输出在 [0, 1] 之间；在非极端 z 值下严格在 (0, 1) 之间

    注：z > ~36 时 float64 饱和至 1.0，这是浮点精度而非数学错误。
    """
    z = np.linspace(-10, 10, 1000)
    out = sigmoid(z)
    assert np.all(out > 0), "sigmoid 输出不应有 ≤ 0 的值"
    assert np.all(out < 1), "sigmoid 输出不应有 ≥ 1 的值"


def test_sigmoid_midpoint():
    """sigmoid(0) 应恰好等于 0.5"""
    assert sigmoid(0) == pytest.approx(0.5)


def test_sigmoid_deriv_max_is_025():
    """sigmoid 导数最大值恰好是 0.25（在 z=0 处取到）

    这是梯度消失问题的数学根源：每经过一层，梯度至多乘以 0.25。
    """
    z = np.linspace(-10, 10, 10000)
    max_deriv = sigmoid_deriv(z).max()
    assert max_deriv == pytest.approx(0.25, abs=1e-4)


def test_sigmoid_deriv_symmetry():
    """sigmoid 导数关于 z=0 对称"""
    z_pos = np.array([0.5, 1.0, 2.0, 3.0])
    z_neg = -z_pos
    np.testing.assert_allclose(sigmoid_deriv(z_pos), sigmoid_deriv(z_neg), rtol=1e-10)


# ── Tests: MSE loss ──────────────────────────────────────────────────────────

def test_mse_loss_zero():
    """预测等于真值时 MSE 应为 0"""
    y = np.array([[0.0], [1.0], [0.5]])
    assert mse_loss(y, y) == pytest.approx(0.0)


def test_mse_loss_positive():
    """预测错误时 MSE 应严格大于 0"""
    y_hat = np.array([[0.9], [0.1], [0.8], [0.9]])
    assert mse_loss(y_hat, XOR_Y) > 0


# ── Tests: forward pass ──────────────────────────────────────────────────────

def test_forward_output_shape():
    """前向传播输出维度应为 (N, 1)"""
    net = TwoLayerNet()
    y_hat = net.forward(XOR_X)
    assert y_hat.shape == (4, 1), f"期望 (4,1)，实际 {y_hat.shape}"


def test_forward_output_in_range():
    """前向传播输出应在 (0, 1) 之间（因为最后一层是 sigmoid）"""
    net = TwoLayerNet()
    y_hat = net.forward(XOR_X)
    assert np.all(y_hat > 0) and np.all(y_hat < 1)


def test_forward_hidden_shape():
    """隐藏层输出维度应为 (N, hidden_size)"""
    net = TwoLayerNet(hidden_size=4)
    net.forward(XOR_X)
    assert net.h.shape == (4, 4)


# ── Tests: backward pass (gradient direction) ────────────────────────────────

def test_backward_reduces_loss():
    """单步反向传播后，损失应下降（验证梯度方向正确）"""
    net = TwoLayerNet(seed=0)
    y_hat_before = net.forward(XOR_X)
    loss_before = mse_loss(y_hat_before, XOR_Y)
    net.backward(XOR_X, XOR_Y, lr=0.5)
    y_hat_after = net.forward(XOR_X)
    loss_after = mse_loss(y_hat_after, XOR_Y)
    assert loss_after < loss_before, (
        f"反向传播后 loss 应下降，实际: before={loss_before:.6f}, after={loss_after:.6f}"
    )


# ── Tests: XOR convergence ───────────────────────────────────────────────────

def test_xor_convergence():
    """10000 轮训练后，网络应能正确分类所有 4 个 XOR 样本（误差 < 0.1）"""
    np.random.seed(42)
    net = TwoLayerNet(seed=42)

    for _ in range(10000):
        net.forward(XOR_X)
        net.backward(XOR_X, XOR_Y, lr=0.5)

    y_hat = net.forward(XOR_X)
    errors = np.abs(y_hat - XOR_Y)
    assert errors.max() < 0.1, (
        f"XOR 收敛失败，最大误差: {errors.max():.4f}\n预测: {y_hat.T}\n真值: {XOR_Y.T}"
    )


def test_xor_convergence_loss_below_threshold():
    """10000 轮训练后，MSE 损失应低于 0.01"""
    np.random.seed(42)
    net = TwoLayerNet(seed=42)

    for _ in range(10000):
        net.forward(XOR_X)
        net.backward(XOR_X, XOR_Y, lr=0.5)

    y_hat = net.forward(XOR_X)
    loss = mse_loss(y_hat, XOR_Y)
    assert loss < 0.01, f"XOR 收敛损失过高: {loss:.6f}"
