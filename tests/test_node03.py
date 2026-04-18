"""
节点 03 测试：反向传播正确性验证
- sigmoid 数值精度
- 前向传播形状和边界值
- 梯度数值验证（解析梯度 vs 有限差分，精确匹配 MSE 梯度）
- 网络能够学会 XOR
"""
import numpy as np
import pytest


# ─────────────── 被测实现（内联，与 notebook 保持一致） ───────────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(a):
    return a * (1.0 - a)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return a1, a2

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def backward(X, y, a1, a2, W2):
    """精确计算 MSE 损失对各权重的梯度（含 2/n 因子）。"""
    n = len(X)
    delta2 = (2.0 / n) * (a2 - y) * sigmoid_deriv(a2)
    dW2 = a1.T @ delta2
    db2 = delta2.sum(axis=0, keepdims=True)
    delta1 = (delta2 @ W2.T) * sigmoid_deriv(a1)
    dW1 = X.T @ delta1
    db1 = delta1.sum(axis=0, keepdims=True)
    return dW1, db1, dW2, db2


# ─────────────────────── fixtures ───────────────────────

@pytest.fixture
def xor_data():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)
    return X, y

@pytest.fixture
def init_params():
    np.random.seed(0)
    W1 = np.random.randn(2, 2) * 0.5
    b1 = np.zeros((1, 2))
    W2 = np.random.randn(2, 1) * 0.5
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2


# ─────────────────────── sigmoid 测试 ───────────────────────

def test_sigmoid_zero():
    """σ(0) 精确等于 0.5"""
    assert sigmoid(0.0) == pytest.approx(0.5)

def test_sigmoid_symmetry():
    """σ(-x) = 1 - σ(x)"""
    for z in [1.0, 2.5, 10.0]:
        assert sigmoid(-z) == pytest.approx(1.0 - sigmoid(z))

def test_sigmoid_range():
    """输出必须在 [0, 1] 之间且为有限值（clip 防止溢出）"""
    z = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    a = sigmoid(z)
    assert np.all(np.isfinite(a))
    assert np.all(a >= 0.0) and np.all(a <= 1.0)

def test_sigmoid_deriv_at_half():
    """σ'(0.5) = 0.5 × (1 - 0.5) = 0.25"""
    assert sigmoid_deriv(0.5) == pytest.approx(0.25)

def test_sigmoid_deriv_max():
    """sigmoid 导数在 a=0.5 时最大（不超过 0.25）"""
    a_vals = np.linspace(0.01, 0.99, 200)
    derivs = sigmoid_deriv(a_vals)
    assert np.max(derivs) == pytest.approx(0.25, abs=1e-4)


# ─────────────────────── 前向传播测试 ───────────────────────

def test_forward_output_shape(xor_data, init_params):
    """前向传播输出形状：(4, 1)"""
    X, _ = xor_data
    W1, b1, W2, b2 = init_params
    a1, a2 = forward(X, W1, b1, W2, b2)
    assert a1.shape == (4, 2)
    assert a2.shape == (4, 1)

def test_forward_output_in_range(xor_data, init_params):
    """前向传播输出必须在 [0, 1] 之间"""
    X, _ = xor_data
    W1, b1, W2, b2 = init_params
    _, a2 = forward(X, W1, b1, W2, b2)
    assert np.all(a2 >= 0) and np.all(a2 <= 1)


# ─────────────────────── 梯度验证（数值 vs 解析） ───────────────────────

def numerical_gradient_W1(X, y, W1, b1, W2, b2, eps=1e-5):
    """用有限差分法计算 dMSE/dW1（验证解析梯度用）。"""
    grad = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_plus = W1.copy(); W1_plus[i, j] += eps
            _, a2_plus = forward(X, W1_plus, b1, W2, b2)

            W1_minus = W1.copy(); W1_minus[i, j] -= eps
            _, a2_minus = forward(X, W1_minus, b1, W2, b2)

            grad[i, j] = (mse_loss(a2_plus, y) - mse_loss(a2_minus, y)) / (2 * eps)
    return grad

def test_gradient_W1_numerical(xor_data, init_params):
    """解析梯度 dW1 与有限差分梯度一致（相对误差 < 1e-4）"""
    X, y = xor_data
    W1, b1, W2, b2 = init_params

    a1, a2 = forward(X, W1, b1, W2, b2)
    dW1_analytic, _, _, _ = backward(X, y, a1, a2, W2)
    dW1_numeric = numerical_gradient_W1(X, y, W1, b1, W2, b2)

    diff = np.abs(dW1_analytic - dW1_numeric)
    scale = np.abs(dW1_analytic) + np.abs(dW1_numeric) + 1e-8
    rel_err = np.max(diff / scale)
    assert rel_err < 1e-4, f"Gradient check failed: rel_err={rel_err:.2e}"


# ─────────────────────── 学会 XOR 的集成测试 ───────────────────────

def test_learns_xor():
    """训练 3000 轮后，网络必须能 100% 正确分类 XOR 的 4 个样本。"""
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)

    np.random.seed(0)
    W1 = np.random.randn(2, 2) * 0.5
    b1 = np.zeros((1, 2))
    W2 = np.random.randn(2, 1) * 0.5
    b2 = np.zeros((1, 1))

    lr = 1.0
    for _ in range(3000):
        a1, a2 = forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X, y, a1, a2, W2)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    _, y_pred = forward(X, W1, b1, W2, b2)
    preds = (y_pred > 0.5).astype(int)
    assert np.all(preds == y.astype(int)), \
        f"Network failed to learn XOR. Predictions: {preds.T}, True: {y.T}"

def test_loss_decreases(xor_data, init_params):
    """训练 1000 轮后，损失应比初始损失减少 80%。"""
    X, y = xor_data
    W1, b1, W2, b2 = init_params

    _, a2 = forward(X, W1, b1, W2, b2)
    initial_loss = mse_loss(a2, y)

    lr = 1.0
    for _ in range(1000):
        a1, a2 = forward(X, W1, b1, W2, b2)
        dW1, db1_g, dW2, db2_g = backward(X, y, a1, a2, W2)
        W1 -= lr * dW1
        b1 -= lr * db1_g
        W2 -= lr * dW2
        b2 -= lr * db2_g

    _, a2 = forward(X, W1, b1, W2, b2)
    final_loss = mse_loss(a2, y)
    assert final_loss < initial_loss * 0.2, \
        f"Loss did not decrease enough: {initial_loss:.4f} → {final_loss:.4f}"
