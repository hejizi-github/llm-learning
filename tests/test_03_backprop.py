"""
Tests for node 03: Backpropagation (1986)
Validates the core backprop implementation logic independently.
"""
import numpy as np
import pytest

np.random.seed(42)

# ── helpers (same as notebook) ──────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(s):
    return s * (1.0 - s)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    h  = sigmoid(z1)
    z2 = h @ W2 + b2
    y_hat = sigmoid(z2)
    return z1, h, z2, y_hat

def backward(X, y, W2, z1, h, y_hat):
    n = X.shape[0]
    dL_dy_hat = -(y - y_hat)
    dL_dz2    = dL_dy_hat * sigmoid_deriv(y_hat)
    dL_dW2    = h.T @ dL_dz2 / n
    dL_db2    = dL_dz2.mean(axis=0, keepdims=True)
    dL_dh     = dL_dz2 @ W2.T
    dL_dz1    = dL_dh * sigmoid_deriv(h)
    dL_dW1    = X.T @ dL_dz1 / n
    dL_db1    = dL_dz1.mean(axis=0, keepdims=True)
    return dL_dW1, dL_db1, dL_dW2, dL_db2

def train_xor(epochs=10000, lr=0.5, hidden=4, seed=42):
    np.random.seed(seed)
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)
    W1 = np.random.randn(2, hidden) * 0.5
    b1 = np.zeros((1, hidden))
    W2 = np.random.randn(hidden, 1) * 0.5
    b2 = np.zeros((1, 1))
    losses = []
    for _ in range(epochs):
        z1, h, z2, y_hat = forward(X, W1, b1, W2, b2)
        losses.append(np.mean((y - y_hat)**2))
        dW1, db1, dW2, db2 = backward(X, y, W2, z1, h, y_hat)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
    _, _, _, final = forward(X, W1, b1, W2, b2)
    acc = sum(int(final[i,0] > 0.5) == int(y[i,0]) for i in range(4)) / 4
    return acc, losses, final

# ── tests ───────────────────────────────────────────────────────────────────

def test_sigmoid_boundary_values():
    """sigmoid(0) = 0.5, edges approach 0 and 1."""
    assert abs(sigmoid(0) - 0.5) < 1e-9
    assert sigmoid(100) > 0.999
    assert sigmoid(-100) < 0.001

def test_sigmoid_deriv_formula():
    """sigmoid'(x) = s(1-s) holds for multiple inputs."""
    for x in [-2.0, 0.0, 1.5, 3.0]:
        s = sigmoid(x)
        expected = s * (1 - s)
        assert abs(sigmoid_deriv(s) - expected) < 1e-12

def test_forward_output_range():
    """All forward-pass outputs must be in (0, 1) (sigmoid guarantee)."""
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    W1 = np.random.randn(2, 4) * 0.5
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 0.5
    b2 = np.zeros((1, 1))
    _, _, _, y_hat = forward(X, W1, b1, W2, b2)
    assert np.all(y_hat > 0) and np.all(y_hat < 1)

def test_loss_decreases_over_training():
    """Loss at epoch 5000 must be less than loss at epoch 0."""
    _, losses, _ = train_xor(epochs=5000)
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

def test_xor_accuracy_perfect():
    """Two-layer backprop network achieves 100% accuracy on XOR."""
    acc, _, _ = train_xor(epochs=10000)
    assert acc == 1.0, f"Expected 100% accuracy, got {acc:.2%}"

def test_single_layer_cannot_solve_xor():
    """Single linear layer (no hidden layer) cannot perfectly solve XOR,
    confirming the Minsky-Papert result from node 02."""
    np.random.seed(0)
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)
    W = np.random.randn(2, 1) * 0.5
    b = np.zeros((1, 1))
    for _ in range(10000):
        p = sigmoid(X @ W + b)
        dW = -(y - p) * sigmoid_deriv(p)
        W -= 0.1 * (X.T @ dW / 4)
        b -= 0.1 * dW.mean()
    final = sigmoid(X @ W + b)
    acc = sum(int(final[i,0] > 0.5) == int(y[i,0]) for i in range(4)) / 4
    assert acc < 1.0, "Single layer solved XOR — unexpected!"

def test_gradient_shapes_match_weights():
    """Gradient tensors must have same shape as the weight tensors."""
    np.random.seed(1)
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)
    W1 = np.random.randn(2, 4) * 0.5
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 0.5
    b2 = np.zeros((1, 1))
    z1, h, _, y_hat = forward(X, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward(X, y, W2, z1, h, y_hat)
    assert dW1.shape == W1.shape
    assert dW2.shape == W2.shape
    assert db1.shape == b1.shape
    assert db2.shape == b2.shape
