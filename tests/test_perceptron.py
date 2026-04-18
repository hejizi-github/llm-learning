"""
Unit tests for the Perceptron learning algorithm.
Perceptron class is replicated here from notebooks/01-perceptron-1958.ipynb
to keep tests self-contained and runnable without Jupyter.
"""
import numpy as np
import pytest


class Perceptron:
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.history = []

    def _step(self, z):
        return (z >= 0).astype(int)

    def predict(self, X):
        X = np.atleast_2d(X)
        z = X @ self.weights + self.bias
        return self._step(z)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = []
        for epoch in range(self.max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.predict(xi)[0]
                error = int(yi) - int(pred)
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1
            self.history.append(errors)
            if errors == 0:
                return self
        return self

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


# --- fixtures ---

AND_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
AND_Y = np.array([0, 0, 0, 1])

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([0, 1, 1, 0])


# --- test cases ---

def test_and_converges():
    """AND gate is linearly separable — Perceptron must reach 100% accuracy."""
    p = Perceptron(learning_rate=0.1, max_epochs=100)
    p.fit(AND_X, AND_Y)
    assert p.accuracy(AND_X, AND_Y) == 1.0, "AND gate should converge to 100% accuracy"


def test_and_history_ends_at_zero():
    """After convergence the last history entry must be 0 errors."""
    p = Perceptron(learning_rate=0.1, max_epochs=100)
    p.fit(AND_X, AND_Y)
    assert p.history[-1] == 0, "Last epoch error count should be 0 after convergence"


def test_xor_does_not_converge():
    """XOR is not linearly separable — Perceptron must NOT reach 100% accuracy."""
    p = Perceptron(learning_rate=0.1, max_epochs=50)
    p.fit(XOR_X, XOR_Y)
    acc = p.accuracy(XOR_X, XOR_Y)
    assert acc < 1.0, "XOR should NOT converge (not linearly separable)"


def test_predict_output_shape():
    """predict() must return an array with the same length as the input."""
    p = Perceptron()
    p.fit(AND_X, AND_Y)
    out = p.predict(AND_X)
    assert out.shape == AND_Y.shape, f"Expected shape {AND_Y.shape}, got {out.shape}"


def test_weights_change_after_fit():
    """Weights should not all remain zero after training on non-trivial data."""
    p = Perceptron()
    p.fit(AND_X, AND_Y)
    assert not np.all(p.weights == 0), "Weights should be updated during training"
