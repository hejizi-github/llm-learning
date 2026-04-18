"""
Unit tests for the Perceptron learning algorithm.
Imports from src/perceptron.py — the single source of truth shared with the notebook.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.perceptron import Perceptron


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
    """XOR is not linearly separable — Perceptron must NOT reach 100% accuracy,
    AND weights must have been updated (ruling out a broken no-op update)."""
    p = Perceptron(learning_rate=0.1, max_epochs=50)
    weights_before = np.zeros(2)
    p.fit(XOR_X, XOR_Y)
    acc = p.accuracy(XOR_X, XOR_Y)
    assert acc < 1.0, "XOR should NOT converge (not linearly separable)"
    # Guard against a broken implementation that never updates weights:
    # at least one epoch must have had errors (the algorithm actually tried)
    assert any(e > 0 for e in p.history), \
        "history should show non-zero errors — weights must have been updated"
    assert not np.array_equal(p.weights, weights_before), \
        "Weights must change during XOR training (algorithm is not a no-op)"


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
