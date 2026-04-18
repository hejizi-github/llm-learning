"""Tests for knowledge node 02: Minsky-Papert 1969 — XOR linear non-separability."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.perceptron import Perceptron

X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y_XOR = np.array([0, 1, 1, 0])

X_AND = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y_AND = np.array([0, 0, 0, 1])

X_OR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y_OR = np.array([0, 1, 1, 1])


def test_perceptron_fails_xor_never_converges():
    """Single-layer perceptron must reach max_epochs without converging on XOR."""
    p = Perceptron(learning_rate=0.1, max_epochs=200)
    p.fit(X_XOR, Y_XOR)
    # If it converged it would have returned early; history length == max_epochs means no convergence
    assert len(p.history) == 200, "Perceptron should run all 200 epochs without converging on XOR"
    # Errors must remain > 0 throughout (never hit the early-return condition)
    assert all(e > 0 for e in p.history), "Every epoch should have errors > 0 on XOR"


def test_xor_accuracy_below_perfect():
    """Single perceptron cannot achieve 100% accuracy on XOR."""
    p = Perceptron(learning_rate=0.1, max_epochs=500)
    p.fit(X_XOR, Y_XOR)
    acc = p.accuracy(X_XOR, Y_XOR)
    assert acc < 1.0, f"Single perceptron must NOT achieve 100% on XOR, got {acc}"


def test_xor_not_linearly_separable_exhaustive():
    """Exhaustive grid search over weights finds zero valid linear classifiers for XOR."""
    vals = np.linspace(-5, 5, 30)
    W1, W2, B = np.meshgrid(vals, vals, vals)
    W1, W2, B = W1.ravel(), W2.ravel(), B.ravel()

    p00 = ((W1*0 + W2*0 + B) >= 0).astype(int)
    p01 = ((W1*0 + W2*1 + B) >= 0).astype(int)
    p10 = ((W1*1 + W2*0 + B) >= 0).astype(int)
    p11 = ((W1*1 + W2*1 + B) >= 0).astype(int)

    correct = (p00 == 0) & (p01 == 1) & (p10 == 1) & (p11 == 0)
    assert correct.sum() == 0, (
        f"Expected 0 valid linear classifiers for XOR, found {correct.sum()}"
    )


def test_and_is_linearly_separable():
    """Single perceptron converges on AND within 100 epochs (linear separability control)."""
    p = Perceptron(learning_rate=0.1, max_epochs=100)
    p.fit(X_AND, Y_AND)
    assert p.accuracy(X_AND, Y_AND) == 1.0, "AND must be perfectly classified"
    assert 0 in p.history, "AND must converge (at least one epoch with 0 errors)"


def test_or_is_linearly_separable():
    """Single perceptron converges on OR (another control for linear separability)."""
    p = Perceptron(learning_rate=0.1, max_epochs=100)
    p.fit(X_OR, Y_OR)
    assert p.accuracy(X_OR, Y_OR) == 1.0, "OR must be perfectly classified"


def test_two_layer_network_solves_xor():
    """Two-layer network (OR + NAND -> AND) correctly computes XOR."""
    Y_NAND = np.array([1, 1, 1, 0])

    p_or   = Perceptron(learning_rate=0.1, max_epochs=100).fit(X_XOR, Y_OR)
    p_nand = Perceptron(learning_rate=0.1, max_epochs=100).fit(X_XOR, Y_NAND)

    X_layer2 = np.column_stack([p_or.predict(X_XOR), p_nand.predict(X_XOR)])
    # The second layer computes AND of its inputs; labels equal Y_XOR = [0,1,1,0]
    Y_AND2 = np.array([0, 1, 1, 0])
    p_and = Perceptron(learning_rate=0.1, max_epochs=100).fit(X_layer2, Y_AND2)

    final = p_and.predict(X_layer2)
    assert np.array_equal(final, Y_XOR), (
        f"Two-layer network must solve XOR perfectly, got {final}"
    )


def test_algebraic_contradiction_no_solution():
    """Verify the algebraic proof: conditions 1+2+3 always violate condition 4."""
    rng = np.random.default_rng(42)
    w1 = rng.uniform(-10, 10, 500_000)
    w2 = rng.uniform(-10, 10, 500_000)
    b  = rng.uniform(-10, 10, 500_000)

    # Conditions 1, 2, 3 satisfied
    mask = (b < 0) & (w2 + b >= 0) & (w1 + b >= 0)
    # Condition 4: w1 + w2 + b < 0 must NEVER hold simultaneously
    violates_cond4 = mask & (w1 + w2 + b < 0)
    assert violates_cond4.sum() == 0, (
        "Algebraic proof failed: found weights satisfying all 4 XOR conditions"
    )
