"""Tests for Attention mechanism (Node 06 — Bahdanau 2015)."""
import numpy as np
import pytest
import sys
import os

# Allow importing helpers from notebook via exec
NB_GEN = os.path.join(os.path.dirname(__file__), "..", "tools", "gen_nb_06.py")


# ── Inline reference implementations (mirrors notebook code) ─────────────────

def softmax(x):
    x_shifted = x - np.max(x)
    e = np.exp(x_shifted)
    return e / e.sum()


class BahdanauAttention:
    def __init__(self, encoder_size, decoder_size, attn_size, seed=0):
        rng = np.random.default_rng(seed)
        self.W_a = rng.standard_normal((attn_size, decoder_size)) * 0.1
        self.U_a = rng.standard_normal((attn_size, encoder_size)) * 0.1
        self.v = rng.standard_normal(attn_size) * 0.1

    def alignment_scores(self, s_prev, encoder_states):
        T = encoder_states.shape[0]
        dec_part = self.W_a @ s_prev
        scores = np.zeros(T)
        for j in range(T):
            enc_part = self.U_a @ encoder_states[j]
            scores[j] = self.v @ np.tanh(dec_part + enc_part)
        return scores

    def forward(self, s_prev, encoder_states):
        scores = self.alignment_scores(s_prev, encoder_states)
        alpha = softmax(scores)
        context = alpha @ encoder_states
        return context, alpha


# ── Softmax tests ─────────────────────────────────────────────────────────────

def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    assert abs(result.sum() - 1.0) < 1e-9


def test_softmax_non_negative():
    x = np.array([-5.0, 0.0, 3.0, 100.0])
    result = softmax(x)
    assert np.all(result >= 0)


def test_softmax_numerical_stability_large_values():
    x = np.array([1000.0, 1001.0, 999.0])
    result = softmax(x)
    assert np.all(np.isfinite(result)), "softmax should not produce inf/nan for large inputs"
    assert abs(result.sum() - 1.0) < 1e-9


def test_softmax_uniform_input_is_uniform():
    x = np.zeros(5)
    result = softmax(x)
    np.testing.assert_allclose(result, np.full(5, 0.2), atol=1e-9)


def test_softmax_peak_at_max():
    x = np.array([1.0, 5.0, 2.0])
    result = softmax(x)
    assert result.argmax() == 1, "max score should get max weight"


# ── BahdanauAttention shape tests ─────────────────────────────────────────────

def test_alignment_scores_shape():
    attn = BahdanauAttention(encoder_size=8, decoder_size=6, attn_size=10)
    enc = np.random.randn(5, 8)
    s = np.random.randn(6)
    scores = attn.alignment_scores(s, enc)
    assert scores.shape == (5,), f"Expected (5,) got {scores.shape}"


def test_attention_alpha_shape():
    attn = BahdanauAttention(encoder_size=8, decoder_size=6, attn_size=10)
    enc = np.random.randn(7, 8)
    s = np.random.randn(6)
    context, alpha = attn.forward(s, enc)
    assert alpha.shape == (7,), f"Expected alpha shape (7,) got {alpha.shape}"


def test_attention_context_shape():
    attn = BahdanauAttention(encoder_size=8, decoder_size=6, attn_size=10)
    enc = np.random.randn(7, 8)
    s = np.random.randn(6)
    context, _ = attn.forward(s, enc)
    assert context.shape == (8,), f"Expected context shape (8,) got {context.shape}"


def test_attention_alpha_is_probability():
    attn = BahdanauAttention(encoder_size=8, decoder_size=6, attn_size=10)
    enc = np.random.randn(5, 8)
    s = np.random.randn(6)
    _, alpha = attn.forward(s, enc)
    assert abs(alpha.sum() - 1.0) < 1e-9, "attention weights must sum to 1"
    assert np.all(alpha >= 0), "attention weights must be non-negative"


# ── Context vector mathematical properties ────────────────────────────────────

def test_onehot_alpha_gives_encoder_state():
    """One-hot alpha => context == the selected encoder state."""
    enc = np.random.randn(5, 8)
    alpha = np.zeros(5)
    alpha[3] = 1.0
    context = alpha @ enc
    np.testing.assert_allclose(context, enc[3], atol=1e-9)


def test_uniform_alpha_gives_mean():
    """Uniform alpha => context == mean of encoder states."""
    enc = np.random.randn(5, 8)
    alpha = np.ones(5) / 5
    context = alpha @ enc
    np.testing.assert_allclose(context, enc.mean(axis=0), atol=1e-9)


def test_context_is_linear_combination():
    """Context vector must lie within convex hull of encoder states."""
    enc = np.eye(4)  # unit vectors
    attn = BahdanauAttention(encoder_size=4, decoder_size=4, attn_size=8)
    s = np.random.randn(4)
    context, alpha = attn.forward(s, enc)
    # context = sum_j alpha_j * e_j; each component should be alpha_j
    np.testing.assert_allclose(context, alpha, atol=1e-9)


# ── Different sequence lengths ────────────────────────────────────────────────

@pytest.mark.parametrize("T", [1, 3, 10, 20])
def test_attention_works_for_various_seq_lengths(T):
    attn = BahdanauAttention(encoder_size=6, decoder_size=4, attn_size=8)
    enc = np.random.randn(T, 6)
    s = np.random.randn(4)
    context, alpha = attn.forward(s, enc)
    assert context.shape == (6,)
    assert alpha.shape == (T,)
    assert abs(alpha.sum() - 1.0) < 1e-9
