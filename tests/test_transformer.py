"""Tests for Transformer (2017) node — Scaled Dot-Product Attention + Positional Encoding + Encoder Block."""
import numpy as np
import pytest


# ── Helpers (duplicated from notebook for test isolation) ──────────────────────

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights


def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))
    positions = np.arange(max_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])
    return PE


def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


# ── Scaled Dot-Product Attention tests ────────────────────────────────────────

class TestScaledDotProductAttention:

    def test_output_shape(self):
        np.random.seed(0)
        Q = np.random.randn(4, 8)
        K = np.random.randn(4, 8)
        V = np.random.randn(4, 8)
        out, weights = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (4, 8)
        assert weights.shape == (4, 4)

    def test_weights_sum_to_one(self):
        np.random.seed(1)
        Q = np.random.randn(5, 16)
        K = np.random.randn(5, 16)
        V = np.random.randn(5, 16)
        _, weights = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones(5), atol=1e-6)

    def test_weights_non_negative(self):
        np.random.seed(2)
        Q = np.random.randn(3, 4)
        K = np.random.randn(3, 4)
        V = np.random.randn(3, 4)
        _, weights = scaled_dot_product_attention(Q, K, V)
        assert np.all(weights >= 0)

    def test_identical_qk_gives_concentrated_weight(self):
        """When Q == K, each position attends most to itself."""
        np.random.seed(3)
        X = np.random.randn(4, 8)
        _, weights = scaled_dot_product_attention(X, X, X)
        # Each diagonal element should be the max in its row
        for i in range(4):
            assert weights[i, i] == pytest.approx(weights[i].max(), abs=1e-6)

    @pytest.mark.parametrize("seq_len,d_k,d_v", [
        (3, 4, 4),
        (6, 16, 8),
        (10, 32, 32),
    ])
    def test_output_shape_parametrized(self, seq_len, d_k, d_v):
        np.random.seed(0)
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_v)
        out, weights = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (seq_len, d_v)
        assert weights.shape == (seq_len, seq_len)

    def test_causal_mask_zeros_upper_triangle(self):
        """With causal mask, upper-triangle weights should be near zero."""
        np.random.seed(4)
        n = 4
        Q = np.random.randn(n, 8)
        K = np.random.randn(n, 8)
        V = np.random.randn(n, 8)
        # Lower-triangular mask (1 = attend, 0 = mask out)
        mask = np.tril(np.ones((n, n)))
        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        upper = weights[np.triu_indices(n, k=1)]
        assert np.all(upper < 1e-6)

    def test_scaling_prevents_extreme_gradients(self):
        """With large d_k, unscaled attention collapses; scaled version should not."""
        np.random.seed(5)
        d_k = 512
        Q = np.random.randn(3, d_k)
        K = np.random.randn(3, d_k)
        V = np.random.randn(3, d_k)
        _, weights = scaled_dot_product_attention(Q, K, V)
        # Weights should not be one-hot (which would indicate softmax saturation)
        max_weights = weights.max(axis=-1)
        assert np.all(max_weights < 0.99), "Scaled attention should avoid saturation"


# ── Positional Encoding tests ──────────────────────────────────────────────────

class TestPositionalEncoding:

    def test_output_shape(self):
        PE = positional_encoding(50, 64)
        assert PE.shape == (50, 64)

    def test_values_bounded(self):
        PE = positional_encoding(100, 128)
        assert PE.min() >= -1.0 - 1e-6
        assert PE.max() <= 1.0 + 1e-6

    def test_each_position_unique(self):
        PE = positional_encoding(20, 32)
        for i in range(20):
            for j in range(20):
                if i != j:
                    assert not np.allclose(PE[i], PE[j])

    def test_sine_cosine_alternation(self):
        """Even dims should be sin, odd dims should be cos at position 0."""
        PE = positional_encoding(10, 8)
        # At position 0: sin(0) = 0 for all even dims
        np.testing.assert_allclose(PE[0, 0::2], 0.0, atol=1e-6)
        # At position 0: cos(0) = 1 for all odd dims
        np.testing.assert_allclose(PE[0, 1::2], 1.0, atol=1e-6)

    @pytest.mark.parametrize("max_len,d_model", [(10, 16), (50, 32), (100, 64)])
    def test_parametrized_shapes(self, max_len, d_model):
        PE = positional_encoding(max_len, d_model)
        assert PE.shape == (max_len, d_model)


# ── Layer Norm tests ───────────────────────────────────────────────────────────

class TestLayerNorm:

    def test_output_mean_near_zero(self):
        np.random.seed(0)
        x = np.random.randn(5, 16) * 10 + 5
        normed = layer_norm(x)
        np.testing.assert_allclose(normed.mean(axis=-1), np.zeros(5), atol=1e-5)

    def test_output_std_near_one(self):
        np.random.seed(1)
        x = np.random.randn(5, 16) * 10
        normed = layer_norm(x)
        np.testing.assert_allclose(normed.std(axis=-1), np.ones(5), atol=1e-5)


# ── Transformer Encoder Block tests ───────────────────────────────────────────

class TestTransformerEncoderBlock:

    def _make_block(self, d_model=16, num_heads=4, d_ff=32):
        """Minimal encoder block using only NumPy."""

        class FFN_:
            def __init__(self, d_model, d_ff, seed=0):
                rng = np.random.default_rng(seed)
                self.W1 = rng.standard_normal((d_model, d_ff)) * 0.1
                self.b1 = np.zeros(d_ff)
                self.W2 = rng.standard_normal((d_ff, d_model)) * 0.1
                self.b2 = np.zeros(d_model)
            def forward(self, x):
                return np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2

        class MHA_:
            def __init__(self, d_model, h, seed=0):
                rng = np.random.default_rng(seed)
                dk = d_model // h
                self.h = h
                self.dk = dk
                self.WQ = [rng.standard_normal((d_model, dk)) * 0.1 for _ in range(h)]
                self.WK = [rng.standard_normal((d_model, dk)) * 0.1 for _ in range(h)]
                self.WV = [rng.standard_normal((d_model, dk)) * 0.1 for _ in range(h)]
                self.WO = rng.standard_normal((d_model, d_model)) * 0.1
            def forward(self, X):
                heads = []
                for i in range(self.h):
                    out, _ = scaled_dot_product_attention(X @ self.WQ[i], X @ self.WK[i], X @ self.WV[i])
                    heads.append(out)
                return np.concatenate(heads, axis=-1) @ self.WO

        class Block:
            def __init__(self, d_model, num_heads, d_ff):
                self.mha = MHA_(d_model, num_heads)
                self.ffn = FFN_(d_model, d_ff)
            def forward(self, X):
                X = layer_norm(X + self.mha.forward(X))
                X = layer_norm(X + self.ffn.forward(X))
                return X

        return Block(d_model, num_heads, d_ff)

    def test_output_shape(self):
        np.random.seed(0)
        block = self._make_block()
        X = np.random.randn(5, 16)
        out = block.forward(X)
        assert out.shape == (5, 16)

    def test_layer_norm_applied(self):
        """Output of encoder block should have mean ~0 and std ~1 per row (layer norm)."""
        np.random.seed(1)
        block = self._make_block()
        X = np.random.randn(5, 16)
        out = block.forward(X)
        np.testing.assert_allclose(out.mean(axis=-1), np.zeros(5), atol=1e-5)
        np.testing.assert_allclose(out.std(axis=-1), np.ones(5), atol=1e-5)

    def test_different_inputs_give_different_outputs(self):
        block = self._make_block()
        X1 = np.random.randn(4, 16)
        X2 = np.random.randn(4, 16)
        out1 = block.forward(X1)
        out2 = block.forward(X2)
        assert not np.allclose(out1, out2)
