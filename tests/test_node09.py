"""
节点09 (Transformer 2017) 的测试。
测试缩放点积注意力、多头注意力、位置编码的数学性质。
"""
import numpy as np
import pytest


# ── 内联实现（与 notebook 一致，避免 import 路径问题）─────────────────────────

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights


def positional_encoding(max_seq_len, d_model):
    PE = np.zeros((max_seq_len, d_model))
    positions = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(0, d_model, 2)[np.newaxis, :]
    freqs = 1.0 / (10000 ** (i / d_model))
    angles = positions * freqs
    PE[:, 0::2] = np.sin(angles)
    PE[:, 1::2] = np.cos(angles)
    return PE


def causal_mask(seq_len):
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


# ── 缩放点积注意力测试 ──────────────────────────────────────────────────────────

class TestScaledDotProductAttention:

    def test_output_shape(self):
        np.random.seed(0)
        seq, d_k, d_v = 4, 8, 8
        Q = np.random.randn(seq, d_k)
        K = np.random.randn(seq, d_k)
        V = np.random.randn(seq, d_v)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (seq, d_v)
        assert weights.shape == (seq, seq)

    def test_weights_sum_to_one(self):
        np.random.seed(1)
        Q = np.random.randn(5, 8)
        K = np.random.randn(5, 8)
        V = np.random.randn(5, 8)
        _, weights = scaled_dot_product_attention(Q, K, V)
        assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-7)

    def test_weights_nonnegative(self):
        np.random.seed(2)
        Q = np.random.randn(4, 6)
        K = np.random.randn(4, 6)
        V = np.random.randn(4, 6)
        _, weights = scaled_dot_product_attention(Q, K, V)
        assert (weights >= 0).all()

    def test_no_nan_inf(self):
        np.random.seed(3)
        Q = np.random.randn(6, 16)
        K = np.random.randn(6, 16)
        V = np.random.randn(6, 16)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert np.isfinite(output).all()
        assert np.isfinite(weights).all()

    def test_batched_shape(self):
        """支持批量 / 多头维度 (..., seq, d)。"""
        np.random.seed(4)
        h, seq, d = 4, 5, 8
        Q = np.random.randn(h, seq, d)
        K = np.random.randn(h, seq, d)
        V = np.random.randn(h, seq, d)
        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (h, seq, d)
        assert weights.shape == (h, seq, seq)
        assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-7)


# ── 因果遮掩测试 ────────────────────────────────────────────────────────────────

class TestCausalMask:

    def test_shape(self):
        mask = causal_mask(5)
        assert mask.shape == (5, 5)
        assert mask.dtype == bool

    def test_upper_triangle_true(self):
        mask = causal_mask(4)
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j], f"mask[{i},{j}] should be True"

    def test_lower_triangle_false(self):
        mask = causal_mask(4)
        for i in range(4):
            for j in range(i + 1):
                assert not mask[i, j], f"mask[{i},{j}] should be False"

    def test_masked_weights_near_zero(self):
        np.random.seed(5)
        seq = 5
        Q = np.random.randn(seq, 8)
        K = np.random.randn(seq, 8)
        V = np.random.randn(seq, 8)
        _, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(seq))
        for i in range(seq):
            for j in range(i + 1, seq):
                assert weights[i, j] < 1e-6, (
                    f"Masked weight at [{i},{j}] = {weights[i,j]:.2e}, expected < 1e-6")

    def test_visible_weights_sum_to_one(self):
        np.random.seed(6)
        seq = 4
        Q = np.random.randn(seq, 8)
        K = np.random.randn(seq, 8)
        V = np.random.randn(seq, 8)
        _, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask(seq))
        assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-6)


# ── 位置编码测试 ────────────────────────────────────────────────────────────────

class TestPositionalEncoding:

    def test_shape(self):
        PE = positional_encoding(20, 32)
        assert PE.shape == (20, 32)

    def test_value_range(self):
        PE = positional_encoding(100, 64)
        assert PE.min() >= -1.0 - 1e-9
        assert PE.max() <= 1.0 + 1e-9

    def test_different_positions_are_different(self):
        PE = positional_encoding(50, 64)
        assert not np.allclose(PE[0], PE[1])
        assert not np.allclose(PE[0], PE[10])
        assert not np.allclose(PE[5], PE[15])

    def test_self_similarity_is_one(self):
        PE = positional_encoding(50, 512)
        pe0 = PE[0]
        sim = (pe0 @ pe0) / (np.linalg.norm(pe0) ** 2)
        assert abs(sim - 1.0) < 1e-8

    def test_near_more_similar_than_far(self):
        """相邻位置比远距位置更相似（余弦相似度）。"""
        PE = positional_encoding(50, 512)
        pe0 = PE[0]
        norm0 = np.linalg.norm(pe0)

        def cosine(i):
            return (PE[i] @ pe0) / (np.linalg.norm(PE[i]) * norm0)

        sim_near = np.mean([cosine(i) for i in range(1, 6)])
        sim_far = np.mean([cosine(i) for i in range(20, 30)])
        assert sim_near > sim_far, (
            f"Near similarity {sim_near:.4f} should > far similarity {sim_far:.4f}")

    def test_even_dims_are_sin_odd_are_cos(self):
        """偶数维用 sin，奇数维用 cos（基本性质验证）。"""
        d = 8
        PE = positional_encoding(1, d)  # position 0
        # At position 0, sin(0) = 0 for even dims
        for dim_idx in range(0, d, 2):
            assert abs(PE[0, dim_idx]) < 1e-9, (
                f"PE[0, {dim_idx}] (even dim) should be 0 (sin(0))")
        # At position 0, cos(0) = 1 for odd dims
        for dim_idx in range(1, d, 2):
            assert abs(PE[0, dim_idx] - 1.0) < 1e-9, (
                f"PE[0, {dim_idx}] (odd dim) should be 1 (cos(0))")


# ── notebook 可执行性测试 ────────────────────────────────────────────────────────

def test_notebook_executes():
    """节点09 notebook 必须零错误执行。"""
    import subprocess
    import os
    nb_path = os.path.join(
        os.path.dirname(__file__),
        "..", "nodes", "09-transformer-2017", "transformer.ipynb"
    )
    nb_path = os.path.abspath(nb_path)
    result = subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute",
         nb_path, "--output", "/tmp/test_node09_executed.ipynb"],
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, (
        f"Notebook execution failed:\n{result.stderr[-1000:]}")
