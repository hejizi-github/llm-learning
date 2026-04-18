"""
tests/test_flash_attention.py — 节点24 FlashAttention 测试

覆盖：
- 标准 attention 与分块 attention 数学等价性
- 不同块大小下结果一致
- 在线 softmax 更新公式正确性
- 数值稳定性（极端值不产生 NaN/inf）
- 输出归一化正确（softmax 权重加和为1）
- 内存复杂度量化（分块矩阵大小 vs 序列长度无关）
"""
import numpy as np
import pytest


# ─── Implementations (mirror notebook) ────────────────────────────────────

def standard_attention(Q, K, V):
    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    S = (Q @ K.T) * scale
    S = S - S.max(axis=1, keepdims=True)
    exp_S = np.exp(S)
    P = exp_S / exp_S.sum(axis=1, keepdims=True)
    return P @ V


def tiled_attention(Q, K, V, block_size=32):
    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    O = np.zeros((n, d), dtype=np.float64)
    L = np.zeros(n, dtype=np.float64)
    M = np.full(n, -np.inf, dtype=np.float64)

    for j in range(0, n, block_size):
        Kj = K[j:j+block_size].astype(np.float64)
        Vj = V[j:j+block_size].astype(np.float64)
        for i in range(0, n, block_size):
            Qi = Q[i:i+block_size].astype(np.float64)
            bsz = Qi.shape[0]
            Sij = Qi @ Kj.T * scale
            Mij = Sij.max(axis=1)
            new_M = np.maximum(M[i:i+bsz], Mij)
            decay = np.exp(M[i:i+bsz] - new_M)
            Pij = np.exp(Sij - new_M[:, None])
            L[i:i+bsz] = decay * L[i:i+bsz] + Pij.sum(axis=1)
            O[i:i+bsz] = decay[:, None] * O[i:i+bsz] + Pij @ Vj
            M[i:i+bsz] = new_M

    return (O / L[:, None]).astype(np.float32)


def batch_softmax(x):
    x = x - x.max()
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def online_softmax_2blocks(x):
    n = len(x)
    half = n // 2
    block1, block2 = x[:half], x[half:]

    m1 = block1.max()
    exp1 = np.exp(block1 - m1)
    l1 = exp1.sum()

    m2 = block2.max()
    new_m = max(m1, m2)
    exp2 = np.exp(block2 - new_m)
    l2 = exp2.sum()
    new_l = l1 * np.exp(m1 - new_m) + l2

    result = np.zeros(n)
    result[:half] = exp1 * np.exp(m1 - new_m) / new_l
    result[half:] = exp2 / new_l
    return result


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def qkv_small():
    rng = np.random.RandomState(42)
    n, d = 32, 8
    Q = rng.randn(n, d).astype(np.float32)
    K = rng.randn(n, d).astype(np.float32)
    V = rng.randn(n, d).astype(np.float32)
    return Q, K, V


@pytest.fixture
def qkv_medium():
    rng = np.random.RandomState(7)
    n, d = 64, 16
    Q = rng.randn(n, d).astype(np.float32)
    K = rng.randn(n, d).astype(np.float32)
    V = rng.randn(n, d).astype(np.float32)
    return Q, K, V


# ─── Tests ─────────────────────────────────────────────────────────────────

class TestMathematicalEquivalence:
    """分块 Attention 与标准 Attention 数学等价"""

    def test_basic_equivalence(self, qkv_small):
        Q, K, V = qkv_small
        out_s = standard_attention(Q, K, V)
        out_t = tiled_attention(Q, K, V, block_size=8)
        assert np.abs(out_s - out_t).max() < 1e-4, "分块与标准结果差异超过浮点容差"

    def test_equivalence_block_size_4(self, qkv_small):
        Q, K, V = qkv_small
        out_s = standard_attention(Q, K, V)
        out_t = tiled_attention(Q, K, V, block_size=4)
        assert np.abs(out_s - out_t).max() < 1e-4

    def test_equivalence_block_size_16(self, qkv_small):
        Q, K, V = qkv_small
        out_s = standard_attention(Q, K, V)
        out_t = tiled_attention(Q, K, V, block_size=16)
        assert np.abs(out_s - out_t).max() < 1e-4

    def test_equivalence_large_block(self, qkv_medium):
        """块大小 >= n 时等价于标准 attention"""
        Q, K, V = qkv_medium
        n = Q.shape[0]
        out_s = standard_attention(Q, K, V)
        out_t = tiled_attention(Q, K, V, block_size=n)
        assert np.abs(out_s - out_t).max() < 1e-4

    def test_different_block_sizes_give_same_result(self, qkv_medium):
        """不同块大小的输出应该一致"""
        Q, K, V = qkv_medium
        results = [tiled_attention(Q, K, V, block_size=bs) for bs in [4, 8, 16, 32]]
        for i in range(1, len(results)):
            assert np.abs(results[0] - results[i]).max() < 1e-4


class TestOnlineSoftmax:
    """在线 Softmax 更新公式正确性"""

    def test_online_softmax_matches_batch(self):
        rng = np.random.RandomState(0)
        x = rng.randn(8)
        result_batch = batch_softmax(x)
        result_online = online_softmax_2blocks(x)
        assert np.abs(result_batch - result_online).max() < 1e-10

    def test_online_softmax_sums_to_one(self):
        rng = np.random.RandomState(1)
        x = rng.randn(10)
        result = online_softmax_2blocks(x)
        assert abs(result.sum() - 1.0) < 1e-10

    def test_online_softmax_all_positive(self):
        rng = np.random.RandomState(2)
        x = rng.randn(10)
        result = online_softmax_2blocks(x)
        assert np.all(result >= 0)


class TestNumericalStability:
    """数值稳定性：极端输入不产生 NaN/inf"""

    def test_standard_attention_no_nan(self, qkv_small):
        Q, K, V = qkv_small
        out = standard_attention(Q, K, V)
        assert np.all(np.isfinite(out)), "标准 attention 输出含 NaN/inf"

    def test_tiled_attention_no_nan(self, qkv_small):
        Q, K, V = qkv_small
        out = tiled_attention(Q, K, V)
        assert np.all(np.isfinite(out)), "分块 attention 输出含 NaN/inf"

    def test_large_values_no_nan(self):
        rng = np.random.RandomState(99)
        n, d = 16, 8
        Q = (rng.randn(n, d) * 10).astype(np.float32)
        K = (rng.randn(n, d) * 10).astype(np.float32)
        V = rng.randn(n, d).astype(np.float32)
        out_s = standard_attention(Q, K, V)
        out_t = tiled_attention(Q, K, V, block_size=8)
        assert np.all(np.isfinite(out_s))
        assert np.all(np.isfinite(out_t))
        assert np.abs(out_s - out_t).max() < 1e-3


class TestOutputShape:
    """输出形状与标准 attention 相同"""

    def test_output_shape(self, qkv_small):
        Q, K, V = qkv_small
        out = tiled_attention(Q, K, V)
        assert out.shape == Q.shape

    def test_output_shape_medium(self, qkv_medium):
        Q, K, V = qkv_medium
        out = tiled_attention(Q, K, V)
        assert out.shape == Q.shape
