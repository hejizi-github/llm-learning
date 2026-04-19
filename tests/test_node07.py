"""
节点07 (Attention 2015) 的测试。
测试 Bahdanau attention 的数学性质：softmax 归一化、上下文向量形状、凸组合约束。
"""
import os
import math
import numpy as np
import pytest

# ── 辅助函数（内联，避免 import 路径问题）──────────────────────────

def softmax(scores):
    scores = np.array(scores, dtype=float)
    exps = np.exp(scores - np.max(scores))
    return exps / exps.sum()


def bahdanau_attention(encoder_states, decoder_state, W1, W2, v):
    T = len(encoder_states)
    scores = np.array([
        v @ np.tanh(W1 @ encoder_states[j] + W2 @ decoder_state)
        for j in range(T)
    ])
    alphas = softmax(scores)
    context = (alphas[:, None] * encoder_states).sum(axis=0)
    return context, alphas


# ── 文件存在性检查 ────────────────────────────────────────────────

def test_node07_readme_exists():
    """README.md 必须存在。"""
    path = os.path.join(os.path.dirname(__file__), '../nodes/07-attention-2015/README.md')
    assert os.path.exists(path), f"找不到 {path}"


def test_node07_notebook_file_exists():
    """notebook 文件必须存在（不验证执行；执行验证通过 tools/notebook-run 完成）。"""
    path = os.path.join(os.path.dirname(__file__), '../nodes/07-attention-2015/attention.ipynb')
    assert os.path.exists(path), f"找不到 {path}"


def test_node07_references_bib_exists():
    """references.bib 必须存在。"""
    path = os.path.join(os.path.dirname(__file__), '../nodes/07-attention-2015/references.bib')
    assert os.path.exists(path), f"找不到 {path}"


def test_node07_bib_contains_bahdanau():
    """references.bib 必须包含 Bahdanau 2015 (arXiv:1409.0473)。"""
    path = os.path.join(os.path.dirname(__file__), '../nodes/07-attention-2015/references.bib')
    content = open(path).read()
    assert '1409.0473' in content, "缺少 Bahdanau 2015 (arXiv:1409.0473)"


def test_node07_bib_contains_sutskever():
    """references.bib 必须包含 Sutskever 2014 (arXiv:1409.3215)。"""
    path = os.path.join(os.path.dirname(__file__), '../nodes/07-attention-2015/references.bib')
    content = open(path).read()
    assert '1409.3215' in content, "缺少 Sutskever 2014 (arXiv:1409.3215)"


# ── softmax 数学性质 ──────────────────────────────────────────────

def test_softmax_sums_to_one():
    """softmax 输出之和严格等于 1（核心归一化性质）。"""
    for scores in [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-5.0, 0.0, 5.0], [100.0, 100.0]]:
        weights = softmax(scores)
        assert abs(weights.sum() - 1.0) < 1e-9, \
            f"scores={scores}: 权重之和 = {weights.sum()}"


def test_softmax_all_non_negative():
    """softmax 所有输出必须非负。"""
    weights = softmax([-10.0, -1.0, 0.0, 1.0, 10.0])
    assert all(weights >= 0), f"存在负权重: {weights}"


def test_softmax_max_score_gets_highest_weight():
    """得分最高的位置应获得最大权重。"""
    scores = [1.0, 5.0, 2.0]
    weights = softmax(scores)
    assert weights.argmax() == 1, f"最大权重位置应为 1，实际为 {weights.argmax()}"


def test_softmax_uniform_scores_equal_weights():
    """所有得分相等时，权重应均等（1/T）。"""
    T = 5
    weights = softmax([0.0] * T)
    for w in weights:
        assert abs(w - 1.0 / T) < 1e-9, f"均匀分布时权重应为 {1/T}，实际为 {w}"


def test_softmax_numerical_stability_large_scores():
    """softmax 对大数值（差值 100）应保持数值稳定，不出现 nan 或 inf。"""
    weights = softmax([0.0, 100.0, 200.0])
    assert not any(math.isnan(w) for w in weights), "出现 nan"
    assert not any(math.isinf(w) for w in weights), "出现 inf"
    assert abs(weights.sum() - 1.0) < 1e-9


# ── Bahdanau attention 数学性质 ───────────────────────────────────

def _make_test_params(T=4, d_h=8, d_a=8, seed=42):
    rng = np.random.RandomState(seed)
    encoder_states = rng.randn(T, d_h)
    decoder_state = rng.randn(d_h)
    W1 = rng.randn(d_a, d_h) * 0.1
    W2 = rng.randn(d_a, d_h) * 0.1
    v = rng.randn(d_a) * 0.1
    return encoder_states, decoder_state, W1, W2, v


def test_attention_weights_sum_to_one():
    """注意力权重之和必须严格等于 1。"""
    enc, dec, W1, W2, v = _make_test_params()
    _, alphas = bahdanau_attention(enc, dec, W1, W2, v)
    assert abs(alphas.sum() - 1.0) < 1e-9, \
        f"注意力权重之和 = {alphas.sum()}"


def test_attention_weights_non_negative():
    """注意力权重必须全部非负。"""
    enc, dec, W1, W2, v = _make_test_params()
    _, alphas = bahdanau_attention(enc, dec, W1, W2, v)
    assert all(alphas >= 0), f"存在负权重"


def test_attention_context_shape():
    """上下文向量形状必须等于隐藏层维度 (d_h,)。"""
    T, d_h = 6, 16
    enc, dec, W1, W2, v = _make_test_params(T=T, d_h=d_h, d_a=d_h)
    context, _ = bahdanau_attention(enc, dec, W1, W2, v)
    assert context.shape == (d_h,), f"上下文向量形状 = {context.shape}，期望 ({d_h},)"


def test_attention_context_is_convex_combination():
    """上下文向量是编码器状态的凸组合，每维必须在 [min, max] 范围内。"""
    enc, dec, W1, W2, v = _make_test_params()
    context, _ = bahdanau_attention(enc, dec, W1, W2, v)
    d_h = enc.shape[1]
    for dim in range(d_h):
        lo, hi = enc[:, dim].min(), enc[:, dim].max()
        val = context[dim]
        assert lo <= val <= hi, \
            f"维度 {dim}: context[{dim}]={val:.4f} 不在 [{lo:.4f}, {hi:.4f}]"


def test_attention_alphas_shape():
    """注意力权重形状必须等于序列长度 (T,)。"""
    T = 7
    enc, dec, W1, W2, v = _make_test_params(T=T)
    _, alphas = bahdanau_attention(enc, dec, W1, W2, v)
    assert alphas.shape == (T,), f"权重形状 = {alphas.shape}，期望 ({T},)"


def test_attention_multistep_each_row_sums_to_one():
    """多步解码时，每步的注意力权重之和均等于 1。"""
    enc, _, W1, W2, v = _make_test_params(T=5, seed=0)
    rng = np.random.RandomState(99)
    n_steps = 4
    for step in range(n_steps):
        s_i = rng.randn(enc.shape[1])
        _, alphas_i = bahdanau_attention(enc, s_i, W1, W2, v)
        assert abs(alphas_i.sum() - 1.0) < 1e-9, \
            f"第 {step+1} 步权重之和 = {alphas_i.sum()}"


def test_attention_single_position():
    """T=1 时，唯一位置的注意力权重必须精确等于 1.0。"""
    rng = np.random.RandomState(7)
    enc = rng.randn(1, 4)
    dec = rng.randn(4)
    W1 = rng.randn(4, 4) * 0.1
    W2 = rng.randn(4, 4) * 0.1
    v = rng.randn(4) * 0.1
    context, alphas = bahdanau_attention(enc, dec, W1, W2, v)
    assert abs(alphas[0] - 1.0) < 1e-9, \
        f"T=1 时唯一权重应为 1.0，实际为 {alphas[0]}"
    np.testing.assert_array_almost_equal(context, enc[0])
