"""
测试 node05 梯度消失相关逻辑。
所有数学断言都有理论依据，见 README 中的公式。
"""
import math
import sys
import os
import pytest

# 把 node05 的辅助函数内联到测试里（避免 import 路径复杂）
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def sigmoid_deriv_arr(x_list):
    return [sigmoid_derivative(x) for x in x_list]


class SimpleRNN:
    def __init__(self, W_h=0.9, W_x=0.5, b=0.0):
        self.W_h = W_h
        self.W_x = W_x
        self.b = b

    def forward(self, inputs):
        h = 0.0
        hidden_states = [h]
        for x_t in inputs:
            z = self.W_h * h + self.W_x * x_t + self.b
            h = sigmoid(z)
            hidden_states.append(h)
        return hidden_states

    def compute_gradient_magnitudes(self, hidden_states):
        T = len(hidden_states) - 1
        grad = 1.0
        magnitudes = []
        for t in range(T, 0, -1):
            h_t = hidden_states[t]
            local_grad = h_t * (1.0 - h_t) * self.W_h
            grad *= local_grad
            magnitudes.append(abs(grad))
        return magnitudes


# ── Tests ──────────────────────────────────────────────────────────────────

def test_sigmoid_output_range():
    """sigmoid 在实数域输出必须在 [0, 1] 内（极端值因浮点精度可能恰好为 0 或 1）。"""
    for x in [-100, -1, 0, 1, 100]:
        val = sigmoid(x)
        assert 0.0 <= val <= 1.0, f"sigmoid({x}) = {val} 不在 [0,1]"
    # 合理范围内必须是严格 (0, 1)
    for x in [-10, -1, 0, 1, 10]:
        val = sigmoid(x)
        assert 0.0 < val < 1.0, f"sigmoid({x}) = {val} 不在严格 (0,1)"

def test_sigmoid_at_zero():
    """sigmoid(0) = 0.5（对称点）。"""
    assert abs(sigmoid(0) - 0.5) < 1e-9

def test_sigmoid_derivative_max_is_0_25():
    """Sigmoid 导数的最大值在 x=0 时取到，等于 0.25。"""
    max_d = max(sigmoid_derivative(x * 0.01) for x in range(-1000, 1001))
    assert abs(max_d - 0.25) < 0.001, f"最大导数 = {max_d}，期望 ≈ 0.25"

def test_sigmoid_derivative_formula():
    """σ'(x) = σ(x) * (1 - σ(x))，用数值差分验证。"""
    eps = 1e-5
    for x in [-2.0, 0.0, 1.5]:
        numerical = (sigmoid(x + eps) - sigmoid(x - eps)) / (2 * eps)
        analytical = sigmoid_derivative(x)
        assert abs(numerical - analytical) < 1e-6, \
            f"x={x}: 数值导数={numerical:.6f}, 解析导数={analytical:.6f}"

def test_exponential_decay():
    """0.25^T 随 T 增加指数下降，T=20 时应 < 1e-10。"""
    val_20 = 0.25 ** 20
    assert val_20 < 1e-10
    val_10 = 0.25 ** 10
    assert val_10 < 1e-5

def test_rnn_forward_shape():
    """RNN 前向：输入 T 步，返回 T+1 个隐藏状态（含初始 h0）。"""
    rnn = SimpleRNN()
    for seq_len in [1, 5, 10, 50]:
        inputs = [0.5] * seq_len
        states = rnn.forward(inputs)
        assert len(states) == seq_len + 1, \
            f"seq_len={seq_len}: 期望 {seq_len+1} 个状态，得到 {len(states)}"

def test_rnn_hidden_states_in_range():
    """RNN 使用 sigmoid 激活，所有隐藏状态应在 (0, 1)。"""
    rnn = SimpleRNN(W_h=0.9, W_x=0.5)
    states = rnn.forward([0.3] * 20)
    for i, h in enumerate(states[1:], 1):
        assert 0.0 < h < 1.0, f"h_{i} = {h} 不在 (0,1)"

def test_gradient_vanishes_after_many_steps():
    """W_h=0.5 时，50步后梯度应接近0（< 1e-5）。"""
    rnn = SimpleRNN(W_h=0.5, W_x=0.3)
    states = rnn.forward([0.5] * 50)
    grads = rnn.compute_gradient_magnitudes(states)
    assert len(grads) == 50
    assert grads[-1] < 1e-5, f"50步后梯度 = {grads[-1]}，期望 < 1e-5"

def test_gradient_larger_near_end():
    """梯度应该在接近当前步（距离短）时更大，越远越小。"""
    rnn = SimpleRNN(W_h=0.9, W_x=0.5)
    states = rnn.forward([0.5] * 30)
    grads = rnn.compute_gradient_magnitudes(states)
    # 距离 1 步的梯度 > 距离 20 步的梯度
    assert grads[0] > grads[19], \
        f"近处梯度({grads[0]:.4f}) 应大于远处梯度({grads[19]:.4f})"

def test_notebook_exists():
    """notebook 文件必须存在。"""
    nb_path = os.path.join(
        os.path.dirname(__file__),
        '../nodes/05-gradient-vanishing-1991/gradient_vanishing.ipynb'
    )
    assert os.path.exists(nb_path), f"找不到 notebook: {nb_path}"

def test_references_bib_exists():
    """references.bib 必须存在且包含三篇核心论文。"""
    bib_path = os.path.join(
        os.path.dirname(__file__),
        '../nodes/05-gradient-vanishing-1991/references.bib'
    )
    assert os.path.exists(bib_path), f"找不到 {bib_path}"
    content = open(bib_path).read()
    assert 'hochreiter1991' in content, "缺少 hochreiter1991"
    assert 'bengio1994' in content, "缺少 bengio1994"
    assert 'hochreiter1997lstm' in content, "缺少 hochreiter1997lstm"

def test_doi_bengio1994_in_bib():
    """Bengio 1994 DOI 必须正确写入 bib。"""
    bib_path = os.path.join(
        os.path.dirname(__file__),
        '../nodes/05-gradient-vanishing-1991/references.bib'
    )
    content = open(bib_path).read()
    assert '10.1109/72.279181' in content, "缺少 Bengio 1994 DOI"

def test_doi_lstm1997_in_bib():
    """LSTM 1997 DOI 必须正确写入 bib。"""
    bib_path = os.path.join(
        os.path.dirname(__file__),
        '../nodes/05-gradient-vanishing-1991/references.bib'
    )
    content = open(bib_path).read()
    assert '10.1162/neco.1997.9.8.1735' in content, "缺少 LSTM 1997 DOI"
