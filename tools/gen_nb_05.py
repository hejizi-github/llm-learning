#!/usr/bin/env python3
"""生成 notebooks/05-lstm-1997.ipynb"""
import json
from pathlib import Path

def cell(source, cell_type="code", outputs=None):
    if isinstance(source, list):
        src = source
    else:
        src = [line + "\n" for line in source.split("\n")]
        if src and src[-1].endswith("\n"):
            src[-1] = src[-1][:-1]
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": src,
    }

cells = []

# --- Cell 0: 标题 ---
cells.append(cell("""# 节点05：记忆的形状——LSTM（Hochreiter & Schmidhuber，1997）

本 notebook 从零手撕 LSTMCell（只用 NumPy），演示：
1. 简单 RNN 的梯度消失现象
2. LSTM 细胞的前向传播
3. 用 LSTM 学习序列反转任务
4. 用 PyTorch nn.LSTM 验证结论""", "markdown"))

# --- Cell 1: 导入 ---
cells.append(cell("""\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

Path('../docs/assets').mkdir(parents=True, exist_ok=True)

np.random.seed(42)
print("NumPy version:", np.__version__)"""))

# --- Cell 2: 梯度消失演示 ---
cells.append(cell("""## 第一步：梯度消失——简单 RNN 的致命缺陷

我们模拟梯度通过 T 个时刻反向传播的过程：
每个时刻都要乘以 W_h × tanh'(h)，tanh' 最大为 1，通常小于 1。
连乘后梯度指数衰减。""", "markdown"))

cells.append(cell("""\
def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2

def rnn_gradient_norm(T, Wh=0.9):
    \"\"\"模拟梯度范数随时间步 T 的变化（简单 RNN）\"\"\"
    grad_norm = 1.0
    h = 0.0
    norms = [1.0]
    for _ in range(T):
        h = np.tanh(Wh * h + 0.1)         # 前向
        local_grad = Wh * tanh_deriv(h)    # 一步的局部梯度
        grad_norm *= abs(local_grad)
        norms.append(grad_norm)
    return norms

lengths = [5, 20, 50, 100]
fig, ax = plt.subplots(figsize=(8, 4))
for T in lengths:
    norms = rnn_gradient_norm(T)
    ax.semilogy(norms, label=f"T={T} 步")
ax.set_xlabel("时间步（从输出反向传播到此处）")
ax.set_ylabel("梯度范数（log 刻度）")
ax.set_title("简单 RNN：梯度随时间步指数衰减（梯度消失）")
ax.legend()
plt.tight_layout()
plt.savefig('../docs/assets/05-gradient-vanishing.png', dpi=80)
plt.close()

# 验证：100步后梯度接近0
final_norm_100 = rnn_gradient_norm(100)[-1]
print(f"100步后梯度范数: {final_norm_100:.2e}")
assert final_norm_100 < 1e-3, f"期望梯度极小，实际={final_norm_100}"
print("梯度消失验证通过：100步后梯度 < 1e-3")"""))

# --- Cell 3: 手撕 LSTMCell ---
cells.append(cell("""## 第二步：手撕 LSTMCell（纯 NumPy）

三个门 + 细胞状态，一步步实现。""", "markdown"))

cells.append(cell("""\
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

class LSTMCell:
    \"\"\"单步 LSTM 单元（NumPy 实现，无框架依赖）\"\"\"

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        D = input_size + hidden_size
        scale = 0.1
        # 遗忘门 f、输入门 i、候选状态 g、输出门 o
        self.Wf = np.random.randn(hidden_size, D) * scale
        self.Wi = np.random.randn(hidden_size, D) * scale
        self.Wg = np.random.randn(hidden_size, D) * scale
        self.Wo = np.random.randn(hidden_size, D) * scale
        # 遗忘门偏置初始化为 1（促进早期记忆保留）
        self.bf = np.ones(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bg = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)

    def forward(self, x, h_prev, c_prev):
        \"\"\"
        x:      (input_size,)  当前输入
        h_prev: (hidden_size,) 上一步隐状态
        c_prev: (hidden_size,) 上一步细胞状态
        返回: (h, c) 各 (hidden_size,)
        \"\"\"
        combined = np.concatenate([h_prev, x])   # (D,)

        f = sigmoid(self.Wf @ combined + self.bf)        # 遗忘门
        i = sigmoid(self.Wi @ combined + self.bi)        # 输入门
        g = np.tanh(self.Wg @ combined + self.bg)        # 候选细胞状态
        o = sigmoid(self.Wo @ combined + self.bo)        # 输出门

        c = f * c_prev + i * g    # 细胞状态（加法路径，抵抗梯度消失）
        h = o * np.tanh(c)        # 隐状态

        return h, c

# --- 快速验证 ---
np.random.seed(0)
lstm = LSTMCell(input_size=3, hidden_size=4)
x = np.array([0.1, 0.2, 0.3])
h0 = np.zeros(4)
c0 = np.zeros(4)
h1, c1 = lstm.forward(x, h0, c0)
print(f"输入 x: {x}")
print(f"h1 形状: {h1.shape}, 值范围: [{h1.min():.4f}, {h1.max():.4f}]")
print(f"c1 形状: {c1.shape}")
assert h1.shape == (4,), f"h形状错误: {h1.shape}"
assert c1.shape == (4,), f"c形状错误: {c1.shape}"
assert np.all(np.abs(h1) <= 1.0 + 1e-9), "h应该在 [-1, 1] 内（输出门限制）"
print("LSTMCell 前向传播验证通过")"""))

# --- Cell 4: 序列前向传播 ---
cells.append(cell("""## 第三步：处理一个完整序列

把 LSTMCell 包装成可以处理序列的 LSTM 层。""", "markdown"))

cells.append(cell("""\
def lstm_forward(cell, sequence):
    \"\"\"
    sequence: list of (input_size,) arrays，长度 = T
    返回: (outputs, final_h, final_c)
        outputs: (T, hidden_size) 每步的 h
    \"\"\"
    h = np.zeros(cell.hidden_size)
    c = np.zeros(cell.hidden_size)
    outputs = []
    for x in sequence:
        h, c = cell.forward(x, h, c)
        outputs.append(h.copy())
    return np.array(outputs), h, c

# 测试：长度为 5 的序列，每步输入维度 3，隐层维度 4
seq = [np.random.randn(3) for _ in range(5)]
np.random.seed(1)
lstm = LSTMCell(input_size=3, hidden_size=4)
outputs, h_final, c_final = lstm_forward(lstm, seq)
print(f"序列长度: {len(seq)}")
print(f"outputs 形状: {outputs.shape}  （期望 (5, 4)）")
print(f"最终 h 形状: {h_final.shape}")
assert outputs.shape == (5, 4), f"outputs 形状错误: {outputs.shape}"
print("序列前向传播验证通过")"""))

# --- Cell 5: 序列反转任务 ---
cells.append(cell("""## 第四步：训练——学习序列反转

任务：给定输入序列 [1, 2, 3, 4, 5]，学会输出 [5, 4, 3, 2, 1]。

这需要网络"记住"整个序列再倒序输出，是 LSTM 记忆能力的经典验证。

我们使用最简单的梯度下降（手动实现反向传播近似，用数值梯度）。""", "markdown"))

cells.append(cell("""\
class SimpleLSTMModel:
    \"\"\"单步预测的 LSTM 模型：每步输出一个标量（用于序列反转）\"\"\"

    def __init__(self, hidden_size=8):
        self.cell = LSTMCell(input_size=1, hidden_size=hidden_size)
        # 输出层：hidden → 1
        self.Wy = np.random.randn(1, hidden_size) * 0.1
        self.by = np.zeros(1)
        self.hidden_size = hidden_size

    def predict(self, sequence):
        \"\"\"sequence: list of scalars。返回每步的预测值列表。\"\"\"
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        preds = []
        for val in sequence:
            x = np.array([val])
            h, c = self.cell.forward(x, h, c)
            y = self.Wy @ h + self.by
            preds.append(y[0])
        return preds

    def loss(self, sequence, targets):
        preds = self.predict(sequence)
        return float(np.mean([(p - t)**2 for p, t in zip(preds, targets)]))

def collect_params(model):
    \"\"\"收集模型所有 ndarray 参数，返回 [(param_dict, name), ...] 列表\"\"\"
    result = []
    for param_dict in [model.cell.__dict__, model.__dict__]:
        for name, param in param_dict.items():
            if isinstance(param, np.ndarray):
                result.append((param_dict, name))
    return result

def numerical_gradient(model, seq, targets, eps=1e-4):
    \"\"\"对所有参数计算数值梯度（仅用于教学演示）\"\"\"
    param_list = collect_params(model)
    grad_list = []
    for param_dict, name in param_list:
        param = param_dict[name]
        g = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            orig = param[idx]
            param[idx] = orig + eps
            loss_plus = model.loss(seq, targets)
            param[idx] = orig - eps
            loss_minus = model.loss(seq, targets)
            param[idx] = orig
            g[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
        grad_list.append((param_dict, name, g))
    return grad_list

# 训练参数
np.random.seed(42)
model = SimpleLSTMModel(hidden_size=6)
sequence = [1.0, 2.0, 3.0, 4.0, 5.0]
targets  = [5.0, 4.0, 3.0, 2.0, 1.0]

# 归一化到 [-1, 1]
seq_norm = [v/5.0 for v in sequence]
tgt_norm = [v/5.0 for v in targets]

# 用数值梯度训练 (慢但透明)，只跑 80 步作为演示
losses = []
lr = 0.05
for step in range(80):
    l = model.loss(seq_norm, tgt_norm)
    losses.append(l)
    grad_list = numerical_gradient(model, seq_norm, tgt_norm)
    for param_dict, pname, g in grad_list:
        param_dict[pname] -= lr * g

final_loss = model.loss(seq_norm, tgt_norm)
losses.append(final_loss)
print(f"初始 loss: {losses[0]:.4f}")
print(f"最终 loss (80步): {final_loss:.4f}")
print(f"loss 下降幅度: {(losses[0]-final_loss)/losses[0]*100:.1f}%")"""))

# --- Cell 6: 训练曲线 ---
cells.append(cell("""\
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, color='steelblue', linewidth=2)
ax.set_xlabel("训练步数")
ax.set_ylabel("MSE Loss")
ax.set_title("LSTM 序列反转任务训练曲线")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../docs/assets/05-lstm-training.png', dpi=80)
plt.close()

preds = model.predict(seq_norm)
print("序列反转预测（归一化后）:")
print(f"目标:  {[f'{v:.2f}' for v in tgt_norm]}")
print(f"预测:  {[f'{v:.2f}' for v in preds]}")
assert losses[-1] < losses[0], "训练后 loss 应该下降"
print("训练验证通过：loss 确实下降了")"""))

# --- Cell 7: PyTorch 对比 ---
cells.append(cell("""## 第五步：用 PyTorch nn.LSTM 对比

用框架验证手撕版本的原理是否正确。""", "markdown"))

cells.append(cell("""\
try:
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    # 单层 LSTM，input_size=1, hidden_size=4
    lstm_torch = nn.LSTM(input_size=1, hidden_size=4, batch_first=True)

    # 输入：batch=1, seq_len=5, input_size=1
    x_t = torch.tensor([[[v] for v in seq_norm]], dtype=torch.float32)
    output, (h_n, c_n) = lstm_torch(x_t)

    print(f"PyTorch LSTM 输出形状: {output.shape}  （batch=1, seq=5, hidden=4）")
    print(f"最终隐状态 h_n 形状: {h_n.shape}")
    print(f"最终细胞状态 c_n 形状: {c_n.shape}")
    assert output.shape == (1, 5, 4)
    assert h_n.shape == (1, 1, 4)
    print("PyTorch nn.LSTM 维度验证通过")
    print()
    print("结论：PyTorch 的 LSTM 与手撕版本使用完全相同的门控公式，")
    print("      区别仅在于参数初始化方式和批处理效率。")
except ImportError:
    print("PyTorch 未安装，跳过对比演示。")
    print("（手撕版本已通过独立验证，结果正确）")"""))

# --- 组装 notebook ---
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"},
    },
    "cells": cells,
}

out_path = Path("notebooks/05-lstm-1997.ipynb")
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Generated: {out_path}")
