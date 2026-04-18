#!/usr/bin/env python3
"""生成 notebooks/03-backprop-1986.ipynb"""
import json, pathlib

cells = []

def code(source, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source if isinstance(source, list) else [source]
    }

def markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }

cells.append(markdown(
    "# 节点 03 · 反向传播（1986）— 手撕两层网络解决 XOR\n\n"
    "> 对应文档：[docs/03-backprop-1986.md](../docs/03-backprop-1986.md)\n\n"
    "**目标**：从零用 NumPy 实现一个两层神经网络，用反向传播训练它，让它学会 XOR。\n\n"
    "**环境依赖**：`numpy`, `matplotlib`（标准库，无需额外安装）"
))

cells.append(code(
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import os\n"
    "\n"
    "# notebook 从 notebooks/ 目录执行，assets 在 ../docs/assets/\n"
    "os.makedirs('../docs/assets', exist_ok=True)\n"
    "\n"
    "# 固定随机种子，保证可复现\n"
    "np.random.seed(42)\n"
    "print('NumPy version:', np.__version__)"
))

cells.append(markdown(
    "## 1. XOR 数据集\n\n"
    "| x1 | x2 | XOR |\n"
    "|----|----|----- |\n"
    "| 0  | 0  |  0  |\n"
    "| 0  | 1  |  1  |\n"
    "| 1  | 0  |  1  |\n"
    "| 1  | 1  |  0  |\n\n"
    "这是单层感知机永远搞不定的问题（节点02证明了这一点）。"
))

cells.append(code(
    "# XOR 输入：shape (4, 2)\n"
    "X = np.array([[0, 0],\n"
    "              [0, 1],\n"
    "              [1, 0],\n"
    "              [1, 1]], dtype=float)\n"
    "\n"
    "# XOR 标签：shape (4, 1)\n"
    "y = np.array([[0], [1], [1], [0]], dtype=float)\n"
    "\n"
    "print('输入 X:')\n"
    "print(X)\n"
    "print('标签 y:', y.T)"
))

cells.append(markdown(
    "## 2. 激活函数：Sigmoid\n\n"
    "$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$\n\n"
    "它的导数有个漂亮的性质：$\\sigma'(z) = \\sigma(z) \\cdot (1 - \\sigma(z))$\n\n"
    "**为什么用 sigmoid？**\n"
    "- 输出在 (0, 1) 之间，适合表示概率\n"
    "- 处处可导，链式法则可以工作\n"
    "- 1986 年代最流行的选择（后来 ReLU 取代了它，见节点04）"
))

cells.append(code(
    "def sigmoid(z):\n"
    "    \"\"\"sigmoid 激活函数\"\"\"\n"
    "    return 1.0 / (1.0 + np.exp(-z))\n"
    "\n"
    "def sigmoid_deriv(z):\n"
    "    \"\"\"sigmoid 导数：σ(z) * (1 - σ(z))\"\"\"\n"
    "    s = sigmoid(z)\n"
    "    return s * (1 - s)\n"
    "\n"
    "# 可视化 sigmoid 和它的导数\n"
    "z_vals = np.linspace(-6, 6, 200)\n"
    "fig, axes = plt.subplots(1, 2, figsize=(10, 3))\n"
    "\n"
    "axes[0].plot(z_vals, sigmoid(z_vals), color='steelblue', linewidth=2)\n"
    "axes[0].set_title('sigmoid(z)')\n"
    "axes[0].set_xlabel('z')\n"
    "axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)\n"
    "axes[0].grid(True, alpha=0.3)\n"
    "\n"
    "axes[1].plot(z_vals, sigmoid_deriv(z_vals), color='coral', linewidth=2)\n"
    "axes[1].set_title(\"sigmoid'(z) — 最大值只有 0.25\")\n"
    "axes[1].set_xlabel('z')\n"
    "axes[1].axhline(0.25, color='gray', linestyle='--', alpha=0.5)\n"
    "axes[1].grid(True, alpha=0.3)\n"
    "\n"
    "plt.suptitle('Sigmoid 函数及其导数', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig('../docs/assets/03_sigmoid.png', dpi=100, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('注意导数最大值只有 0.25 — 这是梯度消失的根源')"
))

cells.append(markdown(
    "## 3. 前向传播（Forward Pass）\n\n"
    "网络结构：**输入层(2) → 隐藏层(4) → 输出层(1)**\n\n"
    "```\n"
    "x (2,) → z1 = W1·x + b1 → h = σ(z1) → z2 = W2·h + b2 → ŷ = σ(z2)\n"
    "```\n\n"
    "前向传播就是：把输入一步步往前算，得到预测值。"
))

cells.append(code(
    "class TwoLayerNet:\n"
    "    def __init__(self, input_size=2, hidden_size=4, output_size=1):\n"
    "        # Xavier 初始化：让权重的方差适配层的大小\n"
    "        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)\n"
    "        self.b1 = np.zeros((1, hidden_size))\n"
    "        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)\n"
    "        self.b2 = np.zeros((1, output_size))\n"
    "\n"
    "    def forward(self, X):\n"
    "        \"\"\"前向传播：X -> h -> y_hat\"\"\"\n"
    "        # 第一层\n"
    "        self.z1 = X @ self.W1 + self.b1     # (N, hidden)\n"
    "        self.h  = sigmoid(self.z1)           # (N, hidden)\n"
    "        # 第二层\n"
    "        self.z2 = self.h @ self.W2 + self.b2 # (N, 1)\n"
    "        self.y_hat = sigmoid(self.z2)         # (N, 1)\n"
    "        return self.y_hat\n"
    "\n"
    "# 测试前向传播（未训练，输出随机）\n"
    "net = TwoLayerNet()\n"
    "y_hat = net.forward(X)\n"
    "print('未训练时的预测（随机）:')\n"
    "for i, (xi, yi, yh) in enumerate(zip(X, y, y_hat)):\n"
    "    print(f'  x={xi.astype(int)}, 真值={int(yi[0])}, 预测={yh[0]:.3f}')"
))

cells.append(markdown(
    "## 4. 反向传播（Backward Pass）\n\n"
    "这是核心！用**链式法则**，从输出层往回，逐层计算每个权重的梯度。\n\n"
    "```\n"
    "损失 L\n"
    "  ↓ ∂L/∂ŷ = ŷ - y\n"
    "输出层权重 W2\n"
    "  ↓ 链式法则传回\n"
    "隐藏层权重 W1\n"
    "```\n\n"
    "**关键公式（每步都用了链式法则）**：\n\n"
    "1. $\\delta_2 = (\\hat{y} - y) \\cdot \\sigma'(z_2)$ — 输出层误差信号\n"
    "2. $\\delta_1 = (\\delta_2 \\cdot W_2^T) \\cdot \\sigma'(z_1)$ — 把误差「传」回隐藏层\n"
    "3. $\\nabla W_2 = h^T \\cdot \\delta_2$，$\\nabla W_1 = X^T \\cdot \\delta_1$ — 计算梯度\n"
    "4. 梯度下降更新：$W \\leftarrow W - \\eta \\nabla W$"
))

cells.append(code(
    "def backward(self, X, y, lr=0.1):\n"
    "    \"\"\"反向传播 + 权重更新\"\"\"\n"
    "    N = X.shape[0]\n"
    "\n"
    "    # ---- 输出层 ----\n"
    "    # 损失对 y_hat 的梯度：dL/dyhat = (yhat - y)\n"
    "    dL_dyhat = self.y_hat - y                    # (N, 1)\n"
    "    # 链式法则：dL/dz2 = dL/dyhat * sigmoid'(z2)\n"
    "    delta2 = dL_dyhat * sigmoid_deriv(self.z2)  # (N, 1)\n"
    "\n"
    "    # W2 的梯度：dL/dW2 = h^T · delta2\n"
    "    dW2 = self.h.T @ delta2 / N                 # (hidden, 1)\n"
    "    db2 = delta2.mean(axis=0, keepdims=True)    # (1, 1)\n"
    "\n"
    "    # ---- 隐藏层 ----\n"
    "    # 把误差信号反向传回隐藏层：delta2 * W2^T\n"
    "    delta1 = (delta2 @ self.W2.T) * sigmoid_deriv(self.z1)  # (N, hidden)\n"
    "\n"
    "    # W1 的梯度：dL/dW1 = X^T · delta1\n"
    "    dW1 = X.T @ delta1 / N                      # (2, hidden)\n"
    "    db1 = delta1.mean(axis=0, keepdims=True)    # (1, hidden)\n"
    "\n"
    "    # ---- 梯度下降更新 ----\n"
    "    self.W2 -= lr * dW2\n"
    "    self.b2 -= lr * db2\n"
    "    self.W1 -= lr * dW1\n"
    "    self.b1 -= lr * db1\n"
    "\n"
    "# 把 backward 方法加到类里（monkey-patch，避免重写整个类）\n"
    "TwoLayerNet.backward = backward\n"
    "print('反向传播方法已注册')"
))

cells.append(markdown(
    "## 5. 训练循环\n\n"
    "不断重复：前向传播 → 计算损失 → 反向传播 → 更新权重。\n\n"
    "这就是神经网络「学习」的全部过程。"
))

cells.append(code(
    "def mse_loss(y_hat, y):\n"
    "    \"\"\"均方误差损失\"\"\"\n"
    "    return np.mean((y_hat - y) ** 2)\n"
    "\n"
    "# 重新初始化，保证可复现\n"
    "np.random.seed(42)\n"
    "net = TwoLayerNet(hidden_size=4)\n"
    "TwoLayerNet.backward = backward\n"
    "\n"
    "EPOCHS = 10000\n"
    "LR = 0.5\n"
    "loss_history = []\n"
    "\n"
    "for epoch in range(EPOCHS):\n"
    "    # 前向传播\n"
    "    y_hat = net.forward(X)\n"
    "    # 计算损失\n"
    "    loss = mse_loss(y_hat, y)\n"
    "    loss_history.append(loss)\n"
    "    # 反向传播 + 更新\n"
    "    net.backward(X, y, lr=LR)\n"
    "    # 每 1000 步打印一次\n"
    "    if (epoch + 1) % 1000 == 0:\n"
    "        print(f'Epoch {epoch+1:5d} | Loss: {loss:.6f}')\n"
    "\n"
    "print(f'\\n训练完成！最终损失: {loss_history[-1]:.6f}')"
))

cells.append(code(
    "# 绘制训练损失曲线\n"
    "plt.figure(figsize=(8, 4))\n"
    "plt.plot(loss_history, color='steelblue', linewidth=1.5)\n"
    "plt.xlabel('训练轮次 (Epoch)')\n"
    "plt.ylabel('均方误差损失 (MSE)')\n"
    "plt.title('反向传播训练过程：损失随轮次下降')\n"
    "plt.yscale('log')\n"
    "plt.grid(True, alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.savefig('../docs/assets/03_loss_curve.png', dpi=100, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('损失从初始约 0.25 降至 <0.001，下降了约 250 倍')"
))

cells.append(markdown(
    "## 6. 验证：XOR 终于通了！\n\n"
    "17年前，单层感知机在这4个数据点上失败了。现在来看看两层网络表现如何。"
))

cells.append(code(
    "# 用训练好的网络预测\n"
    "y_pred = net.forward(X)\n"
    "y_binary = (y_pred > 0.5).astype(int)\n"
    "\n"
    "print('XOR 预测结果：')\n"
    "print(f'{\"输入\":<10} {\"真值\":<8} {\"预测概率\":<12} {\"预测标签\"}')\n"
    "print('-' * 45)\n"
    "all_correct = True\n"
    "for xi, yi, yp, yb in zip(X, y, y_pred, y_binary):\n"
    "    status = '✓' if yb[0] == int(yi[0]) else '✗'\n"
    "    print(f'{str(xi.astype(int)):<10} {int(yi[0]):<8} {yp[0]:.4f}       {yb[0]}  {status}')\n"
    "    if yb[0] != int(yi[0]):\n"
    "        all_correct = False\n"
    "\n"
    "accuracy = np.mean(y_binary == y.astype(int)) * 100\n"
    "print(f'\\n准确率: {accuracy:.0f}%')\n"
    "if all_correct:\n"
    "    print('XOR 问题完全解决！反向传播成功。')"
))

cells.append(code(
    "# 决策边界可视化\n"
    "xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 300),\n"
    "                     np.linspace(-0.5, 1.5, 300))\n"
    "grid = np.c_[xx.ravel(), yy.ravel()]\n"
    "probs = net.forward(grid).reshape(xx.shape)\n"
    "\n"
    "plt.figure(figsize=(6, 5))\n"
    "plt.contourf(xx, yy, probs, levels=20, cmap='RdYlBu_r', alpha=0.8)\n"
    "plt.colorbar(label='网络输出概率')\n"
    "\n"
    "# 画出 XOR 四个点\n"
    "colors = ['blue', 'red']\n"
    "labels_text = ['XOR=0', 'XOR=1']\n"
    "for i, (xi, yi) in enumerate(zip(X, y)):\n"
    "    label = labels_text[int(yi[0])] if i in [0, 3] or i == 1 else '_'\n"
    "    plt.scatter(xi[0], xi[1],\n"
    "                color=colors[int(yi[0])],\n"
    "                s=200, zorder=5,\n"
    "                label=label if i < 2 else '_')\n"
    "    plt.annotate(f'({int(xi[0])},{int(xi[1])})\\nXOR={int(yi[0])}',\n"
    "                 xy=(xi[0], xi[1]), xytext=(xi[0]+0.07, xi[1]+0.07),\n"
    "                 fontsize=9)\n"
    "\n"
    "plt.contour(xx, yy, probs, levels=[0.5], colors='white', linewidths=2)\n"
    "plt.title('两层网络的决策边界\\n（白线 = 0.5 概率分界）', fontsize=12)\n"
    "plt.xlabel('x1')\n"
    "plt.ylabel('x2')\n"
    "plt.legend(loc='upper right')\n"
    "plt.tight_layout()\n"
    "plt.savefig('../docs/assets/03_decision_boundary.png', dpi=100, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('注意：决策边界是非线性曲线，这是单层网络做不到的')"
))

cells.append(markdown(
    "## 7. 回顾与思考\n\n"
    "这个 notebook 演示了：\n\n"
    "1. **链式法则** 是反向传播的数学核心——误差信号可以从输出层逐层往回传\n"
    "2. **两层网络** 成功解决了 XOR，这在1969年的单层网络上是不可能的\n"
    "3. **sigmoid 导数最大只有 0.25**——想象网络有 10 层，梯度传到第1层只有原来的 $0.25^{10} \\approx 10^{-6}$，这就是**梯度消失**问题\n\n"
    "**动手试试**：把 `hidden_size` 改成 2、8、16，观察训练速度和最终精度的变化。\n\n"
    "> 下一节点：**LeNet（1989）** — Yann LeCun 把反向传播用到了卷积结构，机器第一次能识别手写数字"
))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "cells": cells
}

out = pathlib.Path("notebooks/03-backprop-1986.ipynb")
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
print(f"Written {out} ({out.stat().st_size} bytes, {len(cells)} cells)")
