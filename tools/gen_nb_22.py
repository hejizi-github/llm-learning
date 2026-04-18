"""
gen_nb_22.py — generate notebooks/22-lora-2021.ipynb
LoRA: Low-Rank Adaptation — core math and simulation.
Pure NumPy + matplotlib for 14-year-old readers.
"""
import json
import os


def cell(source, cell_type="code"):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = []

# ── Cell 1: Why LoRA? ─────────────────────────────────────────────────────
cells.append(cell("""\
# 节点22：LoRA — 大模型的"轻量级改装"（2021）

**论文**：LoRA: Low-Rank Adaptation of Large Language Models
**arXiv**：2106.09685 | **ICLR 2022**

## 为什么需要 LoRA？

想象一下：GPT-3 有 **1750 亿**个参数，全量微调需要几十张昂贵 GPU。
大多数研究者和公司根本负担不起。

LoRA 的思路：**不修改原始权重 W，而是在旁边学一个低秩"补丁" ΔW = B@A**

- B 形状：d×r（r 远小于 d）
- A 形状：r×k（r 远小于 k）
- 只训练 B 和 A，参数量从 d×k 压缩到 d×r + r×k

本 notebook 用纯 NumPy 演示 LoRA 的核心原理。\
""", "markdown"))

# ── Cell 2: Low-rank intuition ────────────────────────────────────────────
cells.append(cell("""\
## 矩阵低秩分解的直觉

把一个 100×100 的矩阵想象成"100个同学、100门考试的成绩单"。
真正的信息维度可能只有几个（比如：理科能力、文科能力、体育能力）。

**低秩分解**：用两个小矩阵 B（100×4）和 A（4×100）的乘积来近似整个大矩阵。

```
ΔW (100×100) ≈ B (100×4) @ A (4×100)
```

参数量：
- ΔW：100 × 100 = 10000 个
- B + A：100×4 + 4×100 = 800 个
- 压缩比：10000 ÷ 800 = 12.5x

**关键洞察**：微调时权重的"变化量"ΔW，通常是低秩的——不需要完整的矩阵来表示它。\
""", "markdown"))

# ── Cell 3: LoRA core demo ────────────────────────────────────────────────
cells.append(cell("""\
# 演示低秩矩阵：一个大矩阵可以用两个小矩阵近似
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
d, k, r = 100, 100, 4  # d×k矩阵，rank=r
A = np.random.randn(r, k) * 0.02  # 随机初始化，均值0
B = np.zeros((d, r))              # 初始化为0，保证ΔW=BA=0在开始时
delta_W = np.zeros((d, k))  # B@A = 零矩阵（B全为零，不触发matmul警告）
print(f"全量微调需要更新: {d*k} 个参数")
print(f"LoRA 只需要更新: {d*r + r*k} 个参数")
print(f"压缩比: {d*k / (d*r + r*k):.1f}x")
print(f"\\nB 形状: {B.shape}  (d×r)")
print(f"A 形状: {A.shape}  (r×k)")
print(f"ΔW = B@A 形状: {delta_W.shape}  (d×k)")
print(f"ΔW 的 Frobenius 范数: {np.linalg.norm(delta_W):.6f}  (全为零，训练起点=原模型)")
\
"""))

# ── Cell 4: Why B initialized to zero ────────────────────────────────────
cells.append(cell("""\
## 为什么 B 初始化为零？

**设计精髓**：训练开始时 B@A = 零矩阵，所以：

```
W_eff = W_pretrained + B @ A = W_pretrained + 0 = W_pretrained
```

这意味着：**LoRA 模型在训练开始时，行为与原始预训练模型完全相同。**

好处：
1. **稳定性**：从预训练模型的"好基线"开始，不引入随机噪声
2. **能力保留**：预训练知识被完整继承
3. **对比**：如果 B 也随机初始化，B@A 就是随机矩阵，会破坏原有能力

注意：A 用随机高斯初始化（非零），但乘以全零的 B，结果仍为零。
训练开始后，B 逐渐从零开始更新，"补丁"慢慢生长。\
""", "markdown"))

# ── Cell 5: LoRA vs full fine-tuning simulation ───────────────────────────
cells.append(cell("""\
# 模拟 LoRA 微调：在简单线性回归问题上对比全量微调 vs LoRA
#
# 关键设计：目标权重变化 W_target_delta 是 rank=4 的低秩矩阵。
# 这样 LoRA(rank=4) 理论上能完美拟合，演示才能说明 LoRA 有效。
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)  # 抑制特定BLAS版本的误报警告

np.random.seed(0)

# 问题设置：输入 x (k维)，输出 y (d维)，y = W_true @ x + noise
d_out, d_in = 20, 20  # 输出维度、输入维度
rank = 4              # LoRA rank

# 生成"真实目标"权重——关键：目标变化是低秩的（rank=4）
# 只有这样，rank=4 的 LoRA 才能理论上完美拟合，演示才有说服力
r_true = 4
W_target_delta = (np.random.randn(d_out, r_true) @ np.random.randn(r_true, d_in)) * 0.3

# 预训练权重（固定）
W_pretrained = np.random.randn(d_out, d_in)

# 生成训练数据
n_samples = 200
X = np.random.randn(d_in, n_samples)
Y = (W_pretrained + W_target_delta) @ X + np.random.randn(d_out, n_samples) * 0.1

# ── 全量微调 ──────────────────────────────────────────────────────────────
W_full = W_pretrained.copy()
lr_full = 0.005
losses_full = []

for step in range(300):
    Y_pred = W_full @ X
    error = Y_pred - Y
    loss = float(np.mean(error ** 2))
    losses_full.append(loss)
    grad = (2 / n_samples) * error @ X.T
    W_full -= lr_full * grad

# ── LoRA 微调 ─────────────────────────────────────────────────────────────
B_lora = np.zeros((d_out, rank))   # 初始为零
A_lora = np.random.randn(rank, d_in) * 0.02
lr_lora = 0.005
losses_lora = []

for step in range(300):
    W_eff = W_pretrained + B_lora @ A_lora
    Y_pred = W_eff @ X
    error = Y_pred - Y
    loss = float(np.mean(error ** 2))
    losses_lora.append(loss)
    # 梯度反传到 B 和 A（链式法则）
    grad_W = (2 / n_samples) * error @ X.T
    grad_B = grad_W @ A_lora.T
    grad_A = B_lora.T @ grad_W
    B_lora -= lr_lora * grad_B
    A_lora -= lr_lora * grad_A

print("=== 训练结果对比 ===")
print(f"全量微调  最终 Loss: {losses_full[-1]:.4f}  可训练参数: {d_out * d_in}")
print(f"LoRA r={rank}  最终 Loss: {losses_lora[-1]:.4f}  可训练参数: {d_out*rank + rank*d_in}")
print(f"\\nLoRA 压缩比: {d_out * d_in / (d_out*rank + rank*d_in):.1f}x")
print(f"性能差距: {abs(losses_lora[-1] - losses_full[-1]):.4f}  (目标是低秩的，LoRA 理应能接近全量微调)")
\
"""))

# ── Cell 6: Loss curve visualization ─────────────────────────────────────
cells.append(cell("""\
# 可视化：全量微调 vs LoRA 的 loss 曲线（基本重叠）
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

steps = range(1, 301)

# 左图：完整 loss 曲线
axes[0].plot(steps, losses_full, 'b-', linewidth=2, label=f'全量微调（{d_out*d_in}个参数）', alpha=0.8)
axes[0].plot(steps, losses_lora, 'r--', linewidth=2, label=f'LoRA r={rank}（{d_out*rank+rank*d_in}个参数）', alpha=0.8)
axes[0].set_xlabel('训练步数')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('全量微调 vs LoRA：Loss 曲线对比')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# 右图：最后 100 步放大，看收敛行为
axes[1].plot(range(201, 301), losses_full[200:], 'b-', linewidth=2,
             label=f'全量微调（最终：{losses_full[-1]:.4f}）')
axes[1].plot(range(201, 301), losses_lora[200:], 'r--', linewidth=2,
             label=f'LoRA r={rank}（最终：{losses_lora[-1]:.4f}）')
axes[1].set_xlabel('训练步数（后100步）')
axes[1].set_ylabel('MSE Loss')
axes[1].set_title('收敛阶段放大：低秩目标下两者接近')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../docs/assets/22-lora-loss.png', dpi=100, bbox_inches='tight')
plt.close()
print("图已保存：docs/assets/22-lora-loss.png")

# 验证
assert losses_lora[-1] < losses_lora[0] * 0.5, "LoRA 损失应明显下降"
assert losses_full[-1] < losses_full[0] * 0.5, "全量微调损失应明显下降"
print("验证通过：两种方法的 loss 均显著下降")
\
"""))

# ── Cell 7: Parameter efficiency across ranks ─────────────────────────────
cells.append(cell("""\
# 参数效率展示：不同 rank 下的参数量 vs 压缩比
ranks = [1, 2, 4, 8, 16]
print(f"{'rank':>6} | {'LoRA参数':>10} | {'压缩比':>8} | {'占全量比例':>10}")
print("-" * 44)
for r in ranks:
    params = d*r + r*k
    ratio = d*k / params
    pct = params / (d * k) * 100
    print(f"rank={r:2d} | {params:>10,} | {ratio:>7.1f}x | {pct:>9.2f}%")

print()
print(f"全量微调参数量: {d*k:,}")
print()

# 对于 GPT-3 规模（d=k=12288）的估算
d_gpt = 12288
print("GPT-3 单层 Attention 矩阵（d=k=12288）估算：")
print(f"{'rank':>6} | {'LoRA参数':>12} | {'压缩比':>8}")
print("-" * 34)
for r in [4, 8, 16, 32, 64]:
    params_gpt = d_gpt * r + r * d_gpt
    ratio_gpt = d_gpt * d_gpt / params_gpt
    print(f"rank={r:2d} | {params_gpt:>12,} | {ratio_gpt:>7.0f}x")
\
"""))

# ── Cell 8: Historical significance ──────────────────────────────────────
cells.append(cell("""\
## LoRA 的历史意义

### 从论文到生态系统的演变

| 时间 | 事件 |
|------|------|
| **2021-06** | LoRA 论文上传 arXiv（2106.09685） |
| **2022-05** | ICLR 2022 正式发表 |
| **2023-02** | Meta 发布 LLaMA，社区一周内出现 Alpaca-LoRA |
| **2023-03** | Stable Diffusion LoRA 微调成为标准，用户可以"训练自己的画风" |
| **2023-05** | QLoRA 发布：结合量化，单卡 24GB 可微调 65B 模型 |
| **至今** | HuggingFace PEFT 库将 LoRA 列为首选方法 |

### 为什么 LoRA 如此重要？

1. **民主化**：让个人研究者能微调大模型（之前只有大公司能做）
2. **模块化**：一个基础模型 + N 个小 LoRA 适配器，存储效率极高
3. **不降性能**：在多个基准上，LoRA 的效果接近甚至超过全量微调
4. **启发后续**：QLoRA、AdaLoRA、DoRA 等一系列工作都建立在 LoRA 的基础上

### 核心数字（来自原论文）

- GPT-3 175B，LoRA r=4：可训练参数 **< 0.01%**，ROUGE 分数持平全量微调
- RoBERTa、DeBERTa 等 BERT 类模型：LoRA 效果普遍优于 Adapter 方法
- 推理时无额外延迟：B@A 可以直接合并进 W，`W_merged = W + B@A`

**类比**：LoRA 就是大模型的"插件系统"——主程序不动，插件随意加载。\
""", "markdown"))

# ── Assemble and write ────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.8.0"},
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(__file__), "../notebooks/22-lora-2021.ipynb")
out_path = os.path.normpath(out_path)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
