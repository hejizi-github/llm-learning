"""gen_nb_25.py — generate notebooks/25-scaling-laws-2020.ipynb"""
import json, pathlib

NB_PATH = pathlib.Path(__file__).parent.parent / "notebooks" / "25-scaling-laws-2020.ipynb"

CELLS = [
    # ── Cell 0: title
    {"type": "markdown", "source": """\
# 节点25：Scaling Laws——越大越聪明，有规律可循

**来源**：Kaplan et al. (2020), arXiv:2001.08361

本 notebook 用 NumPy 演示幂律关系，帮你建立直觉：
为什么"训练更大的模型"往往是最优策略？

**目标**：理解幂律的形态，以及如何分配有限的计算预算。
"""},

    # ── Cell 1: imports
    {"type": "code", "source": """\
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 固定随机种子，保证可复现
rng = np.random.RandomState(42)
print("✓ 依赖加载完成")
"""},

    # ── Cell 2: power law basics
    {"type": "markdown", "source": """\
## 第一步：什么是幂律？

幂律的形式：$L = a \\cdot x^{-\\alpha}$

在对数-对数坐标下，这变成一条直线：$\\log L = \\log a - \\alpha \\cdot \\log x$

我们先用一个简单例子感受一下。
"""},

    # ── Cell 3: power law visualization
    {"type": "code", "source": """\
def power_law(x, a, alpha):
    \"\"\"幂律函数: L = a * x^(-alpha)\"\"\"
    return a * x ** (-alpha)

# 参数量从 1M 到 100B（跨越 5 个数量级）
N = np.logspace(6, 11, 100)  # 1e6 到 1e11

# Kaplan 2020 报告的参数量幂律指数 alpha_N ≈ 0.076
# 归一化常数 a 设为能让 N=1e6 时 Loss ≈ 4.0
a_N = 4.0 * (1e6 ** 0.076)
alpha_N = 0.076

L_N = power_law(N, a_N, alpha_N)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 线性坐标
axes[0].plot(N / 1e9, L_N, 'b-', linewidth=2)
axes[0].set_xlabel("参数量（十亿）")
axes[0].set_ylabel("Loss")
axes[0].set_title("线性坐标下的 Scaling Law")
axes[0].grid(True, alpha=0.3)

# 对数-对数坐标
axes[1].loglog(N, L_N, 'b-', linewidth=2)
axes[1].set_xlabel("参数量 N")
axes[1].set_ylabel("Loss L(N)")
axes[1].set_title("对数-对数坐标下变成直线！")
axes[1].grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig("../docs/assets/25-scaling-power-law.png", dpi=80, bbox_inches="tight")
plt.show()
print(f"当 N=1B 时，Loss = {power_law(1e9, a_N, alpha_N):.3f}")
print(f"当 N=10B 时，Loss = {power_law(1e10, a_N, alpha_N):.3f}")
print(f"参数量增加10倍，Loss 减少了 {(1 - power_law(1e10, a_N, alpha_N)/power_law(1e9, a_N, alpha_N))*100:.1f}%")
"""},

    # ── Cell 4: three laws comparison
    {"type": "markdown", "source": """\
## 第二步：三大 Scaling Laws 对比

Kaplan et al. 发现了三条独立的幂律：

| 变量 | 幂律指数 | 含义 |
|------|---------|------|
| 参数量 N | α_N ≈ 0.076 | 参数量翻10倍，Loss降16% |
| 数据量 D | α_D ≈ 0.095 | 数据量翻10倍，Loss降20% |
| 计算量 C | α_C ≈ 0.050 | 计算量翻10倍，Loss降11% |

**注意**：这些系数来自论文中跨越7个数量级的大规模实验，我们这里只是演示它们的形态。
"""},

    # ── Cell 5: compare three laws
    {"type": "code", "source": """\
# 三条幂律的对比
x = np.logspace(0, 7, 100)  # 1 到 1e7 倍基准

# 归一化：x=1 时 Loss 都从 4.0 开始
L_params = 4.0 * x ** (-0.076)   # 参数量 scaling
L_data   = 4.0 * x ** (-0.095)   # 数据量 scaling
L_compute = 4.0 * x ** (-0.050)  # 计算量 scaling

fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(x, L_params,   'b-',  linewidth=2, label=f'参数量 (α={0.076})')
ax.loglog(x, L_data,     'r-',  linewidth=2, label=f'数据量 (α={0.095})')
ax.loglog(x, L_compute,  'g-',  linewidth=2, label=f'计算量 (α={0.050})')

ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='基准100倍')
ax.set_xlabel("相对基准的倍数（对数轴）")
ax.set_ylabel("Loss（对数轴）")
ax.set_title("三大 Scaling Laws 对比")
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig("../docs/assets/25-three-laws.png", dpi=80, bbox_inches="tight")
plt.show()

print("在相同倍数下，哪个最有效？")
x_100 = 100  # 增加100倍
for name, alpha in [("参数量", 0.076), ("数据量", 0.095), ("计算量", 0.050)]:
    improvement = (1 - x_100**(-alpha)) * 100
    print(f"  {name} ×100 → Loss 降低 {improvement:.1f}%")
"""},

    # ── Cell 6: optimal allocation
    {"type": "markdown", "source": """\
## 第三步：最优计算分配

假设你有固定的计算预算 C，应该怎么分配？

**Kaplan 2020 的结论**：$N_{\\text{opt}} \\propto C^{0.73}$

这意味着：计算量翻 10 倍时，最优模型大小增加约 5.4 倍（不是 10 倍！）。

**Chinchilla 2022 的修正**：最优比例约为每个参数 20 个训练 token。
"""},

    # ── Cell 7: optimal model size
    {"type": "code", "source": """\
def kaplan_optimal_N(C, scale=1e8):
    \"\"\"Kaplan 2020: N_opt ∝ C^0.73\"\"\"
    return scale * C ** 0.73

def chinchilla_optimal_N(C, tokens_per_param=20):
    \"\"\"Chinchilla 2022: 每个参数约20个token
    C ≈ 6 * N * D (每个参数每个token约6 FLOPs)
    D = 20 * N → C ≈ 6 * N * 20 * N = 120 * N^2
    → N = sqrt(C / 120)
    \"\"\"
    return np.sqrt(C / 120)

# 计算预算从 1e18 到 1e24 FLOPs
C = np.logspace(18, 24, 50)

N_kaplan = kaplan_optimal_N(C)
N_chinchilla = chinchilla_optimal_N(C)

fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(C, N_kaplan,     'b-', linewidth=2, label='Kaplan 2020: N ∝ C^0.73')
ax.loglog(C, N_chinchilla, 'r-', linewidth=2, label='Chinchilla 2022: N = √(C/120)')

# 标注 GPT-3
gpt3_C = 3e23
gpt3_N = 175e9
ax.scatter([gpt3_C], [gpt3_N], color='orange', s=100, zorder=5)
ax.annotate('GPT-3 (175B)', xy=(gpt3_C, gpt3_N),
            xytext=(gpt3_C * 3, gpt3_N * 0.3),
            arrowprops=dict(arrowstyle='->', color='orange'),
            fontsize=9, color='orange')

ax.set_xlabel("计算预算 C (FLOPs)")
ax.set_ylabel("最优模型大小 N（参数量）")
ax.set_title("Kaplan vs Chinchilla：如何分配计算预算？")
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig("../docs/assets/25-optimal-allocation.png", dpi=80, bbox_inches="tight")
plt.show()

# 在 GPT-3 的计算预算下，Chinchilla 建议的最优模型大小
chin_N = chinchilla_optimal_N(gpt3_C)
print(f"在 GPT-3 的计算预算 ({gpt3_C:.1e} FLOPs) 下：")
print(f"  Kaplan 建议：~{kaplan_optimal_N(gpt3_C)/1e9:.0f}B 参数")
print(f"  Chinchilla 建议：~{chin_N/1e9:.1f}B 参数（实际训练了 70B params × 1.4T tokens）")
print(f"  GPT-3 实际：175B 参数（Kaplan 预测是对的，但数据量不够）")
"""},

    # ── Cell 8: fitting power law from data
    {"type": "markdown", "source": """\
## 第四步：从数据中拟合幂律

真正的研究者是如何发现这些幂律的？他们训练了很多不同大小的模型，然后拟合曲线。

我们来模拟这个过程：生成一些假设的"实验数据"，然后用最小二乘法拟合幂律。
"""},

    # ── Cell 9: fitting
    {"type": "code", "source": """\
from numpy.polynomial import polynomial as P

# 模拟实验数据：不同参数量下的 Loss
# 真实值基于幂律 + 添加一点噪声（模拟真实实验的波动）
true_alpha = 0.076
true_a = 1.0

N_exp = np.array([1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10])
noise = rng.randn(len(N_exp)) * 0.02  # 2% 噪声
L_exp = true_a * N_exp ** (-true_alpha) * (1 + noise)

# 在对数空间拟合直线 (log L = b + m * log N)
log_N = np.log10(N_exp)
log_L = np.log10(L_exp)

# 最小二乘拟合
coeffs = np.polyfit(log_N, log_L, 1)
fitted_alpha = -coeffs[0]  # 斜率的绝对值就是 alpha
fitted_a = 10 ** coeffs[1]

print(f"真实 alpha = {true_alpha}")
print(f"拟合 alpha = {fitted_alpha:.4f}")
print(f"误差 = {abs(fitted_alpha - true_alpha)/true_alpha*100:.1f}%")

# 可视化拟合结果
N_line = np.logspace(7, 10, 100)
L_fitted = fitted_a * N_line ** (-fitted_alpha)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(N_exp, L_exp, 'ko', markersize=8, label='模拟实验数据', zorder=5)
ax.loglog(N_line, L_fitted, 'b-', linewidth=2,
          label=f'拟合结果: α = {fitted_alpha:.4f}')
ax.loglog(N_line, true_a * N_line ** (-true_alpha), 'r--', linewidth=1.5,
          label=f'真实值: α = {true_alpha}', alpha=0.7)

ax.set_xlabel("参数量 N")
ax.set_ylabel("Loss")
ax.set_title("从实验数据拟合幂律指数")
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig("../docs/assets/25-fitting.png", dpi=80, bbox_inches="tight")
plt.show()
"""},

    # ── Cell 10: sample efficiency
    {"type": "markdown", "source": """\
## 第五步：大模型更"样本高效"

Kaplan 等人还发现一个反直觉的结论：**大模型用同样数量的训练步骤，Loss 下降得更快。**

这叫做**样本高效性（Sample Efficiency）**。换句话说：如果你只能训练 1000 步，大模型比小模型更划算！
"""},

    # ── Cell 11: sample efficiency demo
    {"type": "code", "source": """\
def training_loss(steps, N, loss0=5.0, alpha=0.4):
    \"\"\"模拟训练曲线：Loss 随步数下降
    大模型（大N）下降更快（alpha 更大）
    这里用简化模型演示趋势
    \"\"\"
    # 大模型的等效步数因子
    size_factor = (N / 1e8) ** 0.2
    effective_steps = steps * size_factor
    return loss0 / (1 + effective_steps / 100) ** alpha

steps = np.linspace(0, 2000, 200)
model_sizes = {
    "1亿参数 (100M)": 1e8,
    "10亿参数 (1B)":  1e9,
    "100亿参数 (10B)": 1e10,
}

fig, ax = plt.subplots(figsize=(8, 5))
for name, N in model_sizes.items():
    losses = [training_loss(s, N) for s in steps]
    ax.plot(steps, losses, linewidth=2, label=name)

ax.axvline(x=500, color='gray', linestyle='--', alpha=0.5)
ax.text(510, 4.5, '500步时比较', fontsize=9, color='gray')
ax.set_xlabel("训练步数")
ax.set_ylabel("Loss")
ax.set_title("大模型 vs 小模型：相同步数下 Loss 更低（样本高效）")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(1, 6)

plt.tight_layout()
plt.savefig("../docs/assets/25-sample-efficiency.png", dpi=80, bbox_inches="tight")
plt.show()

# 在500步时的比较
print("在500步时的 Loss：")
for name, N in model_sizes.items():
    l = training_loss(500, N)
    print(f"  {name}: Loss = {l:.3f}")
"""},

    # ── Cell 12: summary
    {"type": "markdown", "source": """\
## 总结

| 核心发现 | 含义 |
|---------|------|
| Loss ∝ N^(-0.076) | 参数量翻10倍，Loss降16% |
| Loss ∝ D^(-0.095) | 数据量翻10倍，Loss降20% |
| Loss ∝ C^(-0.050) | 计算量翻10倍，Loss降11% |
| N_opt ∝ C^0.73 (Kaplan) | 更多计算 → 更大的模型 |
| N:D = 1:20 (Chinchilla) | 每个参数需要20个token |

**最重要的直觉**：
- AI 能力的提升遵循**可预测的幂律**
- 这意味着 AI 进步是可以**规划和预算**的
- Scaling Laws 是 GPT-3、GPT-4 等大模型存在的理论基础

**下一节**：[节点26] DeepSeek——如何用更少的成本训练出更强的模型？
"""},
]


def make_cell(c):
    if c["type"] == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": c["source"].splitlines(keepends=True),
        }
    else:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": c["source"].splitlines(keepends=True),
        }


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.8.0"},
    },
    "cells": [make_cell(c) for c in CELLS],
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Written: {NB_PATH}")
