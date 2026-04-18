"""
gen_nb_23.py — generate notebooks/23-chain-of-thought-2022.ipynb
Chain-of-Thought Prompting: simulate standard vs CoT reasoning.
Pure Python/NumPy for 14-year-old readers.
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

# ── Cell 0: Title ──────────────────────────────────────────────────────────
cells.append(cell("""\
# 节点23：Chain-of-Thought Prompting（2022）——让大模型"先想再答"

**论文**：Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
**arXiv**：2201.11903  |  作者：Jason Wei 等  |  发表于 NeurIPS 2022

本 notebook 演示 CoT 的核心思想：
1. **直接回答**（Standard）：一步给答案，容易出错
2. **链式推理**（CoT）：分步计算，准确率大幅提升
3. **准确率对比**：可视化两种方法的差距
4. **Zero-shot CoT**："Let's think step by step" 的魔法\
""", "markdown"))

# ── Cell 1: Imports ────────────────────────────────────────────────────────
cells.append(cell("""\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
print("环境就绪，NumPy 版本:", np.__version__)\
"""))

# ── Cell 2: Problem setup intro ────────────────────────────────────────────
cells.append(cell("""\
## Part 1：问题设置——模拟多步数学推理

我们模拟一个场景：让 AI 回答多步算术题（先加法再减法）。

**题目类型（两步运算）**：
```
"有 a 个苹果，买了 b 个，吃了 c 个，还有几个？"
答案 = a + b - c
```

**两种策略**：
- **直接回答（Standard）**：猜一个整数，误差±2之内算对（模拟小模型一步跳答）
- **链式推理（CoT）**：分步计算（先加，再减），只在最后一步允许小误差

我们生成 200 道题，看两种策略的准确率。\
""", "markdown"))

# ── Cell 3: Simulation setup ───────────────────────────────────────────────
cells.append(cell("""\
# 生成 200 道两步算术题
n_problems = 200

# 随机生成题目参数（小整数，避免溢出）
a = np.random.randint(5, 20, size=n_problems)   # 初始数量
b = np.random.randint(1, 10, size=n_problems)    # 买入数量
c = np.random.randint(1, 8,  size=n_problems)    # 消耗数量（保证 a+b-c > 0）

true_answers = a + b - c   # 正确答案

print(f"生成了 {n_problems} 道题")
print(f"示例前3道：")
for i in range(3):
    print(f"  题{i+1}: {a[i]}个苹果，买了{b[i]}个，吃了{c[i]}个 → 正确答案: {true_answers[i]}")
print(f"答案范围：{true_answers.min()} ~ {true_answers.max()}")\
"""))

# ── Cell 4a: Standard prompting intro ─────────────────────────────────────
cells.append(cell("""\
## Part 2：直接回答策略（Standard Prompting）

模拟小模型直接猜答案的行为：
- 它"知道"大概的数量级（基于 a 的大小猜），但会有随机误差
- 误差在 ±(a//3) 之间，模拟没有分步推理时的不稳定性\
""", "markdown"))

# ── Cell 4b: Standard prompting code ──────────────────────────────────────
cells.append(cell("""\
# 直接回答模拟：模型对整个 a+b-c 一步猜，误差随步骤数增加而扩大
error_range = np.maximum(2, a // 3)  # 误差范围与初始量级相关
noise = np.array([
    np.random.randint(-error_range[i], error_range[i] + 1)
    for i in range(n_problems)
])
standard_answers = true_answers + noise

# 正确 = 误差 ≤ 1（允许±1的容忍）
standard_correct = np.abs(standard_answers - true_answers) <= 1

standard_accuracy = standard_correct.mean()
print("=== 直接回答（Standard Prompting）===")
print(f"准确率：{standard_accuracy:.1%}")
print(f"答对：{standard_correct.sum()} / {n_problems}")
print()
print("示例错误案例（前5个错误）：")
wrong_idx = np.where(~standard_correct)[0][:5]
for i in wrong_idx:
    print(f"  题{i+1}: 正确答案={true_answers[i]}，"
          f"模型猜={standard_answers[i]}，"
          f"误差={standard_answers[i]-true_answers[i]:+d}")\
"""))

# ── Cell 5a: CoT intro ────────────────────────────────────────────────────
cells.append(cell("""\
## Part 3：链式推理策略（Chain-of-Thought Prompting）

CoT 强制模型分两步计算：
1. **Step 1**：先算加法 `step1 = a + b`
2. **Step 2**：再算减法 `step2 = step1 - c`

每步只有±1的误差（因为每步都很简单），最终误差更小。\
""", "markdown"))

# ── Cell 5b: CoT code ─────────────────────────────────────────────────────
cells.append(cell("""\
# CoT 模拟：分两步，每步只有小误差
# Step 1: a + b（简单加法，允许±1误差）
step1_noise = np.random.randint(-1, 2, size=n_problems)
step1 = (a + b) + step1_noise    # 中间结果

# Step 2: step1 - c（基于中间结果，允许±1误差）
step2_noise = np.random.randint(-1, 2, size=n_problems)
cot_answers = step1 - c + step2_noise  # 最终答案

# 正确 = 误差 ≤ 1
cot_correct = np.abs(cot_answers - true_answers) <= 1

cot_accuracy = cot_correct.mean()
print("=== 链式推理（Chain-of-Thought）===")
print(f"准确率：{cot_accuracy:.1%}")
print(f"答对：{cot_correct.sum()} / {n_problems}")
print()
print(f"准确率提升：{cot_accuracy - standard_accuracy:+.1%}")
print(f"提升倍数：{cot_accuracy / standard_accuracy:.2f}×")
print()
print("CoT 思路（展示第一道题的推理过程）：")
i = 0
print(f"  问题：有 {a[i]} 个苹果，买了 {b[i]} 个，吃了 {c[i]} 个，还有几个？")
print(f"  Step 1：{a[i]} + {b[i]} = {step1[i]}（中间结果）")
print(f"  Step 2：{step1[i]} - {c[i]} = {cot_answers[i]}")
print(f"  正确答案：{true_answers[i]}")\
"""))

# ── Cell 6: Visualization ─────────────────────────────────────────────────
cells.append(cell("""\
# 可视化：准确率对比 + 误差分布

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：准确率柱状图
methods = ['直接回答\\n(Standard)', '链式推理\\n(CoT)']
accuracies = [standard_accuracy * 100, cot_accuracy * 100]
colors = ['#e74c3c', '#2ecc71']

bars = axes[0].bar(methods, accuracies, color=colors, width=0.4, edgecolor='black')
axes[0].set_ylim(0, 100)
axes[0].set_ylabel('准确率 (%)')
axes[0].set_title('Standard vs CoT：准确率对比')
for bar, acc in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# 右图：误差分布直方图
std_errors = standard_answers - true_answers
cot_errors = cot_answers - true_answers

axes[1].hist(std_errors, bins=range(-8, 9), alpha=0.6, color='#e74c3c',
             label='直接回答', edgecolor='black')
axes[1].hist(cot_errors, bins=range(-8, 9), alpha=0.6, color='#2ecc71',
             label='CoT', edgecolor='black')
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='完全正确')
axes[1].set_xlabel('误差（预测值 - 正确答案）')
axes[1].set_ylabel('题目数量')
axes[1].set_title('误差分布：CoT 更集中在 0 附近')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../docs/assets/23-cot-accuracy.png', dpi=100, bbox_inches='tight')
plt.show()
print("图像已保存")\
"""))

# ── Cell 7: Zero-shot CoT ─────────────────────────────────────────────────
cells.append(cell("""\
## Part 4：Zero-shot CoT——"Let's think step by step" 的威力

Kojima 等人（arXiv:2205.11916）发现：不需要给例子，只需在问题后加一句话，
大模型就会自动生成推理步骤。

下面模拟 Zero-shot CoT 的"魔法咒语"效果：
- **不加咒语**：模型随机选择"直接回答"或"推理"（各50%概率）
- **加了咒语**：模型几乎总是使用分步推理（90%概率）\
""", "markdown"))

cells.append(cell("""\
# 模拟 Zero-shot CoT 的效果
n_test = 1000

# 不加 "Let's think step by step"：模型随机选策略
# 假设50%概率用CoT，50%用直接回答
def solve_without_cue(a_val, b_val, c_val):
    correct = a_val + b_val - c_val
    if np.random.random() < 0.5:
        # CoT路径
        s1 = a_val + b_val + np.random.randint(-1, 2)
        return s1 - c_val + np.random.randint(-1, 2)
    else:
        # 直接猜
        err = np.random.randint(-max(2, a_val//3), max(2, a_val//3)+1)
        return correct + err

# 加了 "Let's think step by step"：模型90%概率用CoT
def solve_with_cue(a_val, b_val, c_val):
    correct = a_val + b_val - c_val
    if np.random.random() < 0.9:  # 90%用CoT
        s1 = a_val + b_val + np.random.randint(-1, 2)
        return s1 - c_val + np.random.randint(-1, 2)
    else:
        err = np.random.randint(-max(2, a_val//3), max(2, a_val//3)+1)
        return correct + err

# 测试
a_t = np.random.randint(5, 20, n_test)
b_t = np.random.randint(1, 10, n_test)
c_t = np.random.randint(1, 8, n_test)
true_t = a_t + b_t - c_t

no_cue_answers = np.array([solve_without_cue(a_t[i], b_t[i], c_t[i]) for i in range(n_test)])
with_cue_answers = np.array([solve_with_cue(a_t[i], b_t[i], c_t[i]) for i in range(n_test)])

no_cue_acc = (np.abs(no_cue_answers - true_t) <= 1).mean()
with_cue_acc = (np.abs(with_cue_answers - true_t) <= 1).mean()

print("=== Zero-shot CoT 效果 ===")
print(f"不加咒语（随机策略）：{no_cue_acc:.1%}")
print(f"加了咒语 (Let us think step by step)：{with_cue_acc:.1%}")
print(f"提升：{with_cue_acc - no_cue_acc:+.1%}")
print()
print("总结：")
print("  CoT 不是让模型更聪明，而是给它一个指引——先想，再回答。")
print("  强制分步能把多步推理错误从每步叠加变成每步独立的小误差。")
print("  这就是为什么一句提示语就能显著提升准确率。")\
"""))

# ── Cell 8: Why CoT works summary ─────────────────────────────────────────
cells.append(cell("""\
## 总结：CoT 的本质

```
直接回答：一步跳到答案，N步的误差全叠加
              误差 ≈ ε₁ + ε₂ + ... + εₙ  （累加，越来越大）

CoT推理：  每步一个小答案，每步误差独立
              误差 ≈ ε₁ 或 ε₂ 或 ... 或 εₙ  （独立，总体更小）
```

核心论文发现：
- GPT-3 + CoT 在 GSM8K 数学题：准确率从 17.9% → 46.9%（提升 2.6×）
- 效果随模型规模扩大：100B+ 参数才有明显提升
- Zero-shot CoT（Kojima 2022）：一句话"Let's think step by step"即可触发

历史意义：CoT 开启了"提示工程"时代——不改变模型权重，只改变提问方式，
就能解锁大模型隐藏的推理能力。\
""", "markdown"))

# ── Notebook structure ─────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"},
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(__file__), "../notebooks/23-chain-of-thought-2022.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written: {os.path.abspath(out_path)}")
print(f"Cells: {len(cells)}")
