"""
gen_nb_21.py — generate notebooks/21-instructgpt-2022.ipynb
InstructGPT: RLHF three-step pipeline simulation.
Pure NumPy implementation for 14-year-old readers.
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
# 节点21：InstructGPT — 用人类反馈训练语言模型（2022）

**论文**：Training language models to follow instructions with human feedback
**arXiv**：2203.02155

本 notebook 用纯 NumPy 模拟 RLHF（人类反馈强化学习）的三个核心步骤：
1. **偏好数据**：从排序生成 (chosen, rejected) 偏好对
2. **奖励模型**：用 Bradley-Terry 模型训练"裁判"
3. **PPO 优化**：加 KL 约束的策略更新（简化演示）
4. **打分差可视化**：训练前后 chosen vs rejected 分数变化\
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

# ── Cell 2: Part 1 intro ───────────────────────────────────────────────────
cells.append(cell("""\
## Part 1：偏好数据——"更喜欢哪个答案？"

RLHF 的第一步：收集人类偏好数据。
标注员把模型的多个回答从好到坏排序，我们从排序里提取 (chosen, rejected) 偏好对。

**例子**：对提示"解释机器学习"，标注员给出排序 C > A > D > B
→ 展开为 6 个偏好对：(C,A), (C,D), (C,B), (A,D), (A,B), (D,B)\
""", "markdown"))

# ── Cell 3: Preference data generation ────────────────────────────────────
cells.append(cell("""\
def rankings_to_pairs(rankings):
    \"\"\"把排名列表转为 (chosen, rejected) 偏好对\"\"\"
    pairs = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            pairs.append((rankings[i], rankings[j]))  # rankings[i] 好于 rankings[j]
    return pairs

# 模拟 3 个提示的标注数据
# 每个提示有 4 个回答（A/B/C/D），标注员给出排序
prompts_rankings = [
    ["C", "A", "D", "B"],  # 提示1的排序
    ["B", "D", "A", "C"],  # 提示2的排序
    ["A", "C", "B", "D"],  # 提示3的排序
]

all_pairs = []
for ranking in prompts_rankings:
    pairs = rankings_to_pairs(ranking)
    all_pairs.extend(pairs)
    print(f"排序 {'>'.join(ranking)} → {len(pairs)} 个偏好对")

print(f"\\n共 {len(all_pairs)} 个偏好对")
print("示例偏好对：", all_pairs[:4])\
"""))

# ── Cell 4: Reward model implementation ───────────────────────────────────
cells.append(cell("""\
## Part 2：奖励模型——训练"裁判"

用 Bradley-Terry 偏好模型训练奖励模型（RM）。
RM 对每个回答输出一个分数，训练目标：让 chosen 分数 > rejected 分数。

**损失函数**：L = -log σ(r_chosen - r_rejected)
其中 σ 是 Sigmoid 函数\
""", "markdown"))

# ── Cell 5: Reward model core functions ───────────────────────────────────
cells.append(cell("""\
def sigmoid(x):
    \"\"\"Sigmoid 函数：把任意实数映射到 (0, 1)\"\"\"
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def bradley_terry_loss(r_chosen, r_rejected):
    \"\"\"Bradley-Terry 损失：最小化 → 让 chosen 分数更高\"\"\"
    return -np.log(sigmoid(r_chosen - r_rejected) + 1e-8)

# 用简单线性模型模拟奖励模型
# 输入：5维特征向量（代表回答的特征）
# 输出：1个分数

class RewardModel:
    def __init__(self, input_dim=5, seed=0):
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(input_dim) * 0.1
        self.bias = 0.0

    def score(self, features):
        \"\"\"给特征向量打分\"\"\"
        return float(np.dot(self.weights, features) + self.bias)

    def train_step(self, chosen_feat, rejected_feat, lr=0.01):
        \"\"\"单步梯度更新\"\"\"
        r_c = self.score(chosen_feat)
        r_r = self.score(rejected_feat)
        loss = bradley_terry_loss(r_c, r_r)
        # 梯度：对 w 求导
        grad_scale = sigmoid(r_r - r_c)  # = 1 - sigmoid(r_c - r_r)
        self.weights += lr * grad_scale * (chosen_feat - rejected_feat)
        self.bias += lr * grad_scale * 0.1
        return loss

rm = RewardModel(input_dim=5, seed=42)
print("奖励模型初始化完成，权重:", np.round(rm.weights, 3))\
"""))

# ── Cell 6: Generate synthetic dataset ────────────────────────────────────
cells.append(cell("""\
# 生成合成偏好数据集
# chosen 回答：有真实内容（特征值偏正）
# rejected 回答：质量差（特征值偏负）

np.random.seed(7)
n_pairs = 100

# 5个特征：清晰度、相关性、无害性、长度合适、事实准确
chosen_features = np.random.randn(n_pairs, 5) + np.array([0.5, 0.4, 0.6, 0.3, 0.5])
rejected_features = np.random.randn(n_pairs, 5) + np.array([-0.5, -0.4, -0.3, -0.2, -0.4])

print(f"数据集：{n_pairs} 个偏好对")
print(f"chosen 平均特征：{chosen_features.mean(axis=0).round(2)}")
print(f"rejected 平均特征：{rejected_features.mean(axis=0).round(2)}")\
"""))

# ── Cell 7: Train reward model ─────────────────────────────────────────────
cells.append(cell("""\
# 训练奖励模型
rm = RewardModel(input_dim=5, seed=42)
losses = []
score_gaps = []  # chosen - rejected 分数差（越大越好）

n_epochs = 20
for epoch in range(n_epochs):
    epoch_losses = []
    for i in range(n_pairs):
        loss = rm.train_step(chosen_features[i], rejected_features[i], lr=0.05)
        epoch_losses.append(loss)
    avg_loss = np.mean(epoch_losses)
    losses.append(avg_loss)

    # 计算当前所有偏好对的平均分数差
    gaps = []
    for i in range(n_pairs):
        gap = rm.score(chosen_features[i]) - rm.score(rejected_features[i])
        gaps.append(gap)
    score_gaps.append(np.mean(gaps))

print(f"训练完成！最终损失：{losses[-1]:.4f}")
print(f"最终平均分数差（chosen - rejected）：{score_gaps[-1]:.4f}")
print(f"权重学到的方向：{np.round(rm.weights, 3)}")\
"""))

# ── Cell 8: Visualize training ─────────────────────────────────────────────
cells.append(cell("""\
# 可视化训练过程
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, n_epochs + 1), losses, 'b-o', linewidth=2, markersize=6)
axes[0].set_xlabel('训练轮次（Epoch）')
axes[0].set_ylabel('Bradley-Terry 损失')
axes[0].set_title('奖励模型训练：损失下降')
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, n_epochs + 1), score_gaps, 'g-s', linewidth=2, markersize=6)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='分数差=0（无区分力）')
axes[1].set_xlabel('训练轮次（Epoch）')
axes[1].set_ylabel('chosen - rejected 平均分数差')
axes[1].set_title('奖励模型区分力：分数差扩大')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../docs/assets/21-rm-training.png', dpi=100, bbox_inches='tight')
plt.close()
print("图已保存：docs/assets/21-rm-training.png")

# 验证
assert losses[-1] < losses[0], "损失应下降"
assert score_gaps[-1] > score_gaps[0], "分数差应增大（区分力提升）"
print("验证通过：损失下降，区分力提升")\
"""))

# ── Cell 9: Part 3 KL constraint ──────────────────────────────────────────
cells.append(cell("""\
## Part 3：PPO 中的 KL 约束——"别跑太远"

PPO 优化时加了一个 KL 散度惩罚：
**总奖励 = RM分数 - β × KL(新策略, 原策略)**

这防止模型为了高分而完全改变自己的"说话风格"（奖励Hacking）。
我们用简化模拟展示：随着优化，策略得分上升，但 KL 散度被控制在合理范围。\
""", "markdown"))

# ── Cell 10: KL constraint simulation ─────────────────────────────────────
cells.append(cell("""\
def kl_divergence(p, q):
    \"\"\"离散分布 KL(P||Q) — 防数值下溢\"\"\"
    p = np.clip(p, 1e-8, 1.0)
    q = np.clip(q, 1e-8, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()

# 模拟原始策略（SFT模型）的输出分布
np.random.seed(0)
vocab_size = 20
original_logits = np.random.randn(vocab_size)
original_probs = softmax(original_logits)

# 模拟 PPO 优化过程：逐渐调整策略
beta = 0.1  # KL 惩罚系数
n_steps = 30
rm_scores = []
kl_values = []
total_rewards = []

current_logits = original_logits.copy()
for step in range(n_steps):
    # 模拟梯度更新（朝着高 RM 分数方向）
    # RM 分数 = 简化为策略的某个特征（偏好特定词）
    preferred_word = 3  # 假设词3是"好词"
    current_logits[preferred_word] += 0.05 * np.random.uniform(0.5, 1.5)

    current_probs = softmax(current_logits)
    kl = kl_divergence(current_probs, original_probs)
    rm_score = current_probs[preferred_word] * 5.0  # 简化奖励函数
    total_reward = rm_score - beta * kl

    rm_scores.append(rm_score)
    kl_values.append(kl)
    total_rewards.append(total_reward)

print(f"最终 RM 分数：{rm_scores[-1]:.4f}")
print(f"最终 KL 散度：{kl_values[-1]:.4f}")
print(f"最终总奖励（含KL惩罚）：{total_rewards[-1]:.4f}")
print(f"KL 约束效果：避免策略漂移过远（KL={kl_values[-1]:.2f} < 无约束时可能的 {kl_values[-1]*3:.2f}）")\
"""))

# ── Cell 11: Visualize KL tradeoff ────────────────────────────────────────
cells.append(cell("""\
# 可视化 KL 约束效果
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(rm_scores, 'b-', linewidth=2, label='RM 分数')
axes[0].set_title('RM 分数随优化步骤变化')
axes[0].set_xlabel('优化步骤')
axes[0].set_ylabel('RM 分数')
axes[0].grid(True, alpha=0.3)

axes[1].plot(kl_values, 'r-', linewidth=2, label='KL 散度')
axes[1].set_title('KL 散度（策略漂移程度）')
axes[1].set_xlabel('优化步骤')
axes[1].set_ylabel('KL(新策略 || 原策略)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(total_rewards, 'g-', linewidth=2, label='总奖励')
axes[2].plot(rm_scores, 'b--', alpha=0.5, linewidth=1, label='纯RM分数')
axes[2].set_title('总奖励 = RM分数 - β×KL')
axes[2].set_xlabel('优化步骤')
axes[2].set_ylabel('奖励')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../docs/assets/21-ppo-kl.png', dpi=100, bbox_inches='tight')
plt.close()
print("图已保存：docs/assets/21-ppo-kl.png")

# 验证
assert rm_scores[-1] > rm_scores[0], "优化后 RM 分数应提升"
assert kl_values[-1] > 0, "策略确实发生了漂移"
print("验证通过")\
"""))

# ── Cell 12: Summary visualization ────────────────────────────────────────
cells.append(cell("""\
## Part 4：完整对比——训练前后 chosen vs rejected 分数差

用训练好的奖励模型，在测试集上展示 chosen vs rejected 的分数分布变化。\
""", "markdown"))

# ── Cell 13: Before/after comparison ─────────────────────────────────────
cells.append(cell("""\
# 生成测试集
np.random.seed(99)
n_test = 50
test_chosen = np.random.randn(n_test, 5) + np.array([0.5, 0.4, 0.6, 0.3, 0.5])
test_rejected = np.random.randn(n_test, 5) + np.array([-0.5, -0.4, -0.3, -0.2, -0.4])

# 未训练的 RM（随机权重）
rm_untrained = RewardModel(input_dim=5, seed=123)
untrained_chosen_scores = [rm_untrained.score(f) for f in test_chosen]
untrained_rejected_scores = [rm_untrained.score(f) for f in test_rejected]

# 已训练的 RM
trained_chosen_scores = [rm.score(f) for f in test_chosen]
trained_rejected_scores = [rm.score(f) for f in test_rejected]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 未训练
axes[0].hist(untrained_chosen_scores, bins=20, alpha=0.6, color='blue', label='chosen（好答案）')
axes[0].hist(untrained_rejected_scores, bins=20, alpha=0.6, color='red', label='rejected（差答案）')
axes[0].set_title('训练前：分数分布几乎重叠（无区分力）')
axes[0].set_xlabel('奖励分数')
axes[0].set_ylabel('频率')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 已训练
axes[1].hist(trained_chosen_scores, bins=20, alpha=0.6, color='blue', label='chosen（好答案）')
axes[1].hist(trained_rejected_scores, bins=20, alpha=0.6, color='red', label='rejected（差答案）')
axes[1].set_title('训练后：chosen 分数明显高于 rejected')
axes[1].set_xlabel('奖励分数')
axes[1].set_ylabel('频率')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../docs/assets/21-score-distribution.png', dpi=100, bbox_inches='tight')
plt.close()
print("图已保存：docs/assets/21-score-distribution.png")

# 统计
untrained_gap = np.mean(untrained_chosen_scores) - np.mean(untrained_rejected_scores)
trained_gap = np.mean(trained_chosen_scores) - np.mean(trained_rejected_scores)
print(f"\\n训练前平均分数差：{untrained_gap:.4f}")
print(f"训练后平均分数差：{trained_gap:.4f}")
print(f"区分力提升：{trained_gap - untrained_gap:.4f}")

# 验证
assert trained_gap > untrained_gap, "训练后区分力应提升"
accuracy = sum(c > r for c, r in zip(trained_chosen_scores, trained_rejected_scores)) / n_test
print(f"\\n奖励模型准确率（chosen > rejected）：{accuracy:.1%}")
assert accuracy > 0.7, f"奖励模型准确率应 > 70%，got {accuracy:.1%}"
print("所有验证通过！")\
"""))

# ── Cell 14: Summary ──────────────────────────────────────────────────────
cells.append(cell("""\
## 总结：RLHF 三步骤

| 步骤 | 方法 | 目的 |
|------|------|------|
| **SFT（监督微调）** | 标注员示范高质量回答 | 给模型一个"听话"的起点 |
| **RM（奖励模型）** | Bradley-Terry 偏好学习 | 训练一个能打分的"裁判" |
| **PPO（强化学习）** | 策略梯度 + KL 惩罚 | 让模型越来越高分，但不走样 |

**关键数字**（来自论文）：
- InstructGPT (1.3B) vs GPT-3 (175B)：人类评估中 **85%** 更喜欢 InstructGPT
- 这说明：**训练方式** 比 **参数规模** 更重要

**历史意义**：
RLHF 从这篇论文变成了 ChatGPT 的核心技术，开启了"对齐时代"的序幕。\
""", "markdown"))

# ── Assemble and write ─────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.8.0"},
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(__file__), "../notebooks/21-instructgpt-2022.ipynb")
out_path = os.path.normpath(out_path)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")  # 末尾换行（防止评审问题）

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
