#!/usr/bin/env python3
"""gen_nb_19.py -- generate notebooks/19-clip-2021.ipynb"""
import json, pathlib

NB = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"}
    },
    "cells": []
}

def code(src): return {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None}
def md(src):   return {"cell_type": "markdown", "metadata": {}, "source": src}

NB["cells"] = [

md("""# 节点19：CLIP — 用语言监督图像（2021）

**论文**：Learning Transferable Visual Models From Natural Language Supervision
**作者**：Radford A, Kim JW, Hallacy C 等（OpenAI）
**arXiv**：2103.00020

本 notebook 用纯 NumPy 手撕 CLIP 的核心机制：
1. 余弦相似度计算
2. N×N 相似度矩阵（配对游戏）
3. InfoNCE 对比损失
4. 梯度下降拉近正样本对
5. Zero-shot 推理模拟
"""),

code("""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)
print("依赖加载完成，NumPy:", np.__version__)
"""),

md("""## Part 1：余弦相似度

CLIP 用余弦相似度衡量图文向量的"方向相似程度"。
值域 [-1, 1]：1 = 完全同向，0 = 垂直无关，-1 = 完全相反。
"""),

code("""def cosine_similarity(a, b):
    \"\"\"计算两个向量的余弦相似度\"\"\"
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# 手算验证（对应文档第9节的例子）
a = np.array([3.0, 4.0])   # 猫图向量
b = np.array([6.0, 8.0])   # 同方向的猫图
c = np.array([4.0, -3.0])  # 汽车图（垂直）

sim_ab = cosine_similarity(a, b)
sim_ac = cosine_similarity(a, c)

print(f"sim(a, b) = {sim_ab:.4f}  ← 应该 = 1.0（方向完全相同）")
print(f"sim(a, c) = {sim_ac:.4f}  ← 应该 = 0.0（垂直，毫无相关）")
assert abs(sim_ab - 1.0) < 1e-6, f"期望 1.0，得到 {sim_ab}"
assert abs(sim_ac - 0.0) < 1e-6, f"期望 0.0，得到 {sim_ac}"
print("余弦相似度验证通过")
"""),

md("""## Part 2：N×N 相似度矩阵（配对游戏）

CLIP 训练时，把 N 张图和 N 段文字的相似度都算出来，形成一个 N×N 矩阵。
**目标**：对角线（正确配对）最高，其他位置（错误配对）最低。

我们用随机初始化的 4 维向量模拟 4 个图文对。
"""),

code("""def l2_normalize(v):
    \"\"\"L2 归一化：让向量模长变成 1\"\"\"
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-8)

def similarity_matrix(image_vecs, text_vecs):
    \"\"\"
    计算 N×N 余弦相似度矩阵
    image_vecs: (N, D) 图像向量矩阵
    text_vecs:  (N, D) 文字向量矩阵
    返回: (N, N) 相似度矩阵，[i,j] = sim(image_i, text_j)
    \"\"\"
    img = l2_normalize(image_vecs)  # 归一化后，余弦相似度 = 点积
    txt = l2_normalize(text_vecs)
    return img @ txt.T  # 矩阵乘法得到所有对的相似度

# 4 个图文对（随机初始化，训练前）
N = 4
D = 8  # 8维向量（实际 CLIP 用 512 维）

# 故意让图文对"匹配"：同一对的向量方向相近
image_vecs = np.random.randn(N, D)
text_vecs = image_vecs + 0.3 * np.random.randn(N, D)  # 加少量噪声

sim_mat = similarity_matrix(image_vecs, text_vecs)

print("相似度矩阵（行=图像，列=文字）:")
print(np.round(sim_mat, 3))
print()
print("对角线（正确配对）:", np.round(np.diag(sim_mat), 3))
print("非对角线均值（错误配对）:", round(float((sim_mat.sum() - np.trace(sim_mat)) / (N*N - N)), 3))

# 可视化
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(sim_mat, cmap='YlOrRd', vmin=-1, vmax=1)
labels = ['猫图', '夜空图', '苹果派图', '自行车图']
text_labels = ['"猫"', '"夜空"', '"苹果"', '"自行车"']
ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_xticklabels(text_labels, fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
ax.set_title('CLIP 相似度矩阵（训练后目标：对角线最亮）', fontsize=10)
for i in range(N):
    for j in range(N):
        ax.text(j, i, f'{sim_mat[i,j]:.2f}', ha='center', va='center', fontsize=9,
                color='white' if sim_mat[i,j] > 0.6 else 'black',
                fontweight='bold' if i == j else 'normal')
plt.colorbar(im, ax=ax, label='余弦相似度')
plt.tight_layout()
plt.savefig('../docs/assets/19-clip-similarity-matrix.png', dpi=100, bbox_inches='tight')
plt.close()
print("图已保存到 docs/assets/19-clip-similarity-matrix.png")
"""),

md("""## Part 3：InfoNCE 对比损失

对于每张图，把它和所有文字的相似度看成 logits，做 Softmax 后取正确配对的对数概率。
**损失越小 = 正确配对的概率越高。**
"""),

code("""def infonce_loss(image_vecs, text_vecs, temperature=0.07):
    \"\"\"
    InfoNCE 损失（CLIP 使用的对比损失）
    temperature: 温度参数，控制分布的尖锐程度（CLIP 论文中约 0.07）
    \"\"\"
    sim_mat = similarity_matrix(image_vecs, text_vecs) / temperature

    N = len(image_vecs)
    labels = np.arange(N)  # 正确配对：第 i 张图对应第 i 段文字

    # 图像到文字的损失
    # 对每一行做 log-softmax，取对角线（正确配对）
    log_softmax_rows = sim_mat - np.log(np.sum(np.exp(sim_mat - sim_mat.max(axis=1, keepdims=True)),
                                               axis=1, keepdims=True)) - sim_mat.max(axis=1, keepdims=True)
    loss_i2t = -log_softmax_rows[np.arange(N), labels].mean()

    # 文字到图像的损失
    log_softmax_cols = sim_mat.T - np.log(np.sum(np.exp(sim_mat.T - sim_mat.T.max(axis=1, keepdims=True)),
                                                  axis=1, keepdims=True)) - sim_mat.T.max(axis=1, keepdims=True)
    loss_t2i = -log_softmax_cols[np.arange(N), labels].mean()

    return (loss_i2t + loss_t2i) / 2.0

# 手算验证（文档第5节的具体数字）
# N=3，第1张图的分数：正确=0.9，错误1=0.1，错误2=0.2
logits_example = np.array([0.9, 0.1, 0.2])
max_l = logits_example.max()
softmax_example = np.exp(logits_example - max_l) / np.exp(logits_example - max_l).sum()
loss_example = -np.log(softmax_example[0])
print(f"手算示例：softmax = {np.round(softmax_example, 3)}")
print(f"正确配对概率 = {softmax_example[0]:.3f}，损失 = {loss_example:.3f}")
print()

# 测试随机初始化时的损失
loss_random = infonce_loss(np.random.randn(N, D), np.random.randn(N, D))
print(f"随机初始化损失: {loss_random:.4f}  ← 应接近 log(N)={np.log(N):.4f}（均匀随机时的期望值）")

# 测试完全正确配对时的损失
perfect_vecs = np.eye(N, D)  # 图文向量完全一致
loss_perfect = infonce_loss(perfect_vecs, perfect_vecs)
print(f"完全正确配对损失: {loss_perfect:.6f}  ← 应接近 0")
"""),

md("""## Part 4：梯度下降让正样本对相似度上升

用简单的数值梯度下降，演示 CLIP 训练的效果：
**正样本对越来越像，负样本对越来越不像。**
"""),

code("""def numerical_gradient(func, vec, eps=1e-4):
    \"\"\"对向量中每个元素计算数值梯度\"\"\"
    grad = np.zeros_like(vec)
    for i in range(len(vec)):
        v_plus = vec.copy(); v_plus[i] += eps
        v_minus = vec.copy(); v_minus[i] -= eps
        grad[i] = (func(v_plus) - func(v_minus)) / (2 * eps)
    return grad

# 训练：2 对图文（3 维向量，便于可视化）
np.random.seed(0)
N_train = 2
D_train = 3

# 初始化：图文向量随机
img = np.random.randn(N_train, D_train) * 0.3
txt = np.random.randn(N_train, D_train) * 0.3

# 演示用较高温度（0.5），这样相似度必须真正提高才能使损失降低
# 论文中用 0.07，但 0.5 更便于观察余弦相似度的实际上升
DEMO_TEMP = 0.5

def infonce_demo(iv, tv):
    return infonce_loss(iv, tv, temperature=DEMO_TEMP)

initial_sim = float(np.diag(similarity_matrix(img, txt)).mean())
print(f"初始正样本相似度: {initial_sim:.4f}")

lr = 0.15
n_steps = 120
losses = []
sim_pos = []  # 正样本对相似度（对角线均值）

for step in range(n_steps):
    loss = infonce_demo(img, txt)
    losses.append(loss)
    sim_pos.append(float(np.diag(similarity_matrix(img, txt)).mean()))

    # 数值梯度更新图像向量
    for i in range(N_train):
        def loss_fn_img(v, _i=i):
            img_copy = img.copy(); img_copy[_i] = v
            return infonce_demo(img_copy, txt)
        grad_img = numerical_gradient(loss_fn_img, img[i])
        img[i] -= lr * grad_img

    # 数值梯度更新文字向量
    for j in range(N_train):
        def loss_fn_txt(v, _j=j):
            txt_copy = txt.copy(); txt_copy[_j] = v
            return infonce_demo(img, txt_copy)
        grad_txt = numerical_gradient(loss_fn_txt, txt[j])
        txt[j] -= lr * grad_txt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(losses, color='#e74c3c', linewidth=2)
ax1.set_xlabel('训练步数'); ax1.set_ylabel('InfoNCE 损失')
ax1.set_title('对比损失随训练下降', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

ax2.plot(sim_pos, color='#2ecc71', linewidth=2, label='正样本对（正确配对）')
ax2.set_xlabel('训练步数'); ax2.set_ylabel('余弦相似度')
ax2.set_title('正样本对相似度随训练上升', fontsize=11)
ax2.set_ylim(-0.2, 1.1)
ax2.legend(); ax2.grid(True, alpha=0.3)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='上限=1.0')

plt.suptitle('CLIP 对比学习训练过程', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../docs/assets/19-clip-training.png', dpi=100, bbox_inches='tight')
plt.close()

final_sim = float(np.diag(similarity_matrix(img, txt)).mean())
print(f"训练结束，正样本对相似度: {final_sim:.4f}  (初始: {initial_sim:.4f})")
print(f"最终损失: {losses[-1]:.4f}（初始损失: {losses[0]:.4f}）")
assert final_sim > initial_sim + 0.1, (
    f"训练后正样本相似度应比初始值高 > 0.1，初始={initial_sim:.4f}，最终={final_sim:.4f}")
assert losses[-1] < losses[0] * 0.5, (
    f"损失应至少下降一半，初始={losses[0]:.4f}，最终={losses[-1]:.4f}")
print("图已保存到 docs/assets/19-clip-training.png")
"""),

md("""## Part 5：Zero-shot 推理模拟

CLIP 最强的能力：对**从未见过标注的新任务**，直接用文字描述来分类。

我们模拟 3 个类别（猫、狗、鸟）的 Zero-shot 推理：
"""),

code("""# 模拟已训练好的 CLIP：
# 图文向量共处于同一语义空间（相同类别的图文向量方向相近）

D_zs = 8
np.random.seed(99)

# "类别原型"向量（代表每个类别的核心语义方向）
cat_dir    = np.array([1.0, 0.2, -0.1, 0.5, 0.3, -0.2, 0.1, 0.4])
dog_dir    = np.array([0.2, 1.0, 0.3, -0.1, 0.4, 0.2, -0.3, 0.1])
bird_dir   = np.array([-0.1, 0.3, 1.0, 0.2, -0.2, 0.5, 0.3, -0.1])

def make_embedding(direction, noise=0.1):
    \"\"\"模拟编码器输出：在类别方向上加少量噪声\"\"\"
    return l2_normalize(direction + noise * np.random.randn(D_zs))

# 文字编码器输出：把类别名编码为向量
text_prompts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
text_embeddings = np.array([
    make_embedding(cat_dir, noise=0.05),   # "猫"的文字嵌入
    make_embedding(dog_dir, noise=0.05),   # "狗"的文字嵌入
    make_embedding(bird_dir, noise=0.05),  # "鸟"的文字嵌入
])

# 待分类的图片（各 3 张，共 9 张）
image_embeddings = np.array([
    make_embedding(cat_dir),   make_embedding(cat_dir),   make_embedding(cat_dir),   # 猫图
    make_embedding(dog_dir),   make_embedding(dog_dir),   make_embedding(dog_dir),   # 狗图
    make_embedding(bird_dir),  make_embedding(bird_dir),  make_embedding(bird_dir),  # 鸟图
])
true_labels = [0,0,0, 1,1,1, 2,2,2]  # 真实类别

# Zero-shot 推理：找最相似的文字向量
def zero_shot_predict(image_emb, text_embs):
    sims = np.array([cosine_similarity(image_emb, t) for t in text_embs])
    return int(np.argmax(sims)), sims

correct = 0
print(f"{'图片':6} {'真实':8} {'预测':8} {'相似度(猫,狗,鸟)':30} {'结果'}")
print("-" * 70)
category_names = ['猫', '狗', '鸟']
for i, (img_emb, true_label) in enumerate(zip(image_embeddings, true_labels)):
    pred, sims = zero_shot_predict(img_emb, text_embeddings)
    result = "✓" if pred == true_label else "✗"
    if pred == true_label:
        correct += 1
    print(f"图片{i+1:2d}  {category_names[true_label]:6} {category_names[pred]:6}  "
          f"({sims[0]:.2f}, {sims[1]:.2f}, {sims[2]:.2f})  {result}")

accuracy = correct / len(true_labels)
print(""); print(f"Zero-shot 准确率: {correct}/{len(true_labels)} = {accuracy:.1%}")
assert accuracy >= 0.8, f"Zero-shot 准确率应 >= 80%，实际 {accuracy:.1%}"
print("Zero-shot 推理验证通过")
"""),

md("""## 总结

| 概念 | 直觉理解 | 核心公式 |
|------|---------|---------|
| **余弦相似度** | 比较向量的方向，不看长度 | $\\text{sim}(a,b) = \\frac{a\\cdot b}{|a||b|}$ |
| **对比学习** | N×N 配对游戏，对角线分数最高 | N×N 相似度矩阵 |
| **InfoNCE 损失** | 正确配对概率越高，损失越低 | $-\\log\\frac{\\exp(\\text{sim}_{ii}/\\tau)}{\\sum_j \\exp(\\text{sim}_{ij}/\\tau)}$ |
| **Zero-shot** | 用文字描述做分类，不需要标注 | argmax(sim(图片, 文字类别)) |

**CLIP 的贡献**：证明了 4 亿网络图文对（无人工标注）可以训练出媲美有监督学习的视觉模型，并且具有强大的 zero-shot 泛化能力。

**历史位置**：CLIP 成为多模态 AI 的基石——Stable Diffusion 的文字条件控制、DALL-E 2 的图文对齐，都建立在 CLIP 编码器之上。

---
*参考文献：Radford et al., arXiv:2103.00020, ICML 2021*
"""),

]

out = pathlib.Path(__file__).parent.parent / "notebooks" / "19-clip-2021.ipynb"
out.write_text(json.dumps(NB, ensure_ascii=False, indent=1))
print(f"写入 {out}，共 {len(NB['cells'])} 个 cell")
