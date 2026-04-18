"""
gen_nb_20.py — generate notebooks/20-dalle2-2022.ipynb
DALL-E 2: Hierarchical Text-Conditional Image Generation with CLIP Latents
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
# 节点20：DALL-E 2 — 用 CLIP 向量生成图像（2022）

**论文**：Hierarchical Text-Conditional Image Generation with CLIP Latents
**arXiv**：2204.06125

本 notebook 用纯 NumPy 模拟 DALL-E 2 的三个核心步骤：
1. **CLIP 语义空间**：文字和图像向量如何在同一空间里"对齐"
2. **Prior**：从文字嵌入生成图像嵌入（DALL-E 2 的关键创新）
3. **扩散解码器**：从图像嵌入逐步去噪生成像素（简化演示）
4. **语义插值**：在 CLIP 空间里"走"从一个概念到另一个概念\
""", "markdown"))

# ── Cell 1: Imports ────────────────────────────────────────────────────────
cells.append(cell("""\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)
print("环境就绪，NumPy 版本:", np.__version__)\
"""))

# ── Cell 2: Part 1 intro ───────────────────────────────────────────────────
cells.append(cell("""\
## Part 1：CLIP 语义空间——文字和图像的"地图"

CLIP 训练完成后，图像和文字会被映射到同一个高维向量空间。
**同一概念的图像向量和文字向量，方向会非常接近。**

我们用低维向量（8维）模拟这个语义空间，演示核心性质。\
""", "markdown"))

# ── Cell 3: CLIP space simulation ─────────────────────────────────────────
cells.append(cell("""\
def l2_normalize(v):
    \"\"\"L2归一化：让向量模长=1，只保留方向信息\"\"\"
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-8)

def cosine_similarity(a, b):
    \"\"\"余弦相似度：两个向量方向有多接近（-1~1）\"\"\"
    a = l2_normalize(np.array(a, dtype=float))
    b = l2_normalize(np.array(b, dtype=float))
    return float(np.dot(a, b))

D = 8   # 向量维度（真实 CLIP 用 512 或 768 维）

# 模拟"已训练好的 CLIP 语义空间"
# 每个概念用一个固定的方向向量代表
# 相似概念的向量方向接近，不同概念的向量方向远离
np.random.seed(0)

# 猫的语义方向（图像向量 ≈ 文字向量，但不完全一样）
cat_base = l2_normalize(np.array([1.0, 0.8, 0.2, -0.1, 0.3, 0.5, -0.2, 0.1]))
cat_img_vec = l2_normalize(cat_base + np.random.randn(D) * 0.15)   # 图像嵌入
cat_txt_vec = l2_normalize(cat_base + np.random.randn(D) * 0.15)   # 文字嵌入

# 狗的语义方向
dog_base = l2_normalize(np.array([0.9, 0.7, 0.3, 0.1, 0.4, 0.4, -0.1, 0.2]))
dog_img_vec = l2_normalize(dog_base + np.random.randn(D) * 0.15)
dog_txt_vec = l2_normalize(dog_base + np.random.randn(D) * 0.15)

# 飞机的语义方向（和动物类很不同）
plane_base = l2_normalize(np.array([-0.5, 0.2, 0.9, 0.7, -0.3, 0.1, 0.8, -0.4]))
plane_img_vec = l2_normalize(plane_base + np.random.randn(D) * 0.15)
plane_txt_vec = l2_normalize(plane_base + np.random.randn(D) * 0.15)

print("=== CLIP 语义空间：相似度矩阵 ===")
print()
concepts = [('猫-图', cat_img_vec), ('猫-文', cat_txt_vec),
            ('狗-图', dog_img_vec), ('狗-文', dog_txt_vec),
            ('飞机-图', plane_img_vec), ('飞机-文', plane_txt_vec)]

labels = [c[0] for c in concepts]
vecs = [c[1] for c in concepts]
sim_matrix = np.array([[cosine_similarity(a, b) for b in vecs] for a in vecs])

for i, row_label in enumerate(labels):
    row_str = "  ".join(f"{sim_matrix[i,j]:+.2f}" for j in range(len(labels)))
    print(f"  {row_label:6s} | {row_str}")

print()
print("关键观察：")
print(f"  猫(图) ↔ 猫(文) 相似度: {cosine_similarity(cat_img_vec, cat_txt_vec):+.4f}  ← 同一概念，接近")
print(f"  猫(图) ↔ 狗(图) 相似度: {cosine_similarity(cat_img_vec, dog_img_vec):+.4f}  ← 相似概念")
print(f"  猫(图) ↔ 飞机(图) 相似度: {cosine_similarity(cat_img_vec, plane_img_vec):+.4f}  ← 不同概念，较远")\
"""))

# ── Cell 4: Part 2 intro ───────────────────────────────────────────────────
cells.append(cell("""\
## Part 2：Prior——从"文字嵌入"生成"图像嵌入"

DALL-E 2 的关键创新：**Prior 网络**。

为什么需要 Prior？

- CLIP 训练后，"a photo of a cat" 的文字向量和猫的图像向量**方向很近，但不完全一样**
- 文字嵌入描述的是"语言里猫的概念"，图像嵌入描述的是"像素里猫的视觉特征"
- Prior 的工作：给定文字嵌入，生成一个"看起来像猫图像嵌入"的向量

**直觉**：Prior 是一个"翻译器"——把"语言坐标"翻译成"图像坐标"，
两者都在 CLIP 空间里，但微妙地不同。\
""", "markdown"))

# ── Cell 5: Prior simulation ───────────────────────────────────────────────
cells.append(cell("""\
def simulate_prior(text_embedding, noise_scale=0.1, n_diffusion_steps=20):
    \"\"\"
    模拟 Prior：文字嵌入 → 图像嵌入

    真实 DALL-E 2 用扩散模型做这步。
    这里用简化的迭代去噪演示核心思想：
    从纯噪声出发，每步向文字嵌入方向"推"一点。
    \"\"\"
    # 从纯噪声开始（像扩散模型的起点）
    x = np.random.randn(len(text_embedding))
    x = l2_normalize(x)

    history = [x.copy()]

    for step in range(n_diffusion_steps):
        # 每一步：向 text_embedding 方向走一小步，加少量噪声
        alpha = (step + 1) / n_diffusion_steps  # 逐渐增大引导强度
        noise = np.random.randn(len(text_embedding)) * noise_scale * (1 - alpha)

        # 插值：x 逐渐向 text_embedding 靠近
        x = l2_normalize((1 - alpha * 0.3) * x + alpha * 0.3 * text_embedding + noise)
        history.append(x.copy())

    return x, history

# 用猫的文字向量生成"图像嵌入"
np.random.seed(7)
cat_img_pred, cat_prior_history = simulate_prior(cat_txt_vec, noise_scale=0.12)

print("=== Prior 演示：从文字嵌入 → 图像嵌入 ===")
print()
sim_to_txt = cosine_similarity(cat_img_pred, cat_txt_vec)
sim_to_img = cosine_similarity(cat_img_pred, cat_img_vec)   # 真实图像嵌入（参考）
sim_to_dog = cosine_similarity(cat_img_pred, dog_txt_vec)
sim_to_plane = cosine_similarity(cat_img_pred, plane_txt_vec)

print(f"Prior 生成的图像嵌入 vs 猫文字嵌入：  {sim_to_txt:+.4f}  ← 应接近（目标）")
print(f"Prior 生成的图像嵌入 vs 猫真实图像嵌入：{sim_to_img:+.4f}  ← 理想情况应接近")
print(f"Prior 生成的图像嵌入 vs 狗文字嵌入：  {sim_to_dog:+.4f}  ← 应较小")
print(f"Prior 生成的图像嵌入 vs 飞机文字嵌入：{sim_to_plane:+.4f}  ← 应最小")
print()

# 追踪过程中的相似度变化
sims_over_time = [cosine_similarity(h, cat_txt_vec) for h in cat_prior_history]
print(f"Prior 去噪过程：相似度 {sims_over_time[0]:+.4f} → {sims_over_time[-1]:+.4f}")
print("（每步逐渐向文字嵌入方向靠近）")

assert sim_to_txt > sim_to_plane, "Prior 生成的嵌入应与输入文字更近（猫 > 飞机）"
assert sims_over_time[-1] > sims_over_time[0], "Prior 过程应使相似度逐步上升"\
"""))

# ── Cell 6: Visualize Prior process ───────────────────────────────────────
cells.append(cell("""\
# 可视化 Prior 去噪过程中的相似度变化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：Prior 过程相似度曲线
sims_cat = [cosine_similarity(h, cat_txt_vec) for h in cat_prior_history]
sims_dog = [cosine_similarity(h, dog_txt_vec) for h in cat_prior_history]
sims_plane = [cosine_similarity(h, plane_txt_vec) for h in cat_prior_history]

steps = list(range(len(cat_prior_history)))
axes[0].plot(steps, sims_cat, 'b-o', markersize=4, label='与猫(文字)', linewidth=2)
axes[0].plot(steps, sims_dog, 'g--s', markersize=4, label='与狗(文字)', linewidth=1.5)
axes[0].plot(steps, sims_plane, 'r:^', markersize=4, label='与飞机(文字)', linewidth=1.5)
axes[0].set_xlabel('Prior 去噪步数')
axes[0].set_ylabel('余弦相似度')
axes[0].set_title('Prior：从噪声出发，逐步接近目标文字嵌入')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axhline(0, color='gray', linestyle='-', alpha=0.3)

# 右图：三个概念的相似度对比（最终 Prior 结果）
labels_bar = ['猫(图真实)', '猫(文字)', '狗(文字)', '飞机(文字)']
values_bar = [sim_to_img, sim_to_txt, sim_to_dog, sim_to_plane]
colors_bar = ['steelblue', 'royalblue', 'seagreen', 'tomato']
bars = axes[1].bar(labels_bar, values_bar, color=colors_bar, alpha=0.8)
axes[1].set_ylabel('余弦相似度（Prior生成嵌入 vs 各概念）')
axes[1].set_title('Prior 生成结果：语义对齐验证')
axes[1].set_ylim(-0.3, 1.05)
axes[1].axhline(0, color='gray', linestyle='-', alpha=0.5)
for bar, val in zip(bars, values_bar):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('../docs/assets/20-dalle2-prior-process.png', dpi=100, bbox_inches='tight')
plt.close()
print("图已保存：docs/assets/20-dalle2-prior-process.png")\
"""))

# ── Cell 7: Part 3 intro ───────────────────────────────────────────────────
cells.append(cell("""\
## Part 3：扩散解码器——从图像嵌入生成像素（简化演示）

Prior 给了我们一个"图像嵌入向量"。
接下来，扩散解码器（Diffusion Decoder）要把这个向量"展开"成真正的像素图像。

**简化演示**：我们不真的生成图像像素（那需要巨大的神经网络），
而是演示扩散去噪的核心思想：**逐步从噪声恢复目标信号**，
同时被图像嵌入向量"引导"。\
""", "markdown"))

# ── Cell 8: Decoder simulation ────────────────────────────────────────────
cells.append(cell("""\
def simulate_decoder(image_embedding, target_size=16, n_steps=30, guidance_scale=3.0):
    \"\"\"
    模拟扩散解码器：图像嵌入 → 像素图像

    简化为：从噪声出发，每步用图像嵌入引导，生成低分辨率灰度图
    target_size: 图像边长（我们用16x16，真实DALL-E 2用64x64再超分辨到1024x1024）
    \"\"\"
    # "目标图像"：由图像嵌入的前 target_size^2 个维度定义的模式
    # （真实中这是神经网络学到的映射，这里用线性投影简化）
    n_pixels = target_size * target_size

    # 用图像嵌入构造一个目标模式（重复平铺嵌入向量以填满像素）
    repeats = (n_pixels // len(image_embedding)) + 1
    target_pattern = np.tile(image_embedding, repeats)[:n_pixels]
    target_pattern = target_pattern.reshape(target_size, target_size)
    # 归一化到 [0,1]
    target_pattern = (target_pattern - target_pattern.min()) / (target_pattern.max() - target_pattern.min() + 1e-8)

    # 从纯噪声开始
    x = np.random.randn(target_size, target_size)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    history = [x.copy()]
    losses = []

    for step in range(n_steps):
        t = 1.0 - (step + 1) / n_steps  # 噪声水平（从1到0）

        # 预测"去噪"方向（向 target_pattern 靠近）
        noise = np.random.randn(target_size, target_size) * t * 0.3
        guidance = guidance_scale * (target_pattern - x)

        # 更新：混合去噪 + 引导信号
        x = x + 0.05 * guidance + noise
        x = np.clip(x, 0, 1)

        mse = float(np.mean((x - target_pattern) ** 2))
        losses.append(mse)
        history.append(x.copy())

    return x, target_pattern, history, losses

# 为"猫"概念生成图像
np.random.seed(3)
cat_decoded, cat_target, cat_history, cat_losses = simulate_decoder(
    cat_img_pred, target_size=16, n_steps=30, guidance_scale=4.0
)

print("=== 扩散解码器演示 ===")
print(f"初始噪声 vs 目标的 MSE: {np.mean((cat_history[0] - cat_target)**2):.4f}")
print(f"最终结果 vs 目标的 MSE: {np.mean((cat_decoded - cat_target)**2):.4f}")
print()
initial_mse = np.mean((cat_history[0] - cat_target)**2)
final_mse = np.mean((cat_decoded - cat_target)**2)
improvement = (initial_mse - final_mse) / initial_mse * 100
print(f"MSE 下降了 {improvement:.1f}%  ← 扩散去噪效果")
print()
print("（真实 DALL-E 2 用 64x64 解码器 + 超分辨率网络到 1024x1024）")

assert final_mse < initial_mse, "扩散解码器应使图像更接近目标（MSE下降）"\
"""))

# ── Cell 9: Visualize decoder ─────────────────────────────────────────────
cells.append(cell("""\
# 可视化扩散解码过程
fig = plt.figure(figsize=(14, 5))

# 去噪过程：选几个时间步展示
steps_to_show = [0, 5, 10, 20, 30]
for idx, step_idx in enumerate(steps_to_show):
    ax = fig.add_subplot(2, len(steps_to_show) + 1, idx + 1)
    ax.imshow(cat_history[step_idx], cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f'步骤 {step_idx}', fontsize=9)
    ax.axis('off')

# 显示目标
ax_target = fig.add_subplot(2, len(steps_to_show) + 1, len(steps_to_show) + 1)
ax_target.imshow(cat_target, cmap='viridis', vmin=0, vmax=1)
ax_target.set_title('目标模式', fontsize=9, color='green')
ax_target.axis('off')

# 下方：MSE曲线
ax_loss = fig.add_subplot(2, 1, 2)
ax_loss.plot(cat_losses, 'b-', linewidth=2)
ax_loss.set_xlabel('扩散步骤')
ax_loss.set_ylabel('MSE（与目标的距离）')
ax_loss.set_title('解码过程：图像逐渐接近目标（MSE下降）')
ax_loss.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../docs/assets/20-dalle2-decoder.png', dpi=100, bbox_inches='tight')
plt.close()
print("图已保存：docs/assets/20-dalle2-decoder.png")\
"""))

# ── Cell 10: Part 4 intro + semantic interpolation ─────────────────────────
cells.append(cell("""\
## Part 4：语义插值——在 CLIP 空间里"行走"

DALL-E 2 的一个有趣性质：可以在 CLIP 语义空间里**插值**，
生成两个概念之间的"中间状态"。

比如：从"猫"的嵌入向量走向"狗"的嵌入向量，
中间的点可能对应"看起来像猫又像狗的动物"。\
""", "markdown"))

cells.append(cell("""\
def slerp(v0, v1, t):
    \"\"\"
    球面线性插值（Slerp）：在单位球面上沿最短路径插值
    比普通线性插值更适合方向向量（不会缩短长度）

    t=0 → v0，t=1 → v1
    \"\"\"
    v0 = l2_normalize(v0)
    v1 = l2_normalize(v1)

    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    omega = np.arccos(dot)

    if abs(omega) < 1e-6:  # 向量几乎相同
        return l2_normalize((1 - t) * v0 + t * v1)

    sin_omega = np.sin(omega)
    return (np.sin((1 - t) * omega) / sin_omega) * v0 + (np.sin(t * omega) / sin_omega) * v1

# 在猫和狗之间插值（10个中间点）
n_interp = 10
t_values = np.linspace(0, 1, n_interp)
interpolated = [slerp(cat_txt_vec, dog_txt_vec, t) for t in t_values]

# 计算每个插值点与各概念的相似度
sims_to_cat = [cosine_similarity(v, cat_txt_vec) for v in interpolated]
sims_to_dog = [cosine_similarity(v, dog_txt_vec) for v in interpolated]

print("=== CLIP 语义插值：猫 → 狗 ===")
print()
print(f"{'t':>5}  {'与猫相似度':>10}  {'与狗相似度':>10}  {'解释':>20}")
print("-" * 55)
for i, t in enumerate(t_values):
    if t < 0.2:
        label = "← 更像猫"
    elif t > 0.8:
        label = "→ 更像狗"
    else:
        label = "↔ 中间状态"
    print(f"  {t:.1f}  {sims_to_cat[i]:>+10.4f}  {sims_to_dog[i]:>+10.4f}  {label}")

print()
print("观察：t=0 时接近猫，t=1 时接近狗，中间平滑过渡")
print("这就是 DALL-E 2 能做'语义插值'生图的数学基础")

# 验证：插值单调性
assert sims_to_cat[0] > sims_to_cat[-1], "t=0 时应与猫更近"
assert sims_to_dog[-1] > sims_to_dog[0], "t=1 时应与狗更近"\
"""))

# ── Cell 11: Visualize interpolation ──────────────────────────────────────
cells.append(cell("""\
# 可视化语义插值
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 左图：相似度随插值参数变化
axes[0].plot(t_values, sims_to_cat, 'b-o', label='与猫文字嵌入', linewidth=2, markersize=6)
axes[0].plot(t_values, sims_to_dog, 'r-s', label='与狗文字嵌入', linewidth=2, markersize=6)
axes[0].set_xlabel('插值参数 t（0=猫，1=狗）')
axes[0].set_ylabel('余弦相似度')
axes[0].set_title('CLIP 空间插值：猫 ↔ 狗')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：DALL-E 2 完整流程图（文字说明）
axes[1].axis('off')
pipeline_lines = [
    "DALL-E 2 生成流程",
    "─────────────────────────────",
    '输入："a photo of a cat"',
    "        │",
    "   ┌────▼────┐",
    "   │  CLIP   │  文字编码器",
    "   │Text Enc │  → 文字嵌入向量",
    "   └────┬────┘",
    "        │ 文字嵌入",
    "   ┌────▼────┐",
    "   │  Prior  │  扩散模型",
    "   │ Network │  文字嵌入 → 图像嵌入",
    "   └────┬────┘",
    "        │ 图像嵌入",
    "   ┌────▼────┐",
    "   │Diffusion│  扩散解码器",
    "   │ Decoder │  图像嵌入 → 64×64像素",
    "   └────┬────┘",
    "        │",
    "   ┌────▼────┐",
    "   │Upsampler│  超分辨率",
    "   │  ×16   │  64×64 → 1024×1024",
    "   └─────────┘",
    "输出：1024×1024 图像",
]
pipeline_text = chr(10).join(pipeline_lines)
axes[1].text(0.05, 0.95, pipeline_text,
            transform=axes[1].transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('../docs/assets/20-dalle2-interpolation.png', dpi=100, bbox_inches='tight')
plt.close()
print("图已保存：docs/assets/20-dalle2-interpolation.png")\
"""))

# ── Cell 12: Summary ──────────────────────────────────────────────────────
cells.append(cell("""\
## 总结

| 组件 | 输入 | 输出 | 关键思想 |
|------|------|------|---------|
| **CLIP 文字编码器** | 文字提示 | 文字嵌入向量 | 把语言映射到语义空间 |
| **Prior（扩散）** | 文字嵌入 | 图像嵌入 | 语言坐标 → 图像坐标 |
| **扩散解码器** | 图像嵌入 | 64×64 像素 | 在嵌入引导下去噪 |
| **超分辨率** | 64×64 | 1024×1024 | 补充视觉细节 |

**DALL-E 2 的核心创新**：Prior 网络。
它解决了"文字嵌入和图像嵌入不完全一样"的问题，
让 CLIP 的语义空间真正成为生成图像的"地图"。

**历史意义**：DALL-E 2（2022年4月）和 Stable Diffusion（2022年8月）
是同一年出现的两条技术路线——一条闭源高质量，一条开源平民化。
两者共同开启了 AIGC 时代。\
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

out_path = os.path.join(os.path.dirname(__file__), "../notebooks/20-dalle2-2022.ipynb")
out_path = os.path.normpath(out_path)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
