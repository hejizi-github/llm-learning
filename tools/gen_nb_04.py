#!/usr/bin/env python3
"""生成 notebooks/04-lenet-1989.ipynb"""
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
cells.append(cell("""# 节点04：卷积神经网络的诞生（LeCun 1989）

本 notebook 从零手撕卷积（conv2d）和池化（max_pool2d），
不依赖任何深度学习框架，只用 NumPy。
最后用一个简单示例演示不同卷积核检测到的特征。""", "markdown"))

# --- Cell 1: 导入 ---
cells.append(cell("""\
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

Path('../docs/assets').mkdir(parents=True, exist_ok=True)
print("NumPy version:", np.__version__)"""))

# --- Cell 2: 手撕 conv2d ---
cells.append(cell("""## 第一步：手撕 2D 卷积""", "markdown"))

cells.append(cell("""\
def conv2d(image, kernel, stride=1, padding=0):
    \"\"\"
    image:  2D NumPy 数组，形状 (H, W)
    kernel: 2D NumPy 数组，形状 (kH, kW)
    stride: 步长（默认 1）
    padding: 四周补零的圈数（默认 0）
    返回: 特征图，形状 ((H+2p-kH)//s+1, (W+2p-kW)//s+1)
    \"\"\"
    H, W = image.shape
    kH, kW = kernel.shape

    # 补零
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
        H, W = image.shape

    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            patch = image[i*stride:i*stride+kH, j*stride:j*stride+kW]
            # 逐元素相乘再求和 = 点积
            output[i, j] = np.sum(patch * kernel)

    return output

# --- 验证 ---
img = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]], dtype=float)
k = np.array([[1, 0],
              [0, -1]], dtype=float)
result = conv2d(img, k)
print("输入:")
print(img)
print("卷积核:", k.tolist())
print("输出 (期望 [[1-5, 2-6],[4-8, 5-9]] = [[-4,-4],[-4,-4]]):")
print(result)
assert result.shape == (2, 2), f"形状错误: {result.shape}"
print("conv2d OK")"""))

# --- Cell 3: 手撕 max_pool2d ---
cells.append(cell("""## 第二步：手撕 Max Pooling""", "markdown"))

cells.append(cell("""\
def max_pool2d(feature_map, pool_size=2, stride=2):
    \"\"\"
    feature_map: 2D 数组 (H, W)
    pool_size:   池化窗口大小（默认 2×2）
    stride:      步长（默认等于 pool_size）
    返回: 池化后的特征图
    \"\"\"
    H, W = feature_map.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            patch = feature_map[i*stride:i*stride+pool_size,
                                j*stride:j*stride+pool_size]
            output[i, j] = np.max(patch)

    return output

# --- 验证 ---
fm = np.array([[1, 3, 2, 4],
               [5, 6, 1, 2],
               [3, 2, 4, 7],
               [1, 0, 5, 3]], dtype=float)
pooled = max_pool2d(fm)
print("输入 (4×4):")
print(fm)
print("2×2 Max Pooling 输出 (期望 [[6,4],[3,7]]):")
print(pooled)
assert np.allclose(pooled, [[6, 4], [3, 7]]), f"值错误: {pooled}"
print("max_pool2d OK")"""))

# --- Cell 4: 不同卷积核的可视化 ---
cells.append(cell("""## 第三步：用不同卷积核检测特征

我们创建一张简单的合成图片（包含横线、竖线），用不同卷积核处理，观察输出。""", "markdown"))

cells.append(cell("""\
# 构造一张 16×16 的测试图片（含横线和竖线）
def make_test_image(size=16):
    img = np.zeros((size, size))
    img[size//2, :] = 1.0          # 中间一条横线
    img[:, size//2] = 1.0          # 中间一条竖线
    return img

img = make_test_image(16)

# 几种不同的 3×3 卷积核
kernels = {
    "横线检测": np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], dtype=float),
    "竖线检测": np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], dtype=float),
    "边缘检测": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float),
    "模糊":     np.ones((3,3))/9.0,
}

fig, axes = plt.subplots(1, len(kernels)+1, figsize=(14, 3))
axes[0].imshow(img, cmap='gray', vmin=-1, vmax=2)
axes[0].set_title("原始图片")
axes[0].axis('off')

for ax, (name, k) in zip(axes[1:], kernels.items()):
    out = conv2d(img, k)
    ax.imshow(out, cmap='RdBu', vmin=-3, vmax=3)
    ax.set_title(name)
    ax.axis('off')

plt.tight_layout()
plt.savefig('../docs/assets/04-conv-filters.png', dpi=80)
plt.close()
print("图片已保存到 docs/assets/04-conv-filters.png")
print("各核输出形状:", [conv2d(img, k).shape for k in kernels.values()])"""))

# --- Cell 5: 多核卷积 + 池化 ---
cells.append(cell("""## 第四步：多个卷积核 → 特征图堆叠 → Max Pooling

真实 CNN 会同时使用多个卷积核，每个核产生一张特征图。
这里演示 2 个核的情况。""", "markdown"))

cells.append(cell("""\
# 用两个核同时处理图片
k_horizontal = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], dtype=float)
k_vertical   = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], dtype=float)

fmap_h = conv2d(img, k_horizontal)
fmap_v = conv2d(img, k_vertical)

print(f"输入图片: {img.shape}")
print(f"特征图 (横线核): {fmap_h.shape}")
print(f"特征图 (竖线核): {fmap_v.shape}")

pool_h = max_pool2d(fmap_h)
pool_v = max_pool2d(fmap_v)

print(f"池化后 (横线): {pool_h.shape}")
print(f"池化后 (竖线): {pool_v.shape}")

# "拉直" 后接全连接
flat = np.concatenate([pool_h.flatten(), pool_v.flatten()])
print(f"拼接后向量长度: {flat.shape[0]}  (即全连接层的输入维度)")

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes[0,0].imshow(fmap_h, cmap='RdBu'); axes[0,0].set_title("横线特征图"); axes[0,0].axis('off')
axes[0,1].imshow(pool_h, cmap='RdBu'); axes[0,1].set_title("横线池化后"); axes[0,1].axis('off')
axes[1,0].imshow(fmap_v, cmap='RdBu'); axes[1,0].set_title("竖线特征图"); axes[1,0].axis('off')
axes[1,1].imshow(pool_v, cmap='RdBu'); axes[1,1].set_title("竖线池化后"); axes[1,1].axis('off')
plt.tight_layout()
plt.savefig('../docs/assets/04-pool-demo.png', dpi=80)
plt.close()
print("图片已保存到 docs/assets/04-pool-demo.png")"""))

# --- Cell 6: 参数量对比 ---
cells.append(cell("""## 第五步：参数量对比 — CNN vs 全连接

用数字说明 CNN 为什么比全连接高效。""", "markdown"))

cells.append(cell("""\
H, W = 32, 32          # 输入图片大小
n_filters = 8          # 卷积核数量
kH, kW = 5, 5          # 核大小
n_fc_neurons = 100     # 全连接层神经元数

# 全连接第一层参数量
fc_params = H * W * n_fc_neurons
print(f"全连接层参数量: {H}×{W} × {n_fc_neurons} = {fc_params:,}")

# 卷积层参数量（权重共享！）
conv_params = kH * kW * n_filters
print(f"卷积层参数量:   {kH}×{kW} × {n_filters} = {conv_params}  (每个核 {kH*kW} 参数，共 {n_filters} 个核)")
print()
print(f"卷积层参数量仅为全连接的 {conv_params/fc_params*100:.2f}%")
print()
print("关键：同一个核在图片的所有位置共享参数，所以 3×3 的核只需要 9 个参数，")
print("      不管图片有多大，参数数量始终是 kH × kW × n_filters。")"""))

# --- Cell 7: 总结 ---
cells.append(cell("""## 总结

| 组件 | 作用 | 为什么重要 |
|------|------|-----------|
| 卷积（conv2d） | 用小核在图片上滑动，检测局部特征 | 权重共享，参数量小 |
| ReLU/激活函数 | 引入非线性 | 没有非线性就是线性变换，无法学复杂特征 |
| 池化（max_pool2d） | 压缩特征图，取局部最大值 | 降维 + 平移不变性 |
| 全连接层 | 组合特征，做最终分类 | 利用卷积提取的特征做决策 |

LeCun 1989 年的 CNN 证明了：**反向传播 + 卷积 = 能学会看图片的网络**。
这个架构在 2012 年以 AlexNet 的形式重新爆发，引领了深度学习革命。

> **配套文档**：[docs/04-lenet-1989.md](../docs/04-lenet-1989.md)""", "markdown"))

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

out_path = Path("notebooks/04-lenet-1989.ipynb")
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Generated: {out_path}")
