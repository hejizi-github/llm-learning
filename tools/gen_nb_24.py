"""
gen_nb_24.py — 生成 notebooks/24-flash-attention-2022.ipynb

节点24：FlashAttention (2022)
- 标准 Attention 实现（含 O(n²) 内存问题展示）
- 在线 Softmax 原理演示
- 分块 Attention 实现（FlashAttention 简化版）
- 数学等价性验证
- 内存使用量对比（理论分析）
"""

import json

NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"}
    },
    "cells": []
}


import uuid

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": uuid.uuid4().hex[:8]}


def code(source):
    return {
        "cell_type": "code", "metadata": {}, "source": source,
        "outputs": [], "execution_count": None, "id": uuid.uuid4().hex[:8]
    }


cells = [

# ── Cell 1: 标题 ──────────────────────────────────────────────────────────
md("""# 节点24：FlashAttention（2022）

**核心问题**：标准 Attention 为什么慢？答案不是"计算量大"，而是"内存来回搬运太多次"。

FlashAttention 的魔法：**同样的数学，不同的执行顺序**——把中间结果放在快速缓存里，不写回慢速内存。

本 Notebook 演示：
1. 标准 Attention 的实现与内存分析
2. 在线 Softmax 原理（分块更新）
3. 分块 Attention 实现
4. 两者结果完全相同的验证
"""),

# ── Cell 2: 导入 ──────────────────────────────────────────────────────────
code("""import numpy as np

np.random.seed(42)
print("NumPy version:", np.__version__)
"""),

# ── Cell 3: 书桌 vs 档案柜的类比 ─────────────────────────────────────────
md("""## 1. 为什么 Attention 慢？书桌 vs 档案柜

想象你在做题：
- **书桌**（SRAM，片上缓存）：很小，只能放几页纸，**超快**
- **档案柜**（HBM，显卡内存）：很大，能放成千上万页，**较慢**

标准 Attention 的流程：
```
1. Q, K, V 存入档案柜
2. 计算 S = QKᵀ，存入档案柜  ← 搬运第1次
3. 读出 S，算 softmax(S)，存入档案柜  ← 搬运第2次
4. 读出 softmax 结果，乘以 V  ← 搬运第3次
5. 结果存入档案柜  ← 搬运第4次
```

每次"搬运"都很耗时。FlashAttention 的想法：**分块，每块在书桌上一次算完，不存中间结果**。
"""),

# ── Cell 4: 标准 Attention 实现 ───────────────────────────────────────────
code("""# ── 标准 Attention 实现 ────────────────────────────────────────────────

def standard_attention(Q, K, V):
    \"\"\"
    标准 Scaled Dot-Product Attention
    输入: Q, K, V 形状均为 [n, d]
    输出: 形状 [n, d]

    内存问题：中间矩阵 S 和 P 都是 [n, n]，当 n 大时占用大量内存
    \"\"\"
    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    # Step 1: 计算相似度矩阵 S = QKᵀ / √d，形状 [n, n]
    S = (Q @ K.T) * scale                         # 这里产生 n² 个数！

    # Step 2: 数值稳定的 Softmax（减去每行最大值防止溢出）
    S_shifted = S - S.max(axis=1, keepdims=True)  # 还是 [n, n] 大小
    exp_S = np.exp(S_shifted)
    P = exp_S / exp_S.sum(axis=1, keepdims=True)  # 还是 [n, n] 大小

    # Step 3: 加权求和
    O = P @ V                                      # 输出 [n, d]
    return O


# 测试：序列长度 n=64，维度 d=16
n, d = 64, 16
Q = np.random.randn(n, d).astype(np.float32)
K = np.random.randn(n, d).astype(np.float32)
V = np.random.randn(n, d).astype(np.float32)

out_standard = standard_attention(Q, K, V)
print(f"输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
print(f"输出形状: {out_standard.shape}")
print(f"输出前3行前4列:\\n{out_standard[:3, :4].round(4)}")

# 内存估算
n_big = 4096
mem_bytes = n_big * n_big * 4  # float32 = 4 bytes
print(f"\\n当 n={n_big} 时，中间矩阵 S/P 大小: {mem_bytes / 1e6:.1f} MB")
print(f"（还只是一个头的内存，多头 Attention 要乘以头数）")
"""),

# ── Cell 5: 在线 Softmax 原理 ─────────────────────────────────────────────
md("""## 2. 在线 Softmax：不看全局也能算对

FlashAttention 的数学关键是**在线更新 softmax**。

普通 softmax 需要先扫一遍找最大值，再扫一遍算指数，再扫一遍归一化——必须看完所有数。

**在线 softmax 只需一遍扫**，分块处理时可以逐步更新。

更新规则：
- 当前最大值：`m`
- 当前归一化系数：`ℓ = Σ exp(xⱼ - m)`

新来一批数（最大值 `m_new`）时：
- `new_m = max(m, m_new)`
- `new_ℓ = ℓ · exp(m - new_m) + ℓ_new · exp(m_new - new_m)`

这样**数学上完全等价**于一次性算完！
"""),

# ── Cell 6: 在线 Softmax 演示 ─────────────────────────────────────────────
code("""# ── 在线 Softmax 演示 ──────────────────────────────────────────────────

def batch_softmax(x):
    '一次性 softmax（标准做法）'
    x = x - x.max()
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def online_softmax_2blocks(x):
    '把 x 分两块，用在线更新算 softmax'
    n = len(x)
    half = n // 2
    block1, block2 = x[:half], x[half:]

    # 处理第一块
    m1 = block1.max()
    exp1 = np.exp(block1 - m1)
    l1 = exp1.sum()

    # 处理第二块，同时更新全局状态
    m2 = block2.max()
    new_m = max(m1, m2)

    # 用衰减系数更新
    exp2 = np.exp(block2 - new_m)
    l2 = exp2.sum()
    new_l = l1 * np.exp(m1 - new_m) + l2  # 修正第一块的系数

    # 合并两块的归一化结果
    result = np.zeros(n)
    result[:half] = exp1 * np.exp(m1 - new_m) / new_l
    result[half:] = exp2 / new_l
    return result


# 验证：两种方法结果相同
x = np.array([1.0, 3.0, 2.0, 5.0, 1.0, 4.0])

result_batch = batch_softmax(x)
result_online = online_softmax_2blocks(x)

print("原始数组:", x)
print("一次性 softmax:", result_batch.round(6))
print("在线 softmax:  ", result_online.round(6))
print(f"最大差异: {np.abs(result_batch - result_online).max():.2e}")
print(f"结果加和: {result_online.sum():.6f}（应该等于1.0）")
print("\\n结论: 两种方法数学上完全等价！")
"""),

# ── Cell 7: 分块 Attention 实现 ───────────────────────────────────────────
md("""## 3. 分块 Attention（FlashAttention 简化版）

有了在线 softmax 的工具，我们可以实现分块 Attention：

```
for 每个 Q 块 (行方向):
    for 每个 K/V 块 (列方向):
        计算这个小块的相似度
        用在线 softmax 更新累积值
        直接更新输出，不存中间 [n,n] 矩阵
```

整个过程，最大的矩阵只有一个小块大小（block_size × block_size）。
"""),

# ── Cell 8: 分块 Attention 代码 ───────────────────────────────────────────
code("""# ── 分块 Attention 实现（FlashAttention 简化版）────────────────────────

def tiled_attention(Q, K, V, block_size=32):
    \"\"\"
    分块 Attention（IO感知版本的简化实现）

    与标准 Attention 数学等价，但：
    - 最大中间矩阵大小：block_size × block_size（而非 n × n）
    - 不存完整的 S 矩阵或 P 矩阵

    参数:
        Q, K, V: 形状均为 [n, d]
        block_size: 每块大小
    返回:
        O: 形状 [n, d]，与 standard_attention 结果相同
    \"\"\"
    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    O = np.zeros((n, d), dtype=np.float64)   # 累积输出
    L = np.zeros(n, dtype=np.float64)         # 归一化系数（分母）
    M = np.full(n, -np.inf, dtype=np.float64) # 每行的当前最大值

    # 外层循环：遍历 K/V 的列块
    for j in range(0, n, block_size):
        Kj = K[j:j+block_size].astype(np.float64)
        Vj = V[j:j+block_size].astype(np.float64)

        # 内层循环：遍历 Q 的行块
        for i in range(0, n, block_size):
            Qi = Q[i:i+block_size].astype(np.float64)
            bsz = Qi.shape[0]                  # 实际块大小（最后一块可能更小）

            # 计算这个小块的相似度得分（只有 bsz × bsz 大小）
            Sij = Qi @ Kj.T * scale

            # 这个小块的行最大值（用于在线 softmax 更新）
            Mij = Sij.max(axis=1)

            # 更新全局最大值
            new_M = np.maximum(M[i:i+bsz], Mij)

            # 衰减系数：旧结果因为最大值增加了，需要乘以 exp(old_M - new_M)
            decay = np.exp(M[i:i+bsz] - new_M)

            # 计算这个块的 exp（相对于新的最大值）
            Pij = np.exp(Sij - new_M[:, None])  # 形状 [bsz, bsz_kv]

            # 在线更新输出和归一化系数
            L[i:i+bsz] = decay * L[i:i+bsz] + Pij.sum(axis=1)
            O[i:i+bsz] = decay[:, None] * O[i:i+bsz] + Pij @ Vj

            # 更新最大值记录
            M[i:i+bsz] = new_M

    # 最终归一化（除以累积的分母）
    O = O / L[:, None]
    return O.astype(np.float32)


# 验证两种实现结果相同
out_tiled = tiled_attention(Q, K, V, block_size=16)

max_diff = np.abs(out_standard - out_tiled).max()
mean_diff = np.abs(out_standard - out_tiled).mean()

print(f"标准 Attention 输出前3行前4列:\\n{out_standard[:3, :4].round(4)}")
print(f"\\n分块 Attention 输出前3行前4列:\\n{out_tiled[:3, :4].round(4)}")
print(f"\\n最大绝对差异: {max_diff:.2e}")
print(f"平均绝对差异: {mean_diff:.2e}")
print(f"\\n结论: {'✓ 两种方法结果相同（误差在浮点精度范围内）' if max_diff < 1e-4 else '✗ 结果不同，检查实现！'}")
"""),

# ── Cell 9: 不同块大小的一致性 ────────────────────────────────────────────
code("""# ── 不同块大小的一致性测试 ──────────────────────────────────────────────

print("测试不同块大小下的一致性：")
print(f"{'块大小':>10} | {'最大差异':>12} | {'结果一致':>8}")
print("-" * 38)

for bs in [4, 8, 16, 32, 64]:
    out = tiled_attention(Q, K, V, block_size=bs)
    diff = np.abs(out_standard - out).max()
    ok = "✓" if diff < 1e-4 else "✗"
    print(f"{bs:>10} | {diff:>12.2e} | {ok:>8}")

print("\\n结论：块大小不影响数值结果，只影响内存使用！")
"""),

# ── Cell 10: 内存对比 ─────────────────────────────────────────────────────
md("""## 4. 内存使用量：理论分析

| 序列长度 n | 标准 Attention（O(n²)） | 分块 Attention（O(n)） | 节省比例 |
|---|---|---|---|
| 512  | 1 MB   | 0.008 MB | 128× |
| 1024 | 4 MB   | 0.016 MB | 256× |
| 2048 | 16 MB  | 0.032 MB | 512× |
| 4096 | 64 MB  | 0.064 MB | 1000× |
| 8192 | 256 MB | 0.128 MB | 2000× |

（假设 float32，单头，块大小 64，只统计注意力权重矩阵的内存）

注：实际 GPU 上的节省来自减少 HBM 读写次数，不只是峰值内存。
"""),

# ── Cell 11: 内存可视化 ───────────────────────────────────────────────────
code("""# ── 内存使用量可视化 ────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
d_model = 64      # 每个头的维度
block_size = 64   # FlashAttention 块大小
bytes_per_float = 4

# 标准 Attention：S 矩阵大小 n × n
standard_mem = [n * n * bytes_per_float / 1e6 for n in seq_lengths]

# 分块 Attention：只需存一个块 block_size × block_size
tiled_mem = [block_size * block_size * bytes_per_float / 1e6 for n in seq_lengths]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：绝对内存
axes[0].plot(seq_lengths, standard_mem, 'r-o', label='标准 Attention O(n²)', linewidth=2)
axes[0].plot(seq_lengths, tiled_mem, 'g-o', label=f'分块 Attention O(1)（块={block_size}）', linewidth=2)
axes[0].set_xlabel('序列长度 n')
axes[0].set_ylabel('中间矩阵内存 (MB)')
axes[0].set_title('注意力权重矩阵内存占用')
axes[0].legend()
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(seq_lengths)
axes[0].set_xticklabels([str(n) for n in seq_lengths], rotation=45)

# 右图：节省倍数
savings = [s / t for s, t in zip(standard_mem, tiled_mem)]
axes[1].bar(range(len(seq_lengths)), savings, color='steelblue', alpha=0.7)
axes[1].set_xticks(range(len(seq_lengths)))
axes[1].set_xticklabels([str(n) for n in seq_lengths], rotation=45)
axes[1].set_xlabel('序列长度 n')
axes[1].set_ylabel('内存节省倍数')
axes[1].set_title('FlashAttention 节省的中间矩阵内存倍数')
for i, v in enumerate(savings):
    axes[1].text(i, v + 5, f'{v:.0f}×', ha='center', fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../docs/assets/24-flash-attention-memory.png', dpi=100, bbox_inches='tight')
plt.show()
print("图表已保存到 docs/assets/24-flash-attention-memory.png")
"""),

# ── Cell 12: 数值稳定性测试 ───────────────────────────────────────────────
code("""# ── 数值稳定性测试 ──────────────────────────────────────────────────────

# 测试极端值（大值、小值、全相同）下两种方法的稳定性
print("数值稳定性测试：")
print(f"{'测试场景':>20} | {'标准 nan/inf':>12} | {'分块 nan/inf':>12} | {'差异':>10}")
print("-" * 62)

def check_stability(Q, K, V, name):
    out_s = standard_attention(Q, K, V)
    out_t = tiled_attention(Q, K, V, block_size=16)
    s_bad = np.any(~np.isfinite(out_s))
    t_bad = np.any(~np.isfinite(out_t))
    diff = np.abs(out_s - out_t).max() if not (s_bad or t_bad) else float('nan')
    print(f"{name:>20} | {'✗ 有异常' if s_bad else '✓ 正常':>12} | {'✗ 有异常' if t_bad else '✓ 正常':>12} | {diff:>10.2e}")

n2, d2 = 32, 8
rng = np.random.RandomState(0)

check_stability(rng.randn(n2, d2).astype(np.float32),
                rng.randn(n2, d2).astype(np.float32),
                rng.randn(n2, d2).astype(np.float32), "正常输入")

check_stability((rng.randn(n2, d2) * 10).astype(np.float32),
                (rng.randn(n2, d2) * 10).astype(np.float32),
                rng.randn(n2, d2).astype(np.float32), "大值输入 (×10)")

check_stability((rng.randn(n2, d2) * 0.01).astype(np.float32),
                (rng.randn(n2, d2) * 0.01).astype(np.float32),
                rng.randn(n2, d2).astype(np.float32), "小值输入 (×0.01)")

check_stability(np.ones((n2, d2), dtype=np.float32),
                np.ones((n2, d2), dtype=np.float32),
                rng.randn(n2, d2).astype(np.float32), "全1输入")
"""),

# ── Cell 13: 总结 ─────────────────────────────────────────────────────────
md("""## 总结

**FlashAttention 教会我们什么？**

1. **算法优化不只是减少 FLOPS**——有时候，同样的计算，换个顺序就能快几倍
2. **硬件感知很重要**——了解内存层次结构（SRAM vs HBM），才能写出"聪明"的代码
3. **数学等价 ≠ 实现等价**——分块 softmax 数学上完全等价，但在 GPU 上快得多
4. **基础工程创新** 往往比花哨的新模型结构影响更大——FlashAttention 让所有用 Attention 的模型都受益

**一句话总结：** FlashAttention = 同样的数学，更少的档案柜来回跑，速度快 2–4×，内存省 5–20×。

---
*论文：Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. arXiv:2205.14135*
"""),

]

NB["cells"] = cells

import os
out_path = os.path.join(os.path.dirname(__file__), "../notebooks/24-flash-attention-2022.ipynb")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(NB, f, ensure_ascii=False, indent=1)

print(f"Generated: {out_path}")
print(f"Total cells: {len(cells)}")
