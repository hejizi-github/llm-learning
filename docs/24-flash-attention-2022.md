# 节点24：FlashAttention（2022）——让注意力机制快 10 倍的工程魔法

> **所处时代**：2022年。Transformer 已经统治了 NLP，但工程师们发现了一个头疼的问题：当序列变长，Attention 就变得极度缓慢——不是因为计算量大，而是因为**显存读写太慢**。Tri Dao 等人提出 FlashAttention，在不改变任何数学的情况下，让 Attention 快了 2–4 倍，内存使用降低了 5–20 倍。

---

## 背景故事：Attention 为什么慢？

### 直觉类比：书桌 vs 档案柜

假设你在做数学作业，你有：
- **书桌**（很小，只能放几页纸）— 对应 GPU 的 SRAM（片上高速缓存）
- **档案柜**（很大，能放几千页纸，但走过去很慢）— 对应 GPU 的 HBM（显卡内存）

标准 Attention 的做法是这样的：

1. 把所有 Q、K、V 矩阵存进档案柜
2. 算 `QKᵀ`，结果存进档案柜
3. 再走到档案柜，把结果取出来算 Softmax
4. 再走到档案柜，把 Softmax 结果取出来乘以 V
5. 最终结果存进档案柜

你看，为了算一次 Attention，你要来回走**好几趟**档案柜。序列越长，档案柜里的东西越多，来回越费时间。

**FlashAttention 的洞察**：
> "我能不能不来回跑？把整个计算分成小块，每块都放在书桌上算完，不用存中间结果到档案柜。"

---

## Attention 的数学（自包含讲解）

### 什么是 Attention？

给定三个矩阵：
- **Q**（Query，"问题"）：形状 `[n, d]`
- **K**（Key，"索引"）：形状 `[n, d]`
- **V**（Value，"答案"）：形状 `[n, d]`

标准 Attention 计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right) V$$

拆开来看：
1. `S = QKᵀ / √d` — 计算每对 token 之间的"相似度分数"，得到 `[n, n]` 矩阵
2. `P = softmax(S)` — 把分数变成概率（每行加起来等于1）
3. `O = PV` — 用概率加权混合 V 的内容

**问题在哪里？** 步骤1产生的 S 和步骤2产生的 P 都是 `[n, n]` 大小的矩阵。当 n=1024 时，这是 100万个数！全都要存到显存里，然后再读出来。

### Softmax 是什么？

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

用中文说：把每个数取指数，然后除以所有数的指数之和，让结果加起来等于1。

**数值稳定技巧**：直接算 `e^x` 当 x 很大（比如 100）时会溢出（变成无穷大）。标准做法是先减去最大值：

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

减去最大值不影响结果（分子分母同时乘以 `e^{-max(x)}`，约掉了），但能防止溢出。

---

## FlashAttention 的核心技巧：分块计算

### 在线 Softmax（Online Softmax）

FlashAttention 的关键数学工具是**在线更新 softmax**——不需要看完所有数，就能逐块算出正确的 softmax。

假设我们分两块处理序列。第一块完成后，我们有：
- `m₁`：第一块的最大值
- `ℓ₁`：第一块的 `∑ e^{x_j - m₁}`

当第二块来了（最大值 `m₂`），我们可以更新：
- 新的全局最大值：`m = max(m₁, m₂)`
- 新的归一化系数：`ℓ = ℓ₁ · e^{m₁ - m} + ℓ₂ · e^{m₂ - m}`

这个更新公式保证了：**分块算的结果和一次性算完完全相同**。

### 为什么这样就能省内存？

标准 Attention：
- 必须存整个 `[n, n]` 的 S 矩阵（用来之后算 Softmax）
- 内存：O(n²)

FlashAttention：
- 每次只处理一小块 Q（行方向）和一小块 K/V（列方向）
- 只需要在书桌上放一小块（块大小 B × B）
- 中间结果不存档案柜，直接在书桌上算完
- 内存：O(n) — 仅需存 Q、K、V 和最终输出

---

## 代码直觉：从标准到分块

### 标准 Attention（你现在能理解的版本）

```python
import numpy as np

def standard_attention(Q, K, V):
    d = Q.shape[1]
    scale = 1.0 / np.sqrt(d)
    
    # S = QKᵀ / √d，形状 [n, n]
    S = Q @ K.T * scale
    
    # 稳定 Softmax：先减最大值
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=1, keepdims=True)
    
    # 输出 = PV
    return P @ V
```

**内存问题**：`S` 和 `P` 都是 `[n, n]`。n=4096 时约 128MB，还没算 Q、K、V。

### 分块 Attention（FlashAttention 的简化版）

```python
def tiled_attention(Q, K, V, block_size=32):
    n, d = Q.shape
    O = np.zeros_like(Q)       # 最终输出
    L = np.zeros(n)             # 归一化系数（分母）
    M = np.full(n, -np.inf)    # 当前最大值记录

    for j in range(0, n, block_size):    # 遍历 K/V 的块
        Kj = K[j:j+block_size]
        Vj = V[j:j+block_size]
        
        for i in range(0, n, block_size):  # 遍历 Q 的块
            Qi = Q[i:i+block_size]
            
            # 计算这个小块的相似度
            Sij = Qi @ Kj.T / np.sqrt(d)
            
            # 在线更新最大值和归一化系数
            Mij = Sij.max(axis=1)
            new_M = np.maximum(M[i:i+block_size], Mij)
            
            # 更新输出（带上旧结果的衰减系数）
            decay = np.exp(M[i:i+block_size] - new_M)
            Pij = np.exp(Sij - new_M[:, None])
            
            L[i:i+block_size] = decay * L[i:i+block_size] + Pij.sum(axis=1)
            O[i:i+block_size] = (decay[:, None] * O[i:i+block_size] + Pij @ Vj)
            M[i:i+block_size] = new_M
    
    # 最终归一化
    return O / L[:, None]
```

**关键点**：整个过程中，我们只在"书桌"上放 Qi、Kj、Vj（小块），从不存完整的 `[n, n]` 矩阵。

---

## 为什么这个工作很重要？

| | 标准 Attention | FlashAttention |
|---|---|---|
| 内存复杂度 | O(n²) | O(n) |
| 内存读写次数（HBM） | O(n²/B) 次 | O(n²/M) 次（M是SRAM大小） |
| 计算结果 | 精确 | **精确相同**（不是近似！） |
| GPT-3 序列4096时速度 | 基准 | 快 2–4× |

注意：FlashAttention **不是近似算法**。它数学上与标准 Attention 完全等价，只是执行顺序不同。

### 对 LLM 的影响

FlashAttention 使得训练更长序列成为可能：
- GPT-3 训练：序列长度 2048
- 使用 FlashAttention 后：可以用 16k、甚至 100k+ 的序列长度
- GPT-4、Claude、LLaMA 等现代大模型都用了类似技术

---

## 参考文献

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022. arXiv:2205.14135

---

## 下一节预告

节点25 将介绍 **Instruction Tuning（指令微调）** 和 **RLHF**——如何通过人类反馈让模型更"听话"，这正是 ChatGPT 成功的核心秘密。
