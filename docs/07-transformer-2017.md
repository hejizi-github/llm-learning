# 节点07：Transformer（2017）——抛弃循环，用注意力统治一切

> **所处时代**：2017 年。Attention 机制已经让机器翻译大幅改善，但模型仍然依赖 RNN/LSTM——因为 RNN 天然处理序列，研究者们认为"序列模型必须按时间步循环"。Vaswani 等人的问题是：**如果我们完全不要循环，只用 Attention，会怎样？**

---

## 背景故事：RNN 的两个根本问题

### 问题一：计算无法并行

LSTM 处理一个 $n$ 词句子时，必须按顺序：先算第 1 步，再算第 2 步……第 $n$ 步。前一步的隐状态 $h_{t-1}$ 是后一步的输入，这个**顺序依赖**让 GPU 的并行能力几乎完全浪费。

一个句子 100 个词？100 步串行。十亿条训练数据？速度成了致命瓶颈。

### 问题二：远距离依赖仍然困难

加了 Attention 之后，解码器每步都能"回头看"所有编码器状态——这解决了**解码**时的信息获取问题。但**编码器内部**仍然是 RNN：

```
"The animal didn't cross the street because it was too tired"
```

"it" 指代 "animal" 还是 "street"？要搞清楚这一点，编码器需要在步骤 1（"The"）和步骤 9（"it"）之间建立联系。RNN 要经过 8 个时间步的信息传递，路径越长，梯度越弱，联系越模糊。

### Transformer 的核心洞察

Bahdanau Attention 是解码器对编码器的注意力：**解码器的每一步去查询所有编码器状态**。

Vaswani 等人问了一个更激进的问题：**能不能直接让每个词对序列中的所有其他词做 Attention？** 也就是——Self-Attention。

如果编码阶段每个词都能直接关注任意其他词（不经过中间步骤），远距离依赖问题就从根本上消失了。而且，每个词的 Self-Attention 计算相互独立，可以完全并行。

---

## 核心机制一：Scaled Dot-Product Attention

### 直觉类比：图书馆查询系统

想象一个图书馆：
- 你有一个**查询（Query）**："我想找关于深度学习历史的书"
- 每本书有一个**键（Key）**："深度学习入门"、"神经网络基础"……
- 每本书有一个**值（Value）**：书的实际内容

系统计算你的查询和每个键的相似度，相似度高的书获得更高权重，最终返回所有书内容的加权组合。

Attention 机制做的正是这件事：
- $\mathbf{Q}$（Query）：我想找什么
- $\mathbf{K}$（Key）：每个位置提供什么标签
- $\mathbf{V}$（Value）：每个位置的实际内容

### 公式

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

逐步拆解（假设序列长度 $n=4$，向量维度 $d_k=64$）：

**Step 1**：计算相似度矩阵
$$\mathbf{S} = \mathbf{Q}\mathbf{K}^\top \quad \text{形状：} (4, 4)$$
$S_{ij}$ = 位置 $i$ 的查询与位置 $j$ 的键的点积，反映"$i$ 应该关注 $j$ 多少"。

**Step 2**：缩放（这是"Scaled"的意思）
$$\mathbf{S}' = \frac{\mathbf{S}}{\sqrt{d_k}}$$

**数学小补丁：为什么要除以 $\sqrt{d_k}$？**

点积是 $d_k$ 个乘积之和。如果每个元素的方差是 1，那么 $d_k$ 个元素相加，方差变成 $d_k$，标准差是 $\sqrt{d_k}$。值越大，送入 softmax 的数越极端，梯度越趋于 0（softmax 饱和）。除以 $\sqrt{d_k}$ 把方差归一化回 1，让 softmax 工作在合理区间。

**Step 3**：Softmax 归一化为注意力权重
$$\mathbf{A} = \text{softmax}(\mathbf{S}') \quad \text{形状：} (4, 4), \text{每行和为 1}$$

**Step 4**：加权聚合值
$$\text{Output} = \mathbf{A}\mathbf{V} \quad \text{形状：} (4, d_v)$$

### 与 Bahdanau Attention 的对比

| 维度 | Bahdanau（加性）Attention | Scaled Dot-Product |
|------|--------------------------|-------------------|
| 计算方式 | $\text{score}(\mathbf{s}, \mathbf{h}) = \mathbf{v}^\top \tanh(\mathbf{W}_1\mathbf{s} + \mathbf{W}_2\mathbf{h})$ | $\mathbf{q} \cdot \mathbf{k} / \sqrt{d_k}$ |
| 额外参数 | 需要 $\mathbf{W}_1, \mathbf{W}_2, \mathbf{v}$ | 无额外参数（Q/K/V 的线性变换另算）|
| 计算效率 | 较慢（逐对计算）| 快（矩阵乘法，GPU 友好）|
| 应用场景 | 跨序列（解码器 → 编码器）| 可用于自注意力（序列内部）|

关键升级：**Dot-Product Attention 是 Bahdanau 加性 Attention 的高效替代**，且通过 $\sqrt{d_k}$ 缩放解决了维度增大时的数值稳定性问题。

---

## 核心机制二：Multi-Head Attention

单头 Attention 每次只从一个"视角"整合信息。Multi-Head Attention 的思路：**并行运行 $h$ 个 Attention 头，每个头用不同的线性投影，捕捉不同类型的依赖关系**。

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$

其中每个头：
$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

**直觉**：一个句子中，"it" 需要同时捕捉语法依赖（"it" 是主语）和语义依赖（"it" 指代 "animal"）。一个头可能专注于语法，另一个头专注于语义指代。8 个头 = 8 种视角同时分析。

原始论文使用 $h=8$ 个头，每个头维度 $d_k = d_v = d_\text{model}/h = 64$（总维度 512）。

---

## 核心机制三：位置编码（Positional Encoding）

Self-Attention 的并行化有代价：**它对位置信息完全无感**。"猫追狗"和"狗追猫"——如果只看词的集合，完全一样。

Transformer 用**位置编码**把位置信息注入输入向量：

$$\text{PE}(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_\text{model}}}\right)$$

$$\text{PE}(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_\text{model}}}\right)$$

**为什么用正弦/余弦？**

- 不同频率的正弦波组合，每个位置得到唯一的编码（类似二进制计数）
- 固定公式，不需要学习新参数
- 对任意长度的序列都能生成位置编码
- 重要性质：$\text{PE}(\text{pos}+k)$ 可以用 $\text{PE}(\text{pos})$ 的线性变换表示，模型可以学会利用相对位置信息

最终输入 = 词嵌入向量 + 位置编码向量（直接相加）

---

## 完整 Encoder 架构

一个 Encoder 层包含：

```
输入
  ↓
[Multi-Head Self-Attention]
  ↓
Add & Norm (残差连接 + 层归一化)
  ↓
[Feed-Forward Network]
  ↓
Add & Norm
  ↓
输出
```

**残差连接**：输出 = $\text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))$。防止深层网络的梯度消失（同 ResNet 的思路）。

**前馈网络（FFN）**：两层线性变换，中间用 ReLU：
$$\text{FFN}(\mathbf{x}) = \max(0,\ \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

原始 Transformer 堆叠 6 个这样的 Encoder 层。

---

## 为什么 Transformer 是革命性的

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 计算并行性 | 无（顺序依赖）| 完全并行（矩阵乘法）|
| 远距离依赖路径长度 | $O(n)$（$n$ 个时间步）| $O(1)$（直接 Attention）|
| 训练速度（相同数据量）| 慢 | 快（GPU 充分利用）|
| 扩展性 | 有限 | 极强（Scale Law 的基础）|

Transformer 发表时（NeurIPS 2017）在英法翻译上超越所有之前的模型，BLEU 分数 41.0，训练成本是此前 SOTA 的十分之一。

---

## 局限与衔接

### Transformer 的局限

1. **对序列长度的平方复杂度**：Self-Attention 计算 $\mathbf{Q}\mathbf{K}^\top$ 的复杂度是 $O(n^2 d)$，处理超长文本（如整本书）代价极高
2. **没有预训练**：原始 Transformer 是端到端的翻译模型，需要大量任务特定数据
3. **位置编码是固定的**：对超出训练长度的序列泛化能力有限

### 它催生了什么

| 年份 | 突破 | 对 Transformer 的改进 |
|------|------|----------------------|
| 2018 | GPT（OpenAI） | 只用 Decoder，无监督预训练 → 迁移学习 |
| 2018 | BERT（Google）| 只用 Encoder，双向 Masked Language Model |
| 2020 | GPT-3 | 1750 亿参数，证明 Scale Law |
| 2022 | ChatGPT | RLHF 对齐，对话式 LLM |

Transformer 不只是一个更好的翻译模型——它是整个现代大语言模型时代的基础架构。

---

## 可运行 Notebook

▶ [07-transformer-2017.ipynb](../notebooks/07-transformer-2017.ipynb)

内容：
1. Scaled Dot-Product Attention（纯 NumPy 手撕）
2. 注意力权重可视化
3. Multi-Head Attention
4. Positional Encoding 可视化
5. Transformer Encoder Block
6. PyTorch nn.MultiheadAttention 对比验证

---

## 引用溯源

- **Vaswani et al., 2017**：Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30. arXiv:[1706.03762](https://arxiv.org/abs/1706.03762)
- **Bahdanau et al., 2015**：见节点06，arXiv:[1409.0473](https://arxiv.org/abs/1409.0473)
- **He et al., 2016（残差连接）**：He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*. arXiv:[1512.03385](https://arxiv.org/abs/1512.03385)

---

*下一节：节点08 — BERT（2018）：双向预训练，语言理解的新范式*
