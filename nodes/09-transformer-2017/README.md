# 节点 09 — Transformer：把循环扔掉，只留注意力

> **突破时间**：2017 年
> **关键人物**：Ashish Vaswani、Noam Shazeer、Niki Parmar 等 8 人（Google Brain / Google Research）
> **核心论文**：Vaswani et al. (2017) [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
> **背景论文**：Ba, Kiros & Hinton (2016) [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)

---

## 故事：一个自相矛盾的修补

### 回顾：我们刚刚修了什么

上一节（节点 07）里，我们看到了 Attention 注意力机制：
> 翻译每个词时，不依赖一个"压缩后的便条"，而是**回头看整个原文**，
> 对每个输入词打分，加权求和。

这个机制很好用——它让翻译器不再遗忘长句子。

但是，它有一个**没有被修掉的前提**：

> **每个词还是要等前一个词处理完，才能开始处理**。

这不是注意力的问题——这是 LSTM / RNN 的天然结构。
就像一条流水线：第1个工人先干，干完才能轮到第2个工人。

---

### 问题：为什么"顺序"会成为瓶颈？

想象你要翻译一句 100 个词的句子。

**RNN + Attention 的做法**：
1. 读第 1 个词，更新隐藏状态
2. 读第 2 个词，更新隐藏状态
3. …一共 100 步，**顺序执行，不能并行**

**两个代价**：

1. **慢**：100 个词必须走完 100 步，才能开始翻译。现代 GPU 有几千个核同时运转，
   但顺序流水线让它们大部分时间在**空等**。

2. **信息还是要经过长链**：
   虽然 Attention 可以直接"看"任意输入词，
   但隐藏状态 h 仍然要一步步传下去——
   第1步的信息要经过 99 层 "隐藏状态中转" 才能影响第100步。
   越长的依赖，越容易出现误差积累。

---

### 2017 年的大胆想法

Vaswani 等人问了一个颠覆性的问题：

> **我们到底需要 RNN 干什么？**
>
> 如果 Attention 已经能让每个词"直接看"任意其他词——
> 那为什么还需要一个依赖顺序的 RNN 来维持状态？

答案：**不需要。**

论文标题就是他们的结论：**"Attention Is All You Need"（注意力就够了）**

把 RNN 彻底扔掉，全部用 Attention 来建模词与词的关系。

这就是 **Transformer**。

---

## 核心概念一：缩放点积注意力

### 类比：图书馆的精准查找

假设你在一个图书馆找书：

- 你手里有一张**查询单**（Query，Q）：「我想找量子力学的入门书」
- 每本书有一个**书脊标签**（Key，K）：「量子物理 / 薛定谔 / 1926」
- 每本书里装的是真正的**内容**（Value，V）：书的内容本身

**查找过程**：
1. 把你的查询单和每本书的标签比较（计算相似度）
2. 相似度高的书，给更多"权重"
3. 按权重取出所有书的内容，加权求和——得到你想要的知识

Attention 机制就是这样工作的：

```
相似度 = Q · K^T
权重 = softmax(相似度 / √d_k)
输出 = 权重 × V
```

**公式写全了**：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 为什么要除以 √d_k？

这个 √d_k 是**缩放因子**。

原因：当向量维度 d_k 很大时（比如 512），点积 Q·K^T 的数值会变得很大，
大到 softmax 的梯度几乎消失（所有注意力权重都挤到最大值那里去了）。

除以 √d_k，就像把数值"归一化"回一个合理的范围，让训练更稳定。

> **直觉**：如果你有 512 个评分维度，每个维度随机加起来，
> 结果的标准差会是 √512 ≈ 22。除以 √512 就把方差压回 1。

### 用 Python 手写一遍（简化版）

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Q: (seq_len_q, d_k)   —— 查询矩阵
    K: (seq_len_k, d_k)   —— 键矩阵
    V: (seq_len_k, d_v)   —— 值矩阵
    """
    d_k = Q.shape[-1]
    
    # 第1步：计算相似度（点积）
    scores = Q @ K.T          # (seq_len_q, seq_len_k)
    
    # 第2步：缩放
    scores = scores / np.sqrt(d_k)
    
    # 第3步：softmax 得到注意力权重
    scores = scores - scores.max(axis=-1, keepdims=True)  # 数值稳定性
    exp_scores = np.exp(scores)
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    # 第4步：加权求和 Value
    output = weights @ V      # (seq_len_q, d_v)
    return output, weights
```

---

## 核心概念二：多头注意力

### 类比：从多个角度看同一句话

一个词在句子里扮演的"角色"可能有好几个：

> 句子：「猫咪轻轻地踩在软软的地毯上」

对于「踩」这个词：
- **语法角度**：「踩」是谓语，主语是「猫咪」
- **时态角度**：动作正在发生（结合上下文）
- **动作幅度角度**：「轻轻地」修饰它，说明动作很轻柔

单头注意力只用一套 Q、K、V——它一次只能关注一种关系。

**多头注意力**：同时用 h 套不同的 Q、K、V，每套各学各的。

```
头1 的 Q1K1V1 ——学语法关系
头2 的 Q2K2V2 ——学时态关系
头3 的 Q3K3V3 ——学动作幅度
...
头h 的 QhKhVh ——学第 h 种关系
```

最后把 h 个结果拼在一起，再用一个线性变换压缩：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

其中每个头：$\text{head}_i = \text{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)$

原论文用了 h = 8 个头，每个头的维度 d_k = d_model/h = 64。

---

## 核心概念三：位置编码

### 问题：Attention 不知道词序

Attention 机制是**完全并行**的——它同时看所有词，不分先后。

但这带来了一个新问题：

> 「猫咬了狗」和「狗咬了猫」
> 对 Attention 来说，词集合完全一样——除非它知道词的**位置**。

RNN 不需要显式地告诉位置，因为它本来就是顺序读的。
Transformer 抛弃了 RNN，所以必须**手动注入位置信息**。

### 解决方案：用波的频率编码位置

Vaswani 等人用了一个数学很优雅的方法：
给每个位置 pos 和每个维度 i，计算一个固定的值：

$$PE_{(pos,\; 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos,\; 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

**直觉**：把每个位置想象成一个时钟：

- 维度 0、1：一个走得很快的时钟（短波，区分相邻词）
- 维度 2、3：一个走得中等的时钟
- 维度 d-2、d-1：一个走得极慢的时钟（长波，区分前半句和后半句）

用很多频率的"时钟叠加"，每个位置就有了唯一的"指纹"。

```python
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(max_len, d_model):
    """计算位置编码矩阵"""
    PE = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]          # (max_len, 1)
    div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)  # (d_model/2,)
    
    PE[:, 0::2] = np.sin(position / div_term)   # 偶数维：正弦
    PE[:, 1::2] = np.cos(position / div_term)   # 奇数维：余弦
    return PE

# 可视化：50个位置，64个维度
PE = positional_encoding(50, 64)
plt.figure(figsize=(12, 5))
plt.imshow(PE.T, aspect='auto', cmap='RdBu_r')
plt.xlabel('位置 (词的序号)')
plt.ylabel('维度')
plt.title('位置编码：每个位置的"指纹"')
plt.colorbar()
plt.tight_layout()
plt.savefig('positional_encoding.png', dpi=100)
print("位置编码热力图已保存")
```

---

## 整体架构

Transformer 的结构分成两半：**编码器**（Encoder）和**解码器**（Decoder）。

```
输入句子（英文）                    输出句子（法文）
    ↓                                   ↑
[位置编码 + Embedding]          [Softmax → 预测下一词]
    ↓                                   ↑
┌─────────────────────┐     ┌───────────────────────────┐
│    编码器（×6层）    │     │      解码器（×6层）         │
│                     │     │                           │
│  多头自注意力        │────→│  多头遮掩自注意力           │
│       ↓             │     │       ↓                   │
│  Add & Norm         │     │  Add & Norm               │
│       ↓             │     │       ↓                   │
│  前馈神经网络        │     │  多头交叉注意力 ←──────────┤
│       ↓             │     │       ↓                   │
│  Add & Norm         │     │  Add & Norm               │
└─────────────────────┘     │       ↓                   │
                             │  前馈神经网络              │
                             │       ↓                   │
                             │  Add & Norm               │
                             └───────────────────────────┘
```

**"自"注意力**（Self-Attention）：Q、K、V 都来自同一个句子——
每个词对自己所在句子里的所有词做注意力，理解上下文。

**"交叉"注意力**（Cross-Attention）：Q 来自解码器，K、V 来自编码器——
翻译时，每个目标词"看"整个源句子决定内容。

**Add & Norm**：
- **Add** = 残差连接（Residual）：输出 = 层输入 + 层变换，防止梯度消失
- **Norm** = Layer Normalization（Ba et al., 2016）：对每个样本的所有维度做归一化，
  稳定训练

**前馈神经网络**：每个词独立经过一个两层 MLP（维度 2048 → 512），
增加非线性变换能力。

---

## 为什么 Transformer 能并行训练？

关键：Transformer 里**没有任何"先算第 i 个，再算第 i+1 个"的依赖**。

自注意力层里，每个词的 Q、K、V 都是用输入矩阵乘以权重矩阵得到的——
可以一次性算出所有词的 Q、K、V，再做一次大矩阵乘法。

矩阵乘法是 GPU 最擅长的操作——几千个核同时算，比顺序执行快几百倍。

| 模型 | 训练 512 词的序列需要的计算步数 |
|------|---------------------------|
| RNN | 512 步（顺序） |
| Transformer | **1 步**（全并行） |

---

## 历史影响

这篇 2017 年的论文，直接催生了：

- **2018年**：BERT（Google）—— 用 Transformer 编码器预训练，席卷 NLP 榜单
- **2018年**：GPT（OpenAI）—— 用 Transformer 解码器做语言模型
- **2020年**：GPT-3 —— 1750 亿参数，只用解码器，能写文章写代码
- **2022年**：ChatGPT —— 基于 GPT，加上人类反馈训练
- **2023年至今**：GPT-4、Claude、Gemini …… 全部基于 Transformer

"Attention Is All You Need" 是过去10年 AI 领域被引用最多的论文之一。

---

## 数学小补丁

### 什么是矩阵乘法 (Q @ K.T)？

如果 Q 是一个 (3, 4) 的矩阵，K 是一个 (5, 4) 的矩阵：

```
Q:  3行4列（3个词，每个词4维向量）
K:  5行4列（5个词，每个词4维向量）
K.T: 4行5列（K 的转置）

Q @ K.T → 3行5列（每对词之间的相似度）
结果[i][j] = Q 的第 i 行 点乘 K 的第 j 行
```

**点积 = 两个向量的"相似度"**：
同方向的向量点积大，垂直的向量点积为 0，反方向点积为负数。

### 什么是 softmax？

softmax 把一组任意数转成"概率"（加起来等于1，每个都 > 0）：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

直觉：把所有数都取 e 的幂次（让小差异变大），再除以总和归一化。

---

## 参考文献

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. arXiv:1607.06450. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
