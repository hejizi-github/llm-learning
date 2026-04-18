# 节点06：Attention 机制（2015）——让网络学会"看重点"

> **所处时代**：2014–2015 年。深度学习刚刚用 LSTM 打通了序列任务（语音识别、机器翻译），但随着句子变长，翻译质量开始下滑。研究者们开始意识到：问题不在 LSTM 本身，而在**如何把信息从编码器传给解码器**。

---

## 背景故事：一个瓶颈，一瓶旧酒装不下

### 序列到序列（Seq2Seq）的基本框架

2014 年，Google Brain 的 Sutskever 等人提出了 **Seq2Seq** 模型：用一个 LSTM **编码器**（Encoder）读入整句输入，把信息压缩成一个固定大小的向量 $\mathbf{c}$，再用另一个 LSTM **解码器**（Decoder）从 $\mathbf{c}$ 出发，逐词生成翻译。

```
"The cat sat on the mat"
        ↓ Encoder
    [c: 一个 512 维向量]
        ↓ Decoder
"猫 坐 在 垫子 上"
```

这个思路很优雅，也真的有效——在短句上效果不错。

### 固定向量的信息瓶颈

但有个根本问题：**不管输入多长，编码器都必须把所有信息压缩进同一个固定大小的向量 $\mathbf{c}$**。

想象一下：你要把一整本书的内容写在一张名片上，然后让别人凭借这张名片翻译整本书。名片再好，也装不下一本书。

实验结果证实了这个直觉：输入句子越长，BLEU 分数（机器翻译质量指标）下降越明显。

### 突破口：Bahdanau 的洞见

2014 年，来自蒙特利尔大学的 Dzmitry Bahdanau 和合作者（Cho、Bengio）问了一个关键问题：

> **"解码器在生成每个词时，真的需要一次性用到整个输入吗？"**

答案是"不"。翻译"cat"时，模型更应该关注输入中"cat"附近的词；翻译"mat"时，应该关注句子末尾。**这就是 Attention 的核心直觉**：让解码器在每一步动态决定要"看"编码器输出的哪个部分。

论文《Neural Machine Translation by Jointly Learning to Align and Translate》发表于 ICLR 2015 \[Bahdanau et al., 2015\]，直接催生了 Transformer（2017），成为现代 NLP 的基石。

---

## 原理讲解：注意力是怎么计算的

### 编码器的输出不再是一个向量

在引入 Attention 之前，编码器只输出最后一个时间步的隐状态 $\mathbf{h}_T$（最后那一格）。

引入 Attention 后，编码器**保留所有时间步的隐状态**：
$$
\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_T
$$
每个 $\mathbf{h}_j$ 都"记住"了输入序列在位置 $j$ 前后的上下文。

### 三步计算：分数 → 权重 → 向量

假设解码器现在在生成第 $i$ 个输出词，上一步的隐状态是 $\mathbf{s}_{i-1}$。

**第一步：计算对齐分数（Alignment Scores）**

对每个编码器位置 $j$，计算一个分数 $e_{ij}$，表示"生成第 $i$ 个词时，应该对输入位置 $j$ 给多少关注"：

$$
e_{ij} = \mathbf{v}^T \tanh\!\left(\mathbf{W}_a \mathbf{s}_{i-1} + \mathbf{U}_a \mathbf{h}_j\right)
$$

这里 $\mathbf{W}_a$、$\mathbf{U}_a$、$\mathbf{v}$ 是**学习到的参数**（训练时优化）。这个公式叫"加性注意力"（Additive Attention）或"Bahdanau Attention"。

直觉：它在问"当前的解码状态 $\mathbf{s}_{i-1}$ 和编码器位置 $j$ 的隐状态 $\mathbf{h}_j$ 有多匹配？"

**第二步：Softmax 归一化（得到注意力权重）**

把所有分数 $e_{i1}, e_{i2}, \ldots, e_{iT}$ 转成概率分布：

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\displaystyle\sum_{k=1}^{T} \exp(e_{ik})}
$$

> **📐 数学小补丁：Softmax**
>
> Softmax 是把一组任意实数转成"概率"（非负且和为 1）的函数。
>
> 比如输入 $[1, 2, 3]$：
> - 先取指数：$[e^1, e^2, e^3] = [2.718, 7.389, 20.09]$
> - 再除以总和 $30.197$：$[0.090, 0.245, 0.665]$
>
> 结果全部非负，且和为 1——这就是一个"概率分布"。
>
> 为什么用指数而不是直接归一化？因为指数会**放大**较大的值，让分布更"尖锐"，网络更容易学到"专注于某几个位置"的行为。

结果 $\alpha_{ij}$ 就是"在生成第 $i$ 个词时，对输入位置 $j$ 的注意力权重"。权重越大，意味着越"专注"那个位置。

**第三步：加权求和（得到 Context Vector）**

$$
\mathbf{c}_i = \sum_{j=1}^{T} \alpha_{ij} \cdot \mathbf{h}_j
$$

这就是一个**加权平均**：把编码器的所有隐状态，按注意力权重混合在一起，得到一个"本次解码专用"的上下文向量 $\mathbf{c}_i$。

解码器用 $[\mathbf{c}_i; \mathbf{s}_{i-1}]$ 来生成下一个词。

### 为什么这样更好

| 方面 | 传统 Seq2Seq | 带 Attention 的 Seq2Seq |
|------|------------|------------------------|
| 信息传递 | 单个固定向量 | 每步动态合成 context |
| 长序列 | 性能显著下降 | 保持稳定 |
| 可解释性 | 黑盒 | 可视化注意力矩阵 |
| 参数量增加 | — | $\mathbf{W}_a, \mathbf{U}_a, \mathbf{v}$（少量） |

---

## 可运行 Notebook

配套 Notebook：[`notebooks/06-attention-2015.ipynb`](../notebooks/06-attention-2015.ipynb)

Notebook 内容：
1. **信息瓶颈演示**：比较 LSTM 编码短序列和长序列的隐状态差异
2. **对齐分数计算**：纯 NumPy 实现 Bahdanau 加性注意力
3. **Softmax 手撕**：数值稳定版 Softmax，防止溢出
4. **Context Vector**：加权求和实现
5. **玩具对齐任务**：序列复制任务，演示注意力的学习过程
6. **注意力热图**：可视化 $\alpha$ 矩阵
7. **PyTorch 对比**：`nn.MultiheadAttention` 验证

---

## 局限与衔接：注意力催生了 Transformer

Bahdanau Attention 极大改善了 Seq2Seq，但还有两个问题：

**问题一：仍然是顺序计算**

编码器 LSTM 必须逐步处理序列，前一个时间步的输出是后一步的输入——**无法并行**。对于 1000 词的文章，这意味着 1000 步串行计算。

**问题二：Attention 只用于 Encoder→Decoder**

注意力让解码器能"看"编码器，但编码器自身的每个词仍然只通过 LSTM 的局部依赖来理解上下文。

**Self-Attention 的解法（2017）**

Vaswani 等人（Google Brain）在《Attention Is All You Need》（2017）中提出了一个大胆的想法：**把 LSTM 完全去掉，只用 Attention**。

- 每个词都能直接和序列中任意词建立注意力关系（Self-Attention）
- 整个序列可以完全并行计算
- 多层堆叠形成 Transformer

这就是 GPT、BERT、ChatGPT 的技术起点。没有 Bahdanau 2014 的那一步，就没有 2017 的 Transformer，也就没有现在的大语言模型。

---

## 引用溯源

| 引用键 | 描述 | 来源 |
|--------|------|------|
| \[Bahdanau et al., 2015\] | Attention 原论文，ICLR 2015 | arXiv:1409.0473 |
| \[Cho et al., 2014\] | Seq2Seq / GRU / Encoder-Decoder 框架 | DOI: 10.3115/v1/D14-1179 |
| \[Hochreiter & Schmidhuber, 1997\] | LSTM 原论文（Attention 的前驱） | DOI: 10.1162/neco.1997.9.8.1735 |

---

*下一节点 → 节点07：Transformer（2017）——去掉 RNN，只用 Attention*
