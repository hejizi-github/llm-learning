# 节点 08 — Word2Vec（词向量）：让词找到它的"邻居"

> **突破时间**：2013 年
> **关键人物**：Tomas Mikolov、Kai Chen、Greg Corrado、Jeffrey Dean（谷歌）
> **核心论文**：
> - Mikolov et al. (2013) *Efficient Estimation of Word Representations in Vector Space* [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
> - Mikolov et al. (2013) *Distributed Representations of Words and Phrases and their Compositionality* [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)
> **背景论文**：
> - Bengio et al. (2003) *A Neural Probabilistic Language Model*, JMLR [jmlr.org/papers/v3/bengio03a.html](https://www.jmlr.org/papers/v3/bengio03a.html)

---

## 故事：词的"身份证"危机

2012 年的 NLP（自然语言处理）工程师有一个头疼的问题：  
**怎么让计算机理解"猫"和"狗"是近义词，而"猫"和"飞机"不是？**

最古老的方法叫 **one-hot 编码**：给每个词分配一个独占的位置，
其余位置全填 0。词典有 10 万个词，就给每个词一个 10 万维的向量，
里面只有一个 1，其余 99,999 个全是 0。

```
"猫"  = [0, 0, 1, 0, 0, ..., 0]   （10 万维）
"狗"  = [0, 0, 0, 1, 0, ..., 0]
"飞机"= [0, 1, 0, 0, 0, ..., 0]
```

这有两个致命问题：

1. **维度爆炸**：10 万维向量塞满内存，计算极慢。
2. **词义盲目**：任意两个词的向量做点积都等于 0，计算机看不出"猫"和"狗"的相似性。

> **一个直觉**：
> 你不需要知道"芒果"的定义，只要见过足够多句子，
> 就知道"芒果很甜"、"芒果是水果"……这跟"香蕉很甜"、"香蕉是水果"太像了。
> **词的意义，藏在它的邻居里**。

---

## 背景：神经语言模型的先驱

Bengio 等人（2003）已经证明，用神经网络可以学出一个 **稠密词向量**（dense vector）——
把每个词映射到几十到几百维的浮点数向量，语义相近的词自动聚在一起。

但 Bengio 的模型训练太慢，在百万词级别的语料上就吃不消，
更别说谷歌手头的十亿词语料了。

---

## 突破：用"邻居"训练词向量

Mikolov 团队的核心洞察：**不需要语言模型的全部能力，只要能预测邻居就够了**。

他们提出了两个极简架构：

| 模型 | 输入 | 预测目标 |
|------|------|---------|
| **CBOW**（连续词袋） | 周围 $k$ 个词 | 中心词 |
| **Skip-gram** | 中心词 | 周围 $k$ 个词 |

这两个模型去掉了隐藏层的非线性激活函数，变成了"几乎是线性"的浅层网络，
训练速度比 Bengio 的模型快了 **10 倍以上**。

---

## 数学：Skip-gram 的四步推导

### 第一步：词向量矩阵

假设词典大小 $V$，词向量维度 $d$（一般取 100～300）。

模型有两个矩阵：
- 输入矩阵 $\mathbf{W} \in \mathbb{R}^{V \times d}$：每行是中心词的向量
- 输出矩阵 $\mathbf{W'} \in \mathbb{R}^{V \times d}$：每行是上下文词的向量

当中心词是 $w_t$ 时，取出其行向量 $\mathbf{v}_{w_t} \in \mathbb{R}^d$。

> **矩阵和行向量是什么？**
>
> 矩阵就是数字表格。$\mathbf{W} \in \mathbb{R}^{V \times d}$ 表示这张表格有 $V$ 行、$d$ 列。
> "行向量"就是表格中某一行的全部数字，长度为 $d$。
> 比如 $d=3$，词"猫"的行向量可能是 $[0.2,\ -0.5,\ 0.8]$。

### 第二步：计算相似分数

对词典里的每一个词 $w$，计算中心词向量与它的**点积（dot product）**：

$$s(w,\ w_t) = \mathbf{v}'_w \cdot \mathbf{v}_{w_t}$$

点积越大，代表两个向量方向越相近，"相似度越高"。

> **点积是什么？**
>
> 两个长度相同的数组，对应位置相乘后求和。
> $$[a_1, a_2, a_3] \cdot [b_1, b_2, b_3] = a_1 b_1 + a_2 b_2 + a_3 b_3$$
> 你可以把它理解为"两个箭头方向有多一致"的量。方向完全相同时最大，垂直时为 0。

### 第三步：Softmax 变成概率

把所有词的分数转成概率分布，让所有词的概率加起来等于 1：

$$P(w \mid w_t) = \frac{e^{s(w,\, w_t)}}{\displaystyle\sum_{w' \in V} e^{s(w',\, w_t)}}$$

这个操作叫 **softmax**。

> **$e$ 和 softmax 是什么？**
>
> $e \approx 2.718$，是一个数学常数（称为"自然底数"，初中不要求记住它的值）。
> $e^x$ 的特点是**永远大于 0**，且随 $x$ 增大而快速增大。
> Softmax 用 $e^x$ 是为了：① 保证所有概率 $> 0$；② 让大分数的词获得更多概率。

### 第四步：最大化上下文词的概率

对于中心词 $w_t$，窗口内的上下文词是 $w_{t-k}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+k}$。

Skip-gram 的目标：让这些真实邻居的概率尽可能大，即最大化：

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \sum_{-k \le j \le k,\, j \ne 0} \log P(w_{t+j} \mid w_t)$$

用**梯度下降**反复更新 $\mathbf{W}$ 和 $\mathbf{W'}$，直到损失最小。

> **梯度下降是什么？**
>
> 想象你站在山上，闭眼摸路想走到最低点。
> 你每次用脚感受哪个方向上坡最陡，然后**反着**那个方向踩一小步（往下走）。
> 重复无数次，最终你会走到谷底。
>
> - "**梯度**"= 当前位置的"最陡**上坡**方向"（函数值增长最快的方向）
> - "**下降**"= 沿着**负梯度**（反方向）走一小步——朝上坡的反面，即下坡方向迈
>
> 代码里写 `W = W - lr × gradient`，减号就是"反着梯度走"。
>
> 在 Word2Vec 里，"山"是损失函数，"谷底"是让预测最准的矩阵参数值。
> 每次"踩一步"就是用数学公式自动调整 $\mathbf{W}$ 和 $\mathbf{W'}$ 里的数字，
> 让预测变得稍微准一点。

---

## 负采样：绕过"求和"的计算地狱

上面 softmax 分母里有 $\sum_{w' \in V}$，每次更新要遍历整个词典（10 万词），**太慢了**。

Mikolov 2013b 提出**负采样（Negative Sampling）**：

> 与其说"让真实邻居概率最大"，
> 不如说"能不能把真实邻居（正样本）和随机噪声词（负样本）区分开来？"

每次训练只看：
- 1 个正样本（真实的上下文词）
- $k$ 个负样本（从词典随机抽出来的词，通常 $k=5 \sim 20$）

目标函数变成了：

$$\mathcal{L}_{\text{neg}} = \log \sigma(\mathbf{v}'_{w_{\text{pos}}} \cdot \mathbf{v}_{w_t})
  + \sum_{i=1}^{k} \log \sigma(-\mathbf{v}'_{w_{\text{neg}_i}} \cdot \mathbf{v}_{w_t})$$

其中 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 是 sigmoid 函数（把任意实数压缩到 0～1 之间）。

---

## 神奇的向量运算

训练后，词向量学到了真实的语义关系：

```
vec("国王") - vec("男人") + vec("女人") ≈ vec("女王")
vec("巴黎") - vec("法国") + vec("日本") ≈ vec("东京")
```

这说明词向量空间捕捉到了"性别"、"首都-国家"这样的语义方向。

---

## 为什么这个想法这么厉害？

1. **速度**：在十亿词语料上，数小时内训练完，比前代快 10 倍以上。
2. **通用性**：词向量可以被迁移到任何 NLP 任务（情感分析、机器翻译、问答）。
3. **涌现性**：模型从没被教过"国王 - 男人 + 女人 = 女王"，它自己学出来了。
4. **奠基作用**：Word2Vec 是后续 GloVe、FastText、以及 Transformer 位置编码的直接前驱。

---

## 局限性

- Softmax 分母遍历整个词典，原始方案在大词典下极慢（负采样才解决了这个问题）。
- 一词一向量：同一个词只有一个向量，无法区分"苹果（水果）"和"苹果（公司）"。
- 窗口固定：只看局部上下文，不能捕捉长距离依赖（这是 Transformer 出现的动机之一）。
- 静态向量：向量训练完就固定了，不随语境变化（ELMo、BERT 解决了这个问题）。

---

## 动手实验

[word2vec.ipynb](./word2vec.ipynb) — 从零手撕 Skip-gram + 负采样，验证"词向量能感知语义"。

---

## 参考文献

- Mikolov et al. (2013a) *Efficient Estimation of Word Representations in Vector Space* [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
- Mikolov et al. (2013b) *Distributed Representations of Words and Phrases and their Compositionality* [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)
- Bengio et al. (2003) *A Neural Probabilistic Language Model*, JMLR [jmlr.org/papers/v3/bengio03a.html](https://www.jmlr.org/papers/v3/bengio03a.html)
