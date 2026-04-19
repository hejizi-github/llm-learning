# 节点 07 — Attention（注意力机制）：让翻译器学会"看"

> **突破时间**：2015 年
> **关键人物**：Dzmitry Bahdanau、Kyunghyun Cho、Yoshua Bengio
> **核心论文**：
> - Bahdanau, Cho & Bengio (2015) [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
> **背景论文**：
> - Sutskever, Vinyals & Le (2014) [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
> - Cho et al. (2014) [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)

---

## 故事：一个越翻越坏的翻译器

### 背景：seq2seq 模型（2014年）

2014 年，Google 的研究者 Sutskever、Vinyals 和 Le
提出了 **seq2seq（序列到序列）** 模型，用于机器翻译：

- **编码器**（encoder）：读完整句源语言，把含义"压缩"成一个固定长度的向量
- **解码器**（decoder）：从这个向量出发，一个词一个词地生成译文

想象编码器是个"笔记员"——
他读完整本书，然后把所有内容压缩成一张便条。
解码器是个"翻译"——他只看这张便条，不能翻书。

对短句来说这还凑合。但对长句子呢？

---

### 问题：一个向量装不下一整句话

Cho et al. (2014) 发现，
当句子变长时，翻译质量会**急剧下降**——
原因很明显：把"一整句话"压进一个固定大小的向量，
长句子的信息就被强行截断或混在一起了。

用数字说话：翻译 15 词以上的句子时，BLEU 分数（翻译质量指标）
从约 25 跌到 10 以下——质量减半。

---

### 解决方案：不要只看一张便条——每次写词都翻一遍书

2015 年，Bahdanau、Cho、Bengio 提出了 **Attention（注意力机制）**：

> **关键想法**：解码器每生成一个词时，
> 不依赖单一的"压缩向量"，
> 而是**对整个输入序列做一次加权平均**，
> 权重由"当前状态和哪个输入词最相关"决定。

类比：
- 旧方法：看一张便条翻译一整本书
- 新方法：翻译每个词时，都可以翻回原书看一眼——但**重点看最相关的那几页**

---

## 直觉：注意力权重是什么？

### 一个翻译例子

翻译 "The bank was robbed" → "那家银行被抢了"

当解码器生成"**银行**"这个词时，
应该重点"关注"源句中的"**bank**"这个词。

当生成"**被抢了**"时，
应该同时关注"**was robbed**"。

注意力机制就是计算这些"关注程度"——
每个源语言词得到一个 0 到 1 之间的权重：
$$\alpha_1, \alpha_2, \alpha_3, \alpha_4$$

权重之和等于 1（像概率一样），表示"我把 100% 的注意力分配给这些位置"。

---

## 数学：三步推导

### 第一步：编码器输出"记忆"

编码器读完输入序列后，
对每个位置 $j$，都保存一个**隐藏状态** $h_j$（不只是最后一个）。

$$h_1, h_2, \ldots, h_T$$

$h_j$ 表示"第 $j$ 个输入词，及其上下文的信息"。

---

### 第二步：打分——哪个位置和当前解码状态最相关？

解码器在生成第 $i$ 个输出词之前，有一个当前状态 $s_{i-1}$。

我们计算 $s_{i-1}$ 和每个编码器状态 $h_j$ 的"相关度得分"：

$$e_{ij} = \mathbf{v}^\top \tanh\!\left(\mathbf{W}_1 \, h_j + \mathbf{W}_2 \, s_{i-1}\right)$$

> **tanh 和 v 是什么？**
>
> $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$，
> 它把任意数压到 $(-1, 1)$ 区间，作用类似 sigmoid。
>
> $\mathbf{v}$、$\mathbf{W}_1$、$\mathbf{W}_2$ 是**可学习的参数**——
> 网络通过反向传播学会"怎么判断相关度"。

$e_{ij}$ 越大，意味着"在生成第 $i$ 个输出词时，第 $j$ 个输入词越重要"。

---

### 第三步：Softmax 把得分变成权重

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\displaystyle\sum_{k=1}^{T} \exp(e_{ik})}$$

> **Softmax 是什么？**
>
> Softmax 把一组任意实数变成"概率分布"：
> - 所有值变为正数（$\exp$ 保证）
> - 所有值加起来等于 1（除以总和保证）
>
> 结果：$\alpha_{ij} \geq 0$，且 $\displaystyle\sum_j \alpha_{ij} = 1$

---

### 第四步：加权求和得到"上下文向量"

$$c_i = \sum_{j=1}^{T} \alpha_{ij} \cdot h_j$$

$c_i$ 就是"第 $i$ 个输出词的上下文"——
它是所有编码器状态的加权平均，权重由注意力决定。

---

### 全流程总结

```
输入序列 → 编码器 → h_1, h_2, ..., h_T（每个位置一个状态）
                         ↓
解码器状态 s_{i-1} → 打分函数 → e_{i1}, e_{i2}, ..., e_{iT}
                         ↓
                      Softmax → α_{i1}, ..., α_{iT}（权重，和=1）
                         ↓
                    加权求和 → c_i（上下文向量）
                         ↓
             解码器用 c_i 生成第 i 个输出词
```

---

## 为什么这个想法这么厉害？

### 1. 解决了信息瓶颈

不再把"整句话"压缩成一个向量——
每次生成时都能"翻回去看"，长句子也不丢信息。

### 2. 可解释性

注意力权重 $\alpha_{ij}$ 可以可视化，
能看到"翻译每个词时，模型在看哪些位置"——
这在之前的 RNN 里是做不到的。

Bahdanau et al. 的论文里就有这样的热力图：
翻译法语→英语时，diagonal（对角线）注意力权重
说明模型自然地学会了词序对齐。

### 3. 打开了 Transformer 的大门

注意力机制本身不依赖 RNN——
2017 年 Vaswani et al. 发现"完全用注意力替代 RNN"
（Transformer），才有了后来的 GPT、BERT……
那是节点09的故事。

---

## 局限性

- **计算量是 $O(T^2)$**：每生成一个词都要看遍整个输入，
  序列越长越慢，长文档仍是瓶颈
- **仍然依赖 RNN**：Bahdanau 的原版还是 RNN + attention，
  不是纯 attention
- **参数更多**：多了 $\mathbf{v}, \mathbf{W}_1, \mathbf{W}_2$ 三组参数要学

---

## 参考文献

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR 2015*. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. *NeurIPS 2014*. [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
- Cho, K., van Merriënboer, B., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP 2014*. [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
