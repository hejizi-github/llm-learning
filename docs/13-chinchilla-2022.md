# 节点13：Chinchilla 缩放定律（2022）

> **发展脉络位置**：GPT-3 和 Gopher 把模型做到了千亿参数，大家都以为"越大越好"。  
> 2022 年，DeepMind 的一组研究者做了一个简单的实验，然后说了一句让整个行业震惊的话：  
> **"你们都训练错了。"**

---

## 一、问题的起点：厨师买了一把超贵的刀，但没买食材

先用一个比喻理解这件事。

想象你是一位厨师，要在有限的预算里做出最美味的菜：

- **刀**（模型参数量）：刀越好，切得越精准
- **食材**（训练数据量）：食材越多越新鲜，菜才越丰富
- **预算**（算力/计算量）：由 GPU 时间决定，固定的

2020 年，OpenAI 的 Kaplan 等人发现了一个规律：**给定固定预算，把更多钱花在买刀（参数量）上，比花在买食材（数据量）上效果更好。**

这个结论导致了 2020-2021 年一场疯狂的"刀具军备竞赛"：

| 模型 | 机构 | 参数量 | 训练数据 |
|------|------|--------|---------|
| GPT-3 | OpenAI | 1750 亿 | 3000 亿 token |
| Gopher | DeepMind | 2800 亿 | 3000 亿 token |
| MT-NLG | NVIDIA/MS | 5300 亿 | 2700 亿 token |

每个团队都在造更大的模型，但训练数据量增长很少——大概都是 3000 亿 token 左右。

---

## 二、DeepMind 的实验：重新做一遍

2022 年，DeepMind 的 Jordan Hoffmann 团队做了一件很简单但很重要的事：**他们没有假设 Kaplan 的结论是对的，而是自己重做了实验。**

他们训练了超过 **400 个语言模型**，参数量从 7000 万到 160 亿不等，数据量从少到多系统地变化，然后仔细测量哪种组合在相同计算量下损失最小。

> **Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022).** *Training Compute-Optimal Large Language Models.*  
> arXiv: [2203.15556](https://arxiv.org/abs/2203.15556)

实验结论用一句话说：

> **在相同计算预算下，参数量和数据量应该等比例增长——根据经验拟合，大约每个参数对应 20 个训练 token（论文方法1/2的估计；参数化模型方法3给出更高的比值，约 50–100）。**

---

## 三、两张图讲清楚整件事

### 3.1 Kaplan 怎么说（旧规律）

Kaplan 等人 2020 年的论文 [[kaplan2020scaling](#ref-kaplan2020scaling)] 给出的结论大致是：

- 每增加 10 倍计算量，参数量应增加约 **5 倍**
- 数据量只需增加约 **2 倍**

也就是说，模型参数的增长速度应该远快于数据量。

### 3.2 Chinchilla 怎么说（新规律）

Hoffmann 等人的实验结论则是：

- 每增加 10 倍计算量，参数量应增加约 **3.16 倍**（√10）
- 数据量也应增加约 **3.16 倍**（√10）

**两者应该等比例增长。**

用表格对比两种策略：

| 计算预算（FLOPs） | Kaplan 建议参数量 | Chinchilla 建议参数量 | Chinchilla 建议数据量 |
|-----------------|-----------------|---------------------|---------------------|
| 1e21 | ~100B | ~11B | ~220B tokens |
| 1e22 | ~300B | ~35B | ~700B tokens |
| 1e23 | ~1000B | ~110B | ~2.2T tokens |

重要发现：**按 Chinchilla 规律，GPT-3（1750 亿参数）要充分训练，需要约 3.5 万亿 token——而不是它实际使用的 3000 亿。GPT-3 只用了最优训练量的 8%。**

---

## 四、数学小补丁：幂律是什么？

> 如果你学过初中代数，这一节就能读懂。

### 4.1 幂律的直觉

"幂律"（Power Law）说的是：如果 X 变大，Y 也变大（或变小），但不是线性的，而是按某个"幂"的比例。

举个例子：
- 面积 = 边长² → 边长增加 2 倍，面积增加 4 倍
- 声音强度 ∝ 距离^(-2) → 距离增加 2 倍，声音变成 1/4

语言模型的损失值（L）随参数量（N）增加而降低，遵循：

```
L(N) ≈ A / N^α
```

其中：
- L 是模型在测试集上的损失（越小越好）
- N 是参数量
- A 是一个常数（和任务难度有关）
- α 大约是 0.07 到 0.34（一个小于 1 的正数）

这意味着：N 增加 10 倍，L 只下降到原来的 10^(-0.34) ≈ 0.46，即减少约一半。

### 4.2 双因子损失函数

Chinchilla 论文用的模型更完整，同时考虑了参数量（N）和数据量（D）：

```
L(N, D) = A / N^α + B / D^β + L_∞
```

其中：
- `A / N^α`：由于参数量不够，无法完全拟合语言结构带来的损失
- `B / D^β`：由于训练数据不够，没见过足够多样的文本带来的损失
- `L_∞`：不管模型多大、数据多多，语言本身的不确定性（最小理论损失，约为 1.69 bits/token）

这个函数告诉我们：N 和 D 都对损失有贡献，两者都要足够大。

### 4.3 最优分配（约束最优化）

训练大模型的计算量（FLOPs）大约等于：

```
C ≈ 6 × N × D
```

（每个参数每个 token 大约需要 6 次浮点运算，包括前向和反向传播）

**问题**：给定固定的计算预算 C，怎么分配 N 和 D 使 L 最小？

这是一个约束最优化问题（高中/大学数学会讲，这里给直觉版）：

**直觉**：想象你要最大化两块土地的总产量，预算固定。如果两块土地的边际回报相同，你应该平均分配。当两者的回报曲线相似（α ≈ β），最优策略就是**等比例分配**。

当 α ≈ β 时，可以推导出：

```
N_opt ∝ C^(1/2)
D_opt ∝ C^(1/2)
```

即：**增加计算预算时，参数量和数据量各增加预算的平方根。**

---

## 五、Chinchilla 实验的结果

按照新规律，DeepMind 训练了 **Chinchilla（小松鼠）**：

| 属性 | Gopher | Chinchilla |
|------|--------|-----------|
| 参数量 | 2800 亿 | **700 亿** |
| 训练数据 | 3000 亿 token | **1.4 万亿 token** |
| 计算量 | 相同 | 相同 |
| 研发团队 | DeepMind | DeepMind |

用相同的计算量，Gopher 把预算主要花在了参数量上，Chinchilla 则平衡了参数和数据。

结果：**Chinchilla 在几乎所有基准测试上都超过了 Gopher，尽管参数量只有它的 1/4。**

例如在常识推理任务（BIG-Bench Hard）和知识问答（MMLU）上：

| 测试 | Gopher (280B) | Chinchilla (70B) |
|------|--------------|-----------------|
| MMLU（平均） | 60.0% | **67.6%** |
| BIG-Bench（平均） | 低 | 高 |

---

## 六、为什么 LLaMA 能用 7B 打败 GPT-3 175B？

Chinchilla 的发现直接改变了 2023 年之后的模型设计思路。

**旧思路（Kaplan 时代）**：
- 目标：用固定计算量训练出最强的模型
- 策略：让模型尽量大
- 问题：模型太大，推理时每次都要用大量 GPU

**新思路（Chinchilla 之后）**：
- 目标：训练一个**推理时高效**的小模型，同时让它足够强
- 策略：用比计算最优更多的数据训练一个相对小的模型（"过训练"）
- 结果：7B 参数的模型，如果用 1T+ token 训练，可以在推理时很便宜，同时性能接近大模型

这就是 LLaMA-7B 为什么能够在许多任务上接近甚至超过 GPT-3 175B 的原因。

| 模型 | 参数量 | 训练数据 | 推理成本 |
|------|--------|---------|---------|
| GPT-3 | 175B | 300B token（严重不足） | 极高 |
| LLaMA-7B | 7B | 1T token（充分训练） | **极低** |

---

## 七、局限与下一步

Chinchilla 规律虽然重要，但也有局限：

1. **仅适用于训练阶段**：它告诉你训练时怎么分配计算资源，但对"如何在推理时高效"没有直接回答。

2. **数据质量没有考虑**：模型训练了多少 token 是一回事，数据质量是另一回事。LLaMA 发现，1T 高质量网络文本 > 3T 低质量文本。

3. **规律会随架构改变**：Chinchilla 实验用的是 Transformer，不同架构可能有不同的最优比例。

4. **引出下一个问题**：  
   如果每个参数需要 20 个 token，那么模型大小和数据需求的上限在哪里？  
   → 这引出了 **数据墙（Data Wall）** 的担忧：互联网上的高质量文本是有限的。

---

## 八、小结

| 概念 | 一句话总结 |
|------|-----------|
| Kaplan 2020 | 给定计算量，优先做大模型 |
| Chinchilla 2022 | 给定计算量，模型和数据要等比例增长 |
| 核心公式 | L(N,D) = A/N^α + B/D^β + L∞，最优时 N∝C^½，D∝C^½ |
| 经验规则 | 每个参数约 20 token（方法1/2经验拟合）；参数化模型方法3约 50–100 |
| 实验验证 | Chinchilla 70B > Gopher 280B（相同算力） |
| 对后世的影响 | LLaMA、Mistral 等模型设计的理论基础 |

---

## 参考文献

<a id="ref-hoffmann2022chinchilla"></a>
**[Hoffmann2022]** Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de las Casas, D., Hendrycks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J. W., Vinyals, O., & Sifre, L. (2022). *Training Compute-Optimal Large Language Models.* arXiv preprint. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

<a id="ref-kaplan2020scaling"></a>
**[Kaplan2020]** Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). *Scaling Laws for Neural Language Models.* arXiv preprint. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

<a id="ref-rae2021gopher"></a>
**[Rae2021]** Rae, J. W., Borgeaud, S., Cai, T., Millican, K., Hoffmann, J., Song, F., Aslanides, J., Henderson, S., Ring, R., Young, S., et al. (2021). *Scaling Language Models: Methods, Analysis & Insights from Training Gopher.* arXiv preprint. [arXiv:2112.11446](https://arxiv.org/abs/2112.11446)

<a id="ref-touvron2023llama"></a>
**[Touvron2023]** Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., & Lample, G. (2023). *LLaMA: Open and Efficient Foundation Language Models.* arXiv preprint. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
