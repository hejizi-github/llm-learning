# 节点 06 — LSTM/GRU：给神经网络装上"记忆闸门"

> **突破时间**：1997 年（LSTM）/ 2014 年（GRU）
> **关键人物**：Sepp Hochreiter、Jürgen Schmidhuber（LSTM）；Kyunghyun Cho、Yoshua Bengio 等（GRU）
> **核心论文**：
> - Hochreiter & Schmidhuber 1997 [[DOI:10.1162/neco.1997.9.8.1735]](https://doi.org/10.1162/neco.1997.9.8.1735)
> - Cho et al. 2014 [[arXiv:1406.1078]](https://arxiv.org/abs/1406.1078)
> - Chung et al. 2014 [[arXiv:1412.3555]](https://arxiv.org/abs/1412.3555)

---

## 故事：一道记忆难题

[节点 05](../05-gradient-vanishing-1991/) 告诉我们：普通 RNN 在序列变长后，梯度会**指数级缩小**——网络忘记了很久之前的信息（Bengio et al., [1994](https://doi.org/10.1109/72.279181)）。

这是一道真实的难题。

1990 年代，研究者想训练 RNN 来完成翻译——  
"银行倒闭了，储户排起了长队" 中，"银行"是金融机构还是河岸？  
要判断，得看整句话的上下文。但 RNN 根本记不住几步前的东西。

---

1997 年，Sepp Hochreiter（在 Jürgen Schmidhuber 指导下）和 Schmidhuber  
提出了解决方案：**LSTM（Long Short-Term Memory，长短期记忆网络）**。

核心想法一句话：  
> **不要让梯度反复乘以同一个权重——改用"闸门"精确控制信息流。**

---

## 闸门是什么？（从直觉说起）

### 类比：水坝的闸门

想象一条河上有一系列水坝，每个水坝有一个闸门：
- 闸门完全打开（=1）→ 水全部通过
- 闸门完全关闭（=0）→ 水全部截住
- 闸门半开（=0.5）→ 一半的水通过

神经网络里的"闸门"是类似的机制：  
一个介于 0 和 1 之间的数字，乘以某个信息——控制"多少比例的信息能通过"。

$$\text{通过的信息} = \text{闸门值} \times \text{原始信息}$$

- $\text{闸门值} = 1$：全通过（保留全部）
- $\text{闸门值} = 0$：全拦截（彻底遗忘）
- $\text{闸门值} = 0.3$：只通过 30%

闸门值由网络自己学习——学会"什么时候该记、什么时候该忘"。

---

## LSTM：三道闸门 + 一条"记忆高速公路"

### 结构图（简化版）

```
                     细胞状态（长期记忆）
           c_{t-1} ──────────────────────────── c_t
                    │           │
                  遗忘闸      输入闸
                    │           │
  上一步输出 h_{t-1} ──── 混合处理 ─── 新输出 h_t
  当前输入   x_t  ──────────────┘
                                        │
                                      输出闸
                                        │
                                      h_t（输出给下一步）
```

LSTM 有**两条"线"**：
1. **细胞状态 $c_t$**：长期记忆，像一条高速公路，信息可以沿它几乎不损耗地长距离传播
2. **隐藏状态 $h_t$**：短期输出，传给下一步

### 三道闸门

**遗忘闸（Forget Gate）**：决定扔掉细胞里多少旧信息

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- $\sigma$（Sigmoid 函数）：把任何输入压到 0 到 1 之间（$\sigma(x) = \frac{1}{1+e^{-x}}$，其中 $e \approx 2.718$）
- $f_t$ 接近 1 → 几乎全保留；接近 0 → 几乎全遗忘
- $[h_{t-1}, x_t]$ 表示把两个向量拼在一起

**输入闸（Input Gate）**：决定把多少新信息写进细胞

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

- $\tanh$（双曲正切函数）：把输入压到 -1 到 1 之间，作为"候选新内容"
- $i_t$ 决定有多少候选内容真的被写进细胞

**细胞更新**：旧信息 × 遗忘 + 新信息 × 输入

$$c_t = f_t \times c_{t-1} + i_t \times \tilde{c}_t$$

这一步是关键！当 $f_t \approx 1$，旧细胞状态几乎完整地传递下去——**梯度不再消失**。

**输出闸（Output Gate）**：决定把多少细胞状态暴露给输出

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \times \tanh(c_t)$$

---

### 为什么梯度不再消失？

普通 RNN 的梯度传递：每步都乘以权重 $W$ 的导数 → 反复相乘 → 指数级缩小。

LSTM 的细胞状态梯度传递：

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

这个导数等于遗忘闸的值。  
如果网络学会让 $f_t \approx 1$（"保持记忆"模式），那么梯度传递时就乘以接近 1 的数——**不再指数级缩小**！

这就是 Hochreiter & Schmidhuber 1997 年的核心洞见：**常数误差旋转木马（Constant Error Carousel）**。

---

## GRU：两道闸门，更简洁的设计

2014 年，Cho 等人在研究机器翻译时，提出了**GRU（Gated Recurrent Unit，门控循环单元）**——一种用于 RNN Encoder-Decoder 的新型门控单元，后来被研究者们视为比 LSTM 更简洁的设计。

他们的论文提出了 RNN Encoder-Decoder 架构，其中引入了 GRU 单元（[arXiv:1406.1078](https://arxiv.org/abs/1406.1078)）。

GRU 把 LSTM 的三道闸门简化为**两道**：

**更新闸（Update Gate）**：同时控制遗忘和输入

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

**重置闸（Reset Gate）**：控制处理新信息时"看多少过去"

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

**候选状态**：

$$\tilde{h}_t = \tanh(W \cdot [r_t \times h_{t-1}, x_t])$$

**更新**：

$$h_t = (1 - z_t) \times h_{t-1} + z_t \times \tilde{h}_t$$

注意：当 $z_t \approx 0$，新状态 $\approx$ 旧状态（保留记忆）；当 $z_t \approx 1$，新状态 $\approx$ 候选内容（更新记忆）。

---

## LSTM vs GRU：哪个更好？

Chung 等人（2014）系统比较了 LSTM 和 GRU 在音乐建模和语音信号建模上的表现：

> "我们发现 GRU 与 LSTM 相当（GRU to be comparable to LSTM）"（[arXiv:1412.3555](https://arxiv.org/abs/1412.3555)）

**实践经验**：
| | LSTM | GRU |
|---|---|---|
| 闸门数量 | 3 | 2 |
| 参数数量 | 更多 | 更少（约 25%）|
| 训练速度 | 较慢 | 较快 |
| 效果 | 通常相近 | 通常相近 |

GRU 常常在数据量有限时比 LSTM 表现更好（参数少 = 更难过拟合）。  
LSTM 在非常长的序列上有时有优势。

**实践中**：两者都试试，看哪个更好。

---

## 感受一下"记忆闸"（Python 伪代码）

```python
import numpy as np

# 假设 sigmoid 和 tanh 都已定义
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 一步 LSTM（极简版，忽略权重矩阵细节）
def lstm_step(x_t, h_prev, c_prev, W_f, W_i, W_c, W_o, b_f, b_i, b_c, b_o):
    combined = np.concatenate([h_prev, x_t])
    
    f = sigmoid(W_f @ combined + b_f)   # 遗忘闸：0~1
    i = sigmoid(W_i @ combined + b_i)   # 输入闸：0~1
    c_cand = np.tanh(W_c @ combined + b_c)  # 候选内容：-1~1
    
    c_new = f * c_prev + i * c_cand     # 细胞更新
    
    o = sigmoid(W_o @ combined + b_o)   # 输出闸：0~1
    h_new = o * np.tanh(c_new)          # 新隐藏状态
    
    return h_new, c_new

# 关键理解：
# f * c_prev 这一项，当 f ≈ 1 时，旧记忆几乎完整保留
# 梯度通过这条路径反向传播时，不再指数级消失
```

完整的交互式演示见 → [`lstm_gru.ipynb`](./lstm_gru.ipynb)

---

## 这个突破带来了什么？

LSTM 发表后，沉寂了近十年——计算资源和数据量的限制让研究者无法充分利用它。

直到 2010 年代：
- **语音识别**：Google 用 LSTM 大幅提升识别准确率（2012-2014）
- **机器翻译**：Cho et al. 的 RNN Encoder-Decoder 开启了神经机器翻译时代
- **文本生成**：各种语言模型开始能生成连贯的段落

LSTM 和 GRU 成为 [节点 07](../07-attention-2015/)（Attention 机制）的直接前驱——  
即使有了 LSTM，处理非常长的序列仍然困难，这催生了下一个突破。

---

## 参考文献

- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. [DOI:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
- Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning Long-Term Dependencies with Gradient Descent is Difficult. *IEEE Transactions on Neural Networks*, 5(2), 157–166. [DOI:10.1109/72.279181](https://doi.org/10.1109/72.279181)
- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP 2014*. [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
- Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *arXiv preprint*. [arXiv:1412.3555](https://arxiv.org/abs/1412.3555)
