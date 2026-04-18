# 节点11：InstructGPT 与 RLHF（2022）——让 AI 学会"听话"

▶ [11-instructgpt-2022.ipynb](../notebooks/11-instructgpt-2022.ipynb)

> **前置节点**：[节点10 GPT-3（2020）](./10-gpt3-2020.md)
>
> **核心问题**：GPT-3 有惊人的能力，但它不听话——你让它写一封道歉信，它可能给你讲一堆道歉信的写法；你让它"有益地回答"，它可能输出有害内容。能力强不等于可用。怎么办？

---

## 1. 时代背景：强大但危险的 GPT-3

2020 年，GPT-3 震惊世界：175B 参数，few-shot 推理，几乎无所不能。
但在真实部署中，用户很快发现了问题：

| 问题 | 例子 |
|------|------|
| **不遵循指令** | "用简单的话解释量子力学" → 给出学术论文式回答 |
| **有害内容** | 在某些提示下输出歧视性、危险性文本 |
| **胡编乱造** | 自信地输出错误的事实（幻觉） |
| **冗长无用** | 重复、废话连篇，不直接回答 |

根本原因：**GPT-3 的目标是"预测下一个 token"，不是"帮助用户"。**
语言模型学会了人类的写作习惯，但没有学会人类的意图。

这就是 **Alignment Problem**（对齐问题）：如何让 AI 的行为与人类的真实意图对齐？

---

## 2. InstructGPT 的答案：RLHF

2022 年，OpenAI 发布论文《Training language models to follow instructions with human feedback》（Ouyang et al., 2022，arXiv:2203.02155）。

他们提出了 **RLHF**（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）——一个三阶段的训练流程：

```
GPT-3（预训练）
    ↓ 阶段1：SFT
SFT 模型（会听话的基础版）
    ↓ 阶段2：Reward Model
RM（学会判断哪个回答更好）
    ↓ 阶段3：PPO
InstructGPT（对齐后的最终版）
```

---

## 3. 阶段一：SFT（监督微调）

**SFT = Supervised Fine-Tuning**，监督微调。

### 3.1 做法

雇佣专业标注员（labeler），给他们看各种提示词（prompt），让他们**亲自写出理想的回答**。
这些"示范数据"用来微调 GPT-3：

```
训练数据格式：
{
  "prompt": "用简单的话解释什么是机器学习",
  "demonstration": "机器学习就是让电脑从例子中学习。
                    比如，给电脑看 1000 张猫的照片和 1000 张狗的照片，
                    它就能学会分辨猫和狗..."
}
```

### 3.2 直觉

就像给新员工培训：不是只告诉他"做好"，而是亲自给他看"好"是什么样的。

SFT 之后，模型学会了基本的"听话"姿势——但这还不够，因为：
- 人工示范成本高，数量有限（约 13000 条）
- 标注员的品味不代表所有用户的品味

---

## 4. 阶段二：Reward Model（奖励模型）

**核心思想**：与其让人写示范，不如让人"挑更好的那个"。

### 4.1 做法

给同一个 prompt，让 SFT 模型生成多个回答（比如 4 个），
然后让标注员**排序**：哪个最好？哪个最差？

```
Prompt: "解释一下什么是黑洞"
回答A: "黑洞是一种引力极强的天体..."（很好）
回答B: "我无法回答这个问题"（很差）
回答C: "黑洞，即 Black Hole，是..."（中等）
回答D: （一大段无关内容）（最差）
排序：A > C > B > D
```

用这些排序数据，训练一个**奖励模型（Reward Model, RM）**：
给它一个 (prompt, 回答) 对，它输出一个数字分数，表示"这个回答有多好"。

### 4.2 数学：Bradley-Terry 偏好模型

**🧮 数学小补丁（初中生友好版）**

假设有两个回答 $y_w$（好的）和 $y_l$（差的）。
我们希望 RM 给 $y_w$ 打的分 $r(y_w)$ 比 $r(y_l)$ 高。

用一个叫 **Bradley-Terry 模型**的方法，把分数转成概率：

$$P(y_w \text{ 比 } y_l \text{ 好}) = \sigma(r(y_w) - r(y_l))$$

其中 $\sigma(x) = \frac{1}{1+e^{-x}}$ 是 Sigmoid 函数——把任意数字压缩到 0 到 1 之间。

当 $r(y_w) > r(y_l)$，差值为正数，$\sigma$ 输出 > 0.5，表示"有信心 $y_w$ 更好"。
当差值很大时，概率趋向 1。

训练目标：最大化以下对数似然（让模型对正确排序有高置信度）：

$$\mathcal{L}_{RM} = -\mathbb{E}\left[\log\sigma(r(y_w) - r(y_l))\right]$$

**直觉翻译**：每次看到"A 比 B 好"的标注，就调整 RM 的参数，让 $r(A)$ 比 $r(B)$ 更大。
反复训练，RM 就学会了人类的偏好品味。

### 4.3 为什么这比直接用 SFT 更强？

- 排序比写示范**便宜得多**：标注员评判 5 个回答只需 1 分钟
- 可以生成大量比较对（约 33000 对），大大扩充数据
- RM 一旦训好，就是一个**可复用的"人类偏好评分器"**

---

## 5. 阶段三：PPO（近端策略优化）

有了 RM，就可以用**强化学习**来训练语言模型了。

### 5.1 强化学习的基本框架

**🧮 数学小补丁：强化学习直觉**

| 概念 | 在 RLHF 中的含义 |
|------|-----------------|
| **智能体（Agent）** | 语言模型 |
| **环境（Environment）** | 用户 + 奖励模型 |
| **状态（State）** | 当前的 prompt |
| **动作（Action）** | 生成的下一个 token |
| **奖励（Reward）** | RM 给完整回答的打分 |

目标：**最大化期望奖励**——让语言模型生成的回答，平均获得更高的 RM 分数。

### 5.2 PPO 的核心思想

PPO（Proximal Policy Optimization，近端策略优化）是一种强化学习算法，
由 Schulman et al. 2017 提出。它的核心思想：

> **每次更新不要太猛**——每一步只允许模型参数改变一点点，避免"学坏了走极端"。

这个约束用 **KL 散度**来实现。

**🧮 数学小补丁：KL 散度**

KL 散度衡量"两个概率分布有多不同"：

$$D_{KL}(P \| Q) = \sum_x P(x) \log\frac{P(x)}{Q(x)}$$

- 当 $P = Q$（完全相同）时，$D_{KL} = 0$
- 当 $P$ 和 $Q$ 差异大时，$D_{KL}$ 很大

### 5.3 RLHF 的目标函数

PPO 在 RLHF 中的实际优化目标：

$$\max_{\pi} \mathbb{E}\left[r_\phi(x, y) - \beta \cdot D_{KL}(\pi(y|x) \| \pi_\text{SFT}(y|x))\right]$$

分解来看：
- $r_\phi(x, y)$：RM 对（prompt $x$，回答 $y$）的打分（越高越好）
- $\beta \cdot D_{KL}(\pi \| \pi_\text{SFT})$：新模型 $\pi$ 与 SFT 模型的 KL 散度（惩罚项）
- $\beta$：平衡系数（通常 0.01~0.1）

**直觉**：既要分数高（让人类满意），又不能跑太偏（保留 GPT-3 的通用能力）。

没有 KL 惩罚，模型会"钻空子"：找到 RM 打高分但完全没用的回答
（比如重复输出某些固定短语），这叫 **Reward Hacking**。

---

## 6. 实验结果：1.3B 打败 175B

InstructGPT 最惊人的结果：

> 经过 RLHF 训练的 **1.3B InstructGPT**，在真实用户评估中，
> **胜率高于 175B 的原始 GPT-3**（约 85% 的评估者更偏好 InstructGPT）。

参数量相差 **134 倍**，但对齐的 1.3B 比未对齐的 175B 更受欢迎！

这说明一件事：**对齐（Alignment）和规模（Scale）同样重要**。

### 6.1 模型系列

| 模型 | 参数量 | 基础 |
|------|--------|------|
| InstructGPT | 1.3B | GPT-3 1.3B + RLHF |
| InstructGPT | 6B | GPT-3 6B + RLHF |
| InstructGPT (davinci) | 175B | GPT-3 175B + RLHF |

---

## 7. ChatGPT 的诞生（2022 年 11 月）

InstructGPT 发表后，OpenAI 用类似的 RLHF 技术，基于 GPT-3.5 训练了 **ChatGPT**，
于 **2022 年 11 月 30 日** 对公众开放。

5 天内用户突破 100 万，两个月内月活用户超过 1 亿——创造了有史以来增长最快的消费者应用纪录。

ChatGPT 并不是技术上最先进的，而是第一个把"对齐"做到足够好、
让普通人都能顺畅使用的 AI 助手。

---

## 8. 局限与下一步

### 8.1 InstructGPT/RLHF 的局限

| 局限 | 原因 |
|------|------|
| **标注员品味有偏差** | RM 学的是特定标注员的偏好，不代表全人类 |
| **Reward Hacking** | 模型会学会取悦 RM 而非真正有帮助 |
| **PPO 训练不稳定** | 强化学习本身难以调试 |
| **仍有幻觉** | RLHF 不能消除 LLM 的幻觉问题 |
| **人工成本高** | 高质量标注很昂贵 |

### 8.2 衍生方向

| 后续方法 | 核心思想 |
|----------|---------|
| **RLAIF** | 用 AI 代替人类打分（Constitutional AI，Anthropic） |
| **DPO** | 直接偏好优化，绕开 RM 和 PPO，更简单稳定 |
| **RLVR** | 用可验证的奖励（数学题是否算对）替代 RM |
| **Self-Play** | 让模型和自己对弈来改进（AlphaGo 思路） |

### 8.3 留下的种子

InstructGPT 证明了：大模型通过后训练（post-training）可以大幅改变行为。
这开启了"**基础模型 + 对齐微调**"的现代 LLM 开发范式，
后续的 Claude、Gemini、LLaMA 等都在这个框架上演化。

---

## 9. 引用溯源

- **[ouyang2022instructgpt]** Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P. F., Leike, J., & Lowe, R. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35, 27730–27744. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

- **[schulman2017ppo]** Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

- **[christiano2017rlhf]** Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *Advances in Neural Information Processing Systems*, 30. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)

---

> **下一节点**：[节点12 Llama 与开源爆炸（2023）](./12-llama-2023.md) — 权重泄漏引发的开源革命：LLaMA、Alpaca、LoRA 与 PEFT 生态
