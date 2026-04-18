# Nicky Case 写作样本分析

## 来源

Nicky Case 的"Explorable Explanations"系列，主要分析：
- *Parable of the Polygons* (ncase.me/polygons) — 讲 Schelling 隔离模型
- *To Build a Better Ballot* (ncase.me/ballot) — 讲投票系统设计

**说明**: 以下引文从实际页面提取，非记忆重建。

---

## 核心写作技法（带真实引文）

### 1. 用悖论开场，不用定义开场

> "This is a story of how harmless choices can make a harmful world."
> — Parable of the Polygons 第一句

不是"本文介绍 Schelling 分离模型"，而是先抛出矛盾：**无害的个人选择 → 有害的集体结果**。
读者在读第一句时就感到困惑，想知道"这怎么可能"。

### 2. 给抽象实体加人格

> "These little cuties are 50% Triangles, 50% Squares, and 100% slightly shapist."

"cuties"（小可爱）把几何形状变成有偏见的邻居。读者对"有偏见的邻居"有直觉，对"随机移动 agent"没有。

### 3. 规则用第一人称/口语，不用数学符号

> "I wanna move if less than 1/3 of my neighbors are like me."

不是 "Agent moves when $\frac{|S_i|}{|N_i|} < \frac{1}{3}$"，而是"我想搬家，如果邻居里同类不到三分之一"。
完全精确，零门槛。

### 4. 用反问制造预测——然后让结果颠覆预测

> "Harmless, right?"
> "Surely their small bias can't affect the larger shape society that much?"

读者刚刚认同了"这偏见很小"，接下来模拟跑出了高度隔离的城市。
**认知冲击比任何解释都更深刻。**

### 5. 物理类比把抽象系统落地

> "Any one person can be 'strategic' by shouting over others, but if _everybody_ is 'strategic', nobody can hear anybody."

把投票博弈变成拥挤房间里的喊叫——读者的肌肉记忆里有这个场景。

### 6. 短句打节奏，长句装细节

短句：
- "Harmless, right?"
- "Sheesh!"
- "Daaaaang"

长句：
- "And a lack of transparency is an even deadlier sin nowadays, when our trust in government is already so low."

**规律**: 短句建立能量，长句消化后果。交替使用，读者不会累。

### 7. 体验先于解释

Ballot 的第一个互动："click & drag the candidates and the voter" — 读者先玩，再被告知名字。
等你知道叫"孔多塞方法"，你已经在脑子里跑过十几次模拟了。

### 8. 把技术问题绑上道德后果

> "how can we expect our elected officials to be honest, when our voting system _itself_ doesn't let us be honest?"

投票系统 bug 不只是数学问题，是**道德问题**。技术准确性服务于更大的价值。

---

## 感知机段落对比（迁移练习）

### 原版（典型教程风格）
> 感知机是一种线性分类器。它有一个权重向量 **w** 和偏置 b，通过计算 $\hat{y} = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$ 来做预测。训练时，用感知机规则更新权重。

### Nicky Case 风格改写

> 想象你是一个审查员，每天要判断一封邮件是不是垃圾邮件。
> 
> 你有两条线索：邮件里"免费"这个词出现了几次？邮件里有没有"恭喜您"？
> 
> 你的判断规则很简单：**"线索总分超过阈值就判垃圾，否则放行。"**
> 
> 第一天你完全猜——每条线索给同样的权重。
> 
> 然后结果来了：你判错了。好，调整一下，"免费"这个词给更高的分。
> 
> 第二天，又错了几封。再调。
> 
> 第七天：**你一封都没判错。**
> 
> 这就是感知机。它不知道规则是什么——**它自己试出来的。**

---

## 可迁移原则（按优先级）

| 优先级 | 原则 | 在 notebook 里怎么用 |
|--------|------|----------------------|
| ★★★ | 悖论开场 | 每个节点的第一段，先抛出反直觉的事实 |
| ★★★ | 预测 + 颠覆 | 在代码运行前，让读者猜结果；代码运行后对比 |
| ★★★ | 第一人称规则 | 算法规则用"我做什么，如果……"写 |
| ★★ | 短句节奏 | 段落里插入 1-2 个短句作为节拍 |
| ★★ | 物理类比 | 权重=重要程度，阈值=分界线，误差=错的代价 |
| ★ | 道德绑定 | 可选；在"为什么这很重要"部分用 |

---

## 最不该抄的错误模仿

1. **抄语气词不抄结构** — "Daaaaang" 不是 Nicky Case 风格，它只是一个词。风格在于**先让读者猜再揭晓答案**的结构。
2. **把交互换成截图** — 他的精髓是读者亲自操作产生顿悟。没有交互时，用**代码实验+提问**替代（"把 learning_rate 改成 0.5 会怎样？"）。
3. **casual 但不准确** — 他从不为了通俗牺牲精确。"I wanna move if less than 1/3" 在数学上是精确的。

---

*分析基于 ncase.me/polygons 和 ncase.me/ballot 实际页面内容，2026-04-19 抓取。*
