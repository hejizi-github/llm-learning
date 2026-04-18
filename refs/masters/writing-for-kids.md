# 给初中生写技术文章的大师原则

> 基于真实下载的大师写作样本分析  
> 样本来源：Andrej Karpathy (karpathy.github.io)、Michael Nielsen (neuralnetworksanddeeplearning.com)  
> 分析日期：2026-04-19  
> 样本文件：refs/masters/samples/

---

## 为什么需要这份文件

前两次迭代的 clean slate 根因相同：
> Agent 读了论文，但写出来是"大学生教程风格"——不是给 14 岁学生看的。

这份文件从真实大师的作品里提炼**可操作的写作规则**。
每条规则后面都有原文引证（不是我脑补的）。

---

## 大师 vs 学院派：最关键的对比

| 大师风格（Karpathy / Nielsen） | 学院派风格（Goodfellow DL Book） |
|---|---|
| 从读者已知的东西开始 | 从定义开始 |
| 问题先行，解法在后 | 概念先行，例子在后 |
| 公式出现在类比之后 | 公式出现在最前 |
| 个人故事/情感 | 中性客观语调 |
| 短句、主动语态 | 长句、被动结构 |
| 明确告诉你"我们将做什么、有多酷" | 假设你已经知道为什么要学 |

---

## 10 条可操作原则

### 原则 1：从读者「已经知道的东西」开始

**禁止**用定义开头。**必须**从读者熟悉的现象入手，再引出技术问题。

✅ Nielsen 的做法：
> "The human visual system is one of the wonders of the world. Consider the following
> sequence of handwritten digits: Most people effortlessly recognize those digits as 504192.
> **That ease is deceptive.**"

分析：先讲一件人人都能感受的事（认字很容易），然后翻转它（其实极其复杂），
这个翻转创造了悬念，读者自然想知道：到底为什么？

**应用到本项目**：写感知机时，不要从「感知机是一种线性分类器」开始。
要从「你的大脑看到一张照片就知道是猫——为什么计算机做不到？」开始。

---

### 原则 2：先展示「旧方法的失败」，再引入新方法

读者只有在看到旧方法行不通之后，才会真正在乎新方法为什么好。

✅ Nielsen 的做法：
> "Simple intuitions about how we recognize shapes — 'a 9 has a loop at the top, and a
> vertical stroke in the bottom right' — turn out to be not so simple to express
> algorithmically. When you try to make such rules precise, you quickly get lost in a
> morass of exceptions and caveats and special cases. **It seems hopeless.**"

分析：让读者亲自经历「手写规则的绝望」，神经网络才显得是一个真正的突破。

**应用到本项目**：每个知识节点，先描述当时人们用的方法，展示它的局限，
然后才引出这个节点的贡献。

---

### 原则 3：先写类比/故事，最后才写公式

公式是总结，不是介绍。读者理解了直觉之后，公式才有意义。

✅ Nielsen 的做法（感知机部分）：
> 先讲「奶酪节」的故事（三个因素：天气、同伴、交通），
> 读者自然思考「这三件事有不同的重要程度」，
> 然后才引入「权重」概念，最后才写出公式。

✅ Karpathy 的做法：
> "My personal experience... is that everything became much clearer when I started
> **ignoring full-page, dense derivations** of backpropagation equations and just started
> writing code."

**应用到本项目**：顺序永远是：
故事/问题 → 直觉 → 类比 → Python 代码 → 数学公式（可选，放最后）

---

### 原则 4：讲作者自己的迷惑和顿悟（个人叙事）

大师不假装自己天生就懂。他们分享自己曾经迷惑、然后豁然开朗的时刻。
这让读者觉得：我也可以走过这个路。

✅ Karpathy（RNN 帖子）：
> "I still remember when I trained my first recurrent network for Image Captioning.
> Within a few dozen minutes... started to generate very nice looking descriptions...
> What made this result so **shocking** at the time was..."

✅ Karpathy（Hacker's Guide）：
> "I will strive to present the algorithms in a way that **I wish I had come across
> when I was starting out.**"

**应用到本项目**：可以用第一人称描述为什么这个问题难、历史上的人怎么被它困住的。
不要假装每件事都显然易见。

---

### 原则 5：用惊喜数字作为「承诺」，在开头吊住好奇心

具体数字比模糊的「效果很好」有力得多。

✅ Nielsen：
> "The program is **just 74 lines long**, and uses no special neural network libraries.
> But this short program can recognize digits with an accuracy over **96 percent**,
> without human intervention."

分析：两个数字（74行、96%）让读者立刻产生具体期待——不是「学完很厉害」，
而是「学完我能写出 74 行识别手写字的程序」。

**应用到本项目**：每个节点的开头，给一个具体的、令人惊讶的数字或结果预告。

---

### 原则 6：立刻用平白语言解释技术词（第一次出现就解释）

技术词必须在第一次出现时立刻解释，解释要短、直白、不绕圈子。

✅ Nielsen：
> "He introduced **weights**, w1, w2, ..., real numbers expressing
> **the importance of the respective inputs to the output**."

分析：boldface 标记新词，紧接着是一个完整句子的白话解释。
没有「如你所知」「你应该了解」。

**禁止**：先用一个词三次，到第四次才解释。

---

### 原则 7：「你」和「我们」——对话感，不要说教感

✅ Nielsen：
> "If **you** attempt to write a computer program to recognize digits..."
> "In this chapter **we'll** write a computer program..."

✅ Karpathy：
> "This post is about sharing some of that magic with **you**."

分析：「你」让读者觉得作者在跟他说话，「我们」让读者觉得自己和作者一起探索。
「学习者」「读者」「学生」这些第三人称会产生距离感，尽量避免。

---

### 原则 8：每段只说一件事，段落短

✅ Nielsen 的段落节奏：
```
[2-3句] 讲人类视觉很神奇
[2句]   翻转：其实很复杂
[2-3句] 试图写规则会陷入绝望
[2句]   神经网络换了一种方法
[1句]   神经网络从例子里自动学规则
```

每个段落一个转折点，不超过 4-5 句话。

**应用到本项目**：写完一段，问自己「这段只说了一件事吗？」如果说了两件，拆开。

---

### 原则 9：类比用「学生日常」，不用「职场/商业」场景

Nielsen 用的是「奶酪节周末要不要去」——
不是「企业决策流程」「股票投资组合」。

**针对 14 岁**：类比必须来自他们真实的生活：
- 考试分数、游戏分数
- 足球裁判吹不吹犯规
- 去不去操场踢球
- 辨别手写字、图片里的猫

**禁止**：「就像公司的绩效考核」「股市涨跌」「工厂流水线」

---

### 原则 10：代码优先于公式

对 14 岁：一段能跑的 Python 代码 > 一页数学推导。

✅ Karpathy：
> "Thus, this tutorial will contain very little math... My exposition will center around
> **code and physical intuitions instead of mathematical derivations**."

**应用到本项目**：每个概念必须有对应的可运行代码。
代码是第一公民，数学公式是代码的注解。

---

## 自检 Rubric（写完任何段落都问这些问题）

| 检查项 | 理想答案 |
|---|---|
| 第一段从读者已知的东西开始了吗？ | 是 |
| 有没有先展示旧方法/问题？ | 是 |
| 公式出现在类比之后吗？ | 是 |
| 有没有让 14 岁学生陌生的词第一次出现时没解释？ | 没有 |
| 有没有「职场/商业/成年人专用」的类比？ | 没有 |
| 每个概念有对应的可运行代码吗？ | 是 |
| 用了「你」或「我们」吗？ | 是 |
| 一段超过 5 句话了吗？ | 没有 |
| 开头有具体惊喜数字或结果预告吗？ | 是 |
| 有没有「显然」「容易知道」「不难看出」这类词？ | 没有 |

---

## 反面教材：学院派写法 vs 大师写法

下面是对比同一个概念「感知机权重」的两种写法：

**❌ 学院派（禁止这样写）：**
> 感知机是一种二值线性分类器，对输入向量 x ∈ ℝⁿ 进行加权求和，
> 通过阈值函数产生输出。权重向量 w 和偏置 b 通过感知机学习规则迭代更新。

**✅ 大师风（本项目要达到的标准）：**
> 想象你在决定要不要去看一场电影。你会考虑几件事：
> 票价贵不贵？有没有喜欢的演员？最近考试压力大不大？
> 这些事情对你的影响力不一样——票价太贵可能直接否决，演员不熟无所谓。
>
> 感知机做的事情完全一样。它把每个输入（比如票价高低、演员好不好）乘以一个数，
> 这个数叫**权重（weight）**——代表这件事有多重要。
> 最后把所有的「重要程度 × 信息」加起来，超过某个门槛就输出 1（去！），
> 否则输出 0（不去）。
>
> 用代码说就是三行：

---

## 信号：是否符合大师标准的简单测试

把你写的一段文字念给一个真实的初中生，或者想象给一个 14 岁的人解释。
如果你需要加「其实这很简单」「你只需要理解」「这本质上就是」——
那就说明你跳过了某个该解释的跃迁。

> **大师不说「其实很简单」。大师让读者自己发现它很简单。**
