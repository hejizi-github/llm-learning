# Active Memory

> 当前知识库的"状态快照"。每次 session 开始必读，session 结束必更新。

## 知识库当前状态

**基础设施**：完成（目录骨架 + 6工具 + README + 策略文件 + **tests/ 22用例**）

**工具列表**：
- `tools/notebook-run` — 跑 notebook 验证
- `tools/cite-verify` — DOI/ISBN/arxiv 验证
- `tools/md-link-check` — md 链接检查
- `tools/depth-score` — 深度评分
- `tools/claude-advisor` — 外部 Claude 多角度分析（新增 20260418-125113）
- `tools/gen_nb_03.py` — 节点03 notebook 生成器（Python 脚本，非 CLI 工具）

**知识节点**：
| # | 文件 | notebook | depth | citations |
|---|------|----------|-------|-----------|
| 01 | docs/01-perceptron-1958.md | notebooks/01-perceptron-1958.ipynb | 5/5 | 4/4 verified |
| 02 | docs/02-minsky-papert-1969.md | notebooks/02-minsky-papert-1969.ipynb | 5/5 | 3/3 verified |
| 03 | docs/03-backprop-1986.md | notebooks/03-backprop-1986.ipynb | 5/5 | 1/1 verified |

**引用库**：refs/references.bib（4条），refs/citations.jsonl（4条全部已验证）

**时间线覆盖**：1958（感知机）→ 1969（XOR证明 + AI寒冬）→ 1986（反向传播 + 多层网络）

**已修复**：docs/01 + docs/02 中 ISBN 格式错误；docs/01 + docs/02 中的"下一节点"链接已修复为实际链接

## 累积 learnings（重要经验，勿覆盖）

- `spec_from_file_location` 对无 `.py` 后缀脚本返回 `None`，需显式传 `loader=importlib.machinery.SourceFileLoader(mod_name, str(path))` 才能加载（20260418-130019）
- `claude -p --model haiku` 是最简调用，`--bare` 会跳过 OAuth keychain 不可用（20260418-125113）
- `--allowedTools ""` 空字符串会被 Claude CLI 报错，应直接省略（20260418-125113）
- APA DOI 查询有时返回 403，需 fallback 到 GET 而非 HEAD（20260418-123514）
- notebook JSON 转义：cell source 中的反斜杠需双重转义（20260418-122128）
- `tools/notebook-run` 接受目录路径，不接受单文件路径（20260418-130735）
- nbconvert 执行 notebook 时工作目录是 `notebooks/`，所以 savefig 路径要用 `../docs/assets/`（20260418-130735）
- monkey-patch 方式（先 class，再 def func，再 Class.method = func）可以拆分 class 到多个 cell，但方法定义时不能有额外缩进（20260418-130735）
- 用 Python 脚本生成 notebook JSON（gen_nb_03.py）比手写 JSON 更易维护，且避免转义问题（20260418-130735）
- claude-advisor 可进一步泛化：管道 md 内容进去做读者可读性评审（来自评审建议，待实现）
- sigmoid 在 z > ~37 时 float64 饱和至 1.0，测试 sigmoid 范围应用 linspace(-10,10) 而非 (-100,100)（20260418-131743）

## 下次 session 建议

**第一优先**：节点 04 — 1989 LeNet（Yann LeCun，卷积神经网络）
- 时间线接续（1986反向传播 → 1989卷积网络）
- 核心内容：卷积操作直觉（空间局部性 + 权重共享）/ 手撕卷积层 / 在 MNIST 上演示
- 需要先 cite-verify LeCun 1989 DOI：10.1162/neco.1989.1.4.541
- 同时补 Werbos (1974) 和 Hopfield (1982) 到 refs/references.bib（评审遗留建议）

**tests/test_backprop.py**：已完成（20260418-131743），test_delta +12（22用例总计）

**PENDING 提案**：`.evolve/proposals/sub-agent-evaluation.md`
- 用 LLM 子 Agent 评估内容质量（响应用户 DIRECTIVE 20260418-123509）
- 等用户审批后实施
