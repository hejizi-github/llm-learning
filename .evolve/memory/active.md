# Active Memory

> 当前知识库的"状态快照"。每次 session 开始必读，session 结束必更新。

## 知识库当前状态

_（空）首次迭代。还没有任何知识节点。_

## 上次 session 的 learnings

_（空）还没有历史 session。_

## 下次 session 建议

**首次迭代必须优先完成：**

1. 搭建仓库骨架：约定目录结构（比如 `docs/` 放 md，`notebooks/` 放 ipynb，`refs/` 放 references.bib + citations.jsonl）
2. 建立自检工具（放到仓库根的 `tools/`）：
   - `tools/notebook-run` — 批量执行 notebook，收集错误
   - `tools/cite-verify` — 核查引用真伪（arxiv id / DOI / 作者 / 年份）
   - `tools/md-link-check` — md 链接有效性
   - `tools/depth-score` — 用统一 rubric 给知识节点打深度分
3. 写一版"知识库总览"（README + index），规划初步组织形式（时间线？主题并行？依赖图？由 Agent 决策并写下理由）
4. 先完成 **1 个** 示范节点（建议从 1943 McCulloch-Pitts 神经元 或 1958 Rosenblatt 感知机 开始），作为后续节点的"质量基线"
5. 跑一遍自检，记录 KPI 基线

**不要做的事**：一次性铺满所有历史节点。先把质量门槛树立起来，后面才能稳步扩张。
