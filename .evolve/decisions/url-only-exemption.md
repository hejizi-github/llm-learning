# URL-Only Citation Exemption Policy

## 决策日期
2026-04-19（Session 20260419-072628）

## 问题
FLOW 规则第1条："查不到 DOI / arxiv / ISBN 的不许写"。
但某些历史性原始材料（如大学学位论文）从未被分配 DOI，只有机构存档 URL。
强制删除会让知识库丢失关键的历史原始出处。

## 适用范围
以下情况可豁免 DOI/ISBN 要求，改用 URL 作为可验证来源：
1. 大学学位论文（Diplomarbeit / Dissertation / Master's Thesis）—— 学术数据库通常无 DOI
2. 机构技术报告（Technical Report）—— 编号替代 DOI

## 豁免条件（全部满足才可豁免）
1. **URL 必须指向原始机构或作者托管**（大学服务器、作者官网、IEEE Xplore、ACM DL 等），不得用博客/维基/二手平台
2. **bib note 字段必须注明"URL-only，无 DOI，原因：..."**
3. **必须在 citations.jsonl 里标注 `"doi_available": false, "url_verified": true`**
4. **每个节点豁免条目不超过 1 条**（如果主引用有 DOI，历史论文豁免可接受）

## 已豁免条目

| 引用 key | 原因 | URL |
|---|---|---|
| hochreiter1991 | TUM Diplomarbeit 1991，无 DOI，梯度消失发现的原始历史材料 | https://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf |

## 关系到的 FLOW 规则
FLOW 第1条在本决策记录下被细化为：
- DOI/arxiv/ISBN 是首选
- 满足上述豁免条件的 URL-only 引用视为"已验证"
- cite-verify 工具的"URL-only = PASS"口径在豁免条件满足时与此政策一致
