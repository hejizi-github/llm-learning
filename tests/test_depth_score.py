"""Tests for tools/depth-score — score_doc logic."""
import pytest


RICH_MD = """
# 感知机 (1958)

## 背景故事

1958年，Frank Rosenblatt 在冷战背景下提出感知机。那个时代的科学家相信...

## 原理讲解

感知机的工作机制很简单：每个输入 $x_i$ 乘以权重 $w_i$，然后求和。

算法步骤如下：
1. 初始化权重
2. 计算加权和 $\\sum w_i x_i$
3. 与阈值比较

## 数学自包含

公式推导：$y = \\text{sign}(\\mathbf{w} \\cdot \\mathbf{x} - \\theta)$

## 可运行 Notebook

[点击运行](../notebooks/01-perceptron-1958.ipynb)

## 局限与衔接

然而，感知机无法解决 XOR 问题，这个局限性导致了第一次 AI 寒冬。
但这个局限也启发了下一个突破：多层网络。

## 引用溯源

- Rosenblatt 1958: [doi:10.1037/h0042519](https://doi.org/10.1037/h0042519)
"""

THIN_MD = """
# Simple document

Just some text without much structure.
"""


def test_rich_doc_gets_max_score(depth_score, tmp_md):
    """A well-structured doc covering all 6 criteria should score 5/5."""
    md = tmp_md(RICH_MD)
    result = depth_score.score_doc(md)
    assert result["star"] == 5
    assert result["raw"] == 6


def test_thin_doc_gets_low_score(depth_score, tmp_md):
    """A document with minimal content should score low (1 or 2)."""
    md = tmp_md(THIN_MD)
    result = depth_score.score_doc(md)
    assert result["star"] <= 2


def test_score_doc_has_all_criteria_keys(depth_score, tmp_md):
    """score_doc result should include all 6 rubric dimensions."""
    md = tmp_md(RICH_MD)
    result = depth_score.score_doc(md)
    expected_keys = {"背景故事", "原理讲解", "数学自包含", "notebook链接", "局限与衔接", "引用溯源"}
    assert set(result["criteria"].keys()) == expected_keys


def test_notebook_link_criterion(depth_score, tmp_md):
    """A doc with .ipynb link should pass the notebook criterion."""
    md = tmp_md("# Test\n\nSee [notebook](../notebooks/test.ipynb) for details.\n")
    result = depth_score.score_doc(md)
    assert result["criteria"]["notebook链接"]["passed"] is True
