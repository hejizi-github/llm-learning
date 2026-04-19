"""Tests for Node08: Word2Vec (2013)"""
import subprocess
import sys
import os
import pickle
import numpy as np
import pytest

NODE_DIR = os.path.join(os.path.dirname(__file__), "..", "nodes", "08-word2vec-2013")


# ── 文件存在性 ──

def test_readme_exists():
    assert os.path.isfile(os.path.join(NODE_DIR, "README.md"))


def test_notebook_exists():
    assert os.path.isfile(os.path.join(NODE_DIR, "word2vec.ipynb"))


def test_references_bib_exists():
    assert os.path.isfile(os.path.join(NODE_DIR, "references.bib"))


# ── README 内容检查 ──

def get_readme():
    with open(os.path.join(NODE_DIR, "README.md"), encoding="utf-8") as f:
        return f.read()


def test_readme_mentions_mikolov():
    assert "Mikolov" in get_readme()


def test_readme_has_arxiv_link_1301():
    assert "1301.3781" in get_readme()


def test_readme_has_arxiv_link_1310():
    assert "1310.4546" in get_readme()


def test_readme_mentions_skip_gram():
    assert "Skip-gram" in get_readme() or "skip-gram" in get_readme()


def test_readme_mentions_negative_sampling():
    content = get_readme()
    assert "负采样" in content or "Negative Sampling" in content


def test_readme_has_limitations_section():
    assert "局限" in get_readme()


def test_readme_has_notebook_link():
    assert "word2vec.ipynb" in get_readme()


# ── references.bib 检查 ──

def get_bib():
    with open(os.path.join(NODE_DIR, "references.bib"), encoding="utf-8") as f:
        return f.read()


def test_bib_has_two_mikolov_entries():
    bib = get_bib()
    assert "1301.3781" in bib
    assert "1310.4546" in bib


def test_bib_has_bengio_background():
    bib = get_bib()
    assert "bengio" in bib.lower()


# ── Notebook 可执行 ──

def test_node08_notebook_executes():
    result = subprocess.run(
        [sys.executable, "tools/notebook-run", "nodes/08-word2vec-2013/word2vec.ipynb"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    assert result.returncode == 0, f"Notebook failed:\n{result.stdout}\n{result.stderr}"


# ── 语义相似性集成测试（从 notebook 导出的词向量） ──

def _load_vectors():
    """加载 notebook 训练后保存的词向量文件。"""
    W_path = os.path.join(NODE_DIR, "word_vectors.npy")
    idx_path = os.path.join(NODE_DIR, "word2idx.pkl")
    assert os.path.isfile(W_path), "缺少 word_vectors.npy，请先运行 notebook"
    assert os.path.isfile(idx_path), "缺少 word2idx.pkl，请先运行 notebook"
    W = np.load(W_path)
    with open(idx_path, "rb") as f:
        word2idx = pickle.load(f)
    return W, word2idx


def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def test_node08_nearest_neighbor_bird():
    """
    '鸟'的最近邻必须是同类动物（猫或狗），而非动词或食物。

    独立性保证：notebook cell-20 只打印了 most_similar('猫', '鱼', '吃')，
    未打印 '鸟' 的最近邻。cell-17 打印了特定词对但不含 '鸟' vs 其他动物的比较。
    cell-18 断言 sim(猫,狗) > sim(猫,吃)，与 '鸟' 无关。
    因此本测试断言的属性在 notebook 任何 cell 都没有被打印或断言过，
    能真正独立捕捉动物词聚类方向的回归。
    """
    W, word2idx = _load_vectors()
    bird_idx = word2idx["鸟"]
    scores = {
        word: _cosine_sim(W[bird_idx], W[idx])
        for word, idx in word2idx.items()
        if word != "鸟"
    }
    top_word = max(scores, key=scores.__getitem__)
    assert top_word in {"猫", "狗"}, (
        f"动物词聚类失败：'鸟'的最近邻应为同类动物(猫/狗)，"
        f"实际为 '{top_word}'（相似度 {scores[top_word]:.4f}）。"
        "可能梯度方向或词向量训练存在问题。"
    )
