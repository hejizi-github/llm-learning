"""Tests for Node08: Word2Vec (2013)"""
import subprocess
import sys
import os
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


# ── 数学性质：余弦相似度 ──

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def test_cosine_self_similarity_is_one():
    rng = np.random.RandomState(0)
    for _ in range(10):
        v = rng.randn(8)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_range_minus_one_to_one():
    rng = np.random.RandomState(1)
    for _ in range(50):
        a = rng.randn(8)
        b = rng.randn(8)
        cs = cosine_similarity(a, b)
        assert -1.0 - 1e-6 <= cs <= 1.0 + 1e-6


def test_cosine_symmetric():
    rng = np.random.RandomState(2)
    a = rng.randn(8)
    b = rng.randn(8)
    assert abs(cosine_similarity(a, b) - cosine_similarity(b, a)) < 1e-12


def test_cosine_opposite_vectors_is_minus_one():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, -v) - (-1.0)) < 1e-6


def test_cosine_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-9


# ── 数学性质：Sigmoid ──

def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def test_sigmoid_at_zero():
    assert abs(sigmoid(0) - 0.5) < 1e-9


def test_sigmoid_output_in_zero_one():
    xs = np.linspace(-10, 10, 200)
    outs = sigmoid(xs)
    assert np.all(outs > 0) and np.all(outs < 1)


def test_sigmoid_monotone():
    xs = np.linspace(-5, 5, 100)
    outs = sigmoid(xs)
    assert np.all(np.diff(outs) > 0)


# ── Notebook 可执行 ──

def test_node08_notebook_executes():
    result = subprocess.run(
        [sys.executable, "tools/notebook-run", "nodes/08-word2vec-2013/word2vec.ipynb"],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    assert result.returncode == 0, f"Notebook failed:\n{result.stdout}\n{result.stderr}"
