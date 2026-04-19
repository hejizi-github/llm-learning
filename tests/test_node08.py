"""Tests for Node08: Word2Vec (2013)"""
import subprocess
import sys
import os
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
