"""Tests for node 04 — LeNet-1989."""
import os
import subprocess
import sys
import pytest

NODE04_DIR = os.path.join(os.path.dirname(__file__), '..', 'nodes', '04-lenet')


def test_readme_exists():
    assert os.path.exists(os.path.join(NODE04_DIR, 'README.md'))


def test_notebook_exists():
    assert os.path.exists(os.path.join(NODE04_DIR, 'lenet.ipynb'))


def test_references_bib_exists():
    assert os.path.exists(os.path.join(NODE04_DIR, 'references.bib'))


def test_readme_has_doi_lecun1989():
    with open(os.path.join(NODE04_DIR, 'README.md')) as f:
        content = f.read()
    assert '10.1162/neco.1989.1.4.541' in content, 'LeCun 1989 DOI missing from README'


def test_readme_has_doi_lecun1998():
    with open(os.path.join(NODE04_DIR, 'README.md')) as f:
        content = f.read()
    assert '10.1109/5.726791' in content, 'LeCun 1998 DOI missing from README'


def test_readme_mentions_convolution():
    with open(os.path.join(NODE04_DIR, 'README.md')) as f:
        content = f.read()
    assert '卷积' in content


def test_readme_mentions_pooling():
    with open(os.path.join(NODE04_DIR, 'README.md')) as f:
        content = f.read()
    assert '池化' in content or 'Pooling' in content


def test_readme_has_target_audience_language():
    """README should mention intuitive explanation for beginners."""
    with open(os.path.join(NODE04_DIR, 'README.md')) as f:
        content = f.read()
    # Should have analogies/story, not just formulas
    assert '银行' in content or '邮政' in content or '类比' in content


def test_bib_has_lecun1989():
    with open(os.path.join(NODE04_DIR, 'references.bib')) as f:
        content = f.read()
    assert '10.1162/neco.1989.1.4.541' in content


def test_notebook_runs():
    result = subprocess.run(
        [sys.executable, 'tools/notebook-run', 'nodes/04-lenet/lenet.ipynb'],
        capture_output=True, text=True,
        cwd=os.path.join(os.path.dirname(__file__), '..')
    )
    assert result.returncode == 0, f'Notebook failed:\n{result.stdout}\n{result.stderr}'
