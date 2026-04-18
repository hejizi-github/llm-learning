"""Tests for tools/cite-verify — focuses on check_readme_references logic.

We patch check_url to avoid real HTTP requests so tests run offline.
"""
import sys
import os
import re
import types
import importlib
import unittest.mock as mock

# Load cite-verify as a module (it has no .py extension)
_cv_path = os.path.join(os.path.dirname(__file__), '..', 'tools', 'cite-verify')
spec = importlib.util.spec_from_loader(
    'cite_verify',
    importlib.machinery.SourceFileLoader('cite_verify', os.path.abspath(_cv_path))
)
cite_verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cite_verify)


def _run_readme_section(lines: list[str], url_ok: bool = True) -> tuple:
    """Write lines into a temp README and call check_readme_references."""
    import tempfile, textwrap
    content = "## 参考文献\n" + "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(content)
        path = f.name
    try:
        with mock.patch.object(cite_verify, 'check_url', return_value=url_ok):
            return cite_verify.check_readme_references(path)
    finally:
        os.unlink(path)


# ── P1: DOI regex should NOT capture trailing punctuation ─────────────────────

def test_doi_trailing_comma_not_captured():
    """DOI: 10.1038/323533a0, — comma must be stripped from URL."""
    captured_urls = []
    def fake_check_url(url, label=""):
        captured_urls.append(url)
        return True

    import tempfile
    lines = ["- LeCun 1989, DOI: 10.1038/323533a0, Nature."]
    content = "## 参考文献\n" + "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(content)
        path = f.name
    try:
        with mock.patch.object(cite_verify, 'check_url', side_effect=fake_check_url):
            unverifiable, url_results, total = cite_verify.check_readme_references(path)
    finally:
        os.unlink(path)

    assert len(captured_urls) == 1, f"Expected 1 URL check, got {captured_urls}"
    assert not captured_urls[0].endswith(','), f"Comma leaked into URL: {captured_urls[0]}"
    assert captured_urls[0] == "https://doi.org/10.1038/323533a0"


def test_doi_trailing_paren_not_captured():
    """(DOI: 10.1038/323533a0) — closing paren must not appear in URL."""
    captured_urls = []
    def fake_check_url(url, label=""):
        captured_urls.append(url)
        return True

    import tempfile
    lines = ["- Nature paper (DOI: 10.1038/323533a0)"]
    content = "## 参考文献\n" + "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(content)
        path = f.name
    try:
        with mock.patch.object(cite_verify, 'check_url', side_effect=fake_check_url):
            unverifiable, url_results, total = cite_verify.check_readme_references(path)
    finally:
        os.unlink(path)

    assert len(captured_urls) == 1, f"Expected 1 check, got {captured_urls}"
    assert not captured_urls[0].endswith(')'), f"Paren leaked into URL: {captured_urls[0]}"
    assert captured_urls[0] == "https://doi.org/10.1038/323533a0"


def test_doi_trailing_semicolon_not_captured():
    """DOI: 10.1038/323533a0; — semicolon must not appear in URL."""
    captured_urls = []
    def fake_check_url(url, label=""):
        captured_urls.append(url)
        return True

    import tempfile
    lines = ["- Paper; DOI: 10.1038/323533a0; reviewed 2024"]
    content = "## 参考文献\n" + "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(content)
        path = f.name
    try:
        with mock.patch.object(cite_verify, 'check_url', side_effect=fake_check_url):
            unverifiable, url_results, total = cite_verify.check_readme_references(path)
    finally:
        os.unlink(path)

    assert len(captured_urls) == 1
    assert not captured_urls[0].endswith(';'), f"Semicolon leaked: {captured_urls[0]}"


# ── P2: doi_match=None must go to unverifiable ────────────────────────────────

def test_doi_keyword_no_number_is_unverifiable():
    """DOI: pending — has_doi=True but no 10.xxxx/... pattern → unverifiable."""
    unverifiable, url_results, total = _run_readme_section(
        ["- Author 2024, DOI: pending, Journal."]
    )
    assert total == 1
    assert len(unverifiable) == 1, f"Expected 1 unverifiable, got {unverifiable}"
    assert len(url_results) == 0


def test_doi_keyword_typo_is_unverifiable():
    """DOI: xyz/nope — has_doi=True but regex can't extract → unverifiable."""
    unverifiable, url_results, total = _run_readme_section(
        ["- Bad ref, DOI: xyz/nope"]
    )
    assert total == 1
    assert len(unverifiable) == 1


# ── P2 (arXiv variant): arxiv_match=None must go to unverifiable ─────────────

def test_arxiv_keyword_no_id_is_unverifiable():
    """arXiv: pending — has_arxiv=True but no NNNN.NNNNN pattern → unverifiable."""
    unverifiable, url_results, total = _run_readme_section(
        ["- Some paper, arXiv: pending"]
    )
    assert total == 1
    assert len(unverifiable) == 1, f"Expected 1 unverifiable, got {unverifiable}"
    assert len(url_results) == 0


def test_arxiv_trailing_paren_not_captured():
    """arXiv:1706.03762) — closing paren must not appear in URL."""
    captured_urls = []
    def fake_check_url(url, label=""):
        captured_urls.append(url)
        return True

    import tempfile
    lines = ["- Attention is All You Need (arXiv:1706.03762)"]
    content = "## 参考文献\n" + "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(content)
        path = f.name
    try:
        with mock.patch.object(cite_verify, 'check_url', side_effect=fake_check_url):
            unverifiable, url_results, total = cite_verify.check_readme_references(path)
    finally:
        os.unlink(path)

    # arXiv regex uses capture group so paren is naturally excluded
    assert len(url_results) == 1
    assert len(captured_urls) == 1
    assert not captured_urls[0].endswith(')'), f"Paren in arXiv URL: {captured_urls[0]}"
    assert captured_urls[0] == "https://arxiv.org/abs/1706.03762"


# ── Regression: normal cases still work ──────────────────────────────────────

def test_clean_doi_still_verified():
    """Clean DOI without trailing punctuation continues to be verified."""
    unverifiable, url_results, total = _run_readme_section(
        ["- LeCun 1989, DOI: 10.1038/323533a0"]
    )
    assert total == 1
    assert len(unverifiable) == 0
    assert len(url_results) == 1


def test_no_identifier_is_unverifiable():
    """Reference with no DOI/arXiv/ISBN/URL → unverifiable."""
    unverifiable, url_results, total = _run_readme_section(
        ["- Some author, Some title, Some journal, 2000."]
    )
    assert total == 1
    assert len(unverifiable) == 1
    assert len(url_results) == 0
