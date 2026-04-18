"""Tests for tools/cite-verify — parse_bib logic (no network calls)."""
import pytest


SAMPLE_BIB = """
@article{rosenblatt1958,
  author    = {Rosenblatt, Frank},
  title     = {The perceptron: A probabilistic model},
  journal   = {Psychological Review},
  year      = {1958},
  doi       = {10.1037/h0042519},
}

@book{minsky1969,
  author    = {Minsky, Marvin and Papert, Seymour},
  title     = {Perceptrons},
  publisher = {MIT Press},
  year      = {1969},
  isbn      = {9780262630702},
}

@article{rumelhart1986,
  author    = {Rumelhart, David E. and Hinton, Geoffrey E. and Williams, Ronald J.},
  title     = {Learning representations by back-propagating errors},
  journal   = {Nature},
  year      = {1986},
  doi       = {10.1038/323533a0},
}
"""


def test_parse_bib_entry_count(cite_verify, tmp_bib):
    """parse_bib should return one entry per @article/@book block."""
    bib = tmp_bib(SAMPLE_BIB)
    entries = cite_verify.parse_bib(bib)
    assert len(entries) == 3


def test_parse_bib_doi_extracted(cite_verify, tmp_bib):
    """parse_bib should extract doi field correctly."""
    bib = tmp_bib(SAMPLE_BIB)
    entries = cite_verify.parse_bib(bib)
    by_key = {e["key"]: e for e in entries}
    assert by_key["rosenblatt1958"]["doi"] == "10.1037/h0042519"


def test_parse_bib_isbn_extracted(cite_verify, tmp_bib):
    """parse_bib should extract isbn for @book entries."""
    bib = tmp_bib(SAMPLE_BIB)
    entries = cite_verify.parse_bib(bib)
    by_key = {e["key"]: e for e in entries}
    assert by_key["minsky1969"]["isbn"] == "9780262630702"


def test_parse_bib_year_extracted(cite_verify, tmp_bib):
    """parse_bib should extract year field as string."""
    bib = tmp_bib(SAMPLE_BIB)
    entries = cite_verify.parse_bib(bib)
    by_key = {e["key"]: e for e in entries}
    assert by_key["rumelhart1986"]["year"] == "1986"


def test_parse_bib_empty_file(cite_verify, tmp_bib):
    """parse_bib on empty file should return empty list."""
    bib = tmp_bib("")
    entries = cite_verify.parse_bib(bib)
    assert entries == []


def test_isbn_clean_logic():
    """ISBN cleaning (strip hyphens/spaces) should produce 13-digit string."""
    raw = "978-0-262-63070-2"
    clean = raw.replace("-", "").replace(" ", "")
    assert clean == "9780262630702"
    assert len(clean) == 13
