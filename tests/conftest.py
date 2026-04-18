"""Shared pytest fixtures for llm-learning tool tests."""
import importlib.util
import importlib.machinery
from pathlib import Path
import pytest

TOOLS_DIR = Path(__file__).parent.parent / "tools"


def load_tool(name: str):
    """Import a tool script (no .py extension) as a module."""
    path = TOOLS_DIR / name
    mod_name = name.replace("-", "_")
    loader = importlib.machinery.SourceFileLoader(mod_name, str(path.absolute()))
    spec = importlib.util.spec_from_file_location(mod_name, path.absolute(), loader=loader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def cite_verify():
    return load_tool("cite-verify")


@pytest.fixture(scope="session")
def depth_score():
    return load_tool("depth-score")


@pytest.fixture
def tmp_bib(tmp_path):
    """Write a minimal .bib file and return its Path."""
    def _make(content: str) -> Path:
        p = tmp_path / "test.bib"
        p.write_text(content, encoding="utf-8")
        return p
    return _make


@pytest.fixture
def tmp_md(tmp_path):
    """Write a markdown file and return its Path."""
    def _make(content: str) -> Path:
        p = tmp_path / "test.md"
        p.write_text(content, encoding="utf-8")
        return p
    return _make
