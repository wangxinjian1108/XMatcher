"""Tests for scripts/_download.py helpers (no network)."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import pytest


_DL_PATH = Path(__file__).resolve().parents[2] / "scripts" / "_download.py"


@pytest.fixture(scope="module")
def dl():
    """Import scripts/_download.py as a module (it lives outside any package)."""
    spec = importlib.util.spec_from_file_location("xmatcher_download", _DL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["xmatcher_download"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_hf_url_sends_bearer_when_token_set(dl, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok-abc")
    req = dl._build_http_request("https://huggingface.co/foo/resolve/main/x.bin")
    assert req.get_header("Authorization") == "Bearer tok-abc"


def test_hf_url_no_header_when_no_token(dl, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    req = dl._build_http_request("https://huggingface.co/foo/resolve/main/x.bin")
    assert req.get_header("Authorization") is None


def test_hugging_face_hub_token_fallback(dl, monkeypatch):
    """The standard HF env var name is also accepted."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok-xyz")
    req = dl._build_http_request("https://huggingface.co/foo/resolve/main/x.bin")
    assert req.get_header("Authorization") == "Bearer tok-xyz"


def test_non_hf_url_never_gets_token(dl, monkeypatch):
    """Token must not leak to non-HuggingFace hosts."""
    monkeypatch.setenv("HF_TOKEN", "tok-abc")
    req = dl._build_http_request("https://example.com/file.bin")
    assert req.get_header("Authorization") is None


def test_subdomain_of_hf_gets_token(dl, monkeypatch):
    """e.g. cdn-lfs.huggingface.co for LFS-served files should also receive the header."""
    monkeypatch.setenv("HF_TOKEN", "tok-abc")
    req = dl._build_http_request("https://cdn-lfs.huggingface.co/blob.bin")
    assert req.get_header("Authorization") == "Bearer tok-abc"


def test_user_agent_is_always_set(dl, monkeypatch):
    """HF / GitHub releases sometimes return 404 for the default Python-urllib UA."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    for url in (
        "https://huggingface.co/foo/resolve/main/x.bin",
        "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/x.pth",
        "https://example.com/file.bin",
    ):
        req = dl._build_http_request(url)
        ua = req.get_header("User-agent")  # urllib normalizes header names
        assert ua and ua.startswith("xmatcher-"), f"UA missing for {url!r}: {ua!r}"
