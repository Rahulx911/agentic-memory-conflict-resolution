"""Tests for the retry-on-rate-limit wrapper in src/memory/embeddings.py.
No network calls: the Voyage client is faked."""

import pytest
from voyageai.error import RateLimitError

from src.memory import embeddings


class _FakeResult:
    def __init__(self, vec: list[float]) -> None:
        self.embeddings = [vec]


def _fake_client(embed_fn):
    return type("_FakeClient", (), {"embed": staticmethod(embed_fn)})()


def test_embed_text_retries_once_on_rate_limit_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_embed(texts, model, input_type):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("rate limited")
        return _FakeResult([0.1, 0.2])

    monkeypatch.setattr(embeddings, "_get_client", lambda: _fake_client(fake_embed))
    monkeypatch.setattr(embeddings.time, "sleep", lambda seconds: None)

    result = embeddings.embed_text("hello")

    assert result == [0.1, 0.2]
    assert calls["n"] == 2


def test_embed_text_raises_after_exhausting_retries(monkeypatch):
    calls = {"n": 0}

    def fake_embed(texts, model, input_type):
        calls["n"] += 1
        raise RateLimitError("rate limited")

    monkeypatch.setattr(embeddings, "_get_client", lambda: _fake_client(fake_embed))
    monkeypatch.setattr(embeddings.time, "sleep", lambda seconds: None)

    with pytest.raises(RateLimitError):
        embeddings.embed_text("hello")

    assert calls["n"] == embeddings.RATE_LIMIT_RETRIES + 1


def test_embed_text_succeeds_immediately_without_retry(monkeypatch):
    calls = {"n": 0}

    def fake_embed(texts, model, input_type):
        calls["n"] += 1
        return _FakeResult([0.5])

    monkeypatch.setattr(embeddings, "_get_client", lambda: _fake_client(fake_embed))

    result = embeddings.embed_text("hello")

    assert result == [0.5]
    assert calls["n"] == 1
