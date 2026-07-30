"""Unit tests for ollama_client.

The HTTP layer is mocked so the whole suite runs with no inference server
present -- including the case where the server is deliberately unreachable.
"""
import requests

import ollama_client


class FakeResponse:
    def __init__(self, text):
        self._text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self._text}


def _patch_post(monkeypatch, text):
    monkeypatch.setattr(requests, "post", lambda url, **kw: FakeResponse(text))


def test_valid_json_response(monkeypatch):
    _patch_post(monkeypatch, '{"en": "a red apple", "category": "food"}')
    assert ollama_client.classify("りんご") == {"en": "a red apple", "category": "food"}


def test_code_fence_response_is_parsed(monkeypatch):
    _patch_post(monkeypatch, '```json\n{"en": "a black cat", "category": "animal"}\n```')
    assert ollama_client.classify("ねこ") == {"en": "a black cat", "category": "animal"}


def test_invalid_category_returns_none(monkeypatch):
    _patch_post(monkeypatch, '{"en": "a ghost", "category": "spirit"}')
    assert ollama_client.classify("おばけ") is None


def test_empty_en_returns_none(monkeypatch):
    _patch_post(monkeypatch, '{"en": "", "category": "nature"}')
    assert ollama_client.classify("ほし") is None


def test_unparsable_returns_none(monkeypatch):
    _patch_post(monkeypatch, 'Sorry, I do not know that word.')
    assert ollama_client.classify("なぞ") is None


def test_connection_error_returns_none(monkeypatch):
    def boom(url, **kw):
        raise requests.exceptions.ConnectionError("inference server is down")

    monkeypatch.setattr(requests, "post", boom)
    assert ollama_client.classify("うみ") is None


def test_empty_word_returns_none():
    assert ollama_client.classify("") is None
    assert ollama_client.classify("   ") is None
