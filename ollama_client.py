"""Minimal Ollama client that forces a small local model to return structured JSON.

Designed for a CPU-only host where the inference server may not be running at all.
Every failure mode (connection refused, timeout, malformed output, out-of-range
value) collapses into ``None`` so the caller can drop that single input and keep
going instead of crashing.
"""
import json

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"
TIMEOUT_SEC = 60  # the first request also pays for loading the model into memory
VALID_CATEGORIES = {"animal", "food", "nature", "object"}

PROMPT = """You convert a single Japanese word into image-generation input.

Return an English descriptive phrase and one category.
Categories: animal / food / nature / object

Examples:
「にじ」→ {{"en": "a colorful rainbow", "category": "nature"}}
「いぬ」→ {{"en": "a friendly puppy dog", "category": "animal"}}
「ケーキ」→ {{"en": "a strawberry shortcake", "category": "food"}}
「いす」→ {{"en": "a wooden chair", "category": "object"}}

Word: 「{word}」

Output JSON only, in the same format as the examples (no explanation):"""


def classify(word: str, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL,
             timeout: int = TIMEOUT_SEC):
    """Return ``{"en": ..., "category": ...}`` or ``None`` if anything goes wrong."""
    word = (word or "").strip()
    if not word:
        return None
    payload = {
        "model": model,
        "prompt": PROMPT.format(word=word),
        "stream": False,
        # temperature=0 for reproducibility, num_predict to cap CPU inference time
        "options": {"temperature": 0, "num_predict": 80},
    }
    try:
        res = requests.post(url, json=payload, timeout=timeout)
        res.raise_for_status()
        raw = res.json()["response"].strip()
        obj = json.loads(extract_json(raw))
    except Exception as e:  # connection refused / timeout / unparsable output
        print(f"[ollama_client] failed word={word!r}: {e}")
        return None

    en = str(obj.get("en", "")).strip()
    category = obj.get("category")
    if not en or category not in VALID_CATEGORIES:
        print(f"[ollama_client] rejected word={word!r}: {obj!r}")
        return None
    return {"en": en, "category": category}


def extract_json(raw: str) -> str:
    """Strip code fences and slice out the outermost JSON object.

    Small models frequently wrap their answer in ```json fences or prepend a
    sentence, so the raw response cannot be handed to json.loads directly.
    """
    raw = raw.replace("```json", "").replace("```", "").strip()
    return raw[raw.index("{"): raw.rindex("}") + 1]
