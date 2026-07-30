#!/usr/bin/env python3
"""Benchmark quantized models served by Ollama on a CPU-only machine.

Measures, under identical prompts and options:
  - latency per request (mean / min / max)
  - classification accuracy on words whose expected category is known
  - output for words with no expected answer, printed for manual review

Usage:
    ollama serve                 # in another terminal
    python3 benchmark.py llama3.2:3b
"""
import sys
import time

from ollama_client import DEFAULT_MODEL, classify

# Words whose category is unambiguous, used to score the model.
LABELLED = {
    "たいよう": "nature",
    "うさぎ": "animal",
    "おうち": "object",
    "クレヨン": "object",
    "うみ": "nature",
    "ちょうちょ": "animal",
}

# Words deliberately left unlabelled: the point is whether the model produces a
# sensible English phrase at all, which only a human can judge.
UNLABELLED = ["うちゅうじん", "かいじゅう", "おばけ", "にんじゃ"]

TARGET_SEC = 10.0


def timed(word: str, model: str):
    t0 = time.time()
    result = classify(word, model=model)
    return result, time.time() - t0


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"model: {model}\n")

    # The first request also loads the model into memory (~20s on this machine),
    # which says nothing about steady-state latency. Warm up and discard it.
    _, warmup_sec = timed("ねこ", model)
    print(f"warm-up (excluded): {warmup_sec:.1f}s\n")

    times = []

    print("== labelled words (scored) ==")
    correct = 0
    for word, expected in LABELLED.items():
        result, sec = timed(word, model)
        times.append(sec)
        got = result["category"] if result else "FAILED"
        if got == expected:
            correct += 1
        mark = "ok " if got == expected else "NG "
        en = result["en"] if result else "-"
        print(f"{mark} {word}: {got} (expected {expected}) / {en} / {sec:.1f}s")
    print(f"\naccuracy: {correct}/{len(LABELLED)}")

    print("\n== unlabelled words (manual review) ==")
    for word in UNLABELLED:
        result, sec = timed(word, model)
        times.append(sec)
        if result:
            print(f" -  {word}: {result['en']} / {result['category']} / {sec:.1f}s")
        else:
            print(f" -  {word}: FAILED / {sec:.1f}s")

    mean = sum(times) / len(times)
    print(f"\n== latency ==")
    print(f"mean {mean:.1f}s / min {min(times):.1f}s / max {max(times):.1f}s "
          f"(target {TARGET_SEC:.0f}s)")
    return 0 if max(times) <= TARGET_SEC else 1


if __name__ == "__main__":
    raise SystemExit(main())
